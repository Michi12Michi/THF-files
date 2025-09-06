from thf_constants import (ATOMIC_MASSES, NAP_ATOMS, THF_ATOMS, TOTAL_SNAPSHOT_ATOMS, THF_CHARGES,)
from collections import deque
import mmap
import numpy as np
import os
import re
import sys
from time import perf_counter

''' Importing DOS module. '''
path_to_DOSClass = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "..", 'Dinamiche', 'CBB\\',))
sys.path.append(path_to_DOSClass)
try:
    from DOSclass import DensityOfStates, create_graph
    from constants import ELEC_ANGS_TO_DEBYE
except Exception as e:
    print(e)

ATOM_PATTERN = re.compile(r"([A-Za-z]+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)")

class Molecule:
    ''' Object provided with generic properties, suitable for both solvent molecules and solute molecule. '''
    def __init__(self, atoms: list[tuple],) -> None:
        ''' Initializes all the properties of a generic molecule, given the coordinates and nature of its atoms.
                atoms is a list of tuples with format (element_symbol: str, np.array). '''
        assert len(atoms) > 0
        self.atoms_list: list[tuple] = atoms
        self.total_mass: float = sum([ATOMIC_MASSES[element] for element, _ in self.atoms_list])   
        self.center_of_mass: np.array = (sum([ATOMIC_MASSES[element]*position for element, position in self.atoms_list]))/(self.total_mass)
        self.dipole_moment: np.array = np.array([0.0, 0.0, 0.0])
        
    def calculate_distance(self, other) -> float:
        ''' Calculates the distance between the centers of mass of two distinct molecules. '''
        return np.linalg.norm(self.center_of_mass - other.center_of_mass)

    def clear(self) -> None:
        ''' Clears all the molecule's properties. '''
        self.atoms_list.clear()
        self.total_mass = 0.0
        self.center_of_mass = np.array([0.0, 0.0, 0,0])
        self.dipole_moment = np.array([0.0, 0.0, 0.0])

class Solute(Molecule):
    def __init__(self, atoms: list[tuple],):
        super().__init__(atoms)
        self.normal_vector: np.array = self.calculate_normal_vector()
        self.best_normal_vector: np.array = self.calculate_best_plane()

    def calculate_normal_vector(self) -> np.array:
        ''' Calculates the Cartesian components of the normal vector with respect to the plane 
            formed by atoms with indices 0, 4 and 9 (corresponding to C2, C3 and C6 of molecule).
        '''
        vector_1: np.array = self.atoms_list[4][1] - self.atoms_list[9][1]
        vector_2: np.array = self.atoms_list[4][1] - self.atoms_list[0][1]
        return np.cross(vector_1, vector_2)
    
    def calculate_best_plane(self) -> np.array:
        ''' Calculates the best-fitting plane for a given group of atoms using the least squares method.
            The calculation is performed by centering the atoms' coordinates around the center of mass (COM),
            ensuring that the COM lies on the resulting plane. The method uses the covariance matrix to determine
            the plane that best fits the distribution of atoms.
        '''
        centered_coordinates = np.array([(i[1] - self.center_of_mass) for i in self.atoms_list])
        covariance_matrix = np.cov(centered_coordinates.T)
        eigvals, eigvecs = np.linalg.eigh(covariance_matrix)
        best_normal_vector = eigvecs[:, np.argmin(eigvals)]
        return best_normal_vector

    def calculate_signed_angle(self, other):
        ''' Given two vectors, the signed angle is given by substracting arccos[(v1 . v2)/(|v1||v2|)] from 90.0 degrees. '''
        sign = np.sign(np.dot(self.best_normal_vector, other.center_of_mass)) # replaced normal_vector
        if sign == 0.0:
            ''' Return None '''
            return None
        return sign*(90 - np.degrees(np.acos((np.dot(self.best_normal_vector, other.dipole_moment))/(np.linalg.norm(self.best_normal_vector)*np.linalg.norm(other.dipole_moment)))))
    
    def clear(self):
        super().clear()
        self.normal_vector = np.array([0.0, 0.0, 0.0])
        self.best_normal_vector = np.array([0.0, 0.0, 0.0])

class Solvent(Molecule):
    def __init__(self, atoms: list[tuple], charges: list):
        super().__init__(atoms)
        self.dipole_moment = self.calculate_dipole_moment(charges)

    def calculate_dipole_moment(self, charges: list):
        ''' Calculates the dipole moment vector using given partial atomic charges (thf_constants.THF_CHARGES).
            PLEASE NOTE: for calculation purposes the sign of the vector is INVERTED. '''
        return (sum([charges[index]*position[1]*(-1) for index, position in enumerate(self.atoms_list)]))

#mciao mici
class Dynamics:
    ''' Evaluates for all snapshots the angles between the plane of the solute and dipole moment vectors of the solvent molecules. '''
    def __init__(self, filename: str, cutoff: float, closest_primes: int = 0, require_total_dipoles: bool = False):
        ''' Reads the xyz file and returns a deque of angles values. '''
        self.angles: deque = deque()
        self.total_dipoles: deque = deque()
        self._snapshot_number: int = 0
        self._atoms_read: int = 0
        self.test_atoms_read: int = 0
        self._temp_NAP: list[tuple] = []
        self._temp_THFs: list[tuple] = []
        self._cutoff = cutoff
        self.closest_primes = closest_primes
        self.require_total_dipoles_norm = require_total_dipoles
        # ADDON
        self.temp_file = open("resultsTEST.txt", "w")
        self.temp_angles: list = []
        # END ADDON
        ''' PLEASE NOTE: opens a folder and reads every file in it. '''
        # ADDON 
                #for file in os.listdir(filename):
        for i in range(25000, 50001):
        # END ADDON
            try:
                # ADD ON
                        # with open(os.path.abspath(os.path.join(filename, file)), "rb") as f:
                with open(os.path.abspath(os.path.join(filename, f"frame_centered_{i}.xyz")), "rb") as f:
                    # END ADDON
                    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    start = 0
                    while start < len(mm):
                        end = mm.find(b'\n', start)
                        if end == -1:  
                            line = mm[start:]
                            start = len(mm)
                        else:
                            line = mm[start:end]
                            start = end + 1  
                        line_str = line.decode('utf-8')
                        if re.search(r"THF_e_NAP", line_str):
                            self._snapshot_number += 1
                            self._atoms_read = 0
                            self._temp_NAP.clear()
                            self._temp_THFs.clear()
                            print(f"--- READING SNAPSHOT #{self._snapshot_number} ---")
                        elif atom_coordinates := re.search(ATOM_PATTERN, line_str):
                            self._atoms_read += 1
                            self.test_atoms_read += 1
                            if 1 <= self._atoms_read <= NAP_ATOMS:
                                self._temp_NAP.append(
                                    (
                                        str(atom_coordinates.group(1)), 
                                        np.array([float(atom_coordinates.group(2)), float(atom_coordinates.group(3)), float(atom_coordinates.group(4))])
                                    )
                                )
                            else:
                                self._temp_THFs.append(
                                    (
                                        str(atom_coordinates.group(1)), 
                                        np.array([float(atom_coordinates.group(2)), float(atom_coordinates.group(3)), float(atom_coordinates.group(4))])
                                    )
                                )
                                pass
                        if self._atoms_read == TOTAL_SNAPSHOT_ATOMS:
                            self._atoms_read = 0
                            solute_molecule = Solute(self._temp_NAP)
                            solvent_molecules = [Solvent(self._temp_THFs[i:i+THF_ATOMS], THF_CHARGES) for i in range(0, len(self._temp_THFs), THF_ATOMS)]
                            ''' Saving dipole moments' norms for solvent molecules (if required). '''
                            if self.require_total_dipoles_norm:
                                total_dipole_on_snapshot = ELEC_ANGS_TO_DEBYE*np.linalg.norm(np.sum([solv.dipole_moment for solv in solvent_molecules], axis=0))
                                self.total_dipoles.append(total_dipole_on_snapshot)
                            ''' The delicate moment in which solvent_molecules list is ordered with ascending distance between coms. '''
                            solvent_molecules.sort(key=lambda obj: obj.calculate_distance(solute_molecule))
                            if self.closest_primes > 0:
                                solvent_molecules_to_process = solvent_molecules[:self.closest_primes]
                            else:
                                solvent_molecules_to_process = solvent_molecules
                            ''' Executing all the angular evaluations. '''
                            for molecule in solvent_molecules_to_process:
                                if solute_molecule.calculate_distance(molecule) < self._cutoff:
                                    # ADDON
                                    self.angles.append(solute_molecule.calculate_signed_angle(molecule))
                                    self.temp_angles.append(solute_molecule.calculate_signed_angle(molecule))
                                    # END ADDON

                            ''' Cleaning all the objects. '''
                            # ADDON
                            self.temp_file.write(f"{i}\t")
                            self.temp_file.write(f"{np.mean(self.temp_angles)}\n")
                            self.temp_angles  = []
                            # END ADDON
                            solute_molecule.clear()
                            solvent_molecules.clear()
                            solvent_molecules_to_process.clear()
            except Exception as e:
                print("----- I/O ERROR -----")
                print(f"An error occurred reading file {filename}: {e}")

if __name__ == "__main__":
    tic = perf_counter()
    # ADDON
            # dynamics = Dynamics("md-files\\centered_nap_anione_thf", cutoff=4.2, require_total_dipoles=False)
    dynamics = Dynamics("md-files\\50k_snapshot_nap_neutro_partendo_da_anione", cutoff=20.0, closest_primes=2)
    print("###########################")
    print(f"Read: {dynamics._snapshot_number} snapshots!")
    print()
    angles_for_DOS = deque(filter(lambda x: x is not None, dynamics.angles))
    DOS = DensityOfStates(
        parameter_name="Angles", 
        parameter_vector=angles_for_DOS, 
        parameter_minimum=min(angles_for_DOS),
        parameter_maximum=max(angles_for_DOS),
        # MOD
        # norm_factor=dynamics._snapshot_number,
        norm_factor=dynamics._snapshot_number,
        stepsize=0.5,
        out_file="nap-neutro-e-thf-2-vicino"
    )

    create_graph(
        parameter_vector_1=DOS.interval_vector, 
        density_vector_1=DOS.density_vector, 
        curve_label1="Charged NAP-THF angles DOS",
        x_label="Degrees",
        y_label="DOS",
        output_filename="nap-carico-e-thf-2-vicino",
    )
    toc = perf_counter()
    print(f"Process ended. It took only {toc-tic:.2f} seconds.")