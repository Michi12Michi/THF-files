# python -m tests.test

import unittest
import numpy as np
import numpy.testing as npt
from main import (Molecule, Solute, Solvent,)
from thf_constants import THF_CHARGES

''' Real data for NAP. '''
NAP_DATA = [
    ("C", np.array([17.39000, 6.48000, 10.60000])),
    ("C", np.array([18.46000, 7.42000, 10.74000])),
    ("C", np.array([18.67000, 8.07000, 12.05000])),
    ("C", np.array([19.90000, 8.80000, 12.38000])),
    ("C", np.array([20.04000, 9.31000, 13.73000])),
    ("H", np.array([17.31000, 5.78000, 9.70000])),
    ("H", np.array([19.10000, 7.57000, 9.87000])),
    ("H", np.array([20.60000, 9.03000, 11.62000])),
    ("H", np.array([20.87000, 9.94000, 14.11000])),
    ("C", np.array([18.94000, 9.14000, 14.62000])),
    ("C", np.array([17.75000, 8.49000, 14.24000])),
    ("C", np.array([17.60000, 7.88000, 13.03000])),
    ("C", np.array([16.49000, 7.02000, 12.78000])),
    ("C", np.array([16.43000, 6.29000, 11.62000])),
    ("H", np.array([19.05000, 9.53000, 15.65000])),
    ("H", np.array([17.00000, 8.38000, 14.97000])),
    ("H", np.array([15.91000, 6.77000, 13.64000])),
    ("H", np.array([15.63000, 5.65000, 11.44000])),
]

''' Real data for THF molecule. '''

THF_DATA = [
    ("O", np.array([0.31000, 14.78000, 3.72000])),
    ("C", np.array([-0.83000, 14.54000, 4.46000])),
    ("C", np.array([-0.29000, 13.82000, 5.72000])),
    ("C", np.array([1.15000, 14.36000, 5.89000])),
    ("C", np.array([1.55000, 14.39000, 4.42000])),
    ("H", np.array([-1.50000, 13.98000, 3.86000])),
    ("H", np.array([-1.32000, 15.48000, 4.73000])),
    ("H", np.array([-0.29000, 12.73000, 5.68000])),
    ("H", np.array([-0.97000, 13.85000, 6.55000])),
    ("H", np.array([1.72000, 13.68000, 6.60000])),
    ("H", np.array([1.23000, 15.38000, 6.45000])),
    ("H", np.array([1.84000, 13.40000, 4.08000])),
    ("H", np.array([2.29000, 15.12000, 4.13000])),        
]

''' Water and benzene molecule as examples. '''

WATER_EX = [
    ("H", np.array([1.0, -2.0, 0.0])),
    ("H", np.array([-1.0, -2.0, 0.0])),
    ("O", np.array([0.0, 0.0, 0.0])),
]

BENZENE_EX = [
    ("C", np.array([0.0, 1.0, 0.0])),
    ("H", np.array([0.0, 2.0, 0.0])),
    ("C", np.array([0.5, 0.5, 0.0])),
    ("H", np.array([1.0, 1.0, 0.0])),
    ("C", np.array([0.5, -0.5, 0.0])),
    ("H", np.array([1.0, -1.0, 0.0])),
    ("C", np.array([0.0, -1.0, 0.0])),
    ("H", np.array([0.0, -2.0, 0.0])),
    ("C", np.array([-0.5, -0.5, 0.0])),
    ("H", np.array([-1.0, -1.0, 0.0])),
    ("C", np.array([-0.5, 0.5, 0.0])),
    ("H", np.array([-1.0, 1.0, 0.0])),
]

class TestTotalMass(unittest.TestCase):
    ''' Testing the value of the total mass vs the hand-calculated one. '''
    def setUp(self):
        self.real_nap = Solute(NAP_DATA)
        self.real_thf = Solvent(THF_DATA, THF_CHARGES)
        self.water_example = Molecule(WATER_EX)

    def test_total_mass(self):
        self.assertAlmostEqual(self.real_nap.total_mass, 128.174)
        self.assertAlmostEqual(self.real_thf.total_mass, 72.107)
        self.assertAlmostEqual(self.water_example.total_mass, 18.015)

class TestMoleculeCenterOfMass(unittest.TestCase):
    ''' Testing the value of the calculated center of mass vs the hand-calculated one. '''
    def setUp(self):
        self.real_nap = Solute(NAP_DATA)
        self.real_thf = Solvent(THF_DATA, THF_CHARGES)
        self.water_example = Molecule(WATER_EX)

    def test_center_of_mass(self):
        npt.assert_allclose(self.real_nap.center_of_mass, np.array([18.1680538174669, 7.88630377455646, 12.5818940658792]), rtol=1e-9, atol=1e-9)
        npt.assert_allclose(self.real_thf.center_of_mass, np.array([0.373903643196916, 14.3806064598444, 4.82669241543817]), rtol=1e-9, atol=1e-9)
        npt.assert_allclose(self.water_example.center_of_mass, np.array([0.0, -0.223813488759, 0.0]), rtol=1e-9, atol=1e-9)

class TestDistanceCentersOfMass(unittest.TestCase):
    ''' Testing the value of the distance between two centers of mass vs the hand-calculated value. '''
    def setUp(self):
        self.real_nap = Solute(NAP_DATA)
        self.real_thf = Solvent(THF_DATA, THF_CHARGES)

    def test_equal_distances(self):
        self.assertEqual(self.real_nap.calculate_distance(self.real_thf), self.real_thf.calculate_distance(self.real_nap))

    def test_correct_value(self):
        self.assertAlmostEqual(self.real_nap.calculate_distance(self.real_thf), 20.4682901198809)

class TestDipoleMoment(unittest.TestCase):
    ''' Testing the values of calculated dipole moments vs simple and hand-calculated ones. '''
    def setUp(self):
        self.benzene_example = Solvent(BENZENE_EX, charges=[-0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1])
        self.water_example = Solvent(WATER_EX, [1.0, 1.0, -2.0])
        self.real_thf = Solvent(THF_DATA, THF_CHARGES)

    def test_dipole_moment(self):
        npt.assert_allclose(self.benzene_example.dipole_moment, np.array([0.0, 0.0, 0.0]), rtol=1e-9, atol=1e-9)
        npt.assert_allclose(self.water_example.dipole_moment, np.array([0.0, 4.0, 0.0]), rtol=1e-9, atol=1e-9)
        npt.assert_allclose(self.real_thf.dipole_moment, np.array([-0.00328800000000001, 0.18335, -0.390155]), rtol=1e-9, atol=1e-9)

@unittest.SkipTest
class TestNormalVector(unittest.TestCase):
    ''' Testing the functionality of the Solute.calculate_normal_vector method. '''
    def setUp(self):
        self.benzene_example = Solute(BENZENE_EX)
        self.NAP_example = Solute(NAP_DATA)

    def test_benzene_normal_vector(self):
        npt.assert_allclose(self.benzene_example.normal_vector, np.array([0.0, 0.0, -2.5]), rtol=1e-9, atol=1e-9)

    def test_NAP_normal_vector(self):
        npt.assert_allclose(self.NAP_example.normal_vector, np.array([0.0, 0.0, -2.5]), rtol=1e-9, atol=1e-9)

class TestZeroAndNinetyDegreesAngle(unittest.TestCase):
    ''' Testing calculated angles. '''
    def setUp(self):
        self.benzene_example = Solute(BENZENE_EX)
        ''' Cyanide ion parallel (1, 3) and perpendicular (2, 4) to the benzene ring plane. '''
        self.cyanide_example_1 = Solvent([("C", np.array([0.0, 0.0, 2.0])), ("N", np.array([0.0, 2.0, 2.0]))], [1.0, -1.0])
        self.cyanide_example_2 = Solvent([("C", np.array([0.0, 0.0, 2.0])), ("N", np.array([0.0, 0.0, 4.0]))], [1.0, -1.0])
        self.cyanide_example_3 = Solvent([("C", np.array([0.0, 0.0, -2.0])), ("N", np.array([0.0, 2.0, -2.0]))], [1.0, -1.0])
        self.cyanide_example_4 = Solvent([("C", np.array([0.0, 0.0, -2.0])), ("N", np.array([0.0, 0.0, -4.0]))], [1.0, -1.0])
        ''' Cyanide ions perpendicular with dipole moments directed towards the plane. '''
        self.cyanide_example_5 = Solvent([("C", np.array([0.0, 0.0, -4.0])), ("N", np.array([0.0, 0.0, -2.0]))], [1.0, -1.0])
        self.cyanide_example_6 = Solvent([("C", np.array([0.0, 0.0, 4.0])), ("N", np.array([0.0, 0.0, 2.0]))], [1.0, -1.0])

    def test_parallel_molecules(self):
        npt.assert_equal(self.benzene_example.calculate_signed_angle(self.cyanide_example_1), np.float64(-0.0))
        npt.assert_equal(self.benzene_example.calculate_signed_angle(self.cyanide_example_3), np.float64(0.0))

    def test_perpendicular_molecules(self):
        npt.assert_equal(self.benzene_example.calculate_signed_angle(self.cyanide_example_2), np.float64(90.0))
        npt.assert_equal(self.benzene_example.calculate_signed_angle(self.cyanide_example_4), np.float64(90.0))
        npt.assert_equal(self.benzene_example.calculate_signed_angle(self.cyanide_example_5), np.float64(-90.0))
        npt.assert_equal(self.benzene_example.calculate_signed_angle(self.cyanide_example_6), np.float64(-90.0))

class TestOtherDegreesAngle(unittest.TestCase):
    ''' Testing calculated angles (non-0 and non-90). '''
    def setUp(self):
        self.benzene_example = Solute(BENZENE_EX)
        ''' Fantastic ions at 45 degrees, with dipole moment directed towards the plane (1, 2) -> negative angle; 
                                          with dipole moment directed far from the plane (3, 4) -> positive angle. '''
        ''' tan(45) = 1 '''
        self.example_1 = Solvent([("C", np.array([4.0, 0.0, 4.0])), ("C", np.array([2.0, 0.0, 2.0]))], [1.0, -1.0])
        self.example_2 = Solvent([("C", np.array([-4.0, 0.0, -4.0])), ("C", np.array([-2.0, 0.0, -2.0]))], [1.0, -1.0])
        self.example_3 = Solvent([("C", np.array([4.0, 0.0, 4.0])), ("C", np.array([2.0, 0.0, 2.0]))], [-1.0, 1.0])
        self.example_4 = Solvent([("C", np.array([-4.0, 0.0, -4.0])), ("C", np.array([-2.0, 0.0, -2.0]))], [-1.0, 1.0])

        ''' Fantastic ions at 30 degrees, with dipole moment directed towards the plane (5, 6) -> negative angle; 
                                          with dipole moment directed far from the plane (7, 8) -> positive angle. '''
        ''' tan(30) = 0.5774 '''
        self.example_5 = Solvent([("C", np.array([3.0, 0.0, 2.0*0.5774 + 1])), ("N", np.array([1.0, 0.0, 1.0]))], [1.0, -1.0])
        self.example_6 = Solvent([("C", np.array([-3.0, 0.0, -(2.0*0.5774 + 1)])), ("N", np.array([-1.0, 0.0, -1.0]))], [1.0, -1.0])
        self.example_7 = Solvent([("C", np.array([3.0, 0.0, 2.0*0.5774 + 1])), ("N", np.array([1.0, 0.0, 1.0]))], [-1.0, 1.0])
        self.example_8 = Solvent([("C", np.array([-3.0, 0.0, -(2.0*0.5774 + 1)])), ("N", np.array([-1.0, 0.0, -1.0]))], [-1.0, 1.0])
    
    def test_negative_angles(self):
        npt.assert_almost_equal(self.benzene_example.calculate_signed_angle(self.example_1), np.float64(-45.0), decimal=2)
        npt.assert_almost_equal(self.benzene_example.calculate_signed_angle(self.example_2), np.float64(-45.0), decimal=2)
        npt.assert_almost_equal(self.benzene_example.calculate_signed_angle(self.example_5), np.float64(-30.0), decimal=2)
        npt.assert_almost_equal(self.benzene_example.calculate_signed_angle(self.example_6), np.float64(-30.0), decimal=2)

    def test_positive_angles(self):
        npt.assert_almost_equal(self.benzene_example.calculate_signed_angle(self.example_3), np.float64(45.0), decimal=2)
        npt.assert_almost_equal(self.benzene_example.calculate_signed_angle(self.example_4), np.float64(45.0), decimal=2)
        npt.assert_almost_equal(self.benzene_example.calculate_signed_angle(self.example_7), np.float64(30.0), decimal=2)
        npt.assert_almost_equal(self.benzene_example.calculate_signed_angle(self.example_8), np.float64(30.0), decimal=2)

class TestDegreesWhenCenterOfMassIsOnThePlane(unittest.TestCase):
    ''' Testing calculated angles when the solvent center of mass is on the same plane as the solute molecule.
        These tests are essential, since these situations are edge-cases: the function Solute.calculate_signed_angle requires a 
        non-zero scalar product between the normal and the center of mass of the solvent. '''
    def setUp(self):
        self.benzene_example = Solute(BENZENE_EX)
        ''' A fantastic series of anions with the center of mass lying in the plane of the solute molecule: 
            1) Dipole moment vector - along z-axis; 
            2)                      + along z-axis;
            3)                      - along x-axis; 
            4)                      diagonal fashion. '''
        self.example_1 = Solvent([("C", np.array([4.0, 0.0, 2.0])), ("C", np.array([4.0, 0.0, -2.0]))], [1.0, -1.0])
        self.example_2 = Solvent([("C", np.array([4.0, 0.0, 2.0])), ("C", np.array([4.0, 0.0, -2.0]))], [-1.0, 1.0])
        self.example_3 = Solvent([("C", np.array([4.0, 5.0, 0.0])), ("C", np.array([-4.0, 5.0, 0.0]))], [1.0, -1.0])
        self.example_4 = Solvent([("C", np.array([1.0, 1.0, 1.0])), ("C", np.array([-1.0, -1.0, -1.0]))], [1.0, -1.0])

    def test_angles(self):
        npt.assert_equal(self.benzene_example.calculate_signed_angle(self.example_1), None)
        npt.assert_equal(self.benzene_example.calculate_signed_angle(self.example_2), None)
        npt.assert_equal(self.benzene_example.calculate_signed_angle(self.example_3), None)
        npt.assert_equal(self.benzene_example.calculate_signed_angle(self.example_4), None)

@unittest.SkipTest
class TestBestFitPlane(unittest.TestCase):
    def setUp(self):
        self.benzene_example = Solute(BENZENE_EX)

    def test_best_normal_vector(self):
        npt.assert_array_almost_equal(self.benzene_example.normal_vector, self.benzene_example.best_normal_vector)

if __name__ == "__main__":
    unittest.main(verbosity=3)