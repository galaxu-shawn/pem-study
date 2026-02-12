#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for ThetaPhiSweepGenerator class.

This module contains comprehensive unit tests for the ThetaPhiSweepGenerator class,
testing initialization, property setters/getters, domain generation, theta/phi value
generation, edge cases, and error handling.
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path to import pem_utilities
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pem_utilities.theta_phi_sweep_generator import ThetaPhiSweepGenerator


class TestThetaPhiSweepGenerator(unittest.TestCase):
    """Unit tests for ThetaPhiSweepGenerator class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.generator = ThetaPhiSweepGenerator()
        # Ensure clean state
        if hasattr(self.generator, 'all_theta_phi_vals'):
            delattr(self.generator, 'all_theta_phi_vals')

    def test_default_initialization(self):
        """Test the default initialization of ThetaPhiSweepGenerator."""
        # Test default values
        self.assertEqual(self.generator.theta_start, 0)
        self.assertEqual(self.generator.theta_stop, 180)
        self.assertEqual(self.generator.theta_step_deg, 1)
        self.assertEqual(self.generator.theta_step_num, 181)
        
        self.assertEqual(self.generator.phi_start, 0)
        self.assertEqual(self.generator.phi_stop, 360)
        self.assertEqual(self.generator.phi_step_deg, 1)
        self.assertEqual(self.generator.phi_step_num, 361)
        
        # Test domain properties
        theta_domain = self.generator.theta_domain
        phi_domain = self.generator.phi_domain
        
        self.assertEqual(theta_domain.shape, (181,))
        self.assertAlmostEqual(theta_domain[0], 0.0)
        self.assertAlmostEqual(theta_domain[-1], 180.0)
        
        self.assertEqual(phi_domain.shape, (361,))
        self.assertAlmostEqual(phi_domain[0], 0.0)
        self.assertAlmostEqual(phi_domain[-1], 360.0)

    def test_custom_ranges(self):
        """Test setting custom theta and phi ranges."""
        # Set custom ranges
        self.generator.theta_start = 30
        self.generator.theta_stop = 150
        self.generator.theta_step_deg = 10
        
        self.generator.phi_start = -180
        self.generator.phi_stop = 180
        self.generator.phi_step_deg = 15
        
        # Verify settings
        self.assertEqual(self.generator.theta_start, 30.0)
        self.assertEqual(self.generator.theta_stop, 150.0)
        self.assertEqual(self.generator.theta_step_deg, 10.0)
        self.assertEqual(self.generator.theta_step_num, 13)
        
        self.assertEqual(self.generator.phi_start, -180.0)
        self.assertEqual(self.generator.phi_stop, 180.0)
        self.assertEqual(self.generator.phi_step_deg, 15.0)
        self.assertEqual(self.generator.phi_step_num, 25)
        
        # Test domains
        theta_domain = self.generator.theta_domain
        phi_domain = self.generator.phi_domain
        
        np.testing.assert_array_almost_equal(theta_domain[:3], [30., 40., 50.])
        np.testing.assert_array_almost_equal(theta_domain[-3:], [130., 140., 150.])
        np.testing.assert_array_almost_equal(phi_domain[:3], [-180., -165., -150.])
        np.testing.assert_array_almost_equal(phi_domain[-3:], [150., 165., 180.])

    def test_step_num_setting(self):
        """Test setting ranges by number of steps."""
        # Set ranges by number of steps
        self.generator.theta_start = 0
        self.generator.theta_stop = 90
        self.generator.theta_step_num = 10
        
        self.generator.phi_start = 0
        self.generator.phi_stop = 180
        self.generator.phi_step_num = 7
        
        # Verify calculated step sizes - based on actual implementation
        # theta_step_deg = (stop - start) / step_num
        self.assertAlmostEqual(self.generator.theta_step_deg, 90.0/(10-1))  # 10.0
        # phi_step_deg = (stop - start) / step_num  
        self.assertAlmostEqual(self.generator.phi_step_deg, 180.0/(7-1), places=2)  # ~25.71
        
        # Test domains
        theta_domain = self.generator.theta_domain
        phi_domain = self.generator.phi_domain
        
        expected_theta = np.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 90.])
        expected_phi = np.array([0., 30., 60., 90., 120., 150., 180.])
        
        np.testing.assert_array_almost_equal(theta_domain, expected_theta)
        np.testing.assert_array_almost_equal(phi_domain, expected_phi)

    def test_get_all_theta_phi_vals(self):
        """Test the get_all_theta_phi_vals method."""
        # Set up a small test case
        self.generator.theta_start = 0
        self.generator.theta_stop = 60
        self.generator.theta_step_num = 4  # 0, 20, 40, 60
        
        self.generator.phi_start = 0
        self.generator.phi_stop = 90
        self.generator.phi_step_num = 4  # 0, 30, 60, 90
        
        # Access domains to ensure they are updated
        _ = self.generator.theta_domain
        _ = self.generator.phi_domain
        
        # Generate all combinations
        self.generator.get_all_theta_phi_vals()
        
        # Verify results
        expected_combinations = 4 * 4  # theta_steps * phi_steps
        self.assertEqual(len(self.generator.all_theta_phi_vals), expected_combinations)
        
        # Test some specific combinations
        self.assertEqual(self.generator.all_theta_phi_vals[0].tolist(), [0.0, 0.0])  # First combination
        self.assertEqual(self.generator.all_theta_phi_vals[1].tolist(), [0.0, 30.0])  # Second combination
        self.assertEqual(self.generator.all_theta_phi_vals[-1].tolist(), [60.0, 90.0])  # Last combination
        
        # Verify shape
        self.assertEqual(self.generator.all_theta_phi_vals.shape, (16, 2))

    def test_edge_cases(self):
        """Test edge cases and special scenarios."""
        # Test case 1: Single theta point
        self.generator.theta_start = 45
        self.generator.theta_stop = 45
        self.generator.theta_step_num = 1
        
        self.generator.phi_start = 0
        self.generator.phi_stop = 180
        self.generator.phi_step_num = 5
        
        # Access domains to ensure they are updated
        theta_domain = self.generator.theta_domain
        phi_domain = self.generator.phi_domain
        
        self.assertEqual(len(theta_domain), 1)
        self.assertEqual(theta_domain[0], 45.0)
        self.assertEqual(len(phi_domain), 5)
        
        self.generator.get_all_theta_phi_vals()
        self.assertEqual(len(self.generator.all_theta_phi_vals), 5)
        
        # All theta values should be 45
        for combination in self.generator.all_theta_phi_vals:
            self.assertEqual(combination[0], 45.0)
        
        # Test case 2: Single phi point
        self.generator.theta_start = 0
        self.generator.theta_stop = 30
        self.generator.theta_step_num = 3
        
        self.generator.phi_start = 90
        self.generator.phi_stop = 90
        self.generator.phi_step_num = 1
        
        # Access domains to ensure they are updated
        _ = self.generator.theta_domain
        _ = self.generator.phi_domain
        
        self.generator.get_all_theta_phi_vals()
        self.assertEqual(len(self.generator.all_theta_phi_vals), 3)
        
        # All phi values should be 90
        for combination in self.generator.all_theta_phi_vals:
            self.assertEqual(combination[1], 90.0)

    def test_theta_validation(self):
        """Test theta parameter validation."""
        # Test negative theta start
        with self.assertRaises(ValueError):
            self.generator.theta_start = -10
        
        # Test theta start > 180
        with self.assertRaises(ValueError):
            self.generator.theta_start = 200
        
        # Test negative theta stop
        with self.assertRaises(ValueError):
            self.generator.theta_stop = -5
        
        # Test theta stop > 180
        with self.assertRaises(ValueError):
            self.generator.theta_stop = 190
        
        # Test negative theta step
        with self.assertRaises(ValueError):
            self.generator.theta_step_deg = -1
        
        # Test zero theta step num
        with self.assertRaises(ValueError):
            self.generator.theta_step_num = 0
        
        # Test negative theta step num
        with self.assertRaises(ValueError):
            self.generator.theta_step_num = -5

    def test_phi_validation(self):
        """Test phi parameter validation."""
        # Test phi start > 360
        with self.assertRaises(ValueError):
            self.generator.phi_start = 400
        
        # Test phi start < -360
        with self.assertRaises(ValueError):
            self.generator.phi_start = -400
        
        # Test phi stop > 360
        with self.assertRaises(ValueError):
            self.generator.phi_stop = 380
        
        # Test phi stop < -360
        with self.assertRaises(ValueError):
            self.generator.phi_stop = -380
        
        # Test zero phi step num
        with self.assertRaises(ValueError):
            self.generator.phi_step_num = 0

    def test_type_validation(self):
        """Test type validation for parameters."""
        # Test non-numeric theta start
        with self.assertRaises(TypeError):
            self.generator.theta_start = "not a number"
        
        # Test non-numeric phi start
        with self.assertRaises(TypeError):
            self.generator.phi_start = "not a number"
        
        # Test non-integer step numbers
        with self.assertRaises(TypeError):
            self.generator.theta_step_num = 5.5
        
        with self.assertRaises(TypeError):
            self.generator.phi_step_num = 3.7

    def test_realistic_radar_scenario(self):
        """Test a realistic radar scanning scenario."""
        # Typical radar scenario: elevation scan from 0-90°, full azimuth scan
        self.generator.theta_start = 0      # Elevation from horizon
        self.generator.theta_stop = 90      # to zenith
        self.generator.theta_step_deg = 2   # 2-degree steps
        
        self.generator.phi_start = -180     # Full azimuth coverage
        self.generator.phi_stop = 180
        self.generator.phi_step_deg = 5     # 5-degree steps
        
        # Verify setup
        self.assertEqual(self.generator.theta_step_num, 46)  # (90-0)/2 + 1
        self.assertEqual(self.generator.phi_step_num, 73)    # (180-(-180))/5 + 1
        
        # Access domains to ensure they are updated
        _ = self.generator.theta_domain
        _ = self.generator.phi_domain
        
        # Generate all positions
        self.generator.get_all_theta_phi_vals()
        
        expected_total = 46 * 73
        self.assertEqual(len(self.generator.all_theta_phi_vals), expected_total)
        
        # Test first and last positions
        first_position = self.generator.all_theta_phi_vals[0]
        last_position = self.generator.all_theta_phi_vals[-1]
        
        # First position should be (0, -180) or close to it
        self.assertAlmostEqual(first_position[0], 0.0)
        # Last position should be (90, 180) or close to it  
        self.assertAlmostEqual(last_position[0], 90.0)

    def test_domain_property_recalculation(self):
        """Test that domains are recalculated when parameters change."""
        # Set initial values
        self.generator.theta_start = 0
        self.generator.theta_stop = 90
        self.generator.theta_step_num = 10
        
        initial_domain = self.generator.theta_domain.copy()
        
        # Change parameters
        self.generator.theta_step_num = 5
        new_domain = self.generator.theta_domain
        
        # Domains should be different
        self.assertFalse(np.array_equal(initial_domain, new_domain))
        self.assertEqual(len(new_domain), 5)

    def test_meshgrid_functionality(self):
        """Test that the meshgrid functionality works correctly."""
        # Set up a simple 2x3 grid
        self.generator.theta_start = 0
        self.generator.theta_stop = 30
        self.generator.theta_step_num = 2  # [0, 30]
        
        self.generator.phi_start = 0
        self.generator.phi_stop = 60
        self.generator.phi_step_num = 3  # [0, 30, 60]
        
        # Access domains to ensure they are updated
        _ = self.generator.theta_domain
        _ = self.generator.phi_domain
        
        self.generator.get_all_theta_phi_vals()
        
        # Should have 2*3 = 6 combinations
        self.assertEqual(len(self.generator.all_theta_phi_vals), 6)
        
        # Check specific combinations (should be ordered by theta first, then phi)
        expected_combinations = [
            [0., 0.], [0., 30.], [0., 60.],
            [30., 0.], [30., 30.], [30., 60.]
        ]
        
        for i, expected in enumerate(expected_combinations):
            np.testing.assert_array_almost_equal(
                self.generator.all_theta_phi_vals[i], expected
            )


class TestThetaPhiSweepGeneratorIntegration(unittest.TestCase):
    """Integration tests for ThetaPhiSweepGenerator class."""

    def test_full_workflow(self):
        """Test a complete workflow from initialization to value generation."""
        generator = ThetaPhiSweepGenerator()
        
        # Configure for a small but realistic scan
        generator.theta_start = 10
        generator.theta_stop = 50
        generator.theta_step_deg = 20  # Should give 3 steps: 10, 30, 50
        
        generator.phi_start = -90
        generator.phi_stop = 90
        generator.phi_step_deg = 90   # Should give 3 steps: -90, 0, 90
        
        # Access domains to ensure they are updated
        _ = generator.theta_domain
        _ = generator.phi_domain
        
        # Generate all combinations
        generator.get_all_theta_phi_vals()
        
        # Should have 3*3 = 9 combinations
        self.assertEqual(len(generator.all_theta_phi_vals), 9)
        
        # Verify all values are within expected ranges
        for theta, phi in generator.all_theta_phi_vals:
            self.assertGreaterEqual(theta, 10)
            self.assertLessEqual(theta, 50)
            self.assertGreaterEqual(phi, -90)
            self.assertLessEqual(phi, 90)


if __name__ == '__main__':
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestThetaPhiSweepGenerator)
    integration_suite = unittest.TestLoader().loadTestsFromTestCase(TestThetaPhiSweepGeneratorIntegration)
    
    # Combine test suites
    combined_suite = unittest.TestSuite([suite, integration_suite])
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(combined_suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)