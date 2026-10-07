"""Regression checks for source-removal topology, without inference models."""
import unittest

import numpy as np

from scripts.prepare_mld4_guides import dilate, fill_mask_holes, source_boundary_control


class SourceBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.food = np.zeros((64, 64), bool)
        self.food[10:54, 10:54] = True
        self.yy, self.xx = np.indices(self.food.shape)

    def test_fill_holes_preserves_exterior(self):
        food = self.food.copy()
        food[24:40, 24:40] = False
        filled = fill_mask_holes(food)
        np.testing.assert_array_equal(filled, self.food)

    def test_interior_removal_does_not_create_closed_cavity(self):
        removed = (self.xx-32)**2 + (self.yy-32)**2 <= 64
        control, record = source_boundary_control(self.food, removed,
            dilate(removed, 5), np.zeros_like(removed), 1)
        self.assertFalse(record['removal_reaches_exterior'])
        self.assertFalse(control.any())

    def test_exterior_removal_creates_clipped_open_boundary(self):
        removed = ((self.xx-52)**2 + (self.yy-32)**2 <= 64) & self.food
        unknown = dilate(removed, 5)
        control, record = source_boundary_control(self.food, removed,
            unknown, np.zeros_like(removed), 1)
        self.assertTrue(record['removal_reaches_exterior'])
        self.assertEqual(record['open_boundary_components'], 1)
        self.assertGreater(np.count_nonzero(control), 0)
        self.assertLessEqual(float(control.max()), 160.001)
        self.assertFalse(control[~unknown].any())

    def test_transported_foreground_is_excluded(self):
        removed = ((self.xx-52)**2 + (self.yy-32)**2 <= 64) & self.food
        unknown = dilate(removed, 5)
        control, _ = source_boundary_control(self.food, removed, unknown, unknown, 1)
        self.assertFalse(control.any())


if __name__ == '__main__':
    unittest.main()
