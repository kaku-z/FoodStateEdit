"""Relative depth preserves geometry order without clipped foreground."""
import unittest
import numpy as np

from scripts.prepare_mld4_head_depth import normalize_depth


class HeadDepthTests(unittest.TestCase):
    def test_monotonic_foreground_and_overlapping_scene_depth(self):
        depth = np.array([[.5, 1., 1.5, 2., 3.], [1.5, np.nan, 0., -1., 2.5]], np.float32)
        foreground = np.zeros_like(depth, bool)
        foreground[0, 1:4] = True
        control, valid, foreground, record = normalize_depth(depth, np.ones_like(depth, bool), foreground)
        self.assertEqual(control[0, 1], 245)
        self.assertEqual(control[0, 3], 70)
        self.assertEqual(control[0, 2], control[1, 0])
        self.assertEqual(control[0, 0], 255)
        self.assertEqual(control[0, 4], 0)
        self.assertLessEqual(control[1, 4], 60)
        self.assertTrue(np.all(np.diff(control[valid][np.argsort(depth[valid])]) <= 0))
        self.assertFalse(valid[1, 1:4].any())
        self.assertTrue(np.all(control[~valid] == 127))

    def test_constant_foreground_is_explicit_and_finite(self):
        depth = np.ones((2, 3), np.float32)
        control, valid, foreground, record = normalize_depth(depth, depth > 0, depth > 0)
        self.assertTrue(record['degenerate_foreground_range'])
        self.assertTrue(np.isfinite(control).all())
        self.assertTrue(np.all(control == 157.5))


if __name__ == '__main__':
    unittest.main()
