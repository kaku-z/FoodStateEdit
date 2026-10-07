"""Deformation tests exercise physical invariants, actuation and contact ablations."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_mld3_deformation import soft_block, spoon_for
from foodstateedit.material_lineage.deformable import solve_deformation, support_gaps
from foodstateedit.material_lineage.spoon_surface import surface_height, surface_height_gradient, surface_domain


class MaterialDeformationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vertices, cls.faces = soft_block(17, 13)
        cls.translation = np.array([.10, .08, .80])
        cls.spoon = spoon_for(cls.vertices, cls.translation)
        cls.full, cls.metrics = solve_deformation(cls.vertices, cls.faces, np.eye(3), cls.translation,
                                                 cls.spoon, iterations=140)

    def test_exact_bowl_obstacle_and_actual_volume(self):
        gaps, inside = support_gaps(self.full, self.spoon)
        self.assertGreater(np.mean(inside & (np.abs(gaps) <= .005 * self.metrics["characteristic_length"])), .02)
        self.assertGreaterEqual(gaps[inside].min(), -1.e-12)
        self.assertLess(self.metrics["volume_relative_error"], .02)
        self.assertEqual(self.metrics["face_orientation_reversal_count"], 0)

    def test_deformation_is_not_a_global_translation(self):
        self.assertGreater(self.metrics["normalized_nonrigid_displacement_rms"], 1.e-3)
        self.assertLess(self.metrics["edge_strain_rms"], .15)
        self.assertEqual(self.full.shape, self.vertices.shape)
        self.assertEqual(self.metrics["internal_material_column_count"], len(self.vertices) // 2)

    def test_gravity_and_contact_ablations_share_the_same_source(self):
        no_gravity, m0 = solve_deformation(self.vertices, self.faces, np.eye(3), self.translation,
                                          self.spoon, iterations=140, gravity_strength=0)
        np.testing.assert_allclose(no_gravity, self.vertices + self.translation, atol=1.e-10)
        no_contact, m1 = solve_deformation(self.vertices, self.faces, np.eye(3), self.translation,
                                          self.spoon, iterations=140, contact=False)
        self.assertGreater(m1["penetration_depth_normalized"], .1)
        self.assertEqual(m1["supported_vertex_fraction"], 0.)
        self.assertLess(m1["normalized_nonrigid_displacement_rms"], 1.e-8)
        self.assertGreater(np.linalg.norm(no_contact - self.full), .1)

    def test_cohesive_food_is_stiffer_than_soft_food(self):
        _, cohesive = solve_deformation(self.vertices, self.faces, np.eye(3), self.translation,
                                        self.spoon, iterations=140, compliance=.08)
        self.assertLess(cohesive["edge_strain_rms"], self.metrics["edge_strain_rms"])
        self.assertLess(cohesive["normalized_nonrigid_displacement_rms"], self.metrics["normalized_nonrigid_displacement_rms"])

    def test_coordinate_rotation_does_not_change_material_physics(self):
        a = .37
        q = np.array([[np.cos(a), 0., np.sin(a)], [0., 1., 0.], [-np.sin(a), 0., np.cos(a)]])
        spoon = dict(self.spoon, spoon_axes_camera=q.tolist())
        result, metrics = solve_deformation(self.vertices @ q.T, self.faces, q, self.translation @ q.T,
                                            spoon, iterations=140)
        np.testing.assert_allclose(result, self.full @ q.T, atol=2.e-8)
        self.assertAlmostEqual(metrics["edge_strain_rms"], self.metrics["edge_strain_rms"], places=7)

    def test_joined_spoon_neck_and_handle_have_consistent_contact_normals(self):
        s = dict(self.spoon, bowl_handle_blend=True)
        a, b = s["radius_ab"]
        center = np.asarray(s["center_local"])
        xy = center + np.array([[.95 * a, .03 * b], [3 * a, .09 * b], [7.7 * a, .04 * b]])
        heights, gradient = surface_height_gradient(xy, s)
        eps = 1.e-6 * a
        numeric = np.column_stack([(surface_height(xy + np.eye(2)[j] * eps, s) -
                                    surface_height(xy - np.eye(2)[j] * eps, s)) / (2 * eps) for j in range(2)])
        np.testing.assert_allclose(gradient, numeric, atol=2.e-7, rtol=2.e-6)
        self.assertTrue(surface_domain(xy, s).all())
        self.assertFalse(surface_domain(xy[1:], self.spoon).any())
        beyond_tip = center + np.array([[9 * a, .055 * b * np.sin(np.pi)]])
        self.assertFalse(surface_domain(beyond_tip, s).any())
        gap, inside = support_gaps(np.c_[xy, heights], s)
        np.testing.assert_allclose(gap, 0., atol=1.e-12)
        self.assertTrue(inside.all())


if __name__ == "__main__":
    unittest.main()
