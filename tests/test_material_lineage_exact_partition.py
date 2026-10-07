"""Actual Boolean versus analytic volumes, with deliberately partial regions."""
import importlib.util
import unittest

import numpy as np

from foodstateedit.material_lineage import MaterialLedger, RigidTransform, sample_convex_mesh
from foodstateedit.material_lineage.exact_partition import partition_convex_mesh


def box_mesh(low=(0, 0, 0), high=(1, 1, 1)):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    vertices = np.asarray(low) + vertices * (np.asarray(high) - np.asarray(low))
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return vertices, faces


@unittest.skipUnless(importlib.util.find_spec("manifold3d"), "Actual Manifold3D Boolean backend is required")
class ExactPartitionTests(unittest.TestCase):
    def test_partial_cells_match_analytic_cut_amount_and_192_cell_budget(self):
        full = box_mesh()
        bite = box_mesh(high=(.4, 1, 1))
        result = partition_convex_mesh(*full, *bite, "cube", density=2)
        self.assertEqual(len(result.source.material_ids), 192)
        self.assertAlmostEqual(result.source.total_mass, 2, places=11)
        self.assertAlmostEqual(result.carried.total_mass, .8, places=10)
        self.assertAlmostEqual(result.remaining.total_mass, 1.2, places=10)
        self.assertTrue(np.any((result.carried_fractions > 0) & (result.carried_fractions < 1)))
        np.testing.assert_allclose(result.carried_fractions + result.remaining_fractions, 1, atol=1e-14)
        MaterialLedger(result.source).validate([result.remaining, result.carried])
        with self.assertRaises(ValueError):
            MaterialLedger(result.source).validate([result.remaining, result.carried, result.carried])

    def test_exact_region_geometry_centroids_separate_from_persistent_anchors(self):
        result = partition_convex_mesh(*box_mesh(), *box_mesh(high=(.4, 1, 1)), "centroids")
        observed = sample_convex_mesh(*box_mesh(), "centroids", refinement=2)
        np.testing.assert_array_equal(result.source.material_coordinates, observed.material_coordinates)
        carried_center = np.nansum(result.carried_geometric_centroids * result.carried_geometric_volumes[:, None], axis=0) / result.carried_geometric_volumes.sum()
        remaining_center = np.nansum(result.remaining_geometric_centroids * result.remaining_geometric_volumes[:, None], axis=0) / result.remaining_geometric_volumes.sum()
        np.testing.assert_allclose(carried_center, [.2, .5, .5], atol=1e-10)
        np.testing.assert_allclose(remaining_center, [.7, .5, .5], atol=1e-10)
        partial = (result.carried_fractions > 1e-8) & (result.carried_fractions < 1 - 1e-8)
        self.assertTrue(np.any(np.linalg.norm(result.carried_geometric_centroids[partial] - result.source.material_coordinates[partial], axis=1) > 1e-4))
        self.assertTrue(np.isnan(result.carried_geometric_centroids[result.carried_geometric_volumes == 0]).all())

    def test_interior_bite_tetrahedron_has_analytic_volume(self):
        vertices = np.array([[.2, .2, .2], [.8, .2, .2], [.2, .8, .2], [.2, .2, .8]])
        faces = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
        result = partition_convex_mesh(*box_mesh(), vertices, faces, "tetra")
        self.assertAlmostEqual(result.carried.total_mass, .6**3 / 6, places=10)
        self.assertLess(result.audit["relative_bite_volume_error_over_full"], 1e-10)

    def test_small_cut_does_not_disappear_at_centroid_classification(self):
        result = partition_convex_mesh(*box_mesh(), *box_mesh(low=(.999, 0, 0)), "thin")
        self.assertAlmostEqual(result.carried.total_mass, .001, places=10)
        midpoint_approximation = np.sum(result.source.masses[result.source.material_coordinates[:, 0] > .999])
        self.assertEqual(midpoint_approximation, 0)
        self.assertGreater(result.carried.total_mass, 0)

    def test_rotated_scaled_translated_solid_preserves_fraction_and_noise(self):
        full_v, full_f = box_mesh()
        bite_v, bite_f = box_mesh(high=(.4, 1, 1))
        base = partition_convex_mesh(full_v, full_f, bite_v, bite_f, "same")
        pose = RigidTransform([[0, -1, 0], [1, 0, 0], [0, 0, 1]], [100, -20, 15])
        scale = 1e-3
        moved = partition_convex_mesh(pose.apply(full_v * scale), full_f,
                                     pose.apply(bite_v * scale), bite_f, "same")
        np.testing.assert_allclose(moved.carried_fractions, base.carried_fractions, atol=1e-8)
        self.assertAlmostEqual(moved.carried.total_mass / moved.source.total_mass, .4, places=9)
        np.testing.assert_array_equal(base.source.noise(907), moved.source.noise(907))

    def test_outside_bite_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            partition_convex_mesh(*box_mesh(), *box_mesh(low=(.8, 0, 0), high=(1.2, 1, 1)), "outside")

    def test_closed_nonconvex_bite_retains_actual_boundary_and_exact_volume(self):
        vertices, faces = box_mesh()
        vertices[6] = [.4, .4, .4]
        triangle = vertices[faces]
        center = vertices.mean(axis=0)
        analytic_boundary_volume = np.einsum("ij,ij->i", triangle[:, 0] - center,
                                            np.cross(triangle[:, 1] - center, triangle[:, 2] - center)).sum() / 6
        result = partition_convex_mesh(*box_mesh(), vertices, faces, "nonconvex")
        self.assertGreater(analytic_boundary_volume, 0)
        self.assertLess(analytic_boundary_volume, 1)
        self.assertAlmostEqual(result.carried.total_mass, analytic_boundary_volume, places=10)
        self.assertAlmostEqual(result.audit["actual_bite_geometric_volume"], analytic_boundary_volume, places=10)
        self.assertLess(result.audit["relative_bite_volume_error_over_full"], 1e-10)
        MaterialLedger(result.source).validate([result.remaining, result.carried])

    def test_open_bite_rejected_by_actual_boolean_backend(self):
        vertices, faces = box_mesh(high=(.4, 1, 1))
        with self.assertRaisesRegex(ValueError, "Manifold3D rejected"):
            partition_convex_mesh(*box_mesh(), vertices, faces[:-1], "open-bite")


if __name__ == "__main__":
    unittest.main()
