"""Source triangulation tests prevent unreferenced or independent ghost bodies."""
import unittest

import numpy as np

from foodstateedit.material_lineage.topology import coherent_closed_patch, patch_triangles


class MaterialTopologyTests(unittest.TestCase):
    def setUp(self):
        self.patch = np.zeros((17, 21), dtype=bool)
        self.patch[2:9, 3:12] = True  # 63 source pixels, 96 triangles.
        self.patch[12:14, 2:7] = True  # A separate 10-vertex, 8-triangle island.
        self.patch[0, 0] = self.patch[16, 20] = True  # No-face source points.

    def test_largest_cohesive_piece_excludes_island_and_no_face_points(self):
        result = coherent_closed_patch(self.patch)
        metrics = result["metrics"]
        self.assertEqual(metrics["retained_source_pixels"], 63)
        self.assertEqual(metrics["source_unreferenced_pixels"], 2)
        self.assertEqual(metrics["discarded_triangulated_island_pixels"], 10)
        self.assertEqual(metrics["triangulated_component_vertex_counts"], [63, 10])
        self.assertEqual(metrics["retained_top_triangles"], 96)
        self.assertEqual(metrics["retained_top_vertex_coverage"], 1.)
        self.assertEqual(result["discarded_mask"].sum(), 12)
        np.testing.assert_array_equal(result["patch"] | result["discarded_mask"], self.patch)

    def test_uv_material_closure_and_source_indices_remain_in_original_order(self):
        yy, xx = np.where(self.patch)
        n = len(xx)
        closure = dict(source_camera_xyz=np.arange(n * 3).reshape(n, 3),
                       source_local_xyz=np.arange(n * 3).reshape(n, 3) + .25,
                       source_uv_pixels=np.c_[xx + .5, yy + .5],
                       floor_local=np.arange(n) / 100., uniform_floor_local=np.arange(n) / 50.,
                       interior_rgb=np.arange(n * 3).reshape(n, 3) % 255,
                       bottom_rgb=np.arange(n * 3).reshape(n, 3) % 199,
                       world_axes=np.eye(3), extent=np.asarray(2.5))
        result = coherent_closed_patch(self.patch, closure)
        kept = result["keep_pixel_indices"]
        np.testing.assert_array_equal(result["closure"]["source_camera_xyz"], closure["source_camera_xyz"][kept])
        np.testing.assert_array_equal(result["closure"]["source_uv_pixels"], closure["source_uv_pixels"][kept])
        np.testing.assert_array_equal(result["closure"]["floor_local"], closure["floor_local"][kept])
        np.testing.assert_array_equal(result["closure"]["bottom_rgb"], closure["bottom_rgb"][kept])
        np.testing.assert_array_equal(result["closure"]["world_axes"], np.eye(3))
        self.assertEqual(result["closure"]["extent"], 2.5)
        self.assertTrue(np.all(np.diff(kept) > 0))
        np.testing.assert_array_equal(result["original_to_kept"][kept], np.arange(63))
        clean_yy, clean_xx = np.where(result["patch"])
        np.testing.assert_array_equal(np.c_[clean_xx, clean_yy], np.c_[xx, yy][kept])

    def test_closed_rebuild_has_no_unreferenced_vertices_or_open_edges(self):
        result = coherent_closed_patch(self.patch)
        f = result["top_faces"]
        n = int(result["patch"].sum())
        edges, counts = np.unique(np.sort(np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1), axis=0, return_counts=True)
        boundary = edges[counts == 1]
        sides = np.array([[a, b, b + n] for a, b in boundary] + [[a, b + n, a + n] for a, b in boundary])
        complete = np.r_[f, f[:, ::-1] + n, sides]
        self.assertEqual(len(np.unique(complete)), 2 * n)
        _, closed_counts = np.unique(np.sort(np.r_[complete[:, [0, 1]], complete[:, [1, 2]], complete[:, [2, 0]]], axis=1), axis=0, return_counts=True)
        self.assertTrue(np.all(closed_counts == 2))

    def test_source_holes_are_preserved_without_target_geometry(self):
        patch = np.zeros((13, 16), dtype=bool)
        patch[1:11, 1:13] = True
        patch[4:7, 5:8] = False
        patch[12, 15] = True
        result = coherent_closed_patch(patch)
        self.assertFalse(result["patch"][4:7, 5:8].any())
        self.assertFalse(result["patch"][12, 15])
        self.assertFalse((result["patch"] & ~patch).any())
        self.assertEqual(result["metrics"]["retained_top_vertex_coverage"], 1.)

    def test_pinched_hole_cannot_create_a_four_face_column_edge(self):
        patch = np.ones((9, 9), dtype=bool)
        patch[:4, :4] = False
        patch[5, 4] = False  # Interior hole almost meets the exterior notch.
        original_faces, original_pixels = patch_triangles(patch)
        edges, counts = np.unique(np.sort(np.r_[original_faces[:, [0, 1]],
            original_faces[:, [1, 2]], original_faces[:, [2, 0]]], axis=1),
            axis=0, return_counts=True)
        degree = np.bincount(edges[counts == 1].ravel(), minlength=len(original_pixels))
        self.assertTrue(np.any(degree > 2))
        closure = dict(source_uv_pixels=original_pixels+.5,
                       source_camera_xyz=np.c_[original_pixels, np.ones(len(original_pixels))],
                       world_axes=np.eye(3), extent=np.asarray(1.))
        result = coherent_closed_patch(patch, closure)
        self.assertGreater(result['metrics']['manifold_trimmed_source_pixels'], 0)
        self.assertTrue(result['metrics']['closed_column_edge_incidence_valid'])
        self.assertFalse((result['patch'] & ~patch).any())
        np.testing.assert_array_equal(result['patch'] | result['discarded_mask'], patch)
        keep = result['keep_pixel_indices']
        np.testing.assert_array_equal(result['closure']['source_uv_pixels'], closure['source_uv_pixels'][keep])
        faces = result['top_faces']; n = int(result['patch'].sum())
        edges, counts = np.unique(np.sort(np.r_[faces[:, [0, 1]], faces[:, [1, 2]],
            faces[:, [2, 0]]], axis=1), axis=0, return_counts=True)
        a, b = edges[counts == 1].T
        complete = np.r_[faces, faces[:, ::-1]+n, np.c_[a,b,b+n], np.c_[a,b+n,a+n]]
        _, incidence = np.unique(np.sort(np.r_[complete[:, [0, 1]], complete[:, [1, 2]],
            complete[:, [2, 0]]], axis=1), axis=0, return_counts=True)
        self.assertTrue(np.all(incidence == 2))

    def test_triangle_construction_matches_original_closed_patch_rule(self):
        f, xy = patch_triangles(self.patch)
        lookup = np.full(self.patch.shape, -1, dtype=int)
        lookup[xy[:, 1], xy[:, 0]] = np.arange(len(xy))
        expected = []
        h, w = self.patch.shape
        for x, y in xy:
            if x + 1 < w and y + 1 < h:
                a, b = lookup[y, x], lookup[y, x + 1]
                c, d = lookup[y + 1, x], lookup[y + 1, x + 1]
                if b >= 0 and c >= 0:
                    expected.append([a, c, b])
                    if d >= 0:
                        expected.append([b, c, d])
        np.testing.assert_array_equal(f, expected)

    def test_untriangulated_selection_requires_source_reselection(self):
        patch = np.eye(7, dtype=bool)
        with self.assertRaisesRegex(ValueError, "no triangulated material"):
            coherent_closed_patch(patch)


if __name__ == "__main__":
    unittest.main()
