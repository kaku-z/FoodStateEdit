import unittest
import numpy as np

from foodstateedit.material_transfer.core import make_cake, transport, render, material_rgb, MaterialState


class MaterialOracleTests(unittest.TestCase):
    def test_complementary_partition_and_no_residual(self):
        source = make_cake(7)
        edited = transport(source, [15, 0, 0])
        before = np.sort(source.grid[source.grid >= 0])
        np.testing.assert_array_equal(before, np.sort(edited.grid[edited.grid >= 0]))
        positions = np.argwhere(np.isin(source.grid, source.selected_ids))
        self.assertTrue(np.all(edited.grid[tuple(positions.T)] == -1))
        np.testing.assert_array_equal(edited.grid[tuple((positions + [15, 0, 0]).T)],
                                      source.grid[tuple(positions.T)])
        unchanged = (source.grid >= 0) & ~np.isin(source.grid, source.selected_ids)
        np.testing.assert_array_equal(edited.grid[unchanged], source.grid[unchanged])

    def test_source_state_is_frozen_and_target_independent(self):
        source = make_cake(3)
        original = source.grid.copy()
        transport(source, [12, 0, 0]); transport(source, [16, 0, 0])
        np.testing.assert_array_equal(source.grid, original)
        with self.assertRaises(ValueError):
            source.grid[0, 0, 0] = 4

    def test_infeasible_path_and_bounds_rejected(self):
        source = make_cake()
        for delta in ([-3, 0, 0], [50, 0, 0], [0.5, 0, 0]):
            with self.assertRaises(ValueError):
                transport(source, delta)

    def test_renderer_first_hit_depth_and_canonical_mapping(self):
        edited = transport(make_cake(), [15, 0, 0])
        result = render(edited, 48)
        visible = result["material_ids"] >= 0
        self.assertGreater(visible.sum(), 0)
        self.assertTrue(np.isfinite(result["depth"][visible]).all())
        q = result["canonical_hits"][visible]
        base = edited.canonical[result["material_ids"][visible]]
        self.assertTrue(np.all(q >= base - 1e-6))
        self.assertTrue(np.all(q <= base + 1 + 1e-6))
        self.assertTrue(np.isnan(result["canonical_hits"][~visible]).all())

    def test_shared_material_field_has_no_destination_input(self):
        q = np.array([[22.2, 25.3, 8.4]])
        normal = np.array([[0., 0., 1.]])
        np.testing.assert_array_equal(material_rgb(q, normal), material_rgb(q.copy(), normal))

    def test_ray_traversal_matches_independent_cube_intersections(self):
        grid = np.full((56, 56, 56), -1, dtype=np.int32)
        positions = np.array([[27, 24, 6], [26, 23, 10], [23, 23, 7]])
        grid[tuple(positions.T)] = np.arange(3)
        state = MaterialState(grid, positions + [3, 0, 0], np.array([0]))
        result = render(state, 65)
        direction = np.array([.32, .42, -1.]); direction /= np.linalg.norm(direction)
        right = np.cross(direction, [0, 1, 0]); right /= np.linalg.norm(right)
        up = np.cross(right, direction)
        xx, yy = np.meshgrid(np.linspace(-34, 34, 65), np.linspace(34, -34, 65))
        origin = (np.array([27., 24., 7.]) + xx[..., None] * right +
                  yy[..., None] * up - 100 * direction).reshape(-1, 3)
        a = (positions[None] - origin[:, None]) / direction
        b = (positions[None] + 1 - origin[:, None]) / direction
        near = np.minimum(a, b).max(axis=2)
        far = np.maximum(a, b).min(axis=2)
        distances = np.where((near <= far) & (near >= 0), near, np.inf)
        expected_depth = distances.min(axis=1)
        expected_ids = np.where(np.isfinite(expected_depth), distances.argmin(axis=1), -1)
        np.testing.assert_array_equal(result["material_ids"].ravel(), expected_ids)
        np.testing.assert_allclose(result["depth"].ravel(), expected_depth, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
