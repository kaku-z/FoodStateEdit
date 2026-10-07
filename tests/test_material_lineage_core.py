"""Analytic and adversarial checks of the discrete lineage core."""
import unittest

import numpy as np

from foodstateedit.material_lineage import (
    MaterialLedger, MaterialState, PairedSeam, RigidTransform, SeamSide,
    paired_seam_from_interface_mesh, rigid_from_g37_report, sample_convex_mesh,
)


def cube_mesh():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return vertices, faces


class MaterialLineageTests(unittest.TestCase):
    def setUp(self):
        self.state = MaterialState.create([[0, 0, 0], [1, 0, 0], [1, 1, 0]],
                                          [2, 3, 5], "food-A")
        self.pose = RigidTransform([[0, -1, 0], [1, 0, 0], [0, 0, 1]], [2, 3, 4])

    def test_fractional_transfer_preserves_each_source_budget(self):
        branches = self.state.transfer([[1, .25, 0], [0, .75, 1]], ["left", "right"])
        self.assertEqual(branches["left"].total_mass, 2.75)
        self.assertEqual(branches["right"].total_mass, 7.25)
        self.assertEqual(MaterialLedger(self.state).validate(list(branches.values()))["allocated_mass"], 10)

    def test_copy_and_discard_allocation_rejected(self):
        for invalid in (np.ones((2, 3)), np.full((2, 3), .1), [[1, 1, -1], [0, 0, 2]]):
            with self.assertRaises(ValueError):
                self.state.transfer(invalid, ["remaining", "carried"])

    def test_external_whole_state_copy_rejected_even_with_distinct_branch_names(self):
        copied = MaterialState(self.state.material_ids, self.state.material_coordinates,
                               self.state.reference_masses, self.state.fractions,
                               self.state.world_positions, "forged-copy")
        with self.assertRaisesRegex(ValueError, "duplicated"):
            MaterialLedger(self.state).validate([self.state, copied])

    def test_equal_total_mass_cannot_hide_per_cell_copy_and_loss(self):
        state = MaterialState.create([[0, 0, 0], [1, 0, 0]], [1, 1], "balanced")
        left = state.partition(np.array([False, True]))["remaining"]
        with self.assertRaisesRegex(ValueError, "duplicated"):
            MaterialLedger(state).validate([left, left])

    def test_mass_and_coordinates_cannot_be_forged(self):
        for masses, coords in ((self.state.reference_masses * 2, self.state.material_coordinates),
                              (self.state.reference_masses, self.state.material_coordinates + 1)):
            forged = MaterialState(self.state.material_ids, coords, masses,
                                   self.state.fractions, self.state.world_positions)
            with self.assertRaises(ValueError):
                MaterialLedger(self.state).validate([forged])

    def test_nested_splits_preserve_original_budget(self):
        first = self.state.transfer([[.4, 1, .5], [.6, 0, .5]], ["a", "b"])
        second = first["a"].transfer(np.full((2, 3), .5), ["c", "d"])
        report = MaterialLedger(self.state).validate([first["b"], *second.values()])
        self.assertAlmostEqual(report["allocated_mass"], 10)

    def test_tiny_fractional_descendant_cannot_be_copied_under_absolute_tolerance(self):
        tiny = self.state.transfer([[1e-14] * 3, [1 - 1e-14] * 3], ["tiny", "rest"])["tiny"]
        with self.assertRaisesRegex(ValueError, "duplicated"):
            MaterialLedger(tiny).validate([tiny, tiny])

    def test_identity_noise_survives_pose_branch_and_row_order(self):
        original = self.state.noise(907, 7)
        moved = self.state.transformed(self.pose)
        np.testing.assert_array_equal(original, moved.noise(907, 7))
        split = moved.partition(np.array([False, True, True]))
        np.testing.assert_array_equal(original[[1, 2]], split["carried"].noise(907, 7))
        fractional = self.state.transfer(np.full((2, 3), .5), ["a", "b"])
        np.testing.assert_array_equal(fractional["a"].noise(907), fractional["b"].noise(907))
        order = [2, 0, 1]
        reordered = MaterialState(tuple(self.state.material_ids[i] for i in order),
                                  self.state.material_coordinates[order], self.state.reference_masses[order],
                                  self.state.fractions[order], self.state.world_positions[order])
        np.testing.assert_array_equal(reordered.noise(907, 7), original[order])
        self.assertFalse(np.array_equal(original, self.state.noise(41, 7)))

    def test_rigid_inverse_and_composition_match_analytic_positions(self):
        moved = self.state.transformed(self.pose)
        np.testing.assert_allclose(moved.world_positions, [[2, 3, 4], [2, 4, 4], [1, 4, 4]])
        np.testing.assert_allclose(moved.transformed(self.pose.inverse()).world_positions,
                                   self.state.world_positions, atol=1e-14)
        translation = RigidTransform(np.eye(3), [1, 2, 3])
        np.testing.assert_allclose(self.pose.then(translation).apply(self.state.world_positions),
                                   translation.apply(self.pose.apply(self.state.world_positions)))
        large_translation = RigidTransform(self.pose.rotation, [1e20, 1e20, 1e20])
        np.testing.assert_array_equal(large_translation.apply_vectors([[1, 0, 0]]), [[0, 1, 0]])

    def test_operation_is_covariant_under_joint_coordinate_rotation(self):
        frame = RigidTransform([[1, 0, 0], [0, 0, -1], [0, 1, 0]], [8, 4, 2])
        conjugate = frame.inverse().then(self.pose).then(frame)
        np.testing.assert_allclose(conjugate.apply(frame.apply(self.state.world_positions)),
                                   frame.apply(self.pose.apply(self.state.world_positions)))

    def test_g37_adapter_uses_centered_rotation_convention(self):
        source = np.array([4, 5, 6])
        destination = np.array([10, 12, 15])
        pose = rigid_from_g37_report({"rotation_food_frame": self.pose.rotation,
                                     "source_center": source, "destination_center": destination})
        expected = (self.state.world_positions - source) @ self.pose.rotation.T + destination
        np.testing.assert_allclose(pose.apply(self.state.world_positions), expected)
        np.testing.assert_allclose(pose.apply(source[None]), destination[None])

    def test_seam_pairs_have_shared_coordinates_opposite_normals_and_matching_noise(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        seam = paired_seam_from_interface_mesh(vertices, np.array([[0, 1, 2], [0, 2, 3]]),
                                               "cut-7", carried_pose=self.pose)
        self.assertEqual(seam.remaining.sample_ids, seam.carried.sample_ids)
        self.assertEqual(len(seam.remaining.sample_ids), 2)
        np.testing.assert_array_equal(seam.remaining.material_coordinates, seam.carried.material_coordinates)
        np.testing.assert_array_equal(seam.remaining.material_normals, -seam.carried.material_normals)
        self.assertAlmostEqual(seam.sample_areas.sum(), 1)
        self.assertLess(seam.correspondence_residual(), 1e-14)
        np.testing.assert_allclose(seam.carried.world_positions, self.pose.apply(seam.remaining.world_positions))
        self.assertEqual(seam.noise(41).shape, (2, 3))
        bad = SeamSide(tuple(reversed(seam.carried.sample_ids)), seam.carried.material_coordinates,
                       seam.carried.material_normals, self.pose)
        with self.assertRaises(ValueError):
            PairedSeam("cut-7", seam.remaining, bad, seam.sample_areas)

    def test_convex_mesh_quadrature_has_analytic_mass_centroid_and_bounds(self):
        vertices, faces = cube_mesh()
        for refinement in (0, 1, 2):
            state = sample_convex_mesh(vertices, faces, "cube", refinement, density=2)
            self.assertEqual(len(state.material_ids), 12 * 4**refinement)
            self.assertAlmostEqual(state.total_mass, 2, places=13)
            centroid = np.sum(state.world_positions * state.masses[:, None], axis=0) / state.total_mass
            np.testing.assert_allclose(centroid, [.5, .5, .5], atol=1e-14)
            self.assertTrue(np.all(state.world_positions > 0) and np.all(state.world_positions < 1))
            cut = state.partition(state.material_coordinates[:, 0] > .5)
            self.assertAlmostEqual(cut["carried"].total_mass, 1, places=13)
        inward = sample_convex_mesh(vertices, faces[:, ::-1], "cube-inward")
        self.assertAlmostEqual(inward.total_mass, 1, places=13)

    def test_convex_sampling_rotation_covariance(self):
        vertices, faces = cube_mesh()
        a = sample_convex_mesh(vertices, faces, "cube")
        b = sample_convex_mesh(self.pose.apply(vertices), faces, "cube")
        np.testing.assert_allclose(b.world_positions, self.pose.apply(a.world_positions), atol=1e-14)
        np.testing.assert_allclose(a.masses, b.masses, atol=1e-14)
        np.testing.assert_array_equal(a.noise(41), b.noise(41))

    def test_invalid_ranges_and_open_mesh_rejected(self):
        for bad_mass in ([2, 3, -5], [2, 3, np.nan]):
            with self.assertRaises(ValueError):
                MaterialState.create(self.state.material_coordinates, bad_mass)
        with self.assertRaises(ValueError):
            RigidTransform(np.diag([-1, 1, 1]), [0, 0, 0])
        vertices, faces = cube_mesh()
        with self.assertRaises(ValueError):
            sample_convex_mesh(vertices, faces[:-1], "open")
        with self.assertRaises(ValueError):
            sample_convex_mesh(vertices, faces, "cube", refinement=-1)
        dented = vertices.copy()
        dented[6] = [.4, .4, .4]
        with self.assertRaises(ValueError):
            sample_convex_mesh(dented, faces, "nonconvex")
        with self.assertRaises(ValueError):
            self.state.partition([0, 1, 1])
        with self.assertRaises(ValueError):
            self.state.noise(-1)
        with self.assertRaises(ValueError):
            self.state.noise(41, channels=0)
        with self.assertRaises(ValueError):
            self.state.reference_masses[0] = 100


if __name__ == "__main__":
    unittest.main()
