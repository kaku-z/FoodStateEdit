import unittest
import numpy as np
from foodstateedit.observed_edit.core import build_state, compose
from foodstateedit.observed_edit.appearance import interior_alpha, relight_observed


class ObservedEditTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(14).integers(0, 256, (120, 160, 3), dtype=np.uint8)
        self.mask = np.zeros((120, 160), bool)
        self.mask[15:35, 20:40] = True
        self.state = build_state(self.image, self.mask, (100, 70), (145, 105))

    def test_unique_source_ids_and_true_pixel_transport(self):
        state = self.state
        self.assertEqual(np.unique(state.source_ids[state.target_mask]).size, self.mask.sum())
        np.testing.assert_array_equal(state.transported_rgb[state.target_mask],
                                      self.image.reshape(-1, 3)[state.source_ids[state.target_mask]])

    def test_source_state_does_not_change_with_destination(self):
        before = self.image.copy()
        other = build_state(self.image, self.mask, (90, 80), (145, 105))
        np.testing.assert_array_equal(self.state.source_mask, other.source_mask)
        np.testing.assert_array_equal(self.image, before)
        self.assertFalse(self.state.source_mask.flags.writeable)

    def test_no_source_leak_inside_removal_core(self):
        proposal = np.full_like(self.image, 7)
        out = compose(self.image, proposal, self.state, True)
        self.assertTrue(np.all(out[self.mask] == 7))

    def test_same_outside_protection_for_both_conditions(self):
        proposal = np.zeros_like(self.image)
        for anchor in (True, False):
            out = compose(self.image, proposal, self.state, anchor)
            np.testing.assert_array_equal(out[~self.state.editable], self.image[~self.state.editable])

    def test_locked_payload_and_free_ablation_differ_only_in_core(self):
        proposal = np.zeros_like(self.image)
        locked = compose(self.image, proposal, self.state, True)
        free = compose(self.image, proposal, self.state, False)
        np.testing.assert_array_equal(locked[~self.state.payload_core], free[~self.state.payload_core])
        np.testing.assert_array_equal(locked[self.state.payload_core], self.state.transported_rgb[self.state.payload_core])

    def test_clipped_or_overlapping_actions_rejected(self):
        for target in ((0, 0), (30, 25)):
            with self.assertRaises(ValueError):
                build_state(self.image, self.mask, target, (145, 105))

    def test_relighting_cannot_edit_outside_payload(self):
        proposal = np.full_like(self.image, 180)
        output, parameters = relight_observed(self.state.transported_rgb, proposal,
                                              self.state.target_mask, self.state.payload_core)
        np.testing.assert_array_equal(output[~self.state.target_mask], proposal[~self.state.target_mask])
        for a, b in parameters:
            self.assertTrue(.85 <= a <= 1.15)
            self.assertTrue(-12 <= b <= 12)

    def test_relighting_keeps_source_texture_order_in_interior(self):
        proposal = np.full_like(self.image, 180)
        output, parameters = relight_observed(self.state.transported_rgb, proposal,
                                              self.state.target_mask, self.state.payload_core)
        interior = interior_alpha(self.state.target_mask) == 1
        original = self.state.transported_rgb[interior].astype(float)
        expected = np.rint(np.clip(original*np.array(parameters)[:,0]+np.array(parameters)[:,1],0,255)).astype(np.uint8)
        np.testing.assert_array_equal(output[interior], expected)


if __name__ == '__main__':
    unittest.main()
