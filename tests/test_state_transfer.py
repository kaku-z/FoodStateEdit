import unittest
from dataclasses import replace

import numpy as np
import torch

from foodstateedit.state_transfer import Camera, Transition, apply_transition, from_mask, project
from foodstateedit.state_transfer.geometry import KEEP, MOVE, REARRANGE, EXPOSE
from foodstateedit.state_transfer.bridge import pack_controls
from foodstateedit.state_transfer.routing import StateTransferAdapter, gather_reference


class StateGeometryTests(unittest.TestCase):
    def setUp(self):
        self.cam = Camera(8, 10, 10, 10, 4.5, 3.5)
        self.depth = np.full((8, 10), 2.)
        mask = np.zeros((8, 10), bool)
        mask[3, 3:6] = True
        self.state = from_mask(mask, self.depth, self.cam)

    def step(self, delta, owner=None):
        n = len(self.state.ids)
        return apply_transition(self.state, Transition(delta, np.zeros(n) if owner is None else owner,
                                                       np.zeros(n), np.ones(n)), self.cam)

    def test_identity_projection(self):
        out = project(self.state, self.cam, self.depth)
        self.assertTrue((out['condition'][..., KEEP] == 1).all())
        self.assertEqual(out['diagnostics']['visible_parcels'], 3)
        self.assertEqual(out['diagnostics']['unknown_exposed_pixels'], 0)

    def test_joint_transfer_can_refill_source_without_duplicating_ids(self):
        delta = np.zeros((3, 3))
        delta[0, 1] = -.4  # selected parcel moves up two pixels
        delta[1, 0] = -.2  # remaining parcel moves into the old source position
        state = self.step(delta, [1, 0, 0])
        out = project(state, self.cam, self.depth)
        c = out['condition']
        self.assertEqual(c[3, 3, REARRANGE], 1)
        self.assertEqual(out['visible_ids'][3, 3], 1)
        # Lift towards camera as well to place the carried surface ahead of background.
        state = replace(state, xyz=state.xyz.copy())
        state.xyz[0] *= .9
        out = project(state, self.cam, self.depth)
        self.assertEqual(out['condition'][1, 3, MOVE], 1)
        self.assertEqual(out['condition'][3, 4, EXPOSE], 1)
        self.assertEqual(out['condition'][3, 4, 8], 0)
        self.assertTrue((out['condition'][3, 4, 4:6] == 2).all())
        np.testing.assert_array_equal(state.ids, self.state.ids)
        self.assertEqual(len(np.unique(out['visible_ids'][out['visible_ids'] >= 0])), 3)

    def test_occluded_identity_not_deleted(self):
        state = replace(self.state, xyz=self.state.xyz.copy())
        state.xyz[1] = self.state.xyz[0]*.8
        out = project(state, self.cam, self.depth)
        self.assertEqual(out['visible_ids'][3, 3], 1)
        self.assertEqual(out['diagnostics']['occluded_parcels'], 1)
        self.assertEqual(len(state.ids), 3)

    def test_tool_occlusion_invalidates_food_correspondence(self):
        td = np.full((8, 10), np.inf)
        td[3, 3] = 1
        out = project(self.state, self.cam, self.depth, utensil_depth=td)
        self.assertEqual(out['visible_ids'][3, 3], -1)
        self.assertEqual(out['condition'][3, 3, 9], 1)
        self.assertTrue((out['condition'][3, 3, 4:6] == 2).all())

    def test_static_scene_occludes_farther_moving_food(self):
        xyz = self.state.xyz.copy()
        xyz[0, 1] -= .4
        xyz[0] *= 2
        out = project(replace(self.state, xyz=xyz), self.cam, self.depth)
        self.assertEqual(out['condition'][1, 3, KEEP], 1)
        self.assertEqual(out['visible_ids'][1, 3], -1)

    def test_out_of_view_does_not_wrap_or_delete(self):
        delta = np.zeros((3, 3)); delta[0, 0] = 100
        state = self.step(delta)
        out = project(state, self.cam, self.depth)
        self.assertEqual(out['diagnostics']['out_of_view'], 1)
        self.assertEqual(out['diagnostics']['parcels'], 3)

    def test_bad_states_fail_closed(self):
        for bad in (replace(self.state, ids=np.array([0, 0, 2])),
                    replace(self.state, confidence=np.array([1, np.nan, 1])),
                    replace(self.state, owner=np.array([0, 4, 0])),
                    replace(self.state, xyz=-self.state.xyz)):
            with self.assertRaises(ValueError):
                project(bad, self.cam, self.depth)

    def test_no_food_and_no_tool_is_identity(self):
        state = from_mask(np.zeros((8, 10), bool), self.depth, self.cam)
        out = project(state, self.cam, self.depth)
        self.assertEqual(out['diagnostics']['parcels'], 0)
        self.assertTrue((out['condition'][..., KEEP] == 1).all())

    def test_input_is_not_mutated_and_bonds_can_be_removed(self):
        xyz = self.state.xyz.copy()
        state = apply_transition(self.state, Transition(np.ones((3, 3)), np.ones(3),
                                 np.ones(3), np.ones(3), np.zeros(len(self.state.edges), bool)), self.cam)
        np.testing.assert_array_equal(xyz, self.state.xyz)
        self.assertEqual(len(state.edges), 0)
        self.assertGreater(len(self.state.edges), 0)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.c = torch.zeros(1, 5, 10)
        self.c[..., KEEP] = 1
        self.c[..., 8] = 1
        self.c[..., 6] = 2
        self.x = torch.randn(1, 5, 4)
        self.memory = torch.randn(1, 3, 4, 4)
        self.model = StateTransferAdapter(4, 3, 8)

    def test_zero_init_is_exact_identity(self):
        out, _ = self.model(self.x, self.memory, self.c)
        self.assertTrue(torch.equal(out, self.x))

    def test_exposed_and_uncertain_tokens_do_not_read_source(self):
        self.c[0, 0, :4] = torch.tensor([0., 0., 0., 1.])
        self.c[0, 0, 4:6] = 2
        self.c[0, 1, 8] = 0
        self.c[0, 2, 9] = 1
        torch.nn.init.normal_(self.model.route_out.weight)
        first, diag = self.model(self.x, self.memory, self.c)
        second, _ = self.model(self.x, self.memory+5, self.c)
        self.assertTrue(torch.equal(first[:, :3], second[:, :3]))
        self.assertTrue((diag['source_gate'][0, :3] == 0).all())
        self.assertFalse(torch.equal(first[:, 3:], second[:, 3:]))

    def test_inactive_prefix_is_protected_even_with_nonzero_adapter(self):
        for p in self.model.parameters():
            torch.nn.init.normal_(p, std=.1)
        active = torch.tensor([[False, False, True, True, True]])
        out, _ = self.model(self.x, self.memory, self.c, active)
        self.assertTrue(torch.equal(out[:, :2], self.x[:, :2]))
        self.assertFalse(torch.equal(out[:, 2:], self.x[:, 2:]))
        off, _ = self.model(self.x, self.memory, self.c, active, strength=0)
        self.assertTrue(torch.equal(off, self.x))

    def test_bilinear_gather_recovers_known_pixel(self):
        memory = torch.arange(9.).reshape(1, 1, 3, 3)
        values, weights = gather_reference(memory, torch.tensor([[[-1., -1.], [0., 0.], [1., 1.]]]))
        out = (values[..., 0]*weights).sum(-1)
        torch.testing.assert_close(out, torch.tensor([[0., 4., 8.]]))

    def test_trainable_outputs_receive_gradients(self):
        out, _ = self.model(self.x, self.memory, self.c)
        ((out-self.x-1)**2).mean().backward()
        for p in (self.model.route_out.weight, self.model.state_net[-1].weight):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_cpu_toy_optimization_reduces_loss(self):
        # Plumbing test only: this is NOT training on food or generation evidence.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.02)
        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            output, _ = self.model(self.x, self.memory, self.c)
            loss = ((output-self.x-.25)**2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0])

    def test_condition_sequence_to_adapter_end_to_end(self):
        cam = Camera(8, 10, 10, 10, 4.5, 3.5)
        depth = np.ones((8, 10))
        state = from_mask(np.ones((8, 10), bool), depth, cam)
        condition = project(state, cam, depth)['condition']
        packed, active = pack_controls(np.stack([condition, condition]), [0, 1], (2, 2), prefix_tokens=4)
        tokens = torch.randn(1, 12, 4)
        out, diag = self.model(tokens, self.memory, torch.from_numpy(packed), torch.from_numpy(active))
        self.assertTrue(torch.equal(out, tokens))
        self.assertEqual(diag['active_tokens'], 8)

    def test_bad_condition_and_cfg_batch_mismatch_rejected(self):
        bad = self.c.clone(); bad[..., 0] = 0
        with self.assertRaises(ValueError):
            self.model(self.x, self.memory, bad)
        with self.assertRaises(ValueError):
            self.model(self.x.repeat(2, 1, 1), self.memory, self.c)
        bad = self.c.clone(); bad[..., 4] = float('nan')
        with self.assertRaises(ValueError):
            self.model(self.x, self.memory, bad)


class BridgeTests(unittest.TestCase):
    def test_layout_prefix_and_time_selection(self):
        c = np.zeros((3, 4, 6, 10), np.float32)
        c[..., 0] = 1
        c[0, ..., 6] = 1; c[1, ..., 6] = 2; c[2, ..., 6] = 3
        packed, active = pack_controls(c, [0, 2], (2, 3), prefix_tokens=6)
        self.assertEqual(packed.shape, (1, 18, 10))
        self.assertFalse(active[0, :6].any())
        self.assertTrue((packed[0, 6:12, 6] == 1).all())
        self.assertTrue((packed[0, 12:, 6] == 3).all())

    def test_implicit_or_invalid_frame_selection_rejected(self):
        c = np.zeros((3, 4, 6, 10))
        for indices in ([], [3], [-1], [0.5]):
            with self.assertRaises(ValueError):
                pack_controls(c, indices, (2, 3))


if __name__ == '__main__':
    unittest.main()
