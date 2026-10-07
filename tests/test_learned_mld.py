"""Invariance, no-action generation, and actual optimization checks for MLD.

These checks establish properties of the architecture, not accuracy on measured
three-dimensional food or photographic realism.  A one-step optimization check
ensures the requested learned modules are actually in the gradient path.
"""
import inspect
import io
import unittest
from dataclasses import asdict, replace

import torch
from torch.nn import functional as F

from foodstateedit.material_lineage.learned_mld import (
    ACTION_CHANNELS, JOINT_CHANNELS, MODEL_FORMAT_VERSION,
    MLDConfig, MLDModel, canonical_to_world,
)


class LearnedMLDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(312)
        self.config = MLDConfig(width=16, heads=4, source_blocks=1, diffusion_blocks=1)
        self.model = MLDModel(self.config).eval()
        self.image = torch.rand(2, 3, 64, 64)
        self.xyz = torch.rand(2, 11, 3) * 2 - 1
        self.uv = self.xyz[..., :2].clone()
        self.action = torch.rand(2, 10) * 2 - 1
        self.noise = torch.randn(2, 11, 4)
        self.time = torch.tensor([0.25, 0.8])

    def full_forward(self, **replace):
        kwargs = dict(image=self.image, canonical_xyz=self.xyz, source_uv=self.uv,
                      action=self.action, noisy_joint=self.noise, normalized_t=self.time)
        kwargs.update(replace)
        return self.model(**kwargs)

    def test_frozen_schema_outputs_are_joint_sdf_and_material(self):
        output = self.full_forward()
        self.assertEqual(len(ACTION_CHANNELS), 10)
        self.assertEqual(len(JOINT_CHANNELS), 4)
        for key, size in (("features", 16), ("sdf", 1), ("rgb", 3), ("occupancy_logits", 1)):
            value = output["source"][key]
            self.assertEqual(value.shape, (2, 11, size))
            self.assertTrue(torch.isfinite(value).all())
        self.assertEqual(output["epsilon"].shape, (2, 11, 4))
        self.assertTrue((output["source"]["rgb"].abs() <= 1).all())

    def test_every_site_budget_is_conserved_without_copying(self):
        transition = self.full_forward()["transition"]
        allocation = transition["allocation"]
        self.assertTrue(torch.equal(allocation.sum(-1), torch.ones(2, 11)))
        self.assertTrue((allocation >= 0).all() and (allocation <= 1).all())
        reference_mass = torch.arange(1, 23, dtype=allocation.dtype).reshape(2, 11)
        branch_mass = reference_mass[..., None] * allocation
        torch.testing.assert_close(branch_mass.sum(-1), reference_mass, rtol=1e-7, atol=1e-6)
        torch.testing.assert_close(allocation[..., 1:2], transition["carried_fraction"], rtol=0, atol=0)
        self.assertEqual(transition["remaining_delta"].shape, (2, 11, 3))
        self.assertEqual(transition["carried_delta"].shape, (2, 11, 3))

    def test_source_and_joint_noise_prediction_are_action_independent(self):
        first = self.full_forward(action=torch.zeros_like(self.action))
        second = self.full_forward(action=self.action * 3)
        for key in first["source"]:
            self.assertTrue(torch.equal(first["source"][key], second["source"][key]))
        self.assertTrue(torch.equal(first["epsilon"], second["epsilon"]))
        self.assertFalse(torch.allclose(first["transition"]["carried_delta"],
                                       second["transition"]["carried_delta"], rtol=1e-5, atol=1e-7))

    def test_generation_api_has_no_action_target_or_visibility_inputs(self):
        self.assertEqual(tuple(inspect.signature(MLDModel.encode_source).parameters),
                         ("self", "image", "canonical_xyz", "source_uv"))
        self.assertEqual(tuple(inspect.signature(MLDModel.denoise).parameters),
                         ("self", "features", "noisy_joint", "normalized_t"))

    def test_token_permutation_preserves_corresponding_predictions(self):
        order = torch.tensor([8, 1, 6, 10, 2, 7, 0, 9, 4, 3, 5])
        first = self.full_forward()
        second = self.full_forward(canonical_xyz=self.xyz[:, order], source_uv=self.uv[:, order],
                                   noisy_joint=self.noise[:, order])
        for key in first["source"]:
            torch.testing.assert_close(first["source"][key][:, order], second["source"][key],
                                       rtol=2e-5, atol=2e-6)
        for key in first["transition"]:
            torch.testing.assert_close(first["transition"][key][:, order], second["transition"][key],
                                       rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(first["epsilon"][:, order], second["epsilon"], rtol=2e-5, atol=2e-6)

    def test_unrelated_batch_member_does_not_change_source(self):
        together = self.full_forward()
        alone = self.full_forward(image=self.image[:1], canonical_xyz=self.xyz[:1],
                                  source_uv=self.uv[:1], action=self.action[:1],
                                  noisy_joint=self.noise[:1], normalized_t=self.time[:1])
        for key in together["source"]:
            torch.testing.assert_close(together["source"][key][:1], alone["source"][key],
                                       rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(together["epsilon"][:1], alone["epsilon"], rtol=2e-5, atol=2e-6)

    def test_source_image_changes_the_conditioning(self):
        first = self.full_forward()
        second = self.full_forward(image=1 - self.image)
        self.assertFalse(torch.allclose(first["source"]["features"], second["source"]["features"]))
        self.assertFalse(torch.allclose(first["source"]["sdf"], second["source"]["sdf"]))
        self.assertFalse(torch.allclose(first["source"]["rgb"], second["source"]["rgb"]))
        self.assertFalse(torch.allclose(first["epsilon"], second["epsilon"]))

    def test_null_command_is_learned_not_hard_masked(self):
        with torch.no_grad():
            self.model.transition_head[-1].bias.copy_(torch.tensor([0.1, 1., 2., 3., 4., 5., 6.]))
            self.model.transition_head[-1].weight.zero_()
        output = self.full_forward(action=torch.zeros_like(self.action))["transition"]
        torch.testing.assert_close(output["remaining_delta"], torch.tensor([1., 2., 3.]).expand(2, 11, 3))
        torch.testing.assert_close(output["carried_delta"], torch.tensor([4., 5., 6.]).expand(2, 11, 3))
        self.assertTrue((output["carried_fraction"] > 0.5).all())

    def test_actual_optimizer_step_updates_each_requested_component(self):
        self.model.train()
        modules = ("image_encoder", "source_transformer", "source_geometry_head",
                   "source_material_head", "transition_head", "denoiser")
        before = {name: [p.detach().clone() for p in getattr(self.model, name).parameters()]
                  for name in modules}
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        output = self.full_forward()
        source, transition = output["source"], output["transition"]
        target_sdf = (self.xyz.norm(dim=-1, keepdim=True) - 0.5) / 0.5
        target_rgb = self.xyz.sin()
        target_occupied = (target_sdf < 0).float()
        target_carried = (self.xyz[..., :1] < 0).float()
        loss = (
            F.mse_loss(source["sdf"], target_sdf)
            + F.mse_loss(source["rgb"], target_rgb)
            + F.binary_cross_entropy_with_logits(source["occupancy_logits"], target_occupied)
            + F.binary_cross_entropy(transition["carried_fraction"], target_carried)
            + F.mse_loss(transition["remaining_delta"], torch.zeros_like(self.xyz))
            + F.mse_loss(transition["carried_delta"], self.action[:, None, 4:7].expand_as(self.xyz))
            + F.mse_loss(output["epsilon"], self.noise)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name in modules:
            gradients = [p.grad for p in getattr(self.model, name).parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients), name)
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0, name)
        optimizer.step()
        for name in modules:
            after = list(getattr(self.model, name).parameters())
            self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before[name], after)), name)

    def test_checkpoint_round_trip_preserves_predictions(self):
        stream = io.BytesIO()
        torch.save({"format_version": MODEL_FORMAT_VERSION, "model_config": asdict(self.config),
                    "state_dict": self.model.state_dict()}, stream)
        stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
        loaded = MLDModel(MLDConfig(**checkpoint["model_config"])).eval()
        loaded.load_state_dict(checkpoint["state_dict"], strict=True)
        expected = self.full_forward()
        actual = loaded(self.image, self.xyz, self.uv, self.action, self.noise, self.time)
        self.assertTrue(torch.equal(expected["epsilon"], actual["epsilon"]))
        self.assertTrue(torch.equal(expected["source"]["sdf"], actual["source"]["sdf"]))

    def test_legacy_config_without_state_dim_preserves_checkpoint_shapes(self):
        legacy_config = asdict(self.config)
        legacy_config.pop("state_dim")
        loaded = MLDModel(MLDConfig(**legacy_config)).eval()
        loaded.load_state_dict(self.model.state_dict(), strict=True)
        self.assertEqual(loaded.config.state_dim, 0)
        self.assertFalse(any(key.startswith("state_encoder.") for key in loaded.state_dict()))
        self.assertEqual(loaded.transition_head[0].in_features, self.config.width * 2)
        expected = self.full_forward()["transition"]["carried_delta"]
        actual = loaded(self.image, self.xyz, self.uv, self.action)["transition"]["carried_delta"]
        self.assertTrue(torch.equal(expected, actual))

    def test_coupled_transition_requires_matching_predicted_or_sampled_state(self):
        model = MLDModel(replace(self.config, state_dim=4)).eval()
        source = model.encode_source(self.image, self.xyz, self.uv)
        for state in (None, torch.zeros(1, 11, 4), torch.zeros(2, 10, 4), torch.zeros(2, 11, 3)):
            with self.assertRaises(ValueError):
                model.transition(source["features"], self.action, state)
        state = torch.cat((source["sdf"], source["rgb"]), dim=-1)
        first = model.transition(source["features"], self.action, state)
        second = model.transition(source["features"], self.action, -state)
        self.assertFalse(torch.allclose(first["carried_fraction"], second["carried_fraction"]))
        self.assertFalse(torch.allclose(first["carried_delta"], second["carried_delta"]))
        self.assertTrue(torch.equal(first["allocation"].sum(-1), torch.ones(2, 11)))
        with self.assertRaises(ValueError):
            self.model.transition(source["features"], self.action, state)

    def test_coupled_default_state_uses_predictions_and_never_noisy_target(self):
        model = MLDModel(replace(self.config, state_dim=4)).eval()
        first = model(self.image, self.xyz, self.uv, self.action, self.noise, self.time)
        changed_noisy_target = model(self.image, self.xyz, self.uv, self.action, -self.noise, self.time)
        predicted = torch.cat((first["source"]["sdf"], first["source"]["rgb"]), dim=-1)
        explicit_predicted = model(self.image, self.xyz, self.uv, self.action, self.noise,
                                   self.time, source_state=predicted)
        explicit_other_state = model(self.image, self.xyz, self.uv, self.action, self.noise,
                                     self.time, source_state=-predicted)
        for key in first["transition"]:
            self.assertTrue(torch.equal(first["transition"][key], changed_noisy_target["transition"][key]))
            self.assertTrue(torch.equal(first["transition"][key], explicit_predicted["transition"][key]))
        self.assertFalse(torch.allclose(first["transition"]["carried_delta"],
                                       explicit_other_state["transition"]["carried_delta"]))
        for key in first["source"]:
            self.assertTrue(torch.equal(first["source"][key], explicit_other_state["source"][key]))
        self.assertTrue(torch.equal(first["epsilon"], explicit_other_state["epsilon"]))

    def test_one_sampled_state_is_immutable_when_replayed_for_actions(self):
        model = MLDModel(replace(self.config, state_dim=4)).eval()
        source = model.encode_source(self.image, self.xyz, self.uv)
        sampled_state = self.noise.tanh().clone()
        before = sampled_state.clone()
        model.transition(source["features"], torch.zeros_like(self.action), sampled_state)
        model.transition(source["features"], self.action, sampled_state)
        model.transition(source["features"], -self.action, sampled_state)
        self.assertTrue(torch.equal(sampled_state, before))

    def test_state_encoder_has_finite_gradients_and_actual_optimizer_update(self):
        model = MLDModel(replace(self.config, state_dim=4)).train()
        before = [p.detach().clone() for p in model.state_encoder.parameters()]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        output = model(self.image, self.xyz, self.uv, self.action)
        transition = output["transition"]
        loss = (
            F.mse_loss(transition["carried_fraction"], (self.xyz[..., :1] < 0).float())
            + F.mse_loss(transition["carried_delta"], self.action[:, None, 4:7].expand_as(self.xyz))
            + transition["remaining_delta"].square().mean()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for module_name in ("state_encoder", "source_geometry_head", "source_material_head"):
            gradients = [p.grad for p in getattr(model, module_name).parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients), module_name)
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0, module_name)
        optimizer.step()
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, model.state_encoder.parameters())))

    def test_coupled_source_and_denoiser_remain_action_independent(self):
        model = MLDModel(replace(self.config, state_dim=4)).eval()
        first = model(self.image, self.xyz, self.uv, self.action, self.noise, self.time)
        second = model(self.image, self.xyz, self.uv, -self.action, self.noise, self.time)
        for key in first["source"]:
            self.assertTrue(torch.equal(first["source"][key], second["source"][key]))
        self.assertTrue(torch.equal(first["epsilon"], second["epsilon"]))

    def test_external_world_frame_preserves_rigid_covariance(self):
        points = self.xyz.double()
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.float64)
        translation = torch.tensor([2., 3., 4.], dtype=torch.float64)
        frame = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]], dtype=torch.float64)
        frame_t = torch.tensor([7., 8., 9.], dtype=torch.float64)
        transformed = canonical_to_world(points, rotation, translation)
        reframed = canonical_to_world(transformed, frame, frame_t)
        combined_rotation = frame @ rotation
        combined_translation = frame @ translation + frame_t
        torch.testing.assert_close(reframed, canonical_to_world(points, combined_rotation, combined_translation),
                                   rtol=0, atol=2e-15)
        self.assertTrue(torch.equal(points, self.xyz.double()))

    def test_invalid_schema_and_partial_denoising_inputs_rejected(self):
        for invalid in ({"width": 15}, {"joint_dim": 7}, {"action_dim": 13}, {"heads": False},
                        {"state_dim": 1}, {"state_dim": -1}, {"state_dim": True}, {"state_dim": "4"}):
            with self.assertRaises(ValueError):
                MLDConfig(**invalid)
        with self.assertRaises(ValueError):
            self.full_forward(normalized_t=None)
        with self.assertRaises(ValueError):
            self.full_forward(image=self.image[..., :32, :32])
        with self.assertRaises(ValueError):
            self.full_forward(action=torch.rand(2, 13))


if __name__ == "__main__":
    unittest.main()
