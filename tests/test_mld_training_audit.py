"""Adversarial tests for training provenance and synthetic split audits."""
from pathlib import Path
import importlib.util
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("audit_mld_training", Path(__file__).resolve().parents[1] / "scripts" / "audit_mld_training.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditFailureTests(unittest.TestCase):
    def test_target_conditioning_is_rejected(self):
        for target in audit.TARGET_ONLY_KEYS:
            with self.subTest(target=target), self.assertRaises(ValueError):
                audit.validate_input_keys(["source_rgb", "canonical_xyz", target])

    def test_same_shape_material_identity_cannot_cross_split(self):
        with self.assertRaises(ValueError):
            audit.validate_split_identity(["shared", "validation", "shared"], np.array([0, 1, 2]))

    def test_split_count_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            audit.validate_split_identity(["a", "b", "c"], np.array([0, 1, 2]), (2, 1, 0))

    def test_target_visible_sampling_is_rejected_as_canonical_grid(self):
        xyz = np.array([[0, 0, 0]], dtype=np.float32)
        with self.assertRaises(ValueError):
            audit.validate_known_grid(xyz, np.zeros((1, 2)), 2)

    def test_fraction_outside_budget_rejected(self):
        with self.assertRaises(ValueError):
            audit.validate_fraction_budget(np.array([[1.01, 0.5]]), np.ones(2))

    def test_stale_or_untrained_module_checkpoint_rejected(self):
        names = ("posterior", "transition", "diffusion")
        initial = {name + ".weight": np.ones((2, 3), np.float32) for name in names}
        final = {name + ".weight": np.ones((2, 3), np.float32) * 2 for name in names}
        final["transition.weight"] = initial["transition.weight"].copy()
        with self.assertRaises(ValueError):
            audit.checkpoint_differences(initial, final, names)

    def test_nonfinite_weights_rejected(self):
        with self.assertRaises(ValueError):
            audit.checkpoint_differences({"module.weight": np.ones(3)}, {"module.weight": np.array([2, np.nan, 2])}, ("module",))

    def test_historical_source_cannot_be_new_probe(self):
        with self.assertRaises(ValueError):
            audit.validate_probe_exclusion([{"file_name": "7442.jpg", "sha256": "a" * 64}], {"source_numeric_ids": ["7442"]})

    def test_duplicate_content_with_different_filename_rejected(self):
        with self.assertRaises(ValueError):
            audit.validate_probe_exclusion([{"file_name": "10.jpg", "sha256": "a" * 64}, {"file_name": "11.jpg", "sha256": "a" * 64}], {})

    def test_changed_modules_are_quantified(self):
        result = audit.checkpoint_differences({"module.weight": np.ones(3)}, {"module.weight": np.array([1., 2., 1.])}, ("module",))
        self.assertEqual(result["module"]["changed_parameters"], 1)
        self.assertEqual(result["module"]["delta_l2"], 1)

    def test_fraction_budget_uses_reference_mass(self):
        result = audit.validate_fraction_budget(np.array([[.1, .8], [.7, .2]]), np.array([.125, .875]))
        self.assertLessEqual(result["maximum_reference_budget_residual"], np.finfo(np.float64).eps)

    def test_noise_reconstruction_is_keyed_by_material_identity(self):
        identities = ["source:lattice:14", "source:lattice:5"]
        first = audit.independent_material_noise(identities, 41)
        reversed_rows = audit.independent_material_noise(identities[::-1], 41)
        self.assertTrue(np.array_equal(first[::-1], reversed_rows))
        self.assertFalse(np.array_equal(first, audit.independent_material_noise(identities, 42)))

    def test_state_coupled_checkpoint_requires_ninth_learned_module(self):
        legacy = audit.expected_modules_for_config({})
        coupled = audit.expected_modules_for_config({"state_dim": 4})
        self.assertEqual(len(legacy), 8)
        self.assertEqual(len(coupled), 9)
        self.assertIn("state_encoder", coupled)
        with self.assertRaises(ValueError):
            audit.expected_modules_for_config({"state_dim": 3})

    def test_actual_small_generator_dataset_replays_without_target_inputs(self):
        generator_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_mld_pretraining_data.py"
        generator = audit.load_generator(generator_path)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "synthetic"
            generator.prepare_dataset(root, scenes=8, grid_size=2, image_size=8)
            result = audit.Audit()
            arrays = audit.audit_dataset(result, root, generator_path)
            try:
                failures = [check for check in result.checks if not check["passed"]]
                self.assertIsNotNone(arrays)
                self.assertEqual(failures, [])
            finally:
                for array in (arrays or {}).values():
                    if getattr(array, "_mmap", None) is not None:
                        array._mmap.close()


if __name__ == "__main__":
    unittest.main()
