import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("posttraining", ROOT / "scripts/prepare_flexible_completion_posttraining.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PosttrainingTests(unittest.TestCase):
    def setUp(self):
        self.execution = json.loads((ROOT / "configs/flexible_completion_execution_recovery_20260906_v1.json").read_text())
        self.template = json.loads((ROOT / "configs/vace_phase_action_isolated_udon_checkpoint_sweep_v1.json").read_text())

    def test_template_only_normalizes_same_model_audit(self):
        normalized = MODULE.normalized_template(self.execution, self.template)
        normalized["runtime"]["model_hash_audit"] = self.template["runtime"]["model_hash_audit"]
        self.assertEqual(normalized, self.template)

    def test_reject_inference_or_model_audit_changes(self):
        for location, key, value in [("inference", "seed", 2), ("runtime", "model_hash_audit_sha256", "bad")]:
            changed = copy.deepcopy(self.template)
            changed[location][key] = value
            with self.assertRaises(ValueError):
                MODULE.normalized_template(self.execution, changed)

    def test_legacy_validator_metadata_is_explicit(self):
        config = MODULE.compatibility_config(self.execution, "planar_uniform")
        self.assertTrue(config["compatibility_only"])
        self.assertEqual(config["actual_unique_sample_count"], 1)
        self.assertEqual(config["training"]["output_root"], self.execution["arms"]["planar_uniform"]["output_root"])

    def test_validation_bound_to_checkpoint_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.safetensors"
            path.write_bytes(b"fixture")
            expected = MODULE.digest(path)
            report = {"status": "complete", "tensor_count": 160, "official_loader_updated_tensor_count": 80,
                      "checkpoint": str(path), "checkpoint_sha256": expected}
            MODULE.check_validation(report, path, expected)
            report["checkpoint_sha256"] = "wrong"
            with self.assertRaises(ValueError):
                MODULE.check_validation(report, path, expected)
            with self.assertRaises(FileExistsError):
                MODULE.write_new(path, {})


if __name__ == "__main__":
    unittest.main()
