import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("recovery_batch", ROOT / "scripts/run_flexible_completion_recovery_batch.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RecoveryBatchTests(unittest.TestCase):
    def test_completed_arm_snapshot_is_hash_bound_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            arm_root = root / "arm"
            arm_root.mkdir()
            config = {"arms": {"planar_uniform": {"output_root": str(arm_root)}}}
            config_path = root / "config.json"
            MODULE.write_new(config_path, config)
            checkpoints = []
            for step in (16, 32):
                checkpoint = arm_root / f"step-{step}.safetensors"
                checkpoint.write_bytes(str(step).encode())
                checkpoints.append({"path": checkpoint.name, "size_bytes": 2, "sha256": MODULE.sha(checkpoint)})
            manifest = {"status": "complete_requires_cross_arm_match_and_checkpoint_validation",
                        "config_sha256": MODULE.sha(config_path), "return_code": 0,
                        "randomness_trace_count": 32, "checkpoints": checkpoints}
            manifest_path = arm_root / "run_manifest.json"
            MODULE.write_new(manifest_path, manifest)
            MODULE.verify_run(config_path, config, "planar_uniform")
            MODULE.snapshot_arm(arm_root, root, "planar_uniform")
            snapshot = MODULE.read(root / "planar_uniform_snapshot.json")
            self.assertEqual(snapshot["archive"]["sha256"], MODULE.sha(root / "planar_uniform.tar"))
            with self.assertRaises(FileExistsError):
                MODULE.snapshot_arm(arm_root, root, "planar_uniform")
            (arm_root / "step-32.safetensors").write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                MODULE.verify_run(config_path, config, "planar_uniform")

    def test_missing_checkpoints_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.json"
            config = {"arms": {"planar_uniform": {"output_root": str(root)}}}
            MODULE.write_new(config_path, config)
            MODULE.write_new(root / "run_manifest.json", {
                "status": "complete_requires_cross_arm_match_and_checkpoint_validation",
                "config_sha256": MODULE.sha(config_path), "return_code": 0,
                "randomness_trace_count": 32, "checkpoints": []})
            with self.assertRaises(ValueError):
                MODULE.verify_run(config_path, config, "planar_uniform")


if __name__ == "__main__":
    unittest.main()
