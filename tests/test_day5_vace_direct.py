import ast
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def worker_constants():
    source = (ROOT / "scripts" / "run_vace_direct_resident_batch.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    wanted = {"METHOD", "SEED", "NUM_FRAMES", "NUM_INFERENCE_STEPS", "VACE_SCALE"}
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                constants[target.id] = ast.literal_eval(node.value)
    if constants.keys() != wanted:
        raise AssertionError(f"Missing constants: {wanted - constants.keys()}")
    return constants, source


class VaceDirectTests(unittest.TestCase):
    def test_frozen_config_matches_worker(self):
        config = json.loads(
            (ROOT / "configs" / "vace_direct_static_proxy_v1.json").read_text()
        )
        constants, source = worker_constants()
        self.assertEqual(config["method"], constants["METHOD"])
        self.assertEqual(config["seed"], constants["SEED"])
        self.assertEqual(config["num_frames"], constants["NUM_FRAMES"])
        self.assertEqual(
            config["num_inference_steps"], constants["NUM_INFERENCE_STEPS"]
        )
        self.assertEqual(config["vace_scale"], constants["VACE_SCALE"])
        self.assertFalse(config["enable_ttm"])
        self.assertIn("enable_ttm=False", source)
        self.assertIn('proxy_dir / "motion_signal.png"', source)
        self.assertIn('proxy_dir / "edit_alpha.png"', source)

    def test_worker_is_offline_resident_and_fail_closed(self):
        _, source = worker_constants()
        self.assertIn('"DIFFSYNTH_SKIP_DOWNLOAD": "true"', source)
        self.assertIn("pipeline_load_count", source)
        self.assertIn("Refusing to reuse output root", source)
        self.assertIn("verify_model_audit", source)
        self.assertIn("extract_and_project", source)

    def test_completed_batch_and_review_are_traceable(self):
        root = ROOT / "results" / "day5_vace_direct_v1"
        summary = json.loads((root / "batch_summary_v1.json").read_text())
        self.assertEqual(summary["method"], "vace_direct_static_proxy")
        self.assertEqual(summary["case_count"], 4)
        self.assertEqual(summary["complete_count"], 4)
        self.assertEqual(summary["technical_failure_count"], 0)
        self.assertTrue(summary["resident_contract_passed"])
        self.assertTrue(summary["all_protected_pixels_exact"])
        self.assertTrue(summary["all_output_hashes_verified_after_transfer"])
        manifests = sorted(root.glob("gpu*/*/seed_1/run_manifest.json"))
        self.assertEqual(len(manifests), 4)
        for path in manifests:
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest["method"], "vace_direct_static_proxy")
            self.assertEqual(manifest["status"], "complete")
            self.assertFalse(manifest["inference"]["enable_ttm"])
            self.assertEqual(
                manifest["inference"]["outside_edit_alpha_max_pixel_difference"], 0
            )
            self.assertEqual(
                manifest["inference"]["pipeline_load_count_at_completion"], 1
            )
            self.assertEqual(
                manifest["environment"]["code_commit"],
                "a9c72f51d97e504dce7cfdb50c1ce212b683ceb0",
            )
            self.assertEqual(
                manifest["environment"]["worker_file"]["sha256"],
                "1c41d1dfca356575b8781a9ef8cb5cfdb15d8487035e56d4cffb19635ff14e9c",
            )
        for worker in summary["workers"]:
            path = root / worker["tracked_path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), worker["sha256"])
        review = json.loads((root / "internal_pilot_review_v1.json").read_text())
        self.assertEqual(review["aggregate"]["provisional_action_success"], 0)
        self.assertEqual(review["aggregate"]["provisional_photo_success"], 0)


if __name__ == "__main__":
    unittest.main()
