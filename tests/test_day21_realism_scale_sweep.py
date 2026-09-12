import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "day21_realism_scale_sweep_v1.json"
RUNNER = ROOT / "scripts" / "run_realism_scale_sweep.py"
DEPENDENCY = ROOT / "scripts" / "run_high_lift_vace_pilot.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Day21RealismScaleSweepTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_only_control_scale_is_swept(self):
        self.assertEqual(
            self.config["scientific_status"],
            "matched_development_diagnostic_of_proxy_control_strength_not_generalization",
        )
        self.assertEqual(self.config["cases"], ["soup", "rice", "cake"])
        self.assertEqual(self.config["control_arm"], "relative3d")
        self.assertEqual(self.config["vace_scales"], [0.6, 0.8])
        self.assertEqual(self.config["existing_baseline"]["vace_scale"], 1.0)
        self.assertEqual(
            self.config["inference"],
            {
                "seed": 1,
                "num_frames": 21,
                "num_inference_steps": 20,
                "fps": 8,
                "enable_ttm": False,
                "lora_enabled": False,
            },
        )

    def test_runner_and_dependency_are_hash_locked(self):
        self.assertEqual(self.config["implementation"]["runner_sha256"], sha256(RUNNER))
        self.assertEqual(
            self.config["implementation"]["dependency_sha256"], sha256(DEPENDENCY)
        )
        source = RUNNER.read_text(encoding="utf-8")
        self.assertEqual(source.count("load_pipeline("), 1)
        self.assertIn("Foreign compute process appeared", source)
        self.assertIn("HF_HUB_OFFLINE", CONFIG.read_text(encoding="utf-8"))
        self.assertNotIn("download", source.lower())

    def test_success_requires_realism_and_action(self):
        gate = self.config["review_gate"]
        self.assertTrue(gate["compare_against_existing_scale_1_baseline"])
        self.assertTrue(gate["require_action_retention"])
        self.assertTrue(gate["require_no_duplicate_payload"])
        self.assertTrue(gate["require_plausible_source_update"])
        self.assertTrue(gate["require_clear_boundary_or_material_improvement"])
        self.assertTrue(gate["uncertain_is_not_pass"])
        self.assertTrue(gate["no_automatic_best_frame_selection"])

    def test_resource_gate_remains_strict(self):
        gate = self.config["resource_gate"]
        self.assertEqual(gate["required_gpu_name"], "NVIDIA RTX A6000")
        self.assertGreaterEqual(gate["min_free_memory_mib"], 48000)
        self.assertLessEqual(gate["max_utilization_percent"], 5)
        self.assertGreaterEqual(gate["min_available_system_memory_mib"], 80000)
        self.assertTrue(gate["require_zero_compute_process"])
        self.assertTrue(gate["forbid_process_termination_or_preemption"])
        self.assertTrue(gate["forbid_a40_or_blackwell"])
        self.assertTrue(gate["forbid_model_download"])
        self.assertTrue(gate["require_new_non_overwriting_paths"])


if __name__ == "__main__":
    unittest.main()
