import hashlib
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = ROOT / "configs/flexible_completion_udon_relative3d_geometry_high_lift_v1.json"
CONFIG = ROOT / "configs/day17_high_lift_vace_pilot_v1.json"
RUNNER = ROOT / "scripts/run_high_lift_vace_pilot.py"


class Day17HighLiftVacePilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_lift_is_more_than_twice_day13(self):
        width, height = self.geometry["amplitude_audit"]["image_size_wh"]
        contact = self.geometry["anchors_normalized"]["contact_pinch_uv"]
        final = self.geometry["anchors_normalized"]["final_pinch_uv"]
        distance = math.hypot((contact[0] - final[0]) * width,
                              (contact[1] - final[1]) * height)
        self.assertAlmostEqual(distance, 150.314, places=3)
        self.assertGreater(self.geometry["amplitude_audit"]["amplitude_ratio"], 2.3)
        self.assertLess(final[1], 0.2)

    def test_geometry_keeps_topology_contract(self):
        constraints = self.geometry["constraints"]
        self.assertTrue(constraints["exactly_two_rigid_chopsticks"])
        self.assertTrue(constraints["one_continuous_flexible_strand"])
        self.assertTrue(constraints["one_fixed_bowl_connection"])
        self.assertTrue(constraints["strand_depth_crosses_both_chopstick_depths"])
        self.assertTrue(constraints["all_rendered_changes_inside_frozen_edit_support"])

    def test_inference_is_single_lora_off_condition(self):
        self.assertEqual(self.config["method"], "relative3d_high_lift_udon_vace_lora_off_v1")
        self.assertEqual(self.config["inference"]["seed"], 1)
        self.assertEqual(self.config["inference"]["num_frames"], 21)
        self.assertEqual(self.config["inference"]["num_inference_steps"], 20)
        self.assertFalse(self.config["inference"]["enable_ttm"])
        self.assertIn('"lora_enabled": False', self.source)
        self.assertNotIn("install_runtime_injection", self.source)

    def test_runner_is_hash_bound_offline_and_nonoverwriting(self):
        actual = hashlib.sha256(RUNNER.read_bytes()).hexdigest()
        self.assertEqual(actual, self.config["implementation"]["runner_sha256"])
        self.assertIn("exist_ok=False", self.source)
        self.assertIn("HF_HUB_OFFLINE", json.dumps(self.config))
        self.assertNotIn("kill(", self.source)
        self.assertNotIn("terminate(", self.source)

    def test_claim_and_manual_review_remain_limited(self):
        self.assertIn("control following only", self.config["claim_limit"])
        self.assertTrue(self.config["review_gate"]["automatic_success_forbidden"])
        self.assertTrue(self.config["review_gate"]["photo_realism_reviewed_separately"])


if __name__ == "__main__":
    unittest.main()
