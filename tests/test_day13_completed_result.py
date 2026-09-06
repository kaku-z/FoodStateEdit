import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/DAY13_FLEXIBLE_COMPLETION_RESULT_20260906.json"


class CompletedResultTests(unittest.TestCase):
    def setUp(self):
        self.result = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_primary_gate_is_recomputed_and_negative(self):
        result = self.result
        baseline = result["conditions"][result["primary_comparison"]["baseline"]]
        proposed = result["conditions"][result["primary_comparison"]["proposed"]]
        topology = 100 * (baseline["topology_mae"] - proposed["topology_mae"]) / baseline["topology_mae"]
        pinch = 100 * (baseline["pinch_mae"] - proposed["pinch_mae"]) / baseline["pinch_mae"]
        self.assertAlmostEqual(topology, result["primary_comparison"]["topology_mae_improvement_percent"], places=10)
        self.assertAlmostEqual(pinch, result["primary_comparison"]["pinch_mae_improvement_percent"], places=10)
        self.assertLess(topology, result["primary_comparison"]["required_percent_each"])
        self.assertLess(pinch, result["primary_comparison"]["required_percent_each"])
        self.assertFalse(result["primary_comparison"]["numeric_gate_pass"])
        self.assertFalse(result["primary_comparison"]["positive_contribution_gate_pass"])

    def test_execution_and_transfer_contracts_are_recorded(self):
        result = self.result
        self.assertEqual(result["pipeline_load_count"], 1)
        self.assertEqual(result["all_conditions_decoded_frames"], 21)
        self.assertEqual(result["all_outside_support_max_pixel_difference"], 0)
        self.assertEqual(result["training_match"]["validated_checkpoint_count"], 6)
        self.assertEqual(result["training_match"]["cross_arm_match_status"], "complete")
        self.assertEqual(result["transfer_audit"]["remote_file_count"], result["transfer_audit"]["local_file_count"])
        self.assertEqual(result["transfer_audit"]["sha256_mismatch"], 0)
        self.assertIn("negative", result["status"])


if __name__ == "__main__":
    unittest.main()
