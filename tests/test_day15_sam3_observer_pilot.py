import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/day15_sam3_observer_pilot_v1.json"
RUNNER = ROOT / "scripts/run_sam3_observer_pilot.py"
SUMMARY = ROOT / "results/day15_sam3_observer_result_20260907.json"


class Day15Sam3ObserverPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_scope_is_observer_only(self):
        self.assertEqual(self.config["schema_version"], "foodstateedit.sam3_observer_pilot.v1")
        self.assertTrue(self.config["execution_policy"]["no_vace_inference"])
        self.assertTrue(self.config["execution_policy"]["do_not_treat_sam3_masks_as_ground_truth"])
        self.assertTrue(self.config["observer"]["manual_review_required"])

    def test_model_is_offline_and_hash_locked(self):
        sam = self.config["sam3"]
        self.assertFalse(sam["load_from_hf"])
        self.assertFalse(sam["qwen_or_evolutionary_agent_used"])
        self.assertEqual(len(sam["checkpoint"]["sha256"]), 64)
        self.assertEqual(len(sam["model_builder_sha256"]), 64)
        self.assertIn("load_from_HF=False", self.source)
        self.assertIn('HF_HUB_OFFLINE', self.source)

    def test_inputs_are_unique_and_hash_locked(self):
        records = self.config["inputs"]
        self.assertEqual(len(records), 7)
        self.assertEqual(len({record["id"] for record in records}), len(records))
        self.assertEqual(len({record["path"] for record in records}), len(records))
        for record in records:
            self.assertGreater(record["bytes"], 0)
            self.assertEqual(len(record["sha256"]), 64)

    def test_prompts_and_counterfactuals_are_frozen(self):
        self.assertEqual(set(self.config["prompt_groups"]), {"strand", "utensil"})
        self.assertEqual(len(self.config["prompt_groups"]["strand"]), 2)
        self.assertEqual(len(self.config["prompt_groups"]["utensil"]), 2)
        self.assertEqual(set(self.config["counterfactuals"]), {
            "strand_restored_to_source", "utensil_restored_to_source",
            "solid_noodle_color_block",
        })

    def test_resource_and_nonoverwrite_gate(self):
        runtime = self.config["runtime"]
        self.assertEqual(runtime["required_gpu_name"], "NVIDIA RTX A6000")
        self.assertGreaterEqual(runtime["minimum_free_memory_mib"], 48000)
        self.assertLessEqual(runtime["maximum_utilization_percent"], 5)
        self.assertTrue(runtime["require_zero_compute_processes"])
        self.assertIn("exist_ok=False", self.source)
        self.assertNotIn("kill(", self.source)
        self.assertNotIn("terminate(", self.source)
        self.assertNotIn('"manual_review_complete": false', self.source)

    def test_gate_contract_is_exact(self):
        expected = {
            "target_has_strand_candidate", "target_has_utensil_candidate",
            "target_strand_beats_source", "target_utensil_beats_source",
            "strand_erasure_lowers_strand_score",
            "utensil_erasure_lowers_utensil_score",
            "solid_block_not_preferred_as_strand",
            "weighted_output_has_strand_candidate",
            "weighted_output_has_utensil_candidate",
        }
        self.assertEqual(set(self.config["required_gates"]), expected)

    def test_completed_summary_keeps_negative_boundary(self):
        summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "observer_gate_failed_do_not_guide_vace")
        self.assertEqual(summary["gate_count"], {"passed": 3, "failed": 6, "total": 9})
        self.assertEqual(sum(summary["automated_gates"].values()), 3)
        self.assertFalse(summary["technical_visual_audit"]["strand_prompt_scope_correct"])
        self.assertFalse(summary["technical_visual_audit"]["two_chopstick_instance_separation_established"])
        self.assertEqual(summary["new_vace_inference_count"], 0)
        self.assertEqual(summary["evidence"]["hash_mismatches_after_pull"], 0)


if __name__ == "__main__":
    unittest.main()
