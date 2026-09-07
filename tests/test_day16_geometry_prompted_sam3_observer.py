import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/day16_geometry_prompted_sam3_observer_v1.json"
RUNNER = ROOT / "scripts/run_geometry_prompted_sam3_observer.py"
SUMMARY = ROOT / "results/day16_geometry_prompted_sam3_result_20260907.json"


class Day16GeometryPromptedSam3ObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_scope_is_feasibility_only(self):
        self.assertEqual(
            self.config["schema_version"],
            "foodstateedit.geometry_prompted_sam3_observer.v1",
        )
        policy = self.config["execution_policy"]
        self.assertTrue(policy["no_vace_inference"])
        self.assertTrue(policy["geometry_labels_are_not_independent_ground_truth"])
        self.assertTrue(policy["guidance_requires_all_gates_plus_independent_labels"])

    def test_three_prompts_are_geometry_frozen(self):
        prompts = self.config["prompts_normalized"]
        self.assertEqual(set(prompts), {"strand", "near_chopstick", "far_chopstick"})
        for prompt in prompts.values():
            self.assertEqual(len(prompt["box_xyxy"]), 4)
            self.assertGreater(len(prompt["positive_points"]), 0)
            self.assertGreater(len(prompt["negative_points"]), 0)
            for value in prompt["box_xyxy"]:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_counterfactual_family_is_fixed(self):
        self.assertEqual(set(self.config["counterfactuals"]), {
            "strand_erased_to_source", "sticks_erased_to_source", "solid_strand_block",
        })

    def test_gate_contract_is_exact(self):
        self.assertEqual(len(self.config["required_gates"]), 9)
        self.assertEqual(len(set(self.config["required_gates"])), 9)
        self.assertIn("weighted_output_has_all_three_structures", self.config["required_gates"])
        self.assertIn("stick_erasure_lowers_alignment", self.config["required_gates"])

    def test_model_is_offline_hash_locked_and_text_free(self):
        sam = self.config["sam3"]
        self.assertFalse(sam["load_from_hf"])
        self.assertFalse(sam["text_prompt_used"])
        self.assertIn("load_from_HF=False", self.source)
        self.assertIn("add_box_and_point_prompt", self.source)
        self.assertNotIn("set_text_prompt", self.source)

    def test_runner_is_fail_closed_and_non_overwriting(self):
        self.assertIn("exist_ok=False", self.source)
        self.assertIn("no_vace_inference", json.dumps(self.config))
        self.assertNotIn("kill(", self.source)
        self.assertNotIn("terminate(", self.source)
        self.assertIn("Gate contract mismatch", self.source)

    def test_completed_summary_keeps_claim_boundary(self):
        summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
        self.assertEqual(summary["gate_count"], {"passed": 7, "failed": 2, "total": 9})
        self.assertEqual(sum(summary["automated_gates"].values()), 7)
        self.assertEqual(summary["execution"]["new_vace_inference_count"], 0)
        self.assertFalse(summary["automated_gates"]["control_sticks_are_distinct"])
        self.assertLess(
            summary["strand_iou"]["relative3d_topology_weighted_step32"],
            summary["strand_iou"]["relative3d_lora_off"],
        )
        self.assertEqual(summary["evidence"]["hash_mismatches_after_pull"], 0)


if __name__ == "__main__":
    unittest.main()
