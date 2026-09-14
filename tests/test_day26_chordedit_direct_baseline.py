import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/day26_chordedit_direct_baseline_v1.json"
RESULT_PATH = ROOT / "results/day26_chordedit_direct_baseline_result_v1.json"
RUNNER_PATH = ROOT / "scripts/run_chordedit_direct_baseline.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Day26ChordEditDirectBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    def test_frozen_comparison_is_same_input_three_seed_and_no_best_of(self) -> None:
        self.assertEqual(self.config["case_order"], ["ramen", "soup", "rice", "cake"])
        self.assertEqual(self.config["seeds"], [1, 2, 3])
        self.assertEqual(self.config["expected_generation_count"], 12)
        fairness = self.config["fairness_contract"]
        self.assertTrue(fairness["inputs_match_day25"])
        self.assertTrue(fairness["allowed_edit_regions_match_day25"])
        self.assertTrue(fairness["single_candidate_per_seed"])
        self.assertTrue(fairness["no_best_of_selection"])
        self.assertTrue(fairness["no_failed_seed_replacement"])

    def test_runner_is_hash_bound_offline_and_fail_closed(self) -> None:
        self.assertEqual(sha256_file(RUNNER_PATH), self.config["implementation"]["runner_sha256"])
        backend = self.config["backend"]
        self.assertTrue(backend["offline_only"])
        self.assertTrue(backend["downloads_forbidden"])
        gate = self.config["resource_gate"]
        self.assertEqual(gate["required_gpu_name"], "NVIDIA RTX A6000")
        self.assertTrue(gate["require_zero_compute_process"])
        self.assertTrue(gate["require_new_non_overwriting_paths"])
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("ChordEditPipeline.from_local_weights", source)
        self.assertIn('HF_HUB_OFFLINE="1"', source)
        self.assertIn("Refusing to overwrite output or preflight evidence", source)

    def test_completed_result_separates_execution_from_visual_failure(self) -> None:
        execution = self.result["execution"]
        review = self.result["provisional_internal_review"]
        self.assertEqual(execution["completed_outputs"], 12)
        self.assertEqual(execution["remote_files_rehashed_locally"], 72)
        self.assertEqual(execution["remote_to_local_sha256_mismatches"], 0)
        self.assertEqual(execution["outside_support_max_pixel_difference"], 0)
        self.assertEqual(review["action_success"], {"pass": 0, "total": 12})
        self.assertEqual(review["preservation_success"], {"pass": 12, "total": 12})
        self.assertEqual(review["strict_end_to_end_success"], {"pass": 0, "total": 12})
        self.assertFalse(self.result["scope"]["independent_blind_review"])

    def test_result_keeps_named_editor_and_held_out_gates_open(self) -> None:
        impact = self.result["formal_benchmark_impact"]
        self.assertTrue(impact["available_external_editor_baseline_added"])
        self.assertFalse(impact["strong_named_editor_requirement_fully_closed"])
        self.assertFalse(impact["held_out_test_unlocked"])
        self.assertEqual(
            impact["missing_named_editors"],
            ["Qwen-Image-Edit", "FLUX Kontext", "ChronoEdit"],
        )
        claim = self.result["claim_limit"].lower()
        for forbidden_claim in ("superiority", "generalization", "physical correctness"):
            self.assertIn(forbidden_claim, claim)


if __name__ == "__main__":
    unittest.main()
