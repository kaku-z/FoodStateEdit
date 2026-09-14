import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/day27_qwen_image_edit_direct_baseline_v1.json"
RESULT_PATH = ROOT / "results/day27_qwen_image_edit_direct_baseline_result_v1.json"
RUNNER_PATH = ROOT / "scripts/run_qwen_image_edit_direct_baseline.py"
REVIEW_MANIFEST_PATH = (
    ROOT / "artifacts/day27_qwen_image_edit_direct_baseline_v1/review_grid_manifest.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Day27QwenImageEditDirectBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        cls.review = json.loads(REVIEW_MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_frozen_run_is_same_input_three_seed_and_no_best_of(self) -> None:
        self.assertEqual(self.config["case_order"], ["ramen", "soup", "rice", "cake"])
        self.assertEqual(self.config["seeds"], [1, 2, 3])
        self.assertEqual(self.config["expected_generation_count"], 12)
        fairness = self.config["fairness_contract"]
        self.assertTrue(fairness["inputs_match_day25_and_day26"])
        self.assertTrue(fairness["single_candidate_per_seed"])
        self.assertTrue(fairness["no_best_of_selection"])
        self.assertTrue(fairness["no_failed_seed_replacement"])

    def test_runner_is_hash_bound_and_offline(self) -> None:
        self.assertEqual(sha256_file(RUNNER_PATH), self.config["implementation"]["runner_sha256"])
        self.assertTrue(self.config["backend"]["offline_only"])
        self.assertTrue(self.config["backend"]["downloads_forbidden"])
        self.assertEqual(self.config["resource_gate"]["required_gpu_name"], "NVIDIA RTX A6000")
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("QwenImageEditPlusPipeline.from_pretrained", source)
        self.assertIn('HF_HUB_OFFLINE="1"', source)
        self.assertIn("Refusing to overwrite output or preflight evidence", source)

    def test_completed_result_separates_model_output_from_failed_composite(self) -> None:
        execution = self.result["execution"]
        endpoint = self.result["endpoint_decision"]
        review = self.result["provisional_internal_review"]
        self.assertEqual(execution["completed_outputs"], 12)
        self.assertEqual(execution["remote_output_files_rehashed_locally"], 62)
        self.assertEqual(execution["remote_output_sha256_mismatches"], 0)
        self.assertEqual(execution["pipeline_load_count"], 1)
        self.assertEqual(self.result["baseline_endpoint"], "raw_qwen.png")
        self.assertFalse(endpoint["support_locked_composite_is_baseline"])
        self.assertFalse(endpoint["masked_composite_preservation_credit_allowed"])
        self.assertEqual(review["photo_success"], {"pass": 12, "total": 12})
        self.assertEqual(review["preservation_success"], {"pass": 0, "total": 12})
        self.assertEqual(review["strict_end_to_end_success"], {"pass": 0, "total": 12})

    def test_review_and_claim_boundaries_remain_open(self) -> None:
        self.assertEqual(self.review["baseline_endpoint"], "raw_qwen.png")
        self.assertEqual(len(self.review["raw_output_preservation_diagnostics"]), 12)
        impact = self.result["formal_benchmark_impact"]
        self.assertTrue(impact["qwen_named_editor_baseline_added"])
        self.assertFalse(impact["strong_named_editor_requirement_fully_closed"])
        self.assertEqual(impact["missing_named_editors"], ["FLUX Kontext", "ChronoEdit"])
        self.assertFalse(impact["held_out_test_unlocked"])
        claim = self.result["claim_limit"].lower()
        for forbidden_claim in ("superiority", "generalization", "physical correctness"):
            self.assertIn(forbidden_claim, claim)


if __name__ == "__main__":
    unittest.main()
