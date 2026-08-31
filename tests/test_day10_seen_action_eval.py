import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "vace_seen_action_lora_compare_v1.json"
DATASET_MANIFEST = ROOT / "artifacts" / "day9_action_pseudo_dataset_v3" / "dataset_manifest.json"
RESULT = ROOT / "results" / "day10_seen_action_lora_result_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Day10SeenActionEvalTests(unittest.TestCase):
    def test_contract_is_seen_family_diagnostic_only(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["method"], "vace_seen_action_lora_same_seed_compare_v1")
        self.assertEqual(config["scientific_status"], "seen_training_family_underfit_diagnostic_not_generalization")
        self.assertEqual(len(config["samples"]), 2)
        self.assertEqual({sample["family"] for sample in config["samples"]}, {"strand", "liquid"})
        self.assertEqual(config["adapter"]["expected_injected_linear_count"], 80)
        self.assertEqual(config["adapter"]["alpha"], 1.0)
        self.assertEqual(config["inference"]["seed"], 1)
        self.assertEqual(config["inference"]["num_frames"], 21)
        self.assertEqual(config["inference"]["num_inference_steps"], 20)
        self.assertFalse(config["inference"]["enable_ttm"])

    def test_sample_files_and_prompts_match_frozen_training_dataset(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(config["dataset"]["manifest_sha256"], sha256_file(DATASET_MANIFEST))
        expected = {sample["sample_id"]: sample for sample in manifest["samples"]}
        for sample in config["samples"]:
            source = expected[sample["sample_id"]]
            self.assertEqual(sample["prompt"], source["prompt"])
            self.assertEqual(sample["dimensions"], [source["canvas"]["width"], source["canvas"]["height"]])
            self.assertEqual(sample["files"]["reference"], source["files"]["vace_reference_image"])
            self.assertEqual(sample["files"]["control"], source["files"]["vace_video"])
            self.assertEqual(sample["files"]["alpha"], source["files"]["edit_alpha"])
            self.assertEqual(sample["files"]["target"], source["files"]["target_keyframe"])

    def test_launchers_are_hashed_resident_and_fail_closed(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        preflight_path = ROOT / config["launcher"]["preflight"]["path"]
        runner_path = ROOT / config["launcher"]["runner"]["path"]
        preflight = preflight_path.read_text(encoding="utf-8")
        runner = runner_path.read_text(encoding="utf-8")
        self.assertEqual(config["launcher"]["preflight"]["sha256"], sha256_file(preflight_path))
        self.assertEqual(config["launcher"]["runner"]["sha256"], sha256_file(runner_path))
        self.assertIn("output_absent", preflight)
        self.assertIn("gpu_gate", preflight)
        self.assertIn("Preflight blocked inference", runner)
        self.assertEqual(runner.count("inference.load_pipeline("), 1)
        self.assertIn('for condition in ("lora_off", "lora_on")', runner)
        self.assertIn("install_runtime_injection(pipe", runner)
        self.assertIn("injected_count(pipe.vace2) != 0", runner)
        self.assertIn("outside_support_max_pixel_difference", runner)
        self.assertIn("lora_on_closer_to_target", runner)
        self.assertNotIn("modelscope download", runner.lower())

    def test_closed_result_records_underfit_without_generalization_claim(self):
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(
            result["scientific_status"],
            "closed_seen_family_underfit_diagnostic_not_generalization",
        )
        self.assertEqual(result["aggregate"]["technical_complete"], 2)
        self.assertEqual(result["aggregate"]["technical_failure"], 0)
        self.assertEqual(result["aggregate"]["lora_on_semantically_improved"], 0)
        self.assertTrue(result["diagnosis"]["adapter_execution_verified"])
        self.assertTrue(result["diagnosis"]["current_adapter_underfit"])
        self.assertFalse(result["decision"]["generalization_claim_allowed"])
        self.assertFalse(result["decision"]["adapter_benefit_claim_allowed"])

    def test_result_hashes_match_pulled_manifests_and_preflights(self):
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        for sample in result["samples"]:
            sample_id = sample["sample_id"]
            manifest = (
                ROOT
                / "artifacts"
                / "day10_seen_action_lora_compare_v1"
                / sample_id
                / "run_manifest.json"
            )
            self.assertEqual(sample["run_manifest_sha256"], sha256_file(manifest))
        self.assertEqual(
            result["samples"][0]["preflight_sha256"],
            sha256_file(ROOT / "results" / "day10_seen_udon_preflight_gp39_v1.json"),
        )
        self.assertEqual(
            result["samples"][1]["preflight_sha256"],
            sha256_file(ROOT / "results" / "day10_seen_spoon_preflight_gp39_v1.json"),
        )


if __name__ == "__main__":
    unittest.main()
