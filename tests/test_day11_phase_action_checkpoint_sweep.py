import copy
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "vace_phase_action_checkpoint_sweep_v1.json"
DERIVED_CONFIG = (
    ROOT
    / "configs"
    / "vace_phase_action_checkpoint_sweep_gp39_retry_v2_20260901T1119Z.json"
)
DATASET_MANIFEST = ROOT / "artifacts" / "day11_phase_action_overfit_dataset_v1" / "dataset_manifest.json"
SPOON_BLOCKED_PREFLIGHT = ROOT / "results" / "day11_phase_action_checkpoint_sweep_preflight_spoon_gp40_blocked_v1.json"
UDON_RACE_PREFLIGHT = ROOT / "results" / "day11_phase_action_checkpoint_sweep_preflight_udon_gp40_race_v1.json"
RUN_ROOT = (
    ROOT
    / "artifacts"
    / "day11_phase_action_checkpoint_sweep_gp39_retry_v2_20260901T1119Z"
)
EVIDENCE_ROOT = (
    ROOT
    / "results"
    / "day11_phase_action_checkpoint_sweep_gp39_retry_v2_20260901T1119Z"
)
RESULT = ROOT / "results" / "day11_phase_action_checkpoint_sweep_result_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Day11PhaseActionCheckpointSweepTests(unittest.TestCase):
    def test_contract_is_seen_phase_capacity_only(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["method"], "vace_phase_action_checkpoint_sweep_v1")
        self.assertEqual(config["scientific_status"], "seen_phase_overfit_capacity_diagnostic_not_generalization")
        self.assertEqual([item["name"] for item in config["adapters"]["conditions"]], ["lora_off", "step_16", "step_32", "step_48", "step_64"])
        self.assertEqual([item["step"] for item in config["adapters"]["conditions"]], [0, 16, 32, 48, 64])
        self.assertEqual(config["inference"]["seed"], 1)
        self.assertEqual(config["inference"]["num_frames"], 21)
        self.assertEqual(config["inference"]["num_inference_steps"], 20)
        self.assertEqual(config["inference"]["review_frame_indices"], [0, 3, 6, 10, 15, 20])
        self.assertFalse(config["inference"]["enable_ttm"])
        self.assertTrue(config["decision_gate"]["blind_fork_requires_clear_seen_semantic_gain"])

    def test_samples_match_phase_varying_dataset(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(config["dataset"]["manifest_sha256"], sha256_file(DATASET_MANIFEST))
        expected = {sample["sample_id"]: sample for sample in manifest["samples"]}
        mapping = {"reference": "vace_reference_image", "control": "vace_video", "alpha": "edit_alpha", "target_video": "video", "target_keyframe": "target_keyframe"}
        for sample in config["samples"]:
            source = expected[sample["sample_id"]]
            self.assertEqual(sample["prompt"], source["prompt"])
            for key, source_key in mapping.items():
                self.assertEqual(sample["files"][key], source["files"][source_key])

    def test_launchers_are_hashed_resident_and_replace_injections(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        preflight_path = ROOT / config["launcher"]["preflight"]["path"]
        runner_path = ROOT / config["launcher"]["runner"]["path"]
        preflight = preflight_path.read_text(encoding="utf-8")
        runner = runner_path.read_text(encoding="utf-8")
        self.assertEqual(config["launcher"]["preflight"]["sha256"], sha256_file(preflight_path))
        self.assertEqual(config["launcher"]["runner"]["sha256"], sha256_file(runner_path))
        for name, expected in config["launcher"]["dependencies"].items():
            self.assertEqual(expected, sha256_file(ROOT / "scripts" / name))
        self.assertIn("output_absent", preflight)
        self.assertIn("gpu_gate", preflight)
        self.assertIn("Preflight blocked inference", runner)
        self.assertEqual(runner.count("inference.load_pipeline("), 1)
        self.assertIn("remove_runtime_injection", runner)
        self.assertIn("install_runtime_injection", runner)
        self.assertIn("target_rgb_mae_inside_support_by_phase", runner)
        self.assertIn("outside_support_max_pixel_difference", runner)
        self.assertNotIn("modelscope download", runner.lower())

    def test_first_launch_is_preserved_as_resource_evidence_not_results(self):
        spoon = json.loads(SPOON_BLOCKED_PREFLIGHT.read_text(encoding="utf-8"))
        udon = json.loads(UDON_RACE_PREFLIGHT.read_text(encoding="utf-8"))
        self.assertFalse(spoon["ready"])
        self.assertEqual(
            [check["id"] for check in spoon["checks"] if not check["passed"]],
            ["gpu_gate"],
        )
        self.assertTrue(udon["ready"])
        self.assertEqual(udon["selected_gpu"], 0)
        self.assertEqual(
            sha256_file(SPOON_BLOCKED_PREFLIGHT),
            "e719b13ce9218ebdff89a21905e246556ebdc6b6ded3d046bab88cce516c2015",
        )
        self.assertEqual(
            sha256_file(UDON_RACE_PREFLIGHT),
            "d5359018d9d875a51fca201da6ba8333fe228ed5f1e215ef5ecf02132a59d0a2",
        )

    def test_gp39_retry_changes_only_immutable_remote_paths(self):
        original = json.loads(CONFIG.read_text(encoding="utf-8"))
        derived = json.loads(DERIVED_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(
            sha256_file(DERIVED_CONFIG),
            "25b85a4d824443d9c51b040f871771f70004f64a5bbb7c4afd041921c8c6d7e0",
        )
        normalized = copy.deepcopy(derived)
        normalized["output_base"] = original["output_base"]
        normalized["adapters"]["training_manifest"]["path"] = original["adapters"][
            "training_manifest"
        ]["path"]
        for original_condition, derived_condition in zip(
            original["adapters"]["conditions"],
            normalized["adapters"]["conditions"],
        ):
            if original_condition["checkpoint"] is not None:
                derived_condition["checkpoint"]["path"] = original_condition["checkpoint"]["path"]
        self.assertEqual(normalized, original)

    def test_conflict_free_retry_completed_and_local_hashes_match(self):
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        expected_conditions = [
            "lora_off",
            "step_16",
            "step_32",
            "step_48",
            "step_64",
        ]
        self.assertEqual(result["aggregate"]["technical_complete"], 2)
        self.assertEqual(result["aggregate"]["technical_failure"], 0)
        self.assertEqual(result["execution"]["manifest_output_files_verified"], 36)
        self.assertEqual(result["execution"]["local_sha256_mismatch_count"], 0)
        for sample in result["samples"]:
            sample_root = RUN_ROOT / sample["sample_id"]
            manifest_path = sample_root / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(sample["run_manifest_sha256"], sha256_file(manifest_path))
            self.assertEqual(
                manifest["status"],
                "complete_requires_seen_phase_action_and_photo_review",
            )
            self.assertEqual(manifest["pipeline_load_count"], 1)
            self.assertEqual(list(manifest["conditions"]), expected_conditions)
            self.assertTrue((sample_root / "COMPLETE").is_file())
            self.assertFalse((sample_root / "RUNNING").exists())
            self.assertFalse((sample_root / "FAILED").exists())
            for condition in manifest["conditions"].values():
                self.assertEqual(condition["decoded_frames"], 21)
                self.assertEqual(condition["outside_support_max_pixel_difference"], 0)
            for condition_name, expected_hash in sample["contact_sheet_sha256"].items():
                review = sample_root / f"{condition_name}_projected_review.png"
                self.assertEqual(sha256_file(review), expected_hash)
            self.assertEqual(len(manifest["outputs"]), 18)
            for record in manifest["outputs"]:
                local_output = sample_root / Path(record["path"]).name
                self.assertGreater(local_output.stat().st_size, 0)
                self.assertEqual(local_output.stat().st_size, record["size_bytes"])
                self.assertEqual(sha256_file(local_output), record["sha256"])

    def test_review_closes_capacity_gate_without_broad_claims(self):
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(
            result["scientific_status"],
            "closed_seen_synthetic_overfit_checkpoint_diagnostic_no_clear_semantic_gain_not_generalization",
        )
        self.assertEqual(
            result["aggregate"]["samples_with_clear_phase_action_contact_improvement"],
            0,
        )
        self.assertEqual(
            result["aggregate"]["samples_with_clear_photo_realism_improvement"],
            0,
        )
        self.assertFalse(result["aggregate"]["clear_seen_semantic_gain"])
        self.assertFalse(result["decision"]["seen_synthetic_overfit_capacity_gate_passed"])
        self.assertFalse(result["decision"]["balanced_multifamily_expansion_allowed"])
        self.assertFalse(result["decision"]["blind_fork_evaluation_allowed"])
        self.assertFalse(result["decision"]["generalization_claim_allowed"])
        for sample in result["samples"]:
            self.assertFalse(
                sample["phase_action_contact_review"][
                    "clear_improvement_over_lora_off"
                ]
            )
            self.assertFalse(
                sample["photo_realism_review"]["clear_improvement_over_lora_off"]
            )

    def test_pulled_retry_preflight_hashes_match_result(self):
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        evidence = result["evidence"]
        expected = {
            "udon_standalone_preflight_sha256": "udon_standalone_preflight.json",
            "udon_runner_preflight_sha256": "udon_runner_preflight.json",
            "spoon_standalone_preflight_sha256": "spoon_standalone_preflight.json",
            "spoon_runner_preflight_sha256": "spoon_runner_preflight.json",
            "derived_config_sha256": "derived_config.json",
        }
        for key, filename in expected.items():
            self.assertEqual(evidence[key], sha256_file(EVIDENCE_ROOT / filename))
        for filename in ("udon_runner_preflight.json", "spoon_runner_preflight.json"):
            preflight = json.loads((EVIDENCE_ROOT / filename).read_text(encoding="utf-8"))
            self.assertTrue(preflight["ready"])


if __name__ == "__main__":
    unittest.main()
