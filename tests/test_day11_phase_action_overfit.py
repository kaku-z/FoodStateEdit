import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adapter_phase_action_overfit_v1.json"
DATASET_ROOT = ROOT / "artifacts" / "day11_phase_action_overfit_dataset_v1"
MANIFEST = DATASET_ROOT / "dataset_manifest.json"
TRAINING_ROOT = ROOT / "artifacts" / "day11_phase_action_overfit_high_noise_lora_gp40_v1"
RUN_MANIFEST = TRAINING_ROOT / "run_manifest.json"
RESULT = ROOT / "results" / "day11_phase_action_overfit_training_result_v1.json"
INDEPENDENT_PREFLIGHT = ROOT / "results" / "day11_phase_action_overfit_preflight_gp40_v1.json"
RUNNER_PREFLIGHT = ROOT / "results" / "day11_phase_action_overfit_runner_preflight_gp40_v1.json"
STEP64_VALIDATION = ROOT / "results" / "day11_phase_action_overfit_step64_validation_gp40_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


class Day11PhaseActionOverfitTests(unittest.TestCase):
    def test_dataset_is_dynamic_bounded_and_claim_limited(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["experiment_variant"], "phase_varying_overfit_sanity_v1")
        self.assertEqual(manifest["sample_count"], 2)
        self.assertEqual(manifest["frames"], 21)
        self.assertEqual(manifest["held_out"]["training_occurrences"], 0)
        self.assertIn("not_physical_ground_truth", manifest["motion_policy"])
        for sample in manifest["samples"]:
            checks = sample["nonidentity_checks"]
            self.assertEqual(sample["frame_count"], 21)
            self.assertEqual(checks["target_outside_edit_support_max_difference"], 0)
            self.assertEqual(checks["control_outside_edit_support_max_difference"], 0)
            self.assertTrue(checks["frame0_equals_reference"])
            self.assertTrue(checks["final_target_equals_day9_keyframe"])
            self.assertTrue(checks["final_control_equals_day9_keyframe"])
            self.assertGreaterEqual(checks["unique_target_frames"], 10)
            self.assertGreaterEqual(checks["unique_control_frames"], 10)
            self.assertLess(checks["edit_support_fraction"], 0.2)
            for record in sample["files"].values():
                path = DATASET_ROOT / record["path"]
                self.assertEqual(path.stat().st_size, record["size_bytes"])
                self.assertEqual(sha256_file(path), record["sha256"])

    def test_config_freezes_a_64_step_overfit_gate_not_generalization(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        training = config["training"]
        self.assertEqual(config["experiment_variant"], "phase_varying_overfit_sanity_v1")
        self.assertIn("cannot establish generalization", config["claim_limit"])
        self.assertEqual(training["expected_optimizer_steps"], 64)
        self.assertEqual(
            training["expected_optimizer_steps"],
            config["dataset"]["sample_count"] * training["dataset_repeat"] * training["epochs"],
        )
        self.assertEqual(training["expected_checkpoint_steps"], [16, 32, 48, 64])
        self.assertTrue(config["evaluation_gate"]["semantic_gain_required_before_blind_test"])
        self.assertEqual(config["held_out"]["training_occurrences"], 0)
        self.assertEqual(sha256_file(MANIFEST), config["dataset"]["manifest_sha256"])

    def test_builder_and_launchers_are_hash_locked_offline_and_fail_closed(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        builder = ROOT / config["trainer"]["dataset_builder"]
        preflight = ROOT / config["launcher"]["preflight"]
        preflight_dependency = ROOT / config["launcher"]["preflight_dependency"]
        runner = ROOT / config["launcher"]["runner"]
        self.assertEqual(sha256_lf(builder), config["trainer"]["dataset_builder_sha256_lf"])
        self.assertEqual(sha256_file(preflight), config["launcher"]["preflight_sha256"])
        self.assertEqual(
            sha256_file(preflight_dependency),
            config["launcher"]["preflight_dependency_sha256"],
        )
        self.assertEqual(sha256_file(runner), config["launcher"]["runner_sha256"])
        builder_text = builder.read_text(encoding="utf-8")
        runner_text = runner.read_text(encoding="utf-8")
        self.assertIn("Refusing to reuse output root", builder_text)
        self.assertIn("phase_schedule", builder_text)
        self.assertIn("outside_union_max_difference", builder_text)
        self.assertIn("Preflight blocked training", runner_text)
        self.assertIn("HF_HUB_OFFLINE", runner_text)
        self.assertNotIn("modelscope download", runner_text.lower())

    def test_training_completed_four_checkpoints_without_unlocking_blind_evaluation(self):
        run = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        validation = json.loads(STEP64_VALIDATION.read_text(encoding="utf-8"))
        self.assertEqual(run["status"], "complete_requires_seen_phase_checkpoint_sweep")
        self.assertEqual(run["return_code"], 0)
        self.assertEqual(run["expected_optimizer_steps"], 64)
        self.assertEqual(
            [record["path"] for record in run["checkpoints"]],
            [f"step-{step}.safetensors" for step in (16, 32, 48, 64)],
        )
        for record in run["checkpoints"]:
            checkpoint = TRAINING_ROOT / record["path"]
            self.assertEqual(checkpoint.stat().st_size, record["size_bytes"])
            self.assertEqual(sha256_file(checkpoint), record["sha256"])
        self.assertEqual(result["training"]["run_manifest_sha256"], sha256_file(RUN_MANIFEST))
        self.assertEqual(
            result["training"]["independent_preflight_sha256"],
            sha256_file(INDEPENDENT_PREFLIGHT),
        )
        self.assertEqual(
            result["training"]["runner_preflight_sha256"], sha256_file(RUNNER_PREFLIGHT)
        )
        self.assertEqual(
            result["step_64_validation"]["report_sha256"], sha256_file(STEP64_VALIDATION)
        )
        self.assertEqual(validation["status"], "complete")
        self.assertEqual(validation["tensor_count"], 160)
        self.assertEqual(validation["pair_count"], 80)
        self.assertEqual(validation["official_loader_updated_tensor_count"], 80)
        self.assertFalse(result["decision"]["blind_fork_evaluation_allowed"])
        self.assertFalse(result["claim_boundary"]["action_success_evaluated"])


if __name__ == "__main__":
    unittest.main()
