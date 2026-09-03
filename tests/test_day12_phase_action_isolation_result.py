import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = (
    ROOT / "artifacts" / "day12_phase_action_isolated_checkpoint_sweep_gp38_v1"
)
EVIDENCE_ROOT = (
    ROOT / "results" / "day12_phase_action_isolated_checkpoint_sweep_gp38_v1"
)
RESULT = ROOT / "results" / "day12_phase_action_isolation_result_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Day12PhaseActionIsolationResultTests(unittest.TestCase):
    def test_sweeps_are_complete_and_pulled_outputs_match_manifests(self):
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
        self.assertEqual(result["execution"]["pulled_run_files_verified"], 40)
        self.assertEqual(
            result["execution"]["pulled_supporting_evidence_files_verified"], 14
        )
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
            self.assertEqual(len(manifest["outputs"]), 18)
            for record in manifest["outputs"]:
                local_output = sample_root / Path(record["path"]).name
                self.assertEqual(local_output.stat().st_size, record["size_bytes"])
                self.assertEqual(sha256_file(local_output), record["sha256"])
            for condition_name, expected_hash in sample["contact_sheet_sha256"].items():
                review = sample_root / f"{condition_name}_projected_review.png"
                self.assertEqual(sha256_file(review), expected_hash)

    def test_all_pulled_supporting_evidence_matches_declared_hashes(self):
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        evidence = result["evidence"]
        expected = {
            "udon_sweep_config_sha256": "udon_sweep_config.json",
            "spoon_sweep_config_sha256": "spoon_sweep_config.json",
            "udon_sweep_runner_preflight_sha256": "udon_sweep_runner_preflight.json",
            "spoon_sweep_runner_preflight_sha256": "spoon_sweep_runner_preflight.json",
            "udon_training_manifest_sha256": "udon_training_manifest.json",
            "spoon_training_manifest_sha256": "spoon_training_manifest.json",
            "udon_step64_validation_sha256": "udon_step64_validation.json",
            "spoon_step64_validation_sha256": "spoon_step64_validation.json",
            "udon_training_standalone_preflight_sha256": "udon_training_standalone_preflight.json",
            "udon_training_runner_preflight_sha256": "udon_training_runner_preflight.json",
            "spoon_training_standalone_preflight_sha256": "spoon_training_standalone_preflight.json",
            "spoon_training_runner_preflight_sha256": "spoon_training_runner_preflight.json",
            "udon_training_launcher_log_sha256": "udon_training_launcher.log",
            "spoon_training_launcher_log_sha256": "spoon_training_launcher.log",
        }
        for key, filename in expected.items():
            self.assertEqual(evidence[key], sha256_file(EVIDENCE_ROOT / filename))
        for filename in (
            "udon_sweep_runner_preflight.json",
            "spoon_sweep_runner_preflight.json",
            "udon_training_standalone_preflight.json",
            "udon_training_runner_preflight.json",
            "spoon_training_standalone_preflight.json",
            "spoon_training_runner_preflight.json",
        ):
            self.assertTrue(
                json.loads((EVIDENCE_ROOT / filename).read_text(encoding="utf-8"))["ready"]
            )

    def test_matched_interference_and_semantic_gates_remain_closed(self):
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(
            result["scientific_status"],
            "closed_seen_synthetic_single_sample_isolation_no_clear_semantic_gain_not_generalization",
        )
        self.assertEqual(
            result["aggregate"][
                "samples_with_clear_phase_action_contact_improvement_over_both_baselines"
            ],
            0,
        )
        self.assertEqual(
            result["aggregate"][
                "samples_with_clear_photo_realism_improvement_over_lora_off"
            ],
            0,
        )
        self.assertFalse(result["aggregate"]["clear_seen_semantic_gain"])
        self.assertFalse(
            result["aggregate"]["cross_sample_interference_diagnosis_supported"]
        )
        self.assertFalse(result["decision"]["matched_interference_gate_passed"])
        self.assertFalse(result["decision"]["single_sample_overfit_capacity_gate_passed"])
        self.assertFalse(result["decision"]["balanced_multifamily_expansion_allowed"])
        self.assertFalse(result["decision"]["blind_fork_evaluation_allowed"])
        self.assertFalse(result["decision"]["generalization_claim_allowed"])
        for sample in result["samples"]:
            self.assertFalse(
                sample["phase_action_contact_review"]["clear_improvement_over_lora_off"]
            )
            self.assertFalse(
                sample["phase_action_contact_review"][
                    "clear_improvement_over_matched_shared_checkpoint"
                ]
            )
            self.assertFalse(
                sample["photo_realism_review"]["clear_improvement_over_lora_off"]
            )


if __name__ == "__main__":
    unittest.main()
