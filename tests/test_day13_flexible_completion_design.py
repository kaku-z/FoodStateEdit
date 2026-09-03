import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "flexible_completion_mechanism_pilot_v0.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Day13FlexibleCompletionDesignTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_design_is_frozen_but_not_executable(self):
        self.assertEqual(
            sha256_file(CONFIG),
            "6eb599dfbb1b7b2b9bdad955f70c9bc4e5d496407408d9f3717cb3e6b76280b5",
        )
        self.assertEqual(
            self.config["scientific_status"],
            "design_frozen_seen_synthetic_mechanism_pilot_no_execution",
        )
        self.assertFalse(self.config["execution_allowed"])
        self.assertFalse(
            self.config["implementation_freeze_gate"]["execution_allowed"]
        )
        self.assertGreater(
            len(self.config["implementation_freeze_gate"]["pending_hashes"]), 0
        )
        self.assertFalse(
            self.config["decision_rules"]["blind_fork_evaluation_allowed"]
        )
        self.assertFalse(
            self.config["decision_rules"]["balanced_family_expansion_allowed"]
        )

    def test_source_sample_and_all_frozen_input_hashes_match(self):
        sample = self.config["sample"]
        dataset_root = ROOT / sample["source_dataset_root"]
        self.assertEqual(
            sha256_file(dataset_root / "dataset_manifest.json"),
            sample["source_dataset_manifest_sha256"],
        )
        self.assertEqual(
            sha256_file(ROOT / sample["source_training_config"]),
            sample["source_training_config_sha256"],
        )
        for record in sample["files"].values():
            self.assertEqual(
                sha256_file(dataset_root / record["path"]), record["sha256"]
            )
        scaffold = self.config["relative_3d_scaffold"]
        self.assertEqual(
            sha256_file(ROOT / scaffold["projection_module"]),
            scaffold["projection_module_sha256"],
        )
        self.assertEqual(
            sha256_file(ROOT / scaffold["prior_fork_config"]),
            scaffold["prior_fork_config_sha256"],
        )

    def test_three_arms_isolate_representation_and_loss(self):
        arms = {arm["id"]: arm for arm in self.config["training_arms"]}
        self.assertEqual(
            list(arms),
            [
                "planar_uniform",
                "relative3d_uniform",
                "relative3d_topology_weighted",
            ],
        )
        self.assertEqual(
            arms["relative3d_uniform"]["control"],
            arms["relative3d_topology_weighted"]["control"],
        )
        self.assertNotEqual(
            arms["relative3d_uniform"]["loss"],
            arms["relative3d_topology_weighted"]["loss"],
        )
        self.assertEqual(
            self.config["evaluation"]["primary_causal_comparison"],
            "relative3d_topology_weighted_step32_vs_relative3d_uniform_step32",
        )

    def test_training_budget_and_randomness_are_matched(self):
        training = self.config["common_training"]
        self.assertEqual(training["training_seed"], 20260903)
        self.assertEqual(training["expected_optimizer_steps"], 32)
        self.assertEqual(training["checkpoint_steps"], [16, 32])
        self.assertEqual(training["lora_rank"], 8)
        self.assertEqual(training["learning_rate"], 0.0001)
        self.assertEqual(training["num_frames"], 21)
        self.assertIn("same initial LoRA tensor hashes", training["matched_randomness_requirements"])
        self.assertIn("same sampled diffusion noise", training["matched_randomness_requirements"])

    def test_weighted_loss_preserves_thin_masks_without_global_scale_change(self):
        loss = self.config["topology_weighted_loss"]
        self.assertEqual(
            loss["raw_weight_formula"],
            "1 + 3*M_strand + 7*M_contact + 5*M_source_connection",
        )
        self.assertIn("adaptive_max_pool", loss["latent_alignment"])
        self.assertIn("by its own mean", loss["normalization"])
        self.assertIn("excluding", loss["first_frame_policy"])

    def test_positive_gate_requires_semantics_metrics_photo_and_preservation(self):
        gate = self.config["positive_contribution_gate"]
        requirements = " ".join(gate["requirements"])
        self.assertTrue(gate["all_required"])
        self.assertFalse(gate["numeric_only_improvement_passes"])
        self.assertFalse(gate["step16_can_select_the_winner"])
        self.assertFalse(gate["additional_seed_can_rescue_failure"])
        self.assertIn("both blinded reviewers", requirements)
        self.assertIn("at least 5 percent", requirements)
        self.assertIn("photo realism", requirements)
        self.assertIn("outside_support_max_pixel_difference equals zero", requirements)

    def test_resource_gate_remains_fail_closed(self):
        gate = self.config["resource_gate"]
        self.assertEqual(gate["execution_mode"], "serial_only")
        self.assertEqual(gate["required_gpu_name"], "NVIDIA RTX A6000")
        self.assertEqual(gate["min_free_memory_mib"], 48000)
        self.assertEqual(gate["max_utilization_percent"], 5)
        self.assertTrue(gate["require_no_compute_process"])
        self.assertTrue(gate["forbid_process_termination_or_preemption"])
        self.assertTrue(gate["forbid_a40_or_blackwell"])
        self.assertTrue(gate["forbid_model_download"])
        self.assertTrue(gate["require_new_non_overwriting_paths"])


if __name__ == "__main__":
    unittest.main()
