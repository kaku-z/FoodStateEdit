import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adapter_action_pseudo_v1.json"
AUDIT = ROOT / "results" / "day9_action_target_audit_v1.json"
SUMMARY = ROOT / "results" / "day9_action_pseudo_dataset_v3" / "summary.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Day9ActionAdapterTests(unittest.TestCase):
    def test_target_audit_has_only_two_disclosed_pseudotargets(self):
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        eligible = [item for item in audit["candidates"] if item["eligible_for_primary_pilot_training"]]
        self.assertEqual(len(eligible), 2)
        self.assertTrue(all(item["classification"] == "eligible_synthetic_pseudotarget" for item in eligible))
        self.assertTrue(all(item["producer"] == "built_in_imagegen" for item in eligible))
        self.assertEqual(audit["decision"]["real_photo_target_count"], 0)
        self.assertFalse(audit["decision"]["generalization_minimum_met"])
        fork = next(item for item in audit["candidates"] if item["candidate_id"] == "pasta_fork_day8_vace")
        self.assertFalse(fork["eligible_for_primary_pilot_training"])
        self.assertEqual(fork["classification"], "held_out_failure_not_training_data")

    def test_config_freezes_nonidentity_pilot_and_blind_fork(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["method"], "foodstateedit_vace_lora_action_pseudo_v1")
        self.assertEqual(config["scientific_status"], "mechanism_pilot_with_synthetic_pseudotargets")
        self.assertEqual(config["dataset"]["sample_count"], 2)
        self.assertEqual(config["dataset"]["real_photo_target_count"], 0)
        self.assertEqual(config["dataset"]["synthetic_pseudotarget_count"], 2)
        self.assertTrue(config["dataset"]["formal_training_minimum_not_met"])
        self.assertEqual(config["held_out"]["family"], "fork_twirl_and_lift")
        self.assertEqual(config["held_out"]["training_occurrences"], 0)
        self.assertEqual(config["training"]["lora_base_model"], "vace")
        self.assertEqual(config["training"]["lora_rank"], 8)
        self.assertEqual(config["training"]["expected_optimizer_steps"], 16)
        self.assertFalse(config["training"]["overwrite"])
        self.assertEqual(config["offline_environment"]["HF_HUB_OFFLINE"], "1")
        self.assertEqual(config["resource_gate"]["required_gpu_name"], "NVIDIA RTX A6000")
        self.assertEqual(config["target_audit"]["sha256"], sha256_file(AUDIT))

    def test_dataset_summary_is_nonidentity_and_exactly_protected(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
        self.assertEqual(summary["dataset_manifest_sha256"], config["dataset"]["manifest_sha256"])
        self.assertEqual(summary["metadata_sha256"], config["dataset"]["metadata_sha256"])
        self.assertEqual(summary["held_out"]["training_occurrences"], 0)
        for sample in summary["samples"]:
            self.assertNotEqual(sample["target_video_sha256"], sample["control_video_sha256"])
            self.assertGreater(sample["target_control_changed_pixel_fraction"], 0)
            self.assertLess(sample["edit_support_fraction"], 0.2)
            self.assertEqual(sample["target_outside_support_max_difference"], 0)
            self.assertEqual(sample["control_outside_support_max_difference"], 0)
        self.assertEqual(summary["failed_builds_preserved"][1]["dataset_id"], "day9_action_pseudo_dataset_v2")

    def test_builder_and_launchers_are_fail_closed(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        builder = (ROOT / config["trainer"]["dataset_builder"]).read_text(encoding="utf-8")
        preflight_path = ROOT / config["launcher"]["preflight"]
        runner_path = ROOT / config["launcher"]["runner"]
        preflight = preflight_path.read_text(encoding="utf-8")
        runner = runner_path.read_text(encoding="utf-8")
        self.assertIn("Refusing to reuse output root", builder)
        self.assertIn("target_outside_edit_support_max_difference", builder)
        self.assertIn("target_video_hash_differs_from_control", builder)
        self.assertIn("fork_twirl_and_lift", builder)
        self.assertNotIn("modelscope download", builder.lower())
        self.assertIn("nonidentity_hash:", preflight)
        self.assertIn("pseudo_target_contract:", preflight)
        self.assertIn("required_gpu_name", preflight)
        self.assertIn("Refusing to overwrite report", preflight)
        self.assertIn("Preflight blocked training", runner)
        self.assertIn('environment["CUDA_VISIBLE_DEVICES"]', runner)
        self.assertIn("expected_checkpoint_names", runner)
        self.assertNotIn("modelscope download", runner.lower())
        self.assertEqual(config["launcher"]["preflight_sha256"], sha256_file(preflight_path))
        self.assertEqual(config["launcher"]["runner_sha256"], sha256_file(runner_path))


if __name__ == "__main__":
    unittest.main()
