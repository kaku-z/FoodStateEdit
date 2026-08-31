import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adapter_action_pseudo_v1.json"
AUDIT = ROOT / "results" / "day9_action_target_audit_v1.json"
SUMMARY = ROOT / "results" / "day9_action_pseudo_dataset_v3" / "summary.json"
FORK_EVAL_CONFIG = ROOT / "configs" / "vace_fork_action_lora_eval_v1.json"
FORK_EVAL_V2_CONFIG = ROOT / "configs" / "vace_fork_action_lora_eval_v2.json"
RESULT = ROOT / "results" / "day9_action_adapter_result_v1.json"


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

    def test_action_checkpoint_validator_is_offline_and_claim_limited(self):
        validator = (ROOT / "scripts" / "validate_adapter_action_lora_checkpoint.py").read_text(encoding="utf-8")
        self.assertIn("--expected-sha256", validator)
        self.assertIn("Refusing to overwrite report", validator)
        self.assertIn('os.environ["CUDA_VISIBLE_DEVICES"] = ""', validator)
        self.assertIn("safe_open", validator)
        self.assertIn("torch.isfinite", validator)
        self.assertIn("pipe.load_lora(pipe.vace", validator)
        self.assertIn("tokenizer_config=None", validator)
        self.assertIn("redirect_common_files=False", validator)
        self.assertIn("synthetic pseudo-targets", validator)

    def test_blind_fork_eval_changes_only_the_frozen_lora(self):
        training = json.loads(CONFIG.read_text(encoding="utf-8"))
        evaluation = json.loads(FORK_EVAL_CONFIG.read_text(encoding="utf-8"))
        day8 = json.loads((ROOT / "configs" / "vace_fork_3d_projection_compare_v0.json").read_text(encoding="utf-8"))
        self.assertEqual(evaluation["method"], "vace_fork_action_lora_blind_eval_v1")
        self.assertEqual(evaluation["scientific_status"], "blind_cross_family_mechanism_pilot_not_generalization")
        self.assertEqual(evaluation["adapter"]["training_method"], training["method"])
        self.assertEqual(evaluation["adapter"]["held_out"]["family"], "fork_twirl_and_lift")
        self.assertEqual(evaluation["adapter"]["held_out"]["training_occurrences"], 0)
        self.assertEqual(evaluation["adapter"]["alpha"], 1.0)
        for key in (
            "width", "height", "num_frames", "fps", "num_inference_steps", "vace_scale",
            "seed", "selected_frame_index", "enable_ttm", "projection_alpha", "prompt",
            "prompt_sha256_utf8", "negative_prompt", "negative_prompt_sha256_utf8",
        ):
            self.assertEqual(evaluation["inference"][key], day8["inference"][key], key)
        self.assertEqual(evaluation["control"]["files"], day8["control"]["files"])

    def test_blind_fork_launchers_are_hashed_and_fail_closed(self):
        evaluation = json.loads(FORK_EVAL_CONFIG.read_text(encoding="utf-8"))
        preflight_path = ROOT / evaluation["launcher"]["preflight"]["path"]
        runner_path = ROOT / evaluation["launcher"]["runner"]["path"]
        preflight = preflight_path.read_text(encoding="utf-8")
        runner = runner_path.read_text(encoding="utf-8")
        self.assertEqual(sha256_file(preflight_path), evaluation["launcher"]["preflight"]["sha256"])
        self.assertEqual(sha256_file(runner_path), evaluation["launcher"]["runner"]["sha256"])
        self.assertIn("output_absent", preflight)
        self.assertIn("fork_training_occurrences", preflight)
        self.assertIn("training_checkpoint_record", preflight)
        self.assertIn("validation_updated_tensors", preflight)
        self.assertIn("gpu_gate", preflight)
        self.assertIn("Preflight blocked inference", runner)
        self.assertIn("pipe.load_lora(pipe.vace", runner)
        self.assertIn('manifest["adapter"]["pipeline_load_count"] = 1', runner)
        self.assertIn('manifest["adapter"]["lora_load_count"] = 1', runner)
        self.assertIn("extract_selected_and_project", runner)
        self.assertNotIn("modelscope download", runner.lower())

    def test_corrected_fork_eval_injects_inside_wrapped_vace_blocks(self):
        evaluation = json.loads(FORK_EVAL_V2_CONFIG.read_text(encoding="utf-8"))
        runner_path = ROOT / evaluation["launcher"]["runner"]["path"]
        runner = runner_path.read_text(encoding="utf-8")
        self.assertEqual(evaluation["execution_revision"], "v2_vace_block_inner_linear_injection")
        self.assertEqual(evaluation["implementation_correction"]["observed_v1_hotload_count"], 0)
        self.assertTrue(evaluation["implementation_correction"]["observed_v1_output_equals_day8_byte_for_byte"])
        self.assertEqual(evaluation["adapter"]["runtime_injection"]["expected_pair_count"], 80)
        self.assertTrue(evaluation["adapter"]["runtime_injection"]["forbid_bf16_base_weight_fusion"])
        self.assertTrue(evaluation["adapter"]["runtime_injection"]["forbid_low_noise_vace2_modification"])
        self.assertEqual(sha256_file(runner_path), evaluation["launcher"]["runner"]["sha256"])
        self.assertIn("EXPECTED_INJECTIONS = 80", runner)
        self.assertIn("resolve_parent(pipe.vace, target)", runner)
        self.assertNotIn("pipe.vace2, checkpoint, alpha", runner)
        self.assertIn("LoRAInjectedLinear", runner)
        self.assertIn("torch.nn.functional.linear(torch.nn.functional.linear", runner)
        self.assertIn("Expected {EXPECTED_INJECTIONS} LoRA pairs", runner)
        self.assertIn("Completed inference did not record all expected LoRA injections", runner)

    def test_completed_blind_result_separates_execution_from_action_failure(self):
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(result["training"]["checkpoint_pair_count"], 80)
        self.assertTrue(result["training"]["all_checkpoint_tensors_nonzero"])
        self.assertEqual(result["zero_effect_v1"]["runtime_hotload_count"], 0)
        self.assertTrue(result["zero_effect_v1"]["edited_2d_equal_to_day8"])
        self.assertEqual(result["corrected_v2"]["runtime_injected_linear_count"], 80)
        self.assertEqual(result["corrected_v2"]["decoded_frames"], 21)
        self.assertEqual(result["corrected_v2"]["outside_motion_support_max_pixel_difference"], 0)
        self.assertTrue(result["corrected_v2"]["lora_on_differs_from_lora_off"])
        self.assertGreater(result["corrected_v2"]["selected_projected_rgb_mae_vs_lora_off_inside_support"], 0)
        self.assertFalse(result["blind_review"]["action_success"])
        self.assertFalse(result["blind_review"]["contact_success"])
        self.assertFalse(result["blind_review"]["photo_realism_success"])
        self.assertFalse(result["blind_review"]["semantic_improvement_over_lora_off"])
        self.assertTrue(result["decision"]["adapter_execution_problem_resolved"])
        self.assertFalse(result["decision"]["blind_cross_family_action_problem_resolved"])
        self.assertFalse(result["decision"]["generalization_claim_allowed"])

        baselines = json.loads((ROOT / "configs" / "baselines_v1.json").read_text(encoding="utf-8"))
        gate = baselines["day9_action_adapter_gate"]
        self.assertEqual(gate["corrected_runtime_injected_linear_count"], 80)
        self.assertEqual(gate["provisional_action_success"], 0)
        self.assertEqual(gate["provisional_photo_success"], 0)
        self.assertEqual(gate["status"], "CLOSED_BLIND_NEGATIVE_NEXT_GATE_SEEN_FAMILY_UNDERFIT_CHECK")


if __name__ == "__main__":
    unittest.main()
