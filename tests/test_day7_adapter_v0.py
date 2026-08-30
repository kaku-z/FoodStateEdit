import ast
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adapter_v0_lora_smoke.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AdapterV0Tests(unittest.TestCase):
    def test_config_is_claim_limited_offline_and_reuses_audited_model(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["scientific_status"], "infrastructure_smoke_not_quality_evidence")
        self.assertTrue(config["dataset"]["formal_training_minimum_not_met"])
        self.assertEqual(config["dataset"]["sample_count"], 2)
        self.assertEqual(
            config["dataset"]["manifest_sha256"],
            "5483f03bd79295f62cc961c3411198fe4c221a0979dbf833f76ba28c6f97b9ae",
        )
        self.assertEqual(config["training"]["stage"], "high_noise")
        self.assertEqual(config["training"]["lora_base_model"], "vace")
        self.assertEqual(config["training"]["epochs"], 1)
        self.assertEqual(config["training"]["dataset_repeat"], 1)
        self.assertFalse(config["training"]["overwrite"])
        self.assertEqual(
            config["training"]["output_root"],
            "/tmp/foodstateedit_day7_adapter_v0_high_noise_lora_smoke_v5",
        )
        self.assertEqual(config["offline_environment"]["DIFFSYNTH_SKIP_DOWNLOAD"], "True")
        self.assertEqual(config["offline_environment"]["HF_HUB_OFFLINE"], "1")
        self.assertEqual(
            config["model"]["hash_audit_sha256"],
            "4c255c04811a2bc1484db5dae01bdc7352ebd4ad0bd7802b13cebe0bc669d34b",
        )

    def test_compatible_trainer_snapshot_is_frozen(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["trainer"]["commit"], "899d2cd5740f50f7b6a1ec2cf7360a20897b1191")
        self.assertEqual(
            config["trainer"]["train_script_sha256"],
            "4a8c9c27c23b217f7438b37c71213df58414c39cb2b6998d3d75a4dd7992bc41",
        )
        self.assertIn("torch 2.4", config["trainer"]["selection_reason"])
        self.assertEqual(config["trainer"]["no_audio_wrapper"], "scripts/run_diffsynth_wan_train_no_audio.py")
        self.assertEqual(
            config["trainer"]["no_audio_wrapper_sha256"],
            "1db6398d61f6cbae1adb896079a38ed3809464ab7ab1b00bab848c55000dff90",
        )

    def test_dataset_builder_is_deterministic_proxy_only_and_fail_closed(self):
        source = (ROOT / "scripts" / "build_adapter_smoke_dataset.py").read_text(encoding="utf-8")
        lowered = source.lower()
        self.assertNotIn("torch", lowered)
        self.assertNotIn("diffusers", lowered)
        self.assertNotIn("imagegen", lowered)
        self.assertIn("deterministic_proxy_identity_plumbing_only", source)
        self.assertIn("sha256_lf_normalized", source)
        self.assertIn("Refusing to reuse output root", source)
        tree = ast.parse(source)
        constants = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"DATASET_ID", "SOURCE_METHOD", "TARGET_POLICY"}
        }
        self.assertEqual(constants["SOURCE_METHOD"], "vace_direct_dynamic_multikey")

    def test_preflight_and_launcher_are_fail_closed(self):
        preflight = (ROOT / "scripts" / "preflight_adapter_training.py").read_text(encoding="utf-8")
        launcher = (ROOT / "scripts" / "run_adapter_lora_smoke.py").read_text(encoding="utf-8")
        validator = (ROOT / "scripts" / "validate_adapter_lora_checkpoint.py").read_text(encoding="utf-8")
        self.assertIn("output_absent", preflight)
        self.assertIn("require_no_compute_process", preflight)
        self.assertIn("nvidia-smi", preflight)
        self.assertIn("trainer_help", preflight)
        self.assertIn("no_audio_wrapper_help", preflight)
        self.assertIn("video_decode:", preflight)
        self.assertIn("--video-decode-smoke", preflight)
        self.assertIn("dataset_manifest_hash", preflight)
        self.assertIn("dataset_builder_hash", preflight)
        self.assertIn("dataset_hash:", preflight)
        self.assertIn("identity_target:", preflight)
        self.assertIn("Refusing to reuse output root", launcher)
        self.assertIn("Preflight blocked the run", launcher)
        self.assertIn('environment["CUDA_VISIBLE_DEVICES"]', launcher)
        self.assertIn('"HF_HUB_OFFLINE"', launcher)
        self.assertIn('"TRANSFORMERS_OFFLINE"', launcher)
        self.assertIn('"--model_paths"', launcher)
        self.assertIn('trainer["no_audio_wrapper"]', launcher)
        self.assertNotIn("modelscope download", launcher.lower())
        self.assertIn("Refusing to overwrite report", validator)
        self.assertIn('os.environ["CUDA_VISIBLE_DEVICES"] = ""', validator)
        self.assertIn("safe_open", validator)
        self.assertIn("torch.isfinite", validator)
        self.assertIn("pipe.load_lora(pipe.vace", validator)
        self.assertIn("tokenizer_config=None", validator)
        self.assertIn("redirect_common_files=False", validator)
        self.assertIn("no visual-quality claim", validator)

    def test_initial_remote_preflight_is_blocked_only_by_gpu(self):
        report_path = ROOT / "results" / "day7_adapter_v0_preflight_initial_v2.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertFalse(report["ready"])
        self.assertIsNone(report["selected_gpu"])
        self.assertEqual(
            report["dataset_manifest_sha256"],
            "5483f03bd79295f62cc961c3411198fe4c221a0979dbf833f76ba28c6f97b9ae",
        )
        failed = [check["id"] for check in report["checks"] if not check["passed"]]
        self.assertEqual(failed, ["gpu_gate"])
        output_absent = next(check for check in report["checks"] if check["id"] == "output_absent")
        self.assertTrue(output_absent["passed"])
        gpu_gate = next(check for check in report["checks"] if check["id"] == "gpu_gate")
        self.assertEqual(gpu_gate["actual"]["safe_gpu_indices"], [])
        self.assertEqual({process["owner"] for process in gpu_gate["actual"]["processes"]}, {"chen-q"})

    def test_gp39_v1_failure_is_preserved_and_pre_model_load(self):
        result_root = ROOT / "results" / "day7_adapter_v0_gp39_failure_missing_librosa_v1"
        manifest = json.loads((result_root / "run_manifest.json").read_text(encoding="utf-8"))
        preflight = json.loads((ROOT / "results" / "day7_adapter_v0_preflight_gp39_v1.json").read_text(encoding="utf-8"))
        log = (result_root / "train.log").read_text(encoding="utf-8")
        self.assertEqual(manifest["status"], "technical_failure")
        self.assertEqual(manifest["return_code"], 1)
        self.assertEqual(manifest["checkpoints"], [])
        self.assertTrue(preflight["ready"])
        self.assertEqual(preflight["selected_gpu"], 0)
        self.assertIn("No module named 'librosa'", log)

    def test_no_audio_wrapper_is_narrow_and_fail_closed(self):
        wrapper = (ROOT / "scripts" / "run_diffsynth_wan_train_no_audio.py").read_text(encoding="utf-8")
        self.assertIn("No-audio wrapper refuses datasets containing input_audio", wrapper)
        self.assertIn('sys.modules["librosa"] = sentinel', wrapper)
        self.assertIn('ModuleSpec("librosa", loader=None)', wrapper)
        self.assertIn("FoodStateEdit no-audio wrapper forbids librosa.load", wrapper)
        self.assertIn("install_imageio_pyav_metadata_compatibility", wrapper)
        self.assertIn("compatible_count_frames", wrapper)
        self.assertIn('metadata.update({"fps": fps, "duration": frame_count / fps, "nframes": frame_count})', wrapper)
        self.assertIn("reader.get_data(0)", wrapper)
        self.assertIn("reader.get_data(frame_count - 1)", wrapper)
        self.assertIn("runpy.run_path", wrapper)
        self.assertNotIn("pip install", wrapper)

    def test_first_wrapper_preflight_failure_is_preserved(self):
        report = json.loads(
            (ROOT / "results" / "day7_adapter_v0_preflight_gp39_wrapper_spec_failure_v2.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(report["ready"])
        self.assertEqual([check["id"] for check in report["checks"] if not check["passed"]], ["no_audio_wrapper_help"])
        output_absent = next(check for check in report["checks"] if check["id"] == "output_absent")
        self.assertTrue(output_absent["passed"])

    def test_gp39_v2_failure_is_preserved_and_pre_model_load(self):
        result_root = ROOT / "results" / "day7_adapter_v0_gp39_failure_librosa_load_v2"
        manifest = json.loads((result_root / "run_manifest.json").read_text(encoding="utf-8"))
        preflight = json.loads((ROOT / "results" / "day7_adapter_v0_preflight_gp39_v2final.json").read_text(encoding="utf-8"))
        log = (result_root / "train.log").read_text(encoding="utf-8")
        self.assertEqual(manifest["status"], "technical_failure")
        self.assertEqual(manifest["checkpoints"], [])
        self.assertTrue(preflight["ready"])
        self.assertIn("module 'librosa' has no attribute 'load'", log)

    def test_gp39_v3_failure_is_preserved_after_model_load(self):
        result_root = ROOT / "results" / "day7_adapter_v0_gp39_failure_pyav_metadata_v3"
        manifest = json.loads((result_root / "run_manifest.json").read_text(encoding="utf-8"))
        preflight = json.loads((ROOT / "results" / "day7_adapter_v0_preflight_gp39_v3.json").read_text(encoding="utf-8"))
        log = (result_root / "train.log").read_text(encoding="utf-8")
        self.assertEqual(manifest["status"], "technical_failure")
        self.assertEqual(manifest["checkpoints"], [])
        self.assertTrue(preflight["ready"])
        self.assertEqual(manifest["log_sha256"], "a698a927b52f826e012eb09a814c83a0d80ceb234b4599a8f1534c45b11df65c")
        self.assertIn("Loading models from", log)
        self.assertIn("unsupported operand type(s) for *: 'NoneType' and 'Fraction'", log)

    def test_gp39_v5_smoke_and_official_checkpoint_load_are_complete(self):
        result_root = ROOT / "results" / "day7_adapter_v0_gp39_complete_v5"
        manifest = json.loads((result_root / "run_manifest.json").read_text(encoding="utf-8"))
        command = json.loads((result_root / "command.json").read_text(encoding="utf-8"))
        preflight = json.loads((ROOT / "results" / "day7_adapter_v0_preflight_gp39_v5.json").read_text(encoding="utf-8"))
        validation = json.loads(
            (ROOT / "results" / "day7_adapter_v0_checkpoint_validation_v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["return_code"], 0)
        self.assertEqual(manifest["selected_physical_gpu"], 0)
        self.assertTrue(preflight["ready"])
        video_checks = [check for check in preflight["checks"] if check["id"].startswith("video_decode:")]
        self.assertEqual(len(video_checks), 4)
        self.assertTrue(all(check["passed"] for check in video_checks))
        self.assertTrue(all(check["actual"]["frame_count"] == 21 for check in video_checks))
        self.assertEqual(command["environment"]["DIFFSYNTH_SKIP_DOWNLOAD"], "True")
        self.assertEqual(command["environment"]["HF_HUB_OFFLINE"], "1")
        self.assertNotIn("input_audio", command["command"])
        self.assertEqual([record["path"] for record in manifest["checkpoints"]], ["step-1.safetensors", "step-2.safetensors"])
        for record in manifest["checkpoints"]:
            checkpoint = result_root / record["path"]
            self.assertEqual(checkpoint.stat().st_size, record["size_bytes"])
            self.assertEqual(sha256_file(checkpoint), record["sha256"])
        self.assertEqual(validation["status"], "complete")
        self.assertEqual(validation["checkpoint_sha256"], manifest["checkpoints"][-1]["sha256"])
        self.assertEqual(validation["tensor_count"], 160)
        self.assertEqual(validation["pair_count"], 80)
        self.assertEqual(validation["rank"], 8)
        self.assertEqual(validation["official_loader_updated_tensor_count"], 80)
        self.assertIn("no visual-quality claim", validation["claim_limit"])


if __name__ == "__main__":
    unittest.main()
