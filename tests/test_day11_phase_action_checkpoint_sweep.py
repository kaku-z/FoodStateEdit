import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "vace_phase_action_checkpoint_sweep_v1.json"
DATASET_MANIFEST = ROOT / "artifacts" / "day11_phase_action_overfit_dataset_v1" / "dataset_manifest.json"
SPOON_BLOCKED_PREFLIGHT = ROOT / "results" / "day11_phase_action_checkpoint_sweep_preflight_spoon_gp40_blocked_v1.json"
UDON_RACE_PREFLIGHT = ROOT / "results" / "day11_phase_action_checkpoint_sweep_preflight_udon_gp40_race_v1.json"


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


if __name__ == "__main__":
    unittest.main()
