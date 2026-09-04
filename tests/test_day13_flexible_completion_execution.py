import csv
import hashlib
import json
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "flexible_completion_execution_v1.json"
DATASET_ROOT = ROOT / "artifacts" / "day13_3d_guided_flexible_completion_udon_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Day13FlexibleCompletionExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.manifest = json.loads(
            (DATASET_ROOT / "dataset_manifest.json").read_text(encoding="utf-8")
        )

    def test_execution_config_is_immutable_seen_only_protocol(self):
        self.assertEqual(
            sha256_file(CONFIG_PATH),
            "5071c11f488667646db33f81c6a38b6066042a350dfb1d8fbb68ec90662753b3",
        )
        self.assertTrue(self.config["execution_allowed"])
        self.assertFalse(self.config["blind_fork_evaluation_allowed"])
        self.assertFalse(self.config["balanced_family_expansion_allowed"])
        self.assertIn("one-sample seen synthetic", self.config["claim_limit"])

    def test_design_geometry_and_all_implementation_hashes_match(self):
        for key in ("design", "geometry"):
            record = self.config[key]
            path = ROOT / record["path"]
            self.assertEqual(path.stat().st_size, record["size_bytes"])
            self.assertEqual(sha256_file(path), record["sha256"])
        for record in self.config["implementation"].values():
            path = ROOT / record["path"]
            self.assertEqual(path.stat().st_size, record["size_bytes"])
            self.assertEqual(sha256_file(path), record["sha256"])

    def test_dataset_manifest_and_every_record_match(self):
        record = self.config["dataset"]["manifest"]
        manifest_path = DATASET_ROOT / record["path"]
        self.assertEqual(manifest_path.stat().st_size, record["size_bytes"])
        self.assertEqual(sha256_file(manifest_path), record["sha256"])
        for group in ("frozen_source_files", "derived_files", "metadata"):
            for file_record in self.manifest[group].values():
                path = DATASET_ROOT / file_record["path"]
                self.assertEqual(path.stat().st_size, file_record["size_bytes"])
                self.assertEqual(sha256_file(path), file_record["sha256"])

    def test_relative_3d_and_mask_contract_is_realized(self):
        self.assertEqual(self.manifest["frame_count"], 21)
        self.assertEqual(self.manifest["geometry_source"], "relative_3d_normalized_pinhole")
        self.assertTrue(self.manifest["strand_depth_crosses_both_chopstick_depths"])
        self.assertEqual(
            self.manifest[
                "outside_frozen_edit_support_max_pixel_difference_before_video_encoding"
            ],
            0,
        )
        records = self.manifest["frame_records"]
        self.assertEqual([record["frame"] for record in records], list(range(21)))
        for record in records[:6]:
            self.assertEqual(record.get("strand_mask_pixels", 0), 0)
            self.assertEqual(record.get("pinch_contact_mask_pixels", 0), 0)
            self.assertEqual(record.get("source_connection_mask_pixels", 0), 0)
        for record in records[6:9]:
            self.assertGreater(record["strand_mask_pixels"], 0)
            self.assertGreater(record["pinch_contact_mask_pixels"], 0)
            self.assertEqual(record["source_connection_mask_pixels"], 0)
        for record in records[9:]:
            self.assertGreater(record["strand_mask_pixels"], 0)
            self.assertGreater(record["pinch_contact_mask_pixels"], 0)
            self.assertGreater(record["source_connection_mask_pixels"], 0)

    def test_three_metadata_views_keep_target_and_relative3d_pair_matched(self):
        rows = {}
        for arm_id, metadata_name in self.manifest["training_arms"].items():
            with (DATASET_ROOT / metadata_name).open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows[arm_id] = list(csv.DictReader(handle))
            self.assertEqual(len(rows[arm_id]), 2)
            self.assertEqual(
                {row["training_row_id"] for row in rows[arm_id]}, {"0", "1"}
            )
        for arm_rows in rows.values():
            self.assertEqual(
                {row["video"] for row in arm_rows},
                {"video/udon_chopsticks_imagegen_pseudo_v1.mp4"},
            )
        self.assertEqual(
            {row["vace_video"] for row in rows["relative3d_uniform"]},
            {row["vace_video"] for row in rows["relative3d_topology_weighted"]},
        )
        self.assertNotIn(
            "flexible_strand_mask_video", rows["relative3d_uniform"][0]
        )
        self.assertIn(
            "flexible_strand_mask_video",
            rows["relative3d_topology_weighted"][0],
        )

    def test_all_derived_videos_decode_to_21_frames(self):
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            self.skipTest("ffprobe is unavailable")
        names = (
            "relative_3d_vace_control_video",
            "flexible_strand_mask_video",
            "pinch_contact_mask_video",
            "source_connection_mask_video",
        )
        for name in names:
            path = DATASET_ROOT / self.manifest["derived_files"][name]["path"]
            result = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_read_frames",
                    "-of",
                    "default=nokey=1:noprint_wrappers=1",
                    str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(int(result.stdout.strip()), 21)

    def test_trainer_explicitly_matches_randomness_and_isolates_loss(self):
        source = (
            ROOT / self.config["implementation"]["matched_trainer"]["path"]
        ).read_text(encoding="utf-8")
        for token in (
            "timestep_generator.manual_seed(timestep_seed)",
            "noise_generator.manual_seed(noise_seed)",
            "generator=shuffle_generator",
            "functional.adaptive_max_pool3d",
            "noise_prediction[:, :, 1:]",
            "training_target[:, :, 1:]",
            "expanded_weight * squared_error",
            "strand_weight * pooled[MASK_KEYS[0]]",
            "contact_weight * pooled[MASK_KEYS[1]]",
            "connection_weight * pooled[MASK_KEYS[2]]",
        ):
            self.assertIn(token, source)
        self.assertEqual(
            self.config["training"]["topology_weights"],
            {"strand": 3.0, "contact": 7.0, "source_connection": 5.0},
        )

    def test_arms_budget_conditions_and_resource_gate_are_frozen(self):
        self.assertEqual(
            list(self.config["arms"]),
            [
                "planar_uniform",
                "relative3d_uniform",
                "relative3d_topology_weighted",
            ],
        )
        self.assertEqual(
            self.config["arms"]["relative3d_uniform"]["control"],
            self.config["arms"]["relative3d_topology_weighted"]["control"],
        )
        self.assertNotEqual(
            self.config["arms"]["relative3d_uniform"]["loss_mode"],
            self.config["arms"]["relative3d_topology_weighted"]["loss_mode"],
        )
        training = self.config["training"]
        self.assertEqual(training["training_seed"], 20260903)
        self.assertEqual(training["expected_optimizer_steps"], 32)
        self.assertEqual(training["expected_checkpoint_steps"], [16, 32])
        self.assertEqual(
            self.config["evaluation"]["conditions"],
            [
                "planar_lora_off",
                "planar_uniform_step32",
                "relative3d_lora_off",
                "relative3d_uniform_step32",
                "relative3d_topology_weighted_step32",
            ],
        )
        gate = self.config["resource_gate"]
        self.assertEqual(gate["required_gpu_name"], "NVIDIA RTX A6000")
        self.assertEqual(gate["min_free_memory_mib"], 48000)
        self.assertEqual(gate["max_utilization_percent"], 5)
        self.assertEqual(gate["min_available_system_memory_mib"], 80000)
        self.assertTrue(gate["require_no_compute_process"])
        self.assertTrue(gate["forbid_model_download"])


if __name__ == "__main__":
    unittest.main()
