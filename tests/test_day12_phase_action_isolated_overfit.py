import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = ROOT / "configs" / "adapter_phase_action_overfit_v1.json"
SOURCE_DATASET = ROOT / "artifacts" / "day11_phase_action_overfit_dataset_v1"
DATASET_BUILDER = ROOT / "scripts" / "build_phase_action_isolated_overfit_dataset.py"
CONFIG_BUILDER = ROOT / "scripts" / "build_phase_action_isolated_overfit_configs.py"
ARMS = {
    "udon": {
        "sample_id": "udon_chopsticks_imagegen_pseudo_v1",
        "family": "strand",
        "dataset": ROOT / "artifacts" / "day12_phase_action_isolated_udon_dataset_v2",
        "config": ROOT / "configs" / "adapter_phase_action_isolated_udon_v1.json",
        "config_sha256": "c6e11696bf55528655677b40d9ebbd8c53fa7d65c4841b34eb8576da9a1e49f3",
        "manifest_sha256": "2297d29d301b395663fd6b67f6c21a79e19a72d8f66583791b7f4ae217c02c54",
        "metadata_sha256": "ff19012d6d610d3ba5a248c9525bf7824dab03d9b203ca7488214953f41b7f3a",
    },
    "spoon": {
        "sample_id": "clear_broth_spoon_imagegen_pseudo_v1",
        "family": "liquid",
        "dataset": ROOT / "artifacts" / "day12_phase_action_isolated_spoon_dataset_v2",
        "config": ROOT / "configs" / "adapter_phase_action_isolated_spoon_v1.json",
        "config_sha256": "9255aa36bcd5d79d4f8c2c1c6e48db43b377df467fed7f8c27d1d453d12b9112",
        "manifest_sha256": "5bc398202fc2636ecdab9d77916fbf52c233af5650601810b4dc4bbc068fbe2c",
        "metadata_sha256": "2937d0bc49dccca22132221feb5a5e298a4b66d49845b25c389a6a5e2498c728",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


class Day12PhaseActionIsolatedOverfitTests(unittest.TestCase):
    def test_builders_are_frozen_deterministic_and_fail_closed(self):
        self.assertEqual(
            sha256_lf(DATASET_BUILDER),
            "3f32618cb32cfb61239d855238cf7184e458db8769db8c8db626531bcd1016fe",
        )
        self.assertEqual(
            sha256_lf(CONFIG_BUILDER),
            "e7e1f4f00bd961ada1eee2efc7e0644f925686344d2ba3b6b35563ef5f28e13f",
        )
        dataset_builder_text = DATASET_BUILDER.read_text(encoding="utf-8")
        config_builder_text = CONFIG_BUILDER.read_text(encoding="utf-8")
        self.assertIn("Refusing to reuse output root", dataset_builder_text)
        self.assertIn("copyfile", dataset_builder_text)
        self.assertIn("rows_are_byte_identical", dataset_builder_text)
        self.assertIn("Refusing to overwrite", config_builder_text)
        self.assertNotIn("image_gen", dataset_builder_text.lower())
        self.assertNotIn("download", dataset_builder_text.lower())

    def test_isolated_datasets_copy_one_sample_without_changing_bytes(self):
        source_manifest = json.loads(
            (SOURCE_DATASET / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        source_samples = {
            sample["sample_id"]: sample for sample in source_manifest["samples"]
        }
        for arm in ARMS.values():
            manifest_path = arm["dataset"] / "dataset_manifest.json"
            metadata_path = arm["dataset"] / "metadata.csv"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(sha256_file(manifest_path), arm["manifest_sha256"])
            self.assertEqual(sha256_file(metadata_path), arm["metadata_sha256"])
            self.assertEqual(manifest["sample_count"], 2)
            self.assertEqual(manifest["unique_sample_count"], 1)
            self.assertEqual(manifest["selected_sample_id"], arm["sample_id"])
            self.assertEqual(manifest["held_out"]["training_occurrences"], 0)
            self.assertEqual(
                manifest["matched_exposure_comparison"]["dedicated_checkpoint_step"],
                32,
            )
            with metadata_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0], rows[1])
            self.assertEqual(len(manifest["samples"]), 2)
            self.assertEqual(
                {sample["sample_id"] for sample in manifest["samples"]},
                {arm["sample_id"]},
            )
            source_sample = source_samples[arm["sample_id"]]
            for sample in manifest["samples"]:
                self.assertEqual(sample["files"], source_sample["files"])
            for record in source_sample["files"].values():
                source = SOURCE_DATASET / record["path"]
                isolated = arm["dataset"] / record["path"]
                self.assertEqual(sha256_file(isolated), sha256_file(source))
                self.assertEqual(isolated.stat().st_size, source.stat().st_size)
            isolated_phase = arm["dataset"] / manifest["phase_schedule"]["path"]
            source_phase = SOURCE_DATASET / source_manifest["phase_schedule"]["path"]
            self.assertEqual(isolated_phase.read_bytes(), source_phase.read_bytes())

    def test_training_configs_change_isolation_paths_not_model_or_hyperparameters(self):
        source = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
        output_roots = set()
        for arm in ARMS.values():
            config = json.loads(arm["config"].read_text(encoding="utf-8"))
            self.assertEqual(sha256_file(arm["config"]), arm["config_sha256"])
            self.assertEqual(
                config["scientific_status"],
                "mechanism_pilot_with_synthetic_pseudotargets",
            )
            self.assertEqual(
                config["isolation_scientific_status"],
                "seen_synthetic_single_sample_isolation_diagnostic",
            )
            self.assertEqual(config["train_families"], [arm["family"]])
            self.assertEqual(config["train_anchors"], [arm["sample_id"]])
            self.assertEqual(config["dataset"]["unique_sample_count"], 1)
            self.assertEqual(config["dataset"]["selected_sample_id"], arm["sample_id"])
            self.assertEqual(config["dataset"]["manifest_sha256"], arm["manifest_sha256"])
            self.assertEqual(config["dataset"]["metadata_sha256"], arm["metadata_sha256"])
            self.assertEqual(config["model"], source["model"])
            self.assertEqual(config["launcher"], source["launcher"])
            self.assertEqual(config["offline_environment"], source["offline_environment"])
            source_training = dict(source["training"])
            isolated_training = dict(config["training"])
            output_roots.add(isolated_training.pop("output_root"))
            source_training.pop("output_root")
            self.assertEqual(isolated_training, source_training)
            self.assertEqual(config["training"]["expected_optimizer_steps"], 64)
            self.assertEqual(config["training"]["expected_checkpoint_steps"], [16, 32, 48, 64])
            self.assertTrue(config["derivation"]["unchanged_training_hyperparameters"])
            self.assertTrue(config["derivation"]["unchanged_model_and_runtime_hashes"])
        self.assertEqual(len(output_roots), 2)

    def test_matched_exposure_gate_blocks_premature_blind_evaluation(self):
        for arm in ARMS.values():
            config = json.loads(arm["config"].read_text(encoding="utf-8"))
            matched = config["evaluation_gate"]["matched_exposure_comparison"]
            self.assertEqual(matched["dedicated_checkpoint_step"], 32)
            self.assertEqual(matched["dedicated_selected_sample_exposures"], 32)
            self.assertEqual(matched["shared_checkpoint_step"], 64)
            self.assertEqual(matched["shared_expected_selected_sample_exposures"], 32)
            self.assertTrue(
                config["decision_gate"][
                    "both_dedicated_samples_require_clear_semantic_gain_before_blind_fork"
                ]
            )
            self.assertIn("cannot establish", config["claim_limit"])


if __name__ == "__main__":
    unittest.main()
