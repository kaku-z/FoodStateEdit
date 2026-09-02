#!/usr/bin/env python3
"""Derive the two frozen Day 12 isolated-overfit training configurations."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


BASE_CONFIG_SHA256 = "80f81aadf5165443ba8e4991a0a8f40b42457aed8bade48e59adc68a9c58dabe"
DATASET_BUILDER_SHA256_LF = "3f32618cb32cfb61239d855238cf7184e458db8769db8c8db626531bcd1016fe"
ARMS = (
    {
        "short_name": "udon",
        "sample_id": "udon_chopsticks_imagegen_pseudo_v1",
        "family": "strand",
        "dataset_name": "day12_phase_action_isolated_udon_dataset_v2",
        "manifest_sha256": "2297d29d301b395663fd6b67f6c21a79e19a72d8f66583791b7f4ae217c02c54",
        "metadata_sha256": "ff19012d6d610d3ba5a248c9525bf7824dab03d9b203ca7488214953f41b7f3a",
    },
    {
        "short_name": "spoon",
        "sample_id": "clear_broth_spoon_imagegen_pseudo_v1",
        "family": "liquid",
        "dataset_name": "day12_phase_action_isolated_spoon_dataset_v2",
        "manifest_sha256": "5bc398202fc2636ecdab9d77916fbf52c233af5650601810b4dc4bbc068fbe2c",
        "metadata_sha256": "2937d0bc49dccca22132221feb5a5e298a4b66d49845b25c389a6a5e2498c728",
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    base_path = repo_root / "configs" / "adapter_phase_action_overfit_v1.json"
    if sha256_file(base_path) != BASE_CONFIG_SHA256:
        raise ValueError("Frozen Day 11 training config hash mismatch")
    builder_path = repo_root / "scripts" / "build_phase_action_isolated_overfit_dataset.py"
    if sha256_lf(builder_path) != DATASET_BUILDER_SHA256_LF:
        raise ValueError("Isolated dataset builder hash mismatch")
    base = json.loads(base_path.read_text(encoding="utf-8"))

    for arm in ARMS:
        dataset_root = repo_root / "artifacts" / arm["dataset_name"]
        manifest_path = dataset_root / "dataset_manifest.json"
        metadata_path = dataset_root / "metadata.csv"
        if sha256_file(manifest_path) != arm["manifest_sha256"]:
            raise ValueError(f"Manifest mismatch for {arm['short_name']}")
        if sha256_file(metadata_path) != arm["metadata_sha256"]:
            raise ValueError(f"Metadata mismatch for {arm['short_name']}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("selected_sample_id") != arm["sample_id"]:
            raise ValueError(f"Selected sample mismatch for {arm['short_name']}")

        config = copy.deepcopy(base)
        config["schema_version"] = "foodstateedit.adapter_phase_action_isolated_overfit.v1"
        config["freeze_date"] = "2026-09-02"
        config["scientific_status"] = "mechanism_pilot_with_synthetic_pseudotargets"
        config["isolation_scientific_status"] = (
            "seen_synthetic_single_sample_isolation_diagnostic"
        )
        config["isolation_variant"] = (
            "single_unique_sample_duplicated_to_two_training_rows_v1"
        )
        config["claim_limit"] = (
            "This dedicated adapter sees one duplicated synthetic pseudo-motion sample. "
            "It is not real-data training and tests isolation and overfit capacity only. "
            "It cannot establish generalization, real-data performance, physical validity, "
            "or paper-level photo realism."
        )
        config["question"] = (
            f"Can a dedicated adapter visibly learn {arm['sample_id']} when cross-sample "
            "gradient interference is removed, with step 32 matched to the shared run's "
            "approximate per-sample exposure count?"
        )
        config["train_families"] = [arm["family"]]
        config["train_anchors"] = [arm["sample_id"]]
        config["trainer"]["dataset_builder"] = (
            "scripts/build_phase_action_isolated_overfit_dataset.py"
        )
        config["trainer"]["dataset_builder_sha256_lf"] = DATASET_BUILDER_SHA256_LF
        config["dataset"].update(
            {
                "local_root": f"artifacts/{arm['dataset_name']}",
                "remote_root": f"/tmp/{arm['dataset_name']}",
                "metadata_sha256": arm["metadata_sha256"],
                "manifest_sha256": arm["manifest_sha256"],
                "source_manifest_sha256": (
                    "b7458dcdd5ec84bb44e8e67212c83680c90b97256a389d755d7c828f29e2bcde"
                ),
                "sample_count": 2,
                "unique_sample_count": 1,
                "selected_sample_id": arm["sample_id"],
                "duplication_policy": "same_sample_duplicated_metadata_row",
            }
        )
        config["training"]["output_root"] = (
            f"/tmp/foodstateedit_day12_phase_action_isolated_{arm['short_name']}_lora_v1"
        )
        config["evaluation_gate"]["seen_samples"] = 1
        config["evaluation_gate"]["matched_exposure_comparison"] = {
            "dedicated_checkpoint_step": 32,
            "dedicated_selected_sample_exposures": 32,
            "shared_checkpoint_step": 64,
            "shared_expected_selected_sample_exposures": 32,
        }
        config["decision_gate"] = {
            "both_dedicated_samples_require_clear_semantic_gain_before_blind_fork": True,
            "balanced_expansion_requires_clear_mechanistic_conclusion": True,
        }
        config["derivation"] = {
            "base_config": "configs/adapter_phase_action_overfit_v1.json",
            "base_config_sha256": BASE_CONFIG_SHA256,
            "config_builder": Path(__file__).name,
            "config_builder_sha256_lf": sha256_lf(Path(__file__).resolve()),
            "allowed_changes": [
                "single-sample dataset paths and hashes",
                "single training family and anchor",
                "new non-overwriting output root",
                "isolation-specific question and claim boundary",
            ],
            "unchanged_training_hyperparameters": True,
            "unchanged_model_and_runtime_hashes": True,
        }
        output_path = (
            repo_root
            / "configs"
            / f"adapter_phase_action_isolated_{arm['short_name']}_v1.json"
        )
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite {output_path}")
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(config, indent=2) + "\n")
        print(output_path.relative_to(repo_root), sha256_file(output_path))


if __name__ == "__main__":
    main()
