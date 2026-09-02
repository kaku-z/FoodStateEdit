#!/usr/bin/env python3
"""Derive the two frozen Day 12 isolated checkpoint-sweep configs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs" / "vace_phase_action_checkpoint_sweep_v1.json"
ARMS = {
    "udon": {
        "sample_id": "udon_chopsticks_imagegen_pseudo_v1",
        "dataset_root": "/tmp/day12_phase_action_isolated_udon_dataset_v2",
        "dataset_manifest_sha256": "2297d29d301b395663fd6b67f6c21a79e19a72d8f66583791b7f4ae217c02c54",
        "training_config_sha256": "c6e11696bf55528655677b40d9ebbd8c53fa7d65c4841b34eb8576da9a1e49f3",
        "training_root": "/tmp/foodstateedit_day12_phase_action_isolated_udon_lora_v1",
        "training_manifest": {
            "path": "/tmp/foodstateedit_day12_phase_action_isolated_udon_lora_v1/run_manifest.json",
            "size_bytes": 2719,
            "sha256": "414751d3ab9f92cc0d8ecbb169896ba5513762d5ce114bda7dc7c0bda3c98176",
        },
        "step64_validation": {
            "path": "/tmp/foodstateedit_day12_phase_action_isolation_udon_step64_validation_gp38_v1_20260902T0512Z.json",
            "size_bytes": 26787,
            "sha256": "f606098deed20bf9ac3eb3f74fc90349509f2e52cb824205838263215fa3bbe3",
        },
        "checkpoint_sha256": {
            16: "671f6fc3a29d3a517014c3512acdf334e92d0a38aa4a43c65c639fcf5782ab8a",
            32: "cec276ab2a9ba367b2b123fba49655a0e2b82b90f015609e98fec7a32ff2e7c6",
            48: "86e77f093c72516cf24adeaaebfdcd6f0b8a8c76d65c5c752208d85048e1dc57",
            64: "9f748fd70663ae6678885dbc0ae1b66d3f5ae74840cfa42c59215cf27607985b",
        },
        "output_base": "/tmp/foodstateedit_day12_phase_action_isolated_udon_checkpoint_sweep_gp38_v1",
    },
    "spoon": {
        "sample_id": "clear_broth_spoon_imagegen_pseudo_v1",
        "dataset_root": "/tmp/day12_phase_action_isolated_spoon_dataset_v2",
        "dataset_manifest_sha256": "5bc398202fc2636ecdab9d77916fbf52c233af5650601810b4dc4bbc068fbe2c",
        "training_config_sha256": "9255aa36bcd5d79d4f8c2c1c6e48db43b377df467fed7f8c27d1d453d12b9112",
        "training_root": "/tmp/foodstateedit_day12_phase_action_isolated_spoon_lora_v1",
        "training_manifest": {
            "path": "/tmp/foodstateedit_day12_phase_action_isolated_spoon_lora_v1/run_manifest.json",
            "size_bytes": 2718,
            "sha256": "3b48b24dbc605a7355d4e1497c80a3d795a6d2ea5ebaa83cb85774781d439768",
        },
        "step64_validation": {
            "path": "/tmp/foodstateedit_day12_phase_action_isolation_spoon_step64_validation_gp38_v1_20260902T0530Z.json",
            "size_bytes": 26788,
            "sha256": "92acdf752563e780ec499cb607be6f62b2bddd2aede6ffa93d1d32e15c23acbf",
        },
        "checkpoint_sha256": {
            16: "e3096719a1836352cb4c536c273494faf059a4639cf8d96881d857f15431c1ad",
            32: "5f0d43747c39ace60a56d3b85c7226b04e457a74a0a6d8a827e513053b889828",
            48: "effbfe11dd23e6d577a404b9ac8a77a49077db2c30752a3570d2fa0344415d45",
            64: "542f76461dd6a50bdb8a8a25cddf1b2a5425e261b6c707ab392b826498eef3df",
        },
        "output_base": "/tmp/foodstateedit_day12_phase_action_isolated_spoon_checkpoint_sweep_gp38_v1",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_config(arm_name: str) -> dict[str, object]:
    arm = ARMS[arm_name]
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    config = copy.deepcopy(source)
    sample = next(
        item for item in source["samples"] if item["sample_id"] == arm["sample_id"]
    )
    config["freeze_date"] = "2026-09-02"
    config["isolation_scientific_status"] = (
        "seen_synthetic_single_sample_isolation_diagnostic"
    )
    config["question"] = (
        f"Does the dedicated {arm['sample_id']} adapter visibly improve its own "
        "seen approach-contact-lift-hold pseudo-sequence, with dedicated step 32 "
        "as the exposure-matched comparison to the shared Day 11 step 64?"
    )
    config["claim_limit"] = (
        "This single-sample seen synthetic pseudo-motion sweep tests isolation and "
        "overfit capacity only. It cannot establish held-out generalization, "
        "real-data performance, physical motion validity, or photo realism."
    )
    config["dataset"].update(
        {
            "remote_root": arm["dataset_root"],
            "manifest_sha256": arm["dataset_manifest_sha256"],
            "sample_count": 2,
            "unique_sample_count": 1,
            "selected_sample_id": arm["sample_id"],
        }
    )
    config["samples"] = [sample]
    config["adapters"]["training_manifest"] = arm["training_manifest"]
    config["adapters"]["step64_validation"] = arm["step64_validation"]
    config["adapters"]["conditions"] = [{"name": "lora_off", "step": 0, "checkpoint": None}]
    for step in (16, 32, 48, 64):
        config["adapters"]["conditions"].append(
            {
                "name": f"step_{step}",
                "step": step,
                "checkpoint": {
                    "path": f"{arm['training_root']}/step-{step}.safetensors",
                    "size_bytes": 15354160,
                    "sha256": arm["checkpoint_sha256"][step],
                },
            }
        )
    config["runtime"]["cache_root"] = (
        "/tmp/foodstateedit_day12_phase_action_checkpoint_sweep_cache_gp38_v1"
    )
    config["output_base"] = arm["output_base"]
    config["comparison_contract"] = {
        "dedicated_training_config_sha256": arm["training_config_sha256"],
        "dedicated_step32_selected_sample_exposures": 32,
        "shared_day11_step64_expected_selected_sample_exposures": 32,
        "matched_exposure_interference_comparison": "dedicated_step32_vs_shared_step64",
        "dedicated_step64_interpretation": "additional_single_sample_overfit_capacity_only",
    }
    config["decision_gate"].update(
        {
            "blind_fork_requires_both_dedicated_samples_clear_seen_semantic_gain": True,
            "numeric_mae_alone_never_unlocks_blind_fork": True,
        }
    )
    return config


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen config: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(build_config(args.arm), indent=2) + "\n"
    output.write_bytes(payload.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
