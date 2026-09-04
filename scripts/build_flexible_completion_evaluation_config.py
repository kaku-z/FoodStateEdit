#!/usr/bin/env python3
"""Freeze the Day 13 evaluation config from completed training evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ARM_IDS = (
    "planar_uniform",
    "relative3d_uniform",
    "relative3d_topology_weighted",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--runtime-template-config", type=Path, required=True)
    parser.add_argument("--cross-arm-match", type=Path, required=True)
    parser.add_argument(
        "--validation",
        action="append",
        required=True,
        help="ARM_ID=/absolute/path/to/step32_validation.json; repeat for all arms",
    )
    parser.add_argument("--output-config", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, stored_path: str | None = None) -> dict[str, object]:
    return {
        "path": str(path) if stored_path is None else stored_path,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    args = parse_args()
    output_path = args.output_config.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation config: {output_path}")
    execution_path = args.execution_config.resolve()
    template_path = args.runtime_template_config.resolve()
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    template = json.loads(template_path.read_text(encoding="utf-8"))
    if execution.get("schema_version") != "foodstateedit.flexible_completion_execution.v1":
        raise ValueError("Unexpected Day 13 execution config")
    if not execution.get("execution_allowed") or execution.get("blind_fork_evaluation_allowed"):
        raise ValueError("Execution config is not a permitted seen-only protocol")
    if template.get("schema_version") != "foodstateedit.vace_phase_action_checkpoint_sweep.v1":
        raise ValueError("Unexpected frozen inference runtime template")

    validations: dict[str, Path] = {}
    for item in args.validation:
        arm_id, separator, raw_path = item.partition("=")
        if not separator or arm_id not in ARM_IDS:
            raise ValueError(f"Invalid --validation value: {item}")
        validations[arm_id] = Path(raw_path).resolve()
    if set(validations) != set(ARM_IDS):
        raise ValueError(f"Need exactly one validation for each arm: {ARM_IDS}")

    cross_arm_path = args.cross_arm_match.resolve()
    cross_arm = json.loads(cross_arm_path.read_text(encoding="utf-8"))
    if (
        cross_arm.get("status") != "complete"
        or not cross_arm.get("initial_trainable_state_identical")
        or not all(cross_arm.get("matched_trace_by_arm", {}).values())
    ):
        raise ValueError("Cross-arm matched-randomness gate has not passed")

    adapters: dict[str, dict[str, object]] = {}
    for arm_id in ARM_IDS:
        training_root = Path(execution["arms"][arm_id]["output_root"])
        manifest_path = training_root / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete_requires_cross_arm_match_and_checkpoint_validation":
            raise ValueError(f"Training arm is incomplete: {arm_id}")
        checkpoint_path = training_root / "step-32.safetensors"
        checkpoint_record = next(
            (
                record
                for record in manifest.get("checkpoints", [])
                if Path(record["path"]).name == "step-32.safetensors"
            ),
            None,
        )
        if checkpoint_record is None:
            raise ValueError(f"Missing step 32 checkpoint record: {arm_id}")
        if (
            checkpoint_path.stat().st_size != checkpoint_record["size_bytes"]
            or sha256_file(checkpoint_path) != checkpoint_record["sha256"]
        ):
            raise ValueError(f"Checkpoint no longer matches its run manifest: {arm_id}")
        validation_path = validations[arm_id]
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if (
            validation.get("status") != "complete"
            or validation.get("tensor_count") != 160
            or validation.get("official_loader_updated_tensor_count") != 80
        ):
            raise ValueError(f"Checkpoint validation failed: {arm_id}")
        adapters[arm_id] = {
            "training_manifest": file_record(manifest_path),
            "checkpoint": file_record(checkpoint_path),
            "validation": file_record(validation_path),
        }

    dataset_root = Path(execution["dataset"]["remote_root"])
    dataset_manifest_path = dataset_root / execution["dataset"]["manifest"]["path"]
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if sha256_file(dataset_manifest_path) != execution["dataset"]["manifest"]["sha256"]:
        raise ValueError("Remote Day 13 dataset manifest hash mismatch")
    source_files = dataset_manifest["frozen_source_files"]
    derived_files = dataset_manifest["derived_files"]
    dataset_files = {
        "reference_image": source_files["vace_reference_image"],
        "planar_control": source_files["planar_vace_control"],
        "relative3d_control": derived_files["relative_3d_vace_control_video"],
        "edit_alpha": source_files["edit_alpha"],
        "target_video": source_files["video"],
        "target_keyframe": source_files["target_keyframe"],
        "phase_schedule": source_files["phase_schedule"],
        "flexible_strand_mask_video": derived_files["flexible_strand_mask_video"],
        "pinch_contact_mask_video": derived_files["pinch_contact_mask_video"],
        "source_connection_mask_video": derived_files["source_connection_mask_video"],
    }
    runtime = json.loads(json.dumps(template["runtime"]))
    runtime["cache_root"] = "/tmp/foodstateedit_day13_flexible_completion_eval_cache_v1"
    implementation_keys = (
        "evaluation_preflight",
        "evaluation_runner",
        "preflight_gpu_dependency",
        "evaluation_io_dependency",
        "evaluation_lora_dependency",
        "evaluation_lora_v2_dependency",
    )
    implementation = {
        key: execution["implementation"][key] for key in implementation_keys
    }
    conditions = [
        {"name": "planar_lora_off", "control": "planar", "step": 0, "checkpoint": None},
        {
            "name": "planar_uniform_step32",
            "control": "planar",
            "step": 32,
            "checkpoint": adapters["planar_uniform"]["checkpoint"],
        },
        {"name": "relative3d_lora_off", "control": "relative3d", "step": 0, "checkpoint": None},
        {
            "name": "relative3d_uniform_step32",
            "control": "relative3d",
            "step": 32,
            "checkpoint": adapters["relative3d_uniform"]["checkpoint"],
        },
        {
            "name": "relative3d_topology_weighted_step32",
            "control": "relative3d",
            "step": 32,
            "checkpoint": adapters["relative3d_topology_weighted"]["checkpoint"],
        },
    ]
    config = {
        "schema_version": "foodstateedit.flexible_completion_evaluation.v1",
        "freeze_date": "2026-09-03",
        "method": "relative3d_topology_weighted_flexible_completion_eval_v1",
        "scientific_status": "seen_synthetic_five_condition_mechanism_evaluation",
        "execution_allowed": True,
        "blind_fork_evaluation_allowed": False,
        "claim_limit": execution["claim_limit"],
        "source_execution_config": file_record(execution_path),
        "implementation": implementation,
        "dataset": {
            "remote_root": str(dataset_root),
            "manifest": file_record(
                dataset_manifest_path,
                stored_path=dataset_manifest_path.relative_to(dataset_root).as_posix(),
            ),
            "files": dataset_files,
        },
        "prompt": dataset_manifest["prompt"],
        "prompt_sha256_utf8": hashlib.sha256(
            dataset_manifest["prompt"].encode("utf-8")
        ).hexdigest(),
        "controls": {
            "planar": dataset_files["planar_control"],
            "relative3d": dataset_files["relative3d_control"],
        },
        "adapters": adapters,
        "cross_arm_match": file_record(cross_arm_path),
        "adapter_alpha": 1.0,
        "conditions": conditions,
        "inference": template["inference"],
        "runtime": runtime,
        "output_root": execution["evaluation"]["output_root"],
        "resource_gate": execution["resource_gate"],
        "positive_gate": {
            "topology_weight_volume_mae_improvement_percent_min": 5.0,
            "pinch_contact_roi_mae_improvement_percent_min": 5.0,
            "both_blinded_reviewers_prefer_weighted_for_pinch_connected_lift_and_final_hold": True,
            "photo_realism_not_worse_for_either_reviewer": True,
            "all_outside_support_max_pixel_difference": 0,
            "numeric_only_improvement_passes": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_config": str(output_path),
                "size_bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
                "conditions": [item["name"] for item in conditions],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
