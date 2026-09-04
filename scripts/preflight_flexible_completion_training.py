#!/usr/bin/env python3
"""Fail-closed preflight for one Day 13 matched-objective training arm."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from preflight_vace_fork_3d_compare import (
    add_check,
    available_memory_mib,
    read_gpu_state,
    sha256_file,
)


SCHEMA_VERSION = "foodstateedit.flexible_completion_execution.v1"
METHOD = "relative3d_topology_weighted_flexible_completion_v1"
ARM_IDS = (
    "planar_uniform",
    "relative3d_uniform",
    "relative3d_topology_weighted",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARM_IDS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def verify_file(
    checks: list[dict[str, object]],
    check_id: str,
    path: Path,
    record: dict[str, object],
) -> None:
    size = path.stat().st_size if path.is_file() else None
    digest = sha256_file(path) if path.is_file() else None
    add_check(checks, f"{check_id}:size", size == record["size_bytes"], size, record["size_bytes"])
    add_check(checks, f"{check_id}:sha256", digest == record["sha256"], digest, record["sha256"])


def count_video_frames(path: Path) -> int | None:
    try:
        reader = imageio.get_reader(path)
        count = int(reader.count_frames())
        reader.close()
        return count
    except Exception:
        return None


def mask_activity(path: Path) -> list[int]:
    reader = imageio.get_reader(path)
    counts = [int((np.asarray(frame)[..., :3].mean(axis=2) > 127).sum()) for frame in reader]
    reader.close()
    return counts


def main() -> int:
    args = parse_args()
    report_path = args.report.resolve()
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite report: {report_path}")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    checks: list[dict[str, object]] = []

    add_check(checks, "schema", config.get("schema_version") == SCHEMA_VERSION, config.get("schema_version"), SCHEMA_VERSION)
    add_check(checks, "method", config.get("method") == METHOD, config.get("method"), METHOD)
    add_check(checks, "execution_allowed", config.get("execution_allowed") is True, config.get("execution_allowed"), True)
    add_check(checks, "blind_fork_forbidden", config.get("blind_fork_evaluation_allowed") is False, config.get("blind_fork_evaluation_allowed"), False)
    add_check(checks, "arm_order", list(config.get("arms", {})) == list(ARM_IDS), list(config.get("arms", {})), list(ARM_IDS))
    arm = config.get("arms", {}).get(args.arm, {})
    add_check(checks, "arm_exists", bool(arm), args.arm, list(ARM_IDS))
    add_check(checks, "dataset_root", str(dataset_root) == config["dataset"]["remote_root"], str(dataset_root), config["dataset"]["remote_root"])
    add_check(checks, "output_root", str(output_root) == arm.get("output_root"), str(output_root), arm.get("output_root"))
    add_check(checks, "output_absent", not output_root.exists(), str(output_root), "Must not exist")

    implementation = config["implementation"]
    for name in ("design", "geometry"):
        record = config[name]
        verify_file(checks, name, repo_root / record["path"], record)
    for name, record in implementation.items():
        path = repo_root / record["path"]
        verify_file(checks, f"implementation:{name}", path, record)

    manifest_path = dataset_root / config["dataset"]["manifest"]["path"]
    verify_file(checks, "dataset_manifest", manifest_path, config["dataset"]["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    add_check(
        checks,
        "dataset_contract",
        manifest.get("schema_version") == "foodstateedit.flexible_completion_dataset.v1"
        and manifest.get("sample_count") == 2
        and manifest.get("unique_sample_count") == 1
        and manifest.get("frame_count") == 21
        and manifest.get("strand_depth_crosses_both_chopstick_depths") is True
        and manifest.get("outside_frozen_edit_support_max_pixel_difference_before_video_encoding") == 0,
        {
            key: manifest.get(key)
            for key in (
                "schema_version",
                "sample_count",
                "unique_sample_count",
                "frame_count",
                "strand_depth_crosses_both_chopstick_depths",
                "outside_frozen_edit_support_max_pixel_difference_before_video_encoding",
            )
        },
        "Frozen one-sample, two-row, 21-frame relative-3D dataset",
    )
    for group in ("frozen_source_files", "derived_files", "metadata"):
        for name, record in manifest.get(group, {}).items():
            verify_file(checks, f"dataset:{group}:{name}", dataset_root / record["path"], record)

    metadata_name = arm.get("metadata")
    metadata_record = manifest.get("metadata", {}).get(Path(metadata_name or "").stem, {})
    add_check(checks, "arm_metadata_record", bool(metadata_record), metadata_name, sorted(manifest.get("metadata", {})))
    if metadata_record:
        verify_file(checks, "arm_metadata", dataset_root / metadata_name, metadata_record)
    expected_loss = "topology_weighted" if args.arm == "relative3d_topology_weighted" else "uniform"
    add_check(checks, "arm_loss", arm.get("loss_mode") == expected_loss, arm.get("loss_mode"), expected_loss)
    add_check(
        checks,
        "matched_relative3d_control",
        config["arms"]["relative3d_uniform"]["control"]
        == config["arms"]["relative3d_topology_weighted"]["control"],
        {
            key: config["arms"][key]["control"]
            for key in ("relative3d_uniform", "relative3d_topology_weighted")
        },
        "Identical relative-3D control",
    )

    for name in (
        "relative_3d_vace_control_video",
        "flexible_strand_mask_video",
        "pinch_contact_mask_video",
        "source_connection_mask_video",
    ):
        record = manifest.get("derived_files", {}).get(name, {})
        video_path = dataset_root / record.get("path", "missing")
        count = count_video_frames(video_path)
        add_check(checks, f"video_frames:{name}", count == 21, count, 21)
    mask_records = manifest.get("derived_files", {})
    activities: dict[str, list[int]] = {}
    for name in (
        "flexible_strand_mask_video",
        "pinch_contact_mask_video",
        "source_connection_mask_video",
    ):
        record = mask_records.get(name, {})
        path = dataset_root / record.get("path", "missing")
        try:
            activities[name] = mask_activity(path)
        except Exception:
            activities[name] = []
    mask_phase_contract = (
        all(len(values) == 21 for values in activities.values())
        and all(all(value == 0 for value in values[:6]) for values in activities.values())
        and all(value > 0 for value in activities["flexible_strand_mask_video"][6:])
        and all(value > 0 for value in activities["pinch_contact_mask_video"][6:])
        and all(value == 0 for value in activities["source_connection_mask_video"][6:9])
        and all(value > 0 for value in activities["source_connection_mask_video"][9:])
    )
    add_check(checks, "topology_mask_phase_contract", mask_phase_contract, activities, "Zero before contact; local contact masks at 6-8; all masks active from frame 9")

    upstream = config["trainer_runtime"]
    upstream_root = Path(upstream["root"])
    for name, record in upstream["files"].items():
        verify_file(checks, f"upstream:{name}", upstream_root / record["path"], record)
    model = config["model"]
    audit_path = Path(model["hash_audit"]["path"])
    verify_file(checks, "model_hash_audit", audit_path, model["hash_audit"])
    audit_text = audit_path.read_text(encoding="utf-8") if audit_path.is_file() else ""
    for record in model["files"]:
        path = Path(model["root"]) / record["path"]
        size = path.stat().st_size if path.is_file() else None
        add_check(checks, f"model_size:{record['path']}", size == record["size_bytes"], size, record["size_bytes"])
        add_check(checks, f"model_audited_hash:{record['path']}", record["sha256"] in audit_text, record["sha256"] in audit_text, True)

    environment = os.environ.copy()
    environment.update(config["offline_environment"])
    environment["PYTHONPATH"] = os.pathsep.join([str(upstream_root), str(repo_root)])
    imported = subprocess.run(
        [
            config["python"],
            "-c",
            "import accelerate,imageio,numpy,peft,safetensors,torch,torchvision,transformers; from examples.wanvideo.model_training.train import WanTrainingModule",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    add_check(checks, "offline_runtime_import", imported.returncode == 0, {"return_code": imported.returncode, "stderr": imported.stderr[-2000:]}, "All frozen training modules import without network")

    gate = config["resource_gate"]
    available_mib = available_memory_mib()
    add_check(checks, "system_memory", available_mib >= gate["min_available_system_memory_mib"], available_mib, f">={gate['min_available_system_memory_mib']}")
    gpus, processes = read_gpu_state()
    selected = next((gpu for gpu in gpus if gpu["index"] == args.gpu_index), None)
    gpu_ready = (
        selected is not None
        and selected["name"] == gate["required_gpu_name"]
        and selected["memory_free_mib"] >= gate["min_free_memory_mib"]
        and selected["utilization_percent"] <= gate["max_utilization_percent"]
        and (not gate["require_no_compute_process"] or not selected["compute_processes"])
    )
    add_check(checks, "gpu_gate", gpu_ready, {"selected": selected, "all_processes": processes}, "Conflict-free A6000 satisfies the frozen threshold")

    ready = all(check["passed"] for check in checks)
    report = {
        "schema_version": "foodstateedit.flexible_completion_training_preflight.v1",
        "ready": ready,
        "arm": args.arm,
        "selected_gpu": args.gpu_index if gpu_ready else None,
        "config_sha256": sha256_file(config_path),
        "checks": checks,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
