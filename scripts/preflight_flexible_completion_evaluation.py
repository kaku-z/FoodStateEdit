#!/usr/bin/env python3
"""Fail-closed preflight for the Day 13 five-condition evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from preflight_vace_fork_3d_compare import (
    add_check,
    available_memory_mib,
    read_gpu_state,
    sha256_file,
)


SCHEMA_VERSION = "foodstateedit.flexible_completion_evaluation.v1"
METHOD = "relative3d_topology_weighted_flexible_completion_eval_v1"
CONDITION_ORDER = [
    "planar_lora_off",
    "planar_uniform_step32",
    "relative3d_lora_off",
    "relative3d_uniform_step32",
    "relative3d_topology_weighted_step32",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
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


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report_path = args.report.resolve()
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite report: {report_path}")
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    checks: list[dict[str, object]] = []

    add_check(checks, "schema", config.get("schema_version") == SCHEMA_VERSION, config.get("schema_version"), SCHEMA_VERSION)
    add_check(checks, "method", config.get("method") == METHOD, config.get("method"), METHOD)
    add_check(checks, "execution_allowed", config.get("execution_allowed") is True, config.get("execution_allowed"), True)
    add_check(checks, "blind_fork_forbidden", config.get("blind_fork_evaluation_allowed") is False, config.get("blind_fork_evaluation_allowed"), False)
    condition_names = [item.get("name") for item in config.get("conditions", [])]
    add_check(checks, "condition_order", condition_names == CONDITION_ORDER, condition_names, CONDITION_ORDER)
    add_check(checks, "dataset_root", str(dataset_root) == config["dataset"]["remote_root"], str(dataset_root), config["dataset"]["remote_root"])
    add_check(checks, "output_root", str(output_root) == config["output_root"], str(output_root), config["output_root"])
    add_check(checks, "output_absent", not output_root.exists(), str(output_root), "Must not exist")

    for name, record in config["implementation"].items():
        verify_file(checks, f"implementation:{name}", repo_root / record["path"], record)
    manifest_record = config["dataset"]["manifest"]
    manifest_path = dataset_root / manifest_record["path"]
    verify_file(checks, "dataset_manifest", manifest_path, manifest_record)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    for logical_name, record in config["dataset"]["files"].items():
        verify_file(checks, f"dataset:{logical_name}", dataset_root / record["path"], record)

    adapter_records = config["adapters"]
    for arm_id, records in adapter_records.items():
        verify_file(checks, f"training_manifest:{arm_id}", Path(records["training_manifest"]["path"]), records["training_manifest"])
        training_manifest_path = Path(records["training_manifest"]["path"])
        training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8")) if training_manifest_path.is_file() else {}
        add_check(checks, f"training_status:{arm_id}", training_manifest.get("status") == "complete_requires_cross_arm_match_and_checkpoint_validation", training_manifest.get("status"), "complete_requires_cross_arm_match_and_checkpoint_validation")
        verify_file(checks, f"checkpoint:{arm_id}", Path(records["checkpoint"]["path"]), records["checkpoint"])
        verify_file(checks, f"validation:{arm_id}", Path(records["validation"]["path"]), records["validation"])
        validation_path = Path(records["validation"]["path"])
        validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.is_file() else {}
        add_check(checks, f"validation_status:{arm_id}", validation.get("status") == "complete", validation.get("status"), "complete")
        add_check(checks, f"validation_tensor_count:{arm_id}", validation.get("tensor_count") == 160 and validation.get("official_loader_updated_tensor_count") == 80, {"tensor_count": validation.get("tensor_count"), "updated": validation.get("official_loader_updated_tensor_count")}, {"tensor_count": 160, "updated": 80})
    verify_file(checks, "cross_arm_match", Path(config["cross_arm_match"]["path"]), config["cross_arm_match"])
    match_path = Path(config["cross_arm_match"]["path"])
    match = json.loads(match_path.read_text(encoding="utf-8")) if match_path.is_file() else {}
    add_check(checks, "cross_arm_match_status", match.get("status") == "complete" and match.get("initial_trainable_state_identical") is True and all(match.get("matched_trace_by_arm", {}).values()), {key: match.get(key) for key in ("status", "initial_trainable_state_identical", "matched_trace_by_arm")}, "All matched randomness checks complete")

    runtime = config["runtime"]
    geoedit_root = Path(runtime["geoedit_root"])
    for relative, expected_hash in runtime["geoedit_files"].items():
        path = geoedit_root / relative
        actual = sha256_file(path) if path.is_file() else None
        add_check(checks, f"runtime:{relative}", actual == expected_hash, actual, expected_hash)
    audit_path = Path(runtime["model_hash_audit"]["path"])
    verify_file(checks, "model_hash_audit", audit_path, runtime["model_hash_audit"])
    audit_text = audit_path.read_text(encoding="utf-8") if audit_path.is_file() else ""
    for relative, record in runtime["model_files"].items():
        path = Path(runtime["model_root"]) / relative
        size = path.stat().st_size if path.is_file() else None
        add_check(checks, f"model_size:{relative}", size == record["size_bytes"], size, record["size_bytes"])
        add_check(checks, f"model_hash_audited:{relative}", record["sha256"] in audit_text, record["sha256"] in audit_text, True)

    environment = os.environ.copy()
    environment.update(runtime["offline_environment"])
    environment.update({"PYTHONPATH": str(geoedit_root), "HF_HOME": runtime["cache_root"], "TRANSFORMERS_CACHE": runtime["cache_root"]})
    imported = subprocess.run(
        [runtime["python"], "-c", "import cv2,numpy,PIL,torch,safetensors; from geoedit import inference"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    add_check(checks, "offline_runtime_import", imported.returncode == 0, {"return_code": imported.returncode, "stderr": imported.stderr[-2000:]}, "Frozen inference runtime imports offline")

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
        "schema_version": "foodstateedit.flexible_completion_evaluation_preflight.v1",
        "ready": ready,
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
