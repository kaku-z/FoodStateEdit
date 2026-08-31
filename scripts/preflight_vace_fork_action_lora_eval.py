#!/usr/bin/env python3
"""Fail-closed preflight for the blind Day 9 fork LoRA evaluation."""

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


METHOD = "vace_fork_action_lora_blind_eval_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = args.report.resolve()
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite report: {report_path}")

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    checks: list[dict[str, object]] = []

    add_check(checks, "method", config.get("method") == METHOD, config.get("method"), METHOD)
    add_check(
        checks,
        "scientific_status",
        config.get("scientific_status") == "blind_cross_family_mechanism_pilot_not_generalization",
        config.get("scientific_status"),
        "Blind fork pilot with fork absent from training; not a generalization claim.",
    )
    add_check(checks, "input_root", str(input_root) == config["control"]["remote_root"], str(input_root), config["control"]["remote_root"])
    add_check(checks, "output_root", str(output_root) == config["output_root"], str(output_root), config["output_root"])
    add_check(checks, "output_absent", not output_root.exists(), str(output_root), "Output root must not exist.")

    launcher_root = Path(__file__).resolve().parent
    for key in ("preflight", "runner"):
        record = config["launcher"][key]
        path = launcher_root / Path(record["path"]).name
        actual = sha256_file(path) if path.is_file() else None
        add_check(checks, f"launcher_hash:{key}", actual == record["sha256"], actual, record["sha256"])

    for name, expected in config["control"]["files"].items():
        path = input_root / name
        size = path.stat().st_size if path.is_file() else None
        actual = sha256_file(path) if path.is_file() else None
        add_check(checks, f"control_size:{name}", size == expected["size_bytes"], size, str(expected["size_bytes"]))
        add_check(checks, f"control_hash:{name}", actual == expected["sha256"], actual, expected["sha256"])

    control_manifest_path = input_root / "run_manifest.json"
    control_manifest = json.loads(control_manifest_path.read_text(encoding="utf-8")) if control_manifest_path.is_file() else {}
    add_check(checks, "control_anchor", control_manifest.get("anchor_id") == "pasta_fork_001", control_manifest.get("anchor_id"), "pasta_fork_001")
    add_check(checks, "control_geometry", control_manifest.get("geometry_source") == "relative_3d_normalized_pinhole", control_manifest.get("geometry_source"), "relative_3d_normalized_pinhole")
    add_check(checks, "control_frames", control_manifest.get("frame_count") == config["inference"]["num_frames"], control_manifest.get("frame_count"), str(config["inference"]["num_frames"]))
    add_check(checks, "depth_crossing", control_manifest.get("selected_depth_crosses_fork") is True, control_manifest.get("selected_depth_crosses_fork"), "Selected helix crosses fork depth.")

    adapter = config["adapter"]
    checkpoint = Path(adapter["checkpoint"]["path"])
    checkpoint_size = checkpoint.stat().st_size if checkpoint.is_file() else None
    checkpoint_hash = sha256_file(checkpoint) if checkpoint.is_file() else None
    add_check(checks, "checkpoint_size", checkpoint_size == adapter["checkpoint"]["size_bytes"], checkpoint_size, str(adapter["checkpoint"]["size_bytes"]))
    add_check(checks, "checkpoint_hash", checkpoint_hash == adapter["checkpoint"]["sha256"], checkpoint_hash, adapter["checkpoint"]["sha256"])
    add_check(checks, "fork_training_occurrences", adapter["held_out"]["training_occurrences"] == 0, adapter["held_out"]["training_occurrences"], "0")
    add_check(checks, "lora_alpha", adapter["alpha"] == 1.0, adapter["alpha"], "1.0")

    training_manifest_path = Path(adapter["training_manifest"]["path"])
    training_manifest_hash = sha256_file(training_manifest_path) if training_manifest_path.is_file() else None
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8")) if training_manifest_path.is_file() else {}
    add_check(checks, "training_manifest_hash", training_manifest_hash == adapter["training_manifest"]["sha256"], training_manifest_hash, adapter["training_manifest"]["sha256"])
    add_check(checks, "training_status", training_manifest.get("status") == "complete_requires_blind_fork_evaluation", training_manifest.get("status"), "complete_requires_blind_fork_evaluation")
    checkpoint_records = {item.get("path"): item for item in training_manifest.get("checkpoints", [])}
    add_check(checks, "training_checkpoint_record", checkpoint_records.get(checkpoint.name, {}).get("sha256") == checkpoint_hash, checkpoint_records.get(checkpoint.name), "Training manifest records the exact evaluation checkpoint.")

    validation_path = Path(adapter["validation"]["path"])
    validation_hash = sha256_file(validation_path) if validation_path.is_file() else None
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.is_file() else {}
    add_check(checks, "validation_hash", validation_hash == adapter["validation"]["sha256"], validation_hash, adapter["validation"]["sha256"])
    add_check(checks, "validation_status", validation.get("status") == "complete", validation.get("status"), "complete")
    add_check(checks, "validation_checkpoint_hash", validation.get("checkpoint_sha256") == checkpoint_hash, validation.get("checkpoint_sha256"), str(checkpoint_hash))
    add_check(checks, "validation_updated_tensors", validation.get("official_loader_updated_tensor_count") == 80, validation.get("official_loader_updated_tensor_count"), "80")

    runtime = config["runtime"]
    geoedit_root = Path(runtime["geoedit_root"])
    for relative, expected_hash in runtime["geoedit_files"].items():
        path = geoedit_root / relative
        actual = sha256_file(path) if path.is_file() else None
        add_check(checks, f"runtime_hash:{relative}", actual == expected_hash, actual, expected_hash)
    audit_path = Path(runtime["model_hash_audit"])
    audit_hash = sha256_file(audit_path) if audit_path.is_file() else None
    audit_text = audit_path.read_text(encoding="utf-8") if audit_path.is_file() else ""
    add_check(checks, "model_audit_hash", audit_hash == runtime["model_hash_audit_sha256"], audit_hash, runtime["model_hash_audit_sha256"])
    model_root = Path(runtime["model_root"])
    for relative, expected in runtime["model_files"].items():
        path = model_root / relative
        size = path.stat().st_size if path.is_file() else None
        add_check(checks, f"model_size:{relative}", size == expected["size_bytes"], size, str(expected["size_bytes"]))
        add_check(checks, f"model_audited_hash:{relative}", expected["sha256"] in audit_text, expected["sha256"] in audit_text, "Expected SHA-256 occurs in frozen audit.")

    import_environment = os.environ.copy()
    import_environment.update(runtime["offline_environment"])
    import_environment["PYTHONPATH"] = str(geoedit_root)
    import_environment["HF_HOME"] = runtime["cache_root"]
    import_environment["TRANSFORMERS_CACHE"] = runtime["cache_root"]
    import_result = subprocess.run(
        [runtime["python"], "-c", "import cv2,numpy,PIL,torch; from geoedit import inference; print('runtime_import_ok')"],
        check=False,
        capture_output=True,
        text=True,
        env=import_environment,
        timeout=60,
    )
    add_check(checks, "runtime_import", import_result.returncode == 0, {"return_code": import_result.returncode, "stderr": import_result.stderr[-2000:]}, "Frozen runtime imports offline.")

    available_mib = available_memory_mib()
    gate = config["resource_gate"]
    add_check(checks, "system_memory", available_mib >= gate["min_available_system_memory_mib"], available_mib, f">={gate['min_available_system_memory_mib']} MiB")
    gpus, processes = read_gpu_state()
    safe_gpus = [
        gpu for gpu in gpus
        if gpu["name"] == gate["required_gpu_name"]
        and gpu["memory_free_mib"] >= gate["min_free_memory_mib"]
        and gpu["utilization_percent"] <= gate["max_utilization_percent"]
        and (not gate["require_no_compute_process"] or not gpu["compute_processes"])
    ]
    selected = next((gpu for gpu in safe_gpus if args.gpu_index is None or gpu["index"] == args.gpu_index), None)
    add_check(checks, "gpu_gate", selected is not None, {"requested": args.gpu_index, "safe_gpu_indices": [gpu["index"] for gpu in safe_gpus], "gpus": gpus, "processes": processes}, "One conflict-free A6000 satisfies the frozen gate.")

    ready = all(check["passed"] for check in checks)
    report = {
        "schema_version": "foodstateedit.vace_fork_action_lora_preflight.v1",
        "ready": ready,
        "selected_gpu": selected["index"] if selected is not None else None,
        "config_sha256": sha256_file(config_path),
        "checks": checks,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
