#!/usr/bin/env python3
"""Fail-closed preflight for one seen phase-action checkpoint-sweep worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from preflight_vace_fork_3d_compare import add_check, available_memory_mib, read_gpu_state, sha256_file


METHOD = "vace_phase_action_checkpoint_sweep_v1"
STATUS = "seen_phase_overfit_capacity_diagnostic_not_generalization"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def verify_record(checks: list[dict[str, object]], check_id: str, record: dict[str, object]) -> None:
    path = Path(record["path"])
    size = path.stat().st_size if path.is_file() else None
    actual = sha256_file(path) if path.is_file() else None
    add_check(checks, f"{check_id}:size", size == record["size_bytes"], size, str(record["size_bytes"]))
    add_check(checks, f"{check_id}:sha256", actual == record["sha256"], actual, record["sha256"])


def main() -> int:
    args = parse_args()
    report_path = args.report.resolve()
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite report: {report_path}")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    samples = {sample["sample_id"]: sample for sample in config["samples"]}
    sample = samples.get(args.sample_id)
    checks: list[dict[str, object]] = []

    add_check(checks, "method", config.get("method") == METHOD, config.get("method"), METHOD)
    add_check(checks, "scientific_status", config.get("scientific_status") == STATUS, config.get("scientific_status"), STATUS)
    add_check(checks, "claim_limit", "cannot establish" in config.get("claim_limit", ""), config.get("claim_limit"), "No generalization or realism claim.")
    add_check(checks, "sample_id", sample is not None, args.sample_id, f"One of {sorted(samples)}")
    expected_output = Path(config["output_base"]) / args.sample_id
    add_check(checks, "dataset_root", str(dataset_root) == config["dataset"]["remote_root"], str(dataset_root), config["dataset"]["remote_root"])
    add_check(checks, "output_root", output_root == expected_output, str(output_root), str(expected_output))
    add_check(checks, "output_absent", not output_root.exists(), str(output_root), "Output root must not exist.")

    launcher_root = Path(__file__).resolve().parent
    for key in ("preflight", "runner"):
        record = config["launcher"][key]
        path = launcher_root / Path(record["path"]).name
        actual = sha256_file(path) if path.is_file() else None
        add_check(checks, f"launcher_hash:{key}", actual == record["sha256"], actual, record["sha256"])
    for name, expected in config["launcher"]["dependencies"].items():
        path = launcher_root / name
        actual = sha256_file(path) if path.is_file() else None
        add_check(checks, f"dependency_hash:{name}", actual == expected, actual, expected)

    manifest_path = dataset_root / "dataset_manifest.json"
    manifest_hash = sha256_file(manifest_path) if manifest_path.is_file() else None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    add_check(checks, "dataset_manifest_hash", manifest_hash == config["dataset"]["manifest_sha256"], manifest_hash, config["dataset"]["manifest_sha256"])
    add_check(
        checks,
        "dataset_contract",
        manifest.get("sample_count") == 2
        and manifest.get("motion_policy") == config["dataset"]["motion_policy"]
        and manifest.get("held_out", {}).get("training_occurrences") == 0,
        {"sample_count": manifest.get("sample_count"), "motion_policy": manifest.get("motion_policy"), "held_out": manifest.get("held_out")},
        "Two pseudo-motion samples with zero fork training occurrences.",
    )
    phase_record = config["dataset"]["phase_schedule"]
    phase_path = dataset_root / phase_record["path"]
    phase_size = phase_path.stat().st_size if phase_path.is_file() else None
    phase_hash = sha256_file(phase_path) if phase_path.is_file() else None
    add_check(checks, "phase_schedule_size", phase_size == phase_record["size_bytes"], phase_size, str(phase_record["size_bytes"]))
    add_check(checks, "phase_schedule_hash", phase_hash == phase_record["sha256"], phase_hash, phase_record["sha256"])
    manifest_samples = {item.get("sample_id"): item for item in manifest.get("samples", [])}
    manifest_sample = manifest_samples.get(args.sample_id, {})
    if sample is not None:
        add_check(checks, "seen_training_sample", manifest_sample.get("split") == "train_overfit_sanity_only", manifest_sample.get("split"), "train_overfit_sanity_only")
        add_check(checks, "sample_prompt", manifest_sample.get("prompt") == sample["prompt"], manifest_sample.get("prompt"), "Frozen sample prompt")
        prompt_hash = hashlib.sha256(sample["prompt"].encode("utf-8")).hexdigest()
        add_check(checks, "sample_prompt_hash", prompt_hash == sample["prompt_sha256_utf8"], prompt_hash, sample["prompt_sha256_utf8"])
        source_keys = {"reference": "vace_reference_image", "control": "vace_video", "alpha": "edit_alpha", "target_video": "video", "target_keyframe": "target_keyframe"}
        for key, manifest_key in source_keys.items():
            expected = sample["files"][key]
            add_check(checks, f"manifest_record:{key}", manifest_sample.get("files", {}).get(manifest_key) == expected, manifest_sample.get("files", {}).get(manifest_key), str(expected))
            path = dataset_root / expected["path"]
            size = path.stat().st_size if path.is_file() else None
            actual = sha256_file(path) if path.is_file() else None
            add_check(checks, f"sample_size:{key}", size == expected["size_bytes"], size, str(expected["size_bytes"]))
            add_check(checks, f"sample_hash:{key}", actual == expected["sha256"], actual, expected["sha256"])

    adapters = config["adapters"]
    verify_record(checks, "training_manifest", adapters["training_manifest"])
    training_manifest_path = Path(adapters["training_manifest"]["path"])
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8")) if training_manifest_path.is_file() else {}
    add_check(checks, "training_status", training_manifest.get("status") == "complete_requires_seen_phase_checkpoint_sweep", training_manifest.get("status"), "complete_requires_seen_phase_checkpoint_sweep")
    observed = {item.get("path"): item for item in training_manifest.get("checkpoints", [])}
    condition_names = [condition["name"] for condition in adapters["conditions"]]
    add_check(checks, "condition_order", condition_names == ["lora_off", "step_16", "step_32", "step_48", "step_64"], condition_names, "Frozen five-condition sweep")
    for condition in adapters["conditions"]:
        checkpoint = condition["checkpoint"]
        if checkpoint is None:
            continue
        verify_record(checks, f"checkpoint:{condition['name']}", checkpoint)
        basename = Path(checkpoint["path"]).name
        add_check(checks, f"training_manifest_checkpoint:{condition['name']}", observed.get(basename, {}).get("sha256") == checkpoint["sha256"], observed.get(basename), checkpoint["sha256"])
    verify_record(checks, "step64_validation", adapters["step64_validation"])
    validation_path = Path(adapters["step64_validation"]["path"])
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.is_file() else {}
    add_check(checks, "validation_status", validation.get("status") == "complete", validation.get("status"), "complete")
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
    for relative, expected in runtime["model_files"].items():
        path = Path(runtime["model_root"]) / relative
        size = path.stat().st_size if path.is_file() else None
        add_check(checks, f"model_size:{relative}", size == expected["size_bytes"], size, str(expected["size_bytes"]))
        add_check(checks, f"model_audited_hash:{relative}", expected["sha256"] in audit_text, expected["sha256"] in audit_text, "Hash occurs in frozen audit.")

    environment = os.environ.copy()
    environment.update(runtime["offline_environment"])
    environment.update({"PYTHONPATH": str(geoedit_root), "HF_HOME": runtime["cache_root"], "TRANSFORMERS_CACHE": runtime["cache_root"]})
    imported = subprocess.run([runtime["python"], "-c", "import cv2,numpy,PIL,torch,safetensors; from geoedit import inference"], env=environment, check=False, capture_output=True, text=True, timeout=60)
    add_check(checks, "runtime_import", imported.returncode == 0, {"return_code": imported.returncode, "stderr": imported.stderr[-2000:]}, "Frozen runtime imports offline.")

    gate = config["resource_gate"]
    available_mib = available_memory_mib()
    add_check(checks, "system_memory", available_mib >= gate["min_available_system_memory_mib"], available_mib, f">={gate['min_available_system_memory_mib']} MiB")
    gpus, processes = read_gpu_state()
    selected = next((gpu for gpu in gpus if gpu["index"] == args.gpu_index), None)
    gpu_ready = selected is not None and selected["name"] == gate["required_gpu_name"] and selected["memory_free_mib"] >= gate["min_free_memory_mib"] and selected["utilization_percent"] <= gate["max_utilization_percent"] and (not gate["require_no_compute_process"] or not selected["compute_processes"])
    add_check(checks, "gpu_gate", gpu_ready, {"requested": args.gpu_index, "selected": selected, "processes": processes}, "Requested conflict-free A6000 satisfies the frozen gate.")

    ready = all(check["passed"] for check in checks)
    report = {"schema_version": "foodstateedit.phase_action_checkpoint_sweep_preflight.v1", "ready": ready, "sample_id": args.sample_id, "selected_gpu": args.gpu_index if gpu_ready else None, "config_sha256": sha256_file(config_path), "checks": checks}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
