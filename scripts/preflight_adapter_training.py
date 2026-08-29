#!/usr/bin/env python3
"""Read-only, fail-closed preflight for the FoodStateEdit VACE-LoRA smoke run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def add_check(checks: list[dict[str, object]], check_id: str, passed: bool, actual: object, criterion: str) -> None:
    checks.append({"id": check_id, "passed": passed, "actual": actual, "criterion": criterion})


def resolve_dataset_file(dataset_root: Path, relative_path: str) -> Path:
    candidate = (dataset_root / relative_path).resolve()
    if candidate != dataset_root and dataset_root not in candidate.parents:
        raise ValueError(f"Dataset path escapes root: {relative_path}")
    return candidate


def read_gpu_state() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    gpus: list[dict[str, object]] = []
    uuid_to_index: dict[str, int] = {}
    for line in gpu_query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        record = {
            "index": int(fields[0]),
            "uuid": fields[1],
            "name": fields[2],
            "memory_total_mib": int(fields[3]),
            "memory_used_mib": int(fields[4]),
            "memory_free_mib": int(fields[5]),
            "utilization_percent": int(fields[6]),
            "compute_processes": [],
        }
        uuid_to_index[fields[1]] = record["index"]
        gpus.append(record)

    process_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    processes: list[dict[str, object]] = []
    for line in process_query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4 or fields[0] not in uuid_to_index:
            continue
        pid = int(fields[1])
        owner_result = subprocess.run(
            ["ps", "-o", "user=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        )
        record = {
            "gpu_index": uuid_to_index[fields[0]],
            "pid": pid,
            "owner": owner_result.stdout.strip() or "unknown",
            "process_name": fields[2],
            "used_memory_mib": int(fields[3]),
        }
        processes.append(record)
        gpus[record["gpu_index"]]["compute_processes"].append(record)
    return gpus, processes


def available_memory_mib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    checks: list[dict[str, object]] = []

    add_check(
        checks,
        "scientific_status",
        config.get("scientific_status") == "infrastructure_smoke_not_quality_evidence",
        config.get("scientific_status"),
        "The two-sample run must be claim-limited to infrastructure smoke evidence.",
    )
    add_check(checks, "output_absent", not args.output_root.exists(), str(args.output_root), "Output root must not exist.")

    trainer = config["trainer"]
    trainer_root = Path(trainer["remote_root"])
    train_script = trainer_root / trainer["train_script"]
    commit_marker = trainer_root / "SOURCE_COMMIT"
    archive_marker = trainer_root / "SOURCE_ARCHIVE_SHA256"
    add_check(checks, "trainer_script", train_script.is_file(), str(train_script), "Frozen training entry point exists.")
    add_check(
        checks,
        "trainer_script_sha256",
        train_script.is_file() and sha256_file(train_script) == trainer["train_script_sha256"],
        sha256_file(train_script) if train_script.is_file() else None,
        trainer["train_script_sha256"],
    )
    add_check(
        checks,
        "trainer_commit_marker",
        commit_marker.is_file() and commit_marker.read_text(encoding="utf-8").strip() == trainer["commit"],
        commit_marker.read_text(encoding="utf-8").strip() if commit_marker.is_file() else None,
        trainer["commit"],
    )
    add_check(
        checks,
        "trainer_archive_marker",
        archive_marker.is_file() and archive_marker.read_text(encoding="utf-8").strip() == trainer["source_archive_sha256"],
        archive_marker.read_text(encoding="utf-8").strip() if archive_marker.is_file() else None,
        trainer["source_archive_sha256"],
    )

    model = config["model"]
    model_root = Path(model["root"])
    audit_path = Path(model["hash_audit"])
    audit_hash = sha256_file(audit_path) if audit_path.is_file() else None
    add_check(checks, "model_audit", audit_hash == model["hash_audit_sha256"], audit_hash, model["hash_audit_sha256"])
    audit_text = audit_path.read_text(encoding="utf-8") if audit_path.is_file() else ""
    for record in model["files"]:
        path = model_root / record["path"]
        size = path.stat().st_size if path.is_file() else None
        add_check(checks, f"model_size:{record['path']}", size == record["size_bytes"], size, str(record["size_bytes"]))
        add_check(
            checks,
            f"model_recorded_hash:{record['path']}",
            record["sha256"] in audit_text,
            record["sha256"] in audit_text,
            "Expected full SHA-256 is present in the frozen audit.",
        )
    for record in model["tokenizer_files"]:
        path = model_root / record["path"]
        size = path.stat().st_size if path.is_file() else None
        actual_hash = sha256_file(path) if path.is_file() else None
        add_check(checks, f"tokenizer_size:{record['path']}", size == record["size_bytes"], size, str(record["size_bytes"]))
        add_check(checks, f"tokenizer_hash:{record['path']}", actual_hash == record["sha256"], actual_hash, record["sha256"])

    dataset_root = args.dataset_root.resolve()
    dataset_manifest_path = dataset_root / config["dataset"]["manifest"]
    metadata_path = dataset_root / config["dataset"]["metadata"]
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8")) if dataset_manifest_path.is_file() else {}
    dataset_manifest_hash = sha256_file(dataset_manifest_path) if dataset_manifest_path.is_file() else None
    add_check(
        checks,
        "dataset_manifest_hash",
        dataset_manifest_hash == config["dataset"]["manifest_sha256"],
        dataset_manifest_hash,
        config["dataset"]["manifest_sha256"],
    )
    provenance = dataset_manifest.get("provenance", {})
    dataset_builder = Path(__file__).resolve().with_name(str(provenance.get("builder", "")))
    builder_hash = sha256_text_lf(dataset_builder) if dataset_builder.is_file() else None
    add_check(
        checks,
        "dataset_builder_hash",
        provenance.get("builder_hash_policy") == "sha256_lf_normalized"
        and builder_hash == provenance.get("builder_sha256"),
        {"path": str(dataset_builder), "sha256_lf_normalized": builder_hash},
        str(provenance.get("builder_sha256")),
    )
    add_check(
        checks,
        "dataset_status",
        dataset_manifest.get("scientific_status") == "infrastructure_smoke_not_quality_evidence",
        dataset_manifest.get("scientific_status"),
        "infrastructure_smoke_not_quality_evidence",
    )
    add_check(
        checks,
        "dataset_sample_count",
        dataset_manifest.get("sample_count") == config["dataset"]["sample_count"],
        dataset_manifest.get("sample_count"),
        str(config["dataset"]["sample_count"]),
    )
    metadata_hash = sha256_file(metadata_path) if metadata_path.is_file() else None
    add_check(
        checks,
        "metadata_hash",
        metadata_hash == dataset_manifest.get("metadata", {}).get("sha256") == config["dataset"]["metadata_sha256"],
        metadata_hash,
        config["dataset"]["metadata_sha256"],
    )
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        add_check(checks, "metadata_rows", len(rows) == config["dataset"]["sample_count"], len(rows), str(config["dataset"]["sample_count"]))
        samples = dataset_manifest.get("samples", [])
        add_check(checks, "manifest_samples", len(samples) == len(rows), len(samples), f"Exactly {len(rows)} samples.")
        for row_index, row in enumerate(rows):
            sample = samples[row_index] if row_index < len(samples) else {}
            manifest_files = sample.get("files", {})
            for key in ("video", "vace_video", "vace_reference_image"):
                record = manifest_files.get(key, {})
                relative_path = row.get(key, "")
                add_check(
                    checks,
                    f"dataset_path:{row_index}:{key}",
                    relative_path == record.get("path"),
                    relative_path,
                    str(record.get("path")),
                )
                try:
                    path = resolve_dataset_file(dataset_root, relative_path)
                except ValueError as error:
                    add_check(checks, f"dataset_file:{row_index}:{key}", False, str(error), "Path remains inside dataset root.")
                    continue
                size = path.stat().st_size if path.is_file() else None
                actual_hash = sha256_file(path) if path.is_file() else None
                add_check(checks, f"dataset_size:{row_index}:{key}", size == record.get("size_bytes"), size, str(record.get("size_bytes")))
                add_check(checks, f"dataset_hash:{row_index}:{key}", actual_hash == record.get("sha256"), actual_hash, str(record.get("sha256")))
            video_hash = manifest_files.get("video", {}).get("sha256")
            control_hash = manifest_files.get("vace_video", {}).get("sha256")
            add_check(
                checks,
                f"identity_target:{row_index}",
                video_hash is not None and video_hash == control_hash,
                {"video": video_hash, "vace_video": control_hash},
                "Infrastructure-smoke target and control hashes are identical.",
            )

    module_status = {name: importlib.util.find_spec(name) is not None for name in config["required_modules"]}
    add_check(checks, "python_modules", all(module_status.values()), module_status, "All required modules importable.")
    help_result = None
    if train_script.is_file():
        help_env = os.environ.copy()
        help_env.update(config["offline_environment"])
        help_env["PYTHONPATH"] = str(trainer_root)
        help_result = subprocess.run(
            [trainer["python"], str(train_script), "--help"],
            check=False,
            capture_output=True,
            text=True,
            env=help_env,
            timeout=60,
        )
    add_check(
        checks,
        "trainer_help",
        help_result is not None and help_result.returncode == 0,
        help_result.stderr[-2000:] if help_result is not None else None,
        "Frozen trainer imports and --help exits 0 under offline environment.",
    )

    available_mib = available_memory_mib()
    add_check(
        checks,
        "system_memory",
        available_mib >= config["resource_gate"]["min_available_system_memory_mib"],
        available_mib,
        f">={config['resource_gate']['min_available_system_memory_mib']} MiB available",
    )
    gpus, processes = read_gpu_state()
    gate = config["resource_gate"]
    safe_gpus = [
        gpu for gpu in gpus
        if gpu["memory_free_mib"] >= gate["min_free_memory_mib"]
        and gpu["utilization_percent"] <= gate["max_utilization_percent"]
        and (not gate["require_no_compute_process"] or not gpu["compute_processes"])
    ]
    selected = None
    if args.gpu_index is not None:
        selected = next((gpu for gpu in safe_gpus if gpu["index"] == args.gpu_index), None)
    elif safe_gpus:
        selected = safe_gpus[0]
    add_check(
        checks,
        "gpu_gate",
        selected is not None,
        {"requested": args.gpu_index, "safe_gpu_indices": [gpu["index"] for gpu in safe_gpus], "gpus": gpus, "processes": processes},
        "At least one explicitly safe A6000 with no compute process.",
    )

    ready = all(check["passed"] for check in checks)
    report = {
        "schema_version": "foodstateedit.adapter_preflight.v0",
        "ready": ready,
        "selected_gpu": selected["index"] if selected is not None else None,
        "config_sha256": sha256_file(args.config.resolve()),
        "dataset_manifest_sha256": dataset_manifest_hash,
        "checks": checks,
    }
    rendered = json.dumps(report, indent=2)
    if args.report is not None:
        if args.report.exists():
            raise FileExistsError(f"Refusing to overwrite report: {args.report}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
