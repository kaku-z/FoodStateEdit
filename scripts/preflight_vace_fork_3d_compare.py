#!/usr/bin/env python3
"""Read-only, fail-closed preflight for the frozen fork 3-D VACE comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


METHOD = "vace_fork_relative_3d_projection_compare_v0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_check(
    checks: list[dict[str, object]],
    check_id: str,
    passed: bool,
    actual: object,
    criterion: str,
) -> None:
    checks.append({
        "id": check_id,
        "passed": bool(passed),
        "actual": actual,
        "criterion": criterion,
    })


def available_memory_mib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def read_gpu_state() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    query = subprocess.run(
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
    uuid_to_record: dict[str, dict[str, object]] = {}
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        record: dict[str, object] = {
            "index": int(fields[0]),
            "uuid": fields[1],
            "name": fields[2],
            "memory_total_mib": int(fields[3]),
            "memory_used_mib": int(fields[4]),
            "memory_free_mib": int(fields[5]),
            "utilization_percent": int(fields[6]),
            "compute_processes": [],
        }
        gpus.append(record)
        uuid_to_record[fields[1]] = record

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
        if len(fields) != 4 or fields[0] not in uuid_to_record:
            continue
        owner = subprocess.run(
            ["ps", "-o", "user=", "-p", fields[1]],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip() or "unknown"
        used_memory = None if fields[3] == "N/A" else int(fields[3])
        process = {
            "gpu_index": uuid_to_record[fields[0]]["index"],
            "pid": int(fields[1]),
            "owner": owner,
            "process_name": fields[2],
            "used_memory_mib": used_memory,
        }
        processes.append(process)
        uuid_to_record[fields[0]]["compute_processes"].append(process)
    return gpus, processes


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
        raise FileExistsError(f"Refusing to overwrite preflight report: {report_path}")

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    checks: list[dict[str, object]] = []

    add_check(checks, "method", config.get("method") == METHOD, config.get("method"), METHOD)
    add_check(
        checks,
        "scientific_status",
        config.get("scientific_status") == "single_anchor_same_seed_render_comparison_not_generalization",
        config.get("scientific_status"),
        "The run remains a one-anchor representation comparison.",
    )
    add_check(checks, "input_root", str(input_root) == config["control"]["remote_root"], str(input_root), config["control"]["remote_root"])
    add_check(checks, "output_root", str(output_root) == config["output_root"], str(output_root), config["output_root"])
    add_check(checks, "output_absent", not output_root.exists(), str(output_root), "Output root must not exist.")

    for name, expected in config["control"]["files"].items():
        path = input_root / name
        size = path.stat().st_size if path.is_file() else None
        actual_hash = sha256_file(path) if path.is_file() else None
        add_check(checks, f"control_size:{name}", size == expected["size_bytes"], size, str(expected["size_bytes"]))
        add_check(checks, f"control_hash:{name}", actual_hash == expected["sha256"], actual_hash, expected["sha256"])

    control_manifest_path = input_root / "run_manifest.json"
    control_manifest = json.loads(control_manifest_path.read_text(encoding="utf-8")) if control_manifest_path.is_file() else {}
    add_check(checks, "control_anchor", control_manifest.get("anchor_id") == config["control"]["anchor_id"], control_manifest.get("anchor_id"), config["control"]["anchor_id"])
    add_check(checks, "geometry_source", control_manifest.get("geometry_source") == config["control"]["geometry_source"], control_manifest.get("geometry_source"), config["control"]["geometry_source"])
    add_check(checks, "control_frames", control_manifest.get("frame_count") == config["inference"]["num_frames"], control_manifest.get("frame_count"), str(config["inference"]["num_frames"]))
    add_check(checks, "selected_frame", control_manifest.get("selected_frame_index") == config["inference"]["selected_frame_index"], control_manifest.get("selected_frame_index"), str(config["inference"]["selected_frame_index"]))
    add_check(checks, "depth_crossing", control_manifest.get("selected_depth_crosses_fork") is True, control_manifest.get("selected_depth_crosses_fork"), "The selected 3-D helix crosses the fork depth.")

    runtime = config["runtime"]
    geoedit_root = Path(runtime["geoedit_root"])
    for relative, expected_hash in runtime["geoedit_files"].items():
        path = geoedit_root / relative
        actual_hash = sha256_file(path) if path.is_file() else None
        add_check(checks, f"runtime_hash:{relative}", actual_hash == expected_hash, actual_hash, expected_hash)

    audit_path = Path(runtime["model_hash_audit"])
    audit_hash = sha256_file(audit_path) if audit_path.is_file() else None
    add_check(checks, "model_audit_hash", audit_hash == runtime["model_hash_audit_sha256"], audit_hash, runtime["model_hash_audit_sha256"])
    audit_text = audit_path.read_text(encoding="utf-8") if audit_path.is_file() else ""
    model_root = Path(runtime["model_root"])
    for relative, expected in runtime["model_files"].items():
        path = model_root / relative
        size = path.stat().st_size if path.is_file() else None
        add_check(checks, f"model_size:{relative}", size == expected["size_bytes"], size, str(expected["size_bytes"]))
        add_check(checks, f"model_audited_hash:{relative}", expected["sha256"] in audit_text, expected["sha256"] in audit_text, "Expected SHA-256 is present in the frozen full-file audit.")

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
    add_check(checks, "runtime_import", import_result.returncode == 0, {"return_code": import_result.returncode, "stdout": import_result.stdout[-1000:], "stderr": import_result.stderr[-2000:]}, "Frozen offline runtime imports without loading a model.")

    available_mib = available_memory_mib()
    gate = config["resource_gate"]
    add_check(checks, "system_memory", available_mib >= gate["min_available_system_memory_mib"], available_mib, f">={gate['min_available_system_memory_mib']} MiB available")
    gpus, processes = read_gpu_state()
    safe_gpus = [
        gpu for gpu in gpus
        if gpu["name"] == gate["required_gpu_name"]
        and gpu["memory_free_mib"] >= gate["min_free_memory_mib"]
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
        "One A6000 has >=48,000 MiB free, <=5% utilization, and no compute process.",
    )

    ready = all(check["passed"] for check in checks)
    report = {
        "schema_version": "foodstateedit.vace_fork_3d_preflight.v0",
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
