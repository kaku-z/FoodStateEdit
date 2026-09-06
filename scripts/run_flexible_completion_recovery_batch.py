#!/usr/bin/env python3
"""One-shot serial Day 13 recovery batch; no retries, scheduling, or expansion.

Adopt the explicitly identified first-arm launcher, then run the remaining
frozen arms, validate checkpoints, and run the frozen five-condition evaluation.
Stop on any failure. A successful batch still requires independent human review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ARMS = ("planar_uniform", "relative3d_uniform", "relative3d_topology_weighted")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, data):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def verify_run(config_path, config, arm):
    root = Path(config["arms"][arm]["output_root"])
    manifest = read(root / "run_manifest.json")
    if (manifest.get("status") != "complete_requires_cross_arm_match_and_checkpoint_validation"
            or manifest.get("config_sha256") != sha(config_path)
            or manifest.get("return_code") != 0
            or manifest.get("randomness_trace_count") != 32):
        raise ValueError(f"Incomplete or mismatched training arm: {arm}")
    if {r["path"] for r in manifest.get("checkpoints", [])} != {"step-16.safetensors", "step-32.safetensors"}:
        raise ValueError(f"Missing or extra checkpoints: {arm}")
    for record in manifest["checkpoints"]:
        path = root / record["path"]
        if path.stat().st_size != record["size_bytes"] or sha(path) != record["sha256"]:
            raise ValueError(f"Checkpoint differs from manifest: {path}")
    return root


def snapshot_arm(root, destination, arm):
    records = [{"path": p.relative_to(root).as_posix(), "size_bytes": p.stat().st_size,
                "sha256": sha(p)} for p in sorted(root.rglob("*")) if p.is_file()]
    archive = destination / f"{arm}.tar"
    with tarfile.open(archive, "x") as handle:
        for record in records:
            handle.add(root / record["path"], arcname=f"{arm}/{record['path']}", recursive=False)
    write_new(destination / f"{arm}_snapshot.json", {"files": records,
              "archive": {"path": str(archive), "size_bytes": archive.stat().st_size, "sha256": sha(archive)}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--first-arm-pid", type=int, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--posttraining-sha256", required=True)
    args = parser.parse_args()
    config_path = args.execution_config.resolve()
    config = read(config_path)
    repo = config_path.parents[1]
    if (config.get("schema_version") != "foodstateedit.flexible_completion_execution.v1"
            or config.get("execution_allowed") is not True
            or config.get("blind_fork_evaluation_allowed") is not False
            or config["resource_gate"]["execution_mode"] != "serial_only"):
        raise ValueError("Not the authorized serial seen-only protocol")
    for name in ("training_runner", "training_preflight", "evaluation_runner", "evaluation_preflight"):
        record = config["implementation"][name]
        if sha(repo / record["path"]) != record["sha256"]:
            raise ValueError(f"Frozen implementation mismatch: {name}")
    preparer = repo / "scripts/prepare_flexible_completion_posttraining.py"
    if sha(preparer) != args.posttraining_sha256:
        raise ValueError("Posttraining preparation script mismatch")
    batch = args.batch_root.resolve()
    batch.mkdir(parents=True, exist_ok=False)
    snapshots = batch / "snapshots"
    snapshots.mkdir()
    write_new(batch / "command.json", {"argv": os.sys.argv, "config_sha256": sha(config_path),
              "batch_script_sha256": sha(__file__), "posttraining_sha256": sha(preparer),
              "started_at": datetime.now(timezone.utc).isoformat(), "retries_allowed": False})
    try:
        first_root = Path(config["arms"][ARMS[0]]["output_root"])
        deadline = time.monotonic() + 3600
        while not (first_root / "run_manifest.json").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("First-arm adoption wait exceeded one hour; no new job launched")
            process_path = Path(f"/proc/{args.first_arm_pid}")
            if not process_path.exists() or process_path.stat().st_uid != os.getuid():
                raise RuntimeError("Adopted launcher no longer exists or has another owner")
            command = (process_path / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            if str(config_path) not in command or "run_flexible_completion_training.py" not in command or "--arm planar_uniform" not in command:
                raise RuntimeError("Adopted PID is not the expected frozen first-arm launcher")
            time.sleep(15)
        snapshot_arm(verify_run(config_path, config, ARMS[0]), snapshots, ARMS[0])
        print("planar_uniform complete and snapshotted", flush=True)
        python = config["python"]
        for arm in ARMS[1:]:
            command = [python, "-u", str(repo / config["implementation"]["training_runner"]["path"]),
                       "--config", str(config_path), "--dataset-root", config["dataset"]["remote_root"],
                       "--arm", arm, "--output-root", config["arms"][arm]["output_root"],
                       "--gpu-index", str(args.gpu_index), "--preflight-report", str(batch / f"{arm}_preflight.json")]
            with (batch / f"{arm}_launcher.log").open("x", encoding="utf-8") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            snapshot_arm(verify_run(config_path, config, arm), snapshots, arm)
            print(f"{arm} complete and snapshotted", flush=True)
        post = batch / "posttraining"
        with (batch / "posttraining.log").open("x", encoding="utf-8") as log:
            subprocess.run([python, "-u", str(preparer), "--execution-config", str(config_path),
                            "--runtime-template-config", str(repo / "configs/vace_phase_action_isolated_udon_checkpoint_sweep_v1.json"),
                            "--output-root", str(post)], stdout=log, stderr=subprocess.STDOUT, check=True)
        print("All checkpoints validated; evaluation frozen", flush=True)
        with (batch / "evaluation_launcher.log").open("x", encoding="utf-8") as log:
            subprocess.run([python, "-u", str(repo / config["implementation"]["evaluation_runner"]["path"]),
                            "--config", str(post / "evaluation_config.json"),
                            "--dataset-root", config["dataset"]["remote_root"],
                            "--output-root", config["evaluation"]["output_root"],
                            "--gpu-index", str(args.gpu_index), "--preflight-report", str(batch / "evaluation_preflight.json")],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        write_new(batch / "COMPLETE.json", {"status": "evaluation_complete_requires_independent_review",
                  "finished_at": datetime.now(timezone.utc).isoformat()})
        print("Evaluation complete. Effectiveness is not claimed; independent review required.", flush=True)
    except Exception:
        write_new(batch / "FAILED.json", {"status": "technical_failure_preserved_no_retry",
                  "traceback": traceback.format_exc(), "finished_at": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    main()
