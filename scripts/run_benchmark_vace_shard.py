#!/usr/bin/env python3
"""Run one frozen pilot or held-out native/planar/relative3D VACE shard."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

try:
    from run_high_lift_vace_pilot import (
        project_outputs,
        resource_preflight,
        run_text,
        sha256,
        validate_file,
        write_json,
    )
except ModuleNotFoundError:  # package import used by the local test suite
    from scripts.run_high_lift_vace_pilot import (
        project_outputs,
        resource_preflight,
        run_text,
        sha256,
        validate_file,
        write_json,
    )


SCHEMA = "foodstateedit.benchmark_vace_execution.v1"
RUN_SCHEMA = "foodstateedit.benchmark_vace_shard_run.v1"


def foreign_process_check(gpu_uuid: str) -> str:
    raw = run_text([
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    foreign = []
    for line in raw.splitlines():
        parts = [value.strip() for value in line.split(",")]
        if len(parts) >= 4 and parts[0] == gpu_uuid and int(parts[1]) != os.getpid():
            foreign.append(line)
    if foreign:
        raise RuntimeError("Foreign compute process appeared; preserving own partial shard: " + repr(foreign))
    return raw


def condition_specs(config: dict, family: str) -> list[dict]:
    scale = float(config["material_scale_policy"][family])
    specs = [
        {"condition": "native_scale_1p0", "control_mode": "native", "vace_scale": 1.0},
        {"condition": "planar_scale_1p0", "control_mode": "planar", "vace_scale": 1.0},
        {"condition": "relative3d_scale_1p0", "control_mode": "relative3d", "vace_scale": 1.0},
    ]
    if scale != 1.0:
        specs.append({
            "condition": "relative3d_material_adaptive",
            "control_mode": "relative3d",
            "vace_scale": scale,
        })
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA or not config.get("execution_allowed"):
        raise ValueError("Unexpected or disabled benchmark config")
    if args.shard_id not in config["shards"]:
        raise ValueError(f"Unknown shard: {args.shard_id}")
    expected_output = Path(config["output_root_template"].format(shard_id=args.shard_id))
    expected_preflight = Path(config["preflight_template"].format(shard_id=args.shard_id))
    if args.output_root.resolve() != expected_output.resolve() or args.preflight_report.resolve() != expected_preflight.resolve():
        raise ValueError("Output or preflight path differs from frozen config")
    if sha256(Path(__file__)) != config["implementation"]["runner_sha256"]:
        raise ValueError("Runner hash mismatch")
    dependency = Path(__file__).with_name("run_high_lift_vace_pilot.py")
    if sha256(dependency) != config["implementation"]["dependency_sha256"]:
        raise ValueError("Projection/preflight dependency hash mismatch")

    dataset_root = Path(config["dataset_root"])
    validate_file(dataset_root / "dataset_manifest.json", config["dataset_manifest"])
    dataset = json.loads((dataset_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    if dataset.get("split") != config["dataset_split"] or dataset.get("case_count") != config["dataset_case_count"]:
        raise ValueError("Dataset split or case count differs from the frozen config")
    case_index = {case["case_id"]: case for case in dataset["cases"]}
    shard_cases = config["shards"][args.shard_id]
    if len(shard_cases) != len(set(shard_cases)) or any(case_id not in case_index for case_id in shard_cases):
        raise ValueError("Invalid shard membership")
    for case_id in shard_cases:
        for record in case_index[case_id]["files"].values():
            validate_file(dataset_root / record["path"], record)

    runtime = config["runtime"]
    validate_file(Path(runtime["model_hash_audit"]["path"]), runtime["model_hash_audit"])
    audit_text = Path(runtime["model_hash_audit"]["path"]).read_text(encoding="utf-8")
    for relative, record in runtime["model_files"].items():
        validate_file(Path(runtime["model_root"]) / relative, record, hash_file=False)
        if record["sha256"] not in audit_text or relative not in audit_text:
            raise ValueError(f"Model audit does not bind {relative}")
    for relative, expected_hash in runtime["geoedit_files"].items():
        if sha256(Path(runtime["geoedit_root"]) / relative) != expected_hash:
            raise ValueError(f"GeoEdit runtime mismatch: {relative}")

    preflight = resource_preflight(config, args.output_root, args.preflight_report, args.gpu_index)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    expected_jobs = [
        {"case_id": case_id, "seed": seed, **spec}
        for case_id in shard_cases
        for seed in config["inference"]["seeds"]
        for spec in condition_specs(config, case_index[case_id]["family"])
    ]
    manifest = {
        "schema_version": RUN_SCHEMA,
        "status": "loading_pipeline",
        "scientific_status": config["scientific_status"],
        "claim_limit": config["claim_limit"],
        "shard_id": args.shard_id,
        "case_ids": shard_cases,
        "host": socket.gethostname(),
        "physical_gpu": args.gpu_index,
        "gpu_uuid": preflight["gpu"]["uuid"],
        "started_at": started_at,
        "config_sha256": sha256(args.config),
        "runner_sha256": sha256(Path(__file__)),
        "preflight_sha256": sha256(args.preflight_report),
        "pipeline_load_count": 0,
        "expected_jobs": expected_jobs,
        "completed_jobs": [],
    }
    write_json(output_root / "run_manifest.json", manifest)
    (output_root / "RUNNING").write_text(started_at + "\n", encoding="utf-8")

    os.environ.update(runtime["offline_environment"])
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu_index),
        HF_HOME=config["cache_root_template"].format(shard_id=args.shard_id),
        TRANSFORMERS_CACHE=config["cache_root_template"].format(shard_id=args.shard_id),
        DIFFSYNTH_MODEL_BASE_PATH=str(Path(runtime["model_root"]).parent.parent),
    )
    sys.path.insert(0, runtime["geoedit_root"])

    try:
        foreign_process_check(preflight["gpu"]["uuid"])
        from geoedit import inference as geoedit_inference
        import torch

        with (
            (output_root / "pipeline_stdout.log").open("x") as stdout,
            (output_root / "pipeline_stderr.log").open("x") as stderr,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            pipe = geoedit_inference.load_pipeline(geoedit_inference.resolve_vram_limit(None))
        manifest["pipeline_load_count"] = 1
        inference = config["inference"]

        for job in expected_jobs:
            foreign_process_check(preflight["gpu"]["uuid"])
            case = case_index[job["case_id"]]
            case_root = output_root / job["case_id"]
            job_root = case_root / f"seed_{job['seed']}" / job["condition"]
            job_root.mkdir(parents=True, exist_ok=False)
            manifest.update(status="running", active_job=job)
            write_json(output_root / "run_manifest.json", manifest)
            reference_path = dataset_root / case["files"]["reference.png"]["path"]
            alpha_path = dataset_root / case["files"]["edit_alpha.png"]["path"]
            control_paths = {
                "native": reference_path,
                "planar": dataset_root / case["files"]["planar.mp4"]["path"],
                "relative3d": dataset_root / case["files"]["relative3d.mp4"]["path"],
            }
            reference = Image.open(reference_path).convert("RGB")
            width, height = reference.size
            begin = time.monotonic()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            with (
                (job_root / "stdout.log").open("x") as stdout,
                (job_root / "stderr.log").open("x") as stderr,
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                control_frames = geoedit_inference.load_video_or_image(
                    control_paths[job["control_mode"]], height, width, inference["num_frames"]
                )
                mask_frames = geoedit_inference.load_video_or_image(alpha_path, height, width, inference["num_frames"])
                video = pipe(
                    prompt=case["prompt"],
                    negative_prompt=case["negative_prompt"],
                    height=height,
                    width=width,
                    num_frames=inference["num_frames"],
                    num_inference_steps=inference["num_inference_steps"],
                    vace_video=control_frames,
                    vace_video_mask=mask_frames,
                    vace_reference_image=reference,
                    vace_scale=job["vace_scale"],
                    enable_ttm=False,
                    seed=job["seed"],
                    tiled=True,
                )
                geoedit_inference.save_video(video, str(job_root / "raw.mp4"), fps=inference["fps"], quality=5)
                del video, control_frames, mask_frames
            projected = project_outputs(
                job_root / "raw.mp4", reference_path, alpha_path, job_root,
                inference["num_frames"], inference["fps"], inference["review_frame_indices"],
            )
            foreign_process_check(preflight["gpu"]["uuid"])
            record = {
                **job,
                "family": case["family"],
                "status": "complete_requires_blind_review",
                "frames": inference["num_frames"],
                "steps": inference["num_inference_steps"],
                "lora_enabled": False,
                "ttm_enabled": False,
                "wall_time_seconds": round(time.monotonic() - begin, 3),
                "cuda_max_memory_allocated_mib": round(torch.cuda.max_memory_allocated() / 1048576, 2) if torch.cuda.is_available() else None,
                "cuda_max_memory_reserved_mib": round(torch.cuda.max_memory_reserved() / 1048576, 2) if torch.cuda.is_available() else None,
                "outside_support_max_pixel_difference": projected["outside_support_max_pixel_difference"],
                "files": {
                    path.name: {
                        "path": path.relative_to(output_root).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                    for path in job_root.iterdir() if path.is_file()
                },
            }
            write_json(job_root / "condition_manifest.json", record)
            manifest["completed_jobs"].append(record)
            write_json(output_root / "run_manifest.json", manifest)

        manifest["status"] = "complete_requires_blind_review"
        (output_root / "COMPLETE").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
        code = 0
    except Exception as exc:
        manifest.update(status="technical_failure_preserved", error=repr(exc))
        (output_root / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        (output_root / "FAILED").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
        code = 3

    manifest.pop("active_job", None)
    manifest.update(finished_at=datetime.now(timezone.utc).isoformat(), wall_time_seconds=round(time.monotonic() - started, 3))
    (output_root / "RUNNING").unlink(missing_ok=True)
    write_json(output_root / "run_manifest.json", manifest)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
