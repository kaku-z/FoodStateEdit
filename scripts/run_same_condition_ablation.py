#!/usr/bin/env python3
"""Run the missing Day 25 same-condition development ablation jobs for one case."""
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

from run_high_lift_vace_pilot import (
    project_outputs,
    resource_preflight,
    run_text,
    sha256,
    validate_file,
    write_json,
)


SCHEMA = "foodstateedit.same_condition_ablation.v1"
RUN_SCHEMA = "foodstateedit.same_condition_ablation_run.v1"
CASE_ORDER = ("ramen", "soup", "rice", "cake")
CONDITION_SPECS = {
    "native_scale_1p0": ("native", 1.0),
    "planar_scale_1p0": ("planar", 1.0),
    "relative3d_scale_1p0": ("relative3d", 1.0),
    "relative3d_scale_0p6": ("relative3d", 0.6),
    "relative3d_scale_0p8": ("relative3d", 0.8),
}


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
        raise RuntimeError(
            "Foreign compute process appeared; stop own worker without preemption: "
            + repr(foreign)
        )
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case-id", choices=CASE_ORDER, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA or not config.get("execution_allowed"):
        raise ValueError("Unexpected or disabled ablation config")
    if tuple(config["case_order"]) != CASE_ORDER:
        raise ValueError("The frozen case order changed")
    if set(config["unique_generation_conditions"]) != set(CONDITION_SPECS):
        raise ValueError("The frozen unique condition set changed")
    expected_output = Path(config["output_root_template"].format(case_id=args.case_id))
    expected_preflight = Path(config["preflight_template"].format(case_id=args.case_id))
    if args.output_root.resolve() != expected_output.resolve():
        raise ValueError("Output root differs from the frozen case path")
    if args.preflight_report.resolve() != expected_preflight.resolve():
        raise ValueError("Preflight path differs from the frozen case path")
    if sha256(Path(__file__)) != config["implementation"]["runner_sha256"]:
        raise ValueError("Runner hash mismatch")
    dependency = Path(__file__).with_name("run_high_lift_vace_pilot.py")
    if sha256(dependency) != config["implementation"]["dependency_sha256"]:
        raise ValueError("Projection/preflight dependency hash mismatch")

    dataset_root = Path(config["dataset_root"])
    validate_file(dataset_root / "dataset_manifest.json", config["dataset_manifest"])
    dataset = json.loads((dataset_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    aliases = config.get("dataset_case_aliases", {})
    if aliases != {"ramen": "noodle"}:
        raise ValueError("The frozen dataset-case alias changed")
    dataset_case_id = aliases.get(args.case_id, args.case_id)
    case = next(item for item in dataset["cases"] if item["case_id"] == dataset_case_id)
    for record in case["files"].values():
        validate_file(dataset_root / record["path"], record)

    frozen_jobs = config["new_generation_jobs"][args.case_id]
    jobs = [(int(job["seed"]), job["condition"]) for job in frozen_jobs]
    if len(jobs) != len(set(jobs)) or any(condition not in CONDITION_SPECS for _, condition in jobs):
        raise ValueError("Invalid or duplicate frozen jobs")

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
    manifest = {
        "schema_version": RUN_SCHEMA,
        "status": "loading_pipeline",
        "scientific_status": config["scientific_status"],
        "claim_limit": config["claim_limit"],
        "case_id": args.case_id,
        "dataset_case_id": dataset_case_id,
        "host": socket.gethostname(),
        "physical_gpu": args.gpu_index,
        "gpu_uuid": preflight["gpu"]["uuid"],
        "started_at": started_at,
        "config_sha256": sha256(args.config),
        "runner_sha256": sha256(Path(__file__)),
        "preflight_sha256": sha256(args.preflight_report),
        "pipeline_load_count": 0,
        "expected_jobs": [{"seed": seed, "condition": condition} for seed, condition in jobs],
        "completed_conditions": [],
    }
    write_json(output_root / "run_manifest.json", manifest)
    (output_root / "RUNNING").write_text(started_at + "\n", encoding="utf-8")

    os.environ.update(runtime["offline_environment"])
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu_index),
        HF_HOME=config["cache_root_template"].format(case_id=args.case_id),
        TRANSFORMERS_CACHE=config["cache_root_template"].format(case_id=args.case_id),
        DIFFSYNTH_MODEL_BASE_PATH=str(Path(runtime["model_root"]).parent.parent),
    )
    sys.path.insert(0, runtime["geoedit_root"])

    try:
        foreign_process_check(preflight["gpu"]["uuid"])
        from geoedit import inference as geoedit_inference

        with (
            (output_root / "pipeline_stdout.log").open("x") as stdout,
            (output_root / "pipeline_stderr.log").open("x") as stderr,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            pipe = geoedit_inference.load_pipeline(geoedit_inference.resolve_vram_limit(None))
        manifest["pipeline_load_count"] = 1

        reference_path = dataset_root / case["files"]["reference.png"]["path"]
        alpha_path = dataset_root / case["files"]["edit_alpha.png"]["path"]
        control_paths = {
            "native": reference_path,
            "planar": dataset_root / case["files"]["planar.mp4"]["path"],
            "relative3d": dataset_root / case["files"]["relative3d.mp4"]["path"],
        }
        reference = Image.open(reference_path).convert("RGB")
        width, height = reference.size
        inference = config["inference"]

        for seed, condition in jobs:
            foreign_process_check(preflight["gpu"]["uuid"])
            control_mode, scale = CONDITION_SPECS[condition]
            condition_root = output_root / f"seed_{seed}" / condition
            condition_root.mkdir(parents=True, exist_ok=False)
            manifest.update(status="running", active_job={"seed": seed, "condition": condition})
            write_json(output_root / "run_manifest.json", manifest)
            begin = time.monotonic()
            with (
                (condition_root / "stdout.log").open("x") as stdout,
                (condition_root / "stderr.log").open("x") as stderr,
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                control_frames = geoedit_inference.load_video_or_image(
                    control_paths[control_mode], height, width, inference["num_frames"]
                )
                mask_frames = geoedit_inference.load_video_or_image(
                    alpha_path, height, width, inference["num_frames"]
                )
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
                    vace_scale=scale,
                    enable_ttm=False,
                    seed=seed,
                    tiled=True,
                )
                geoedit_inference.save_video(video, str(condition_root / "raw.mp4"), fps=inference["fps"], quality=5)
                del video, control_frames, mask_frames
            projected = project_outputs(
                condition_root / "raw.mp4",
                reference_path,
                alpha_path,
                condition_root,
                inference["num_frames"],
                inference["fps"],
                inference["review_frame_indices"],
            )
            foreign_process_check(preflight["gpu"]["uuid"])
            record = {
                "condition": condition,
                "control_mode": control_mode,
                "vace_scale": scale,
                "status": "complete_requires_frozen_ablation_mapping_and_review",
                "seed": seed,
                "frames": inference["num_frames"],
                "steps": inference["num_inference_steps"],
                "lora_enabled": False,
                "ttm_enabled": False,
                "wall_time_seconds": round(time.monotonic() - begin, 3),
                "outside_support_max_pixel_difference": projected["outside_support_max_pixel_difference"],
                "files": {
                    path.name: {
                        "path": path.relative_to(output_root).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                    for path in condition_root.iterdir()
                    if path.is_file()
                },
            }
            write_json(condition_root / "condition_manifest.json", record)
            manifest["completed_conditions"].append(record)
            write_json(output_root / "run_manifest.json", manifest)

        manifest["status"] = "complete_requires_ablation_mapping_and_review"
        (output_root / "COMPLETE").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
        code = 0
    except Exception as exc:
        manifest.update(status="technical_failure_preserved", error=repr(exc))
        (output_root / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        (output_root / "FAILED").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
        code = 3

    manifest.pop("active_job", None)
    manifest.update(
        finished_at=datetime.now(timezone.utc).isoformat(),
        wall_time_seconds=round(time.monotonic() - started, 3),
    )
    (output_root / "RUNNING").unlink(missing_ok=True)
    write_json(output_root / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
