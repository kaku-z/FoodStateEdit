#!/usr/bin/env python3
"""Run one fail-closed, same-seed VACE control-scale sweep for one food case."""
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


SCHEMA = "foodstateedit.realism_scale_sweep.v1"
RUN_SCHEMA = "foodstateedit.realism_scale_sweep_run.v1"
CASES = ("soup", "rice", "cake")
SCALES = (0.6, 0.8)


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


def condition_name(scale: float) -> str:
    return f"vace_scale_{str(scale).replace('.', 'p')}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case-id", choices=CASES, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA or not config.get("execution_allowed"):
        raise ValueError("Unexpected or disabled realism-scale config")
    if tuple(config["cases"]) != CASES:
        raise ValueError("The frozen case set changed")
    if tuple(config["vace_scales"]) != SCALES:
        raise ValueError("The frozen scale sweep changed")
    expected_output = Path(config["output_root_template"].format(case_id=args.case_id))
    if args.output_root.resolve() != expected_output.resolve():
        raise ValueError("Output root differs from the frozen case path")
    if sha256(Path(__file__)) != config["implementation"]["runner_sha256"]:
        raise ValueError("Runner hash mismatch")
    dependency = Path(__file__).with_name("run_high_lift_vace_pilot.py")
    if sha256(dependency) != config["implementation"]["dependency_sha256"]:
        raise ValueError("Projection/preflight dependency hash mismatch")

    dataset_root = Path(config["dataset_root"])
    validate_file(dataset_root / "dataset_manifest.json", config["dataset_manifest"])
    dataset = json.loads((dataset_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    cases = {item["case_id"]: item for item in dataset["cases"]}
    case = cases[args.case_id]
    for record in case["files"].values():
        validate_file(dataset_root / record["path"], record)

    inference = config["inference"]
    expected_inference = {
        "seed": 1,
        "num_frames": 21,
        "num_inference_steps": 20,
        "fps": 8,
        "enable_ttm": False,
        "lora_enabled": False,
    }
    if inference != expected_inference:
        raise ValueError("Inference settings differ from the frozen matched baseline")

    runtime = config["runtime"]
    audit_path = Path(runtime["model_hash_audit"]["path"])
    validate_file(audit_path, runtime["model_hash_audit"])
    audit_text = audit_path.read_text(encoding="utf-8")
    for relative, record in runtime["model_files"].items():
        validate_file(Path(runtime["model_root"]) / relative, record, hash_file=False)
        if record["sha256"] not in audit_text or relative not in audit_text:
            raise ValueError(f"Model audit does not bind {relative}")
    for relative, expected_hash in runtime["geoedit_files"].items():
        if sha256(Path(runtime["geoedit_root"]) / relative) != expected_hash:
            raise ValueError(f"GeoEdit runtime mismatch: {relative}")

    preflight = resource_preflight(
        config, args.output_root, args.preflight_report, args.gpu_index
    )
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
        "host": socket.gethostname(),
        "physical_gpu": args.gpu_index,
        "gpu_uuid": preflight["gpu"]["uuid"],
        "started_at": started_at,
        "config_sha256": sha256(args.config),
        "runner_sha256": sha256(Path(__file__)),
        "preflight_sha256": sha256(args.preflight_report),
        "pipeline_load_count": 0,
        "expected_conditions": [condition_name(scale) for scale in SCALES],
        "completed_conditions": [],
    }
    write_json(output_root / "run_manifest.json", manifest)
    (output_root / "RUNNING").write_text(started_at + "\n", encoding="utf-8")

    os.environ.update(runtime["offline_environment"])
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu_index),
        HF_HOME=config["cache_root"],
        TRANSFORMERS_CACHE=config["cache_root"],
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
            pipe = geoedit_inference.load_pipeline(
                geoedit_inference.resolve_vram_limit(None)
            )
        manifest["pipeline_load_count"] = 1

        reference_path = dataset_root / case["files"]["reference.png"]["path"]
        alpha_path = dataset_root / case["files"]["edit_alpha.png"]["path"]
        control_path = dataset_root / case["files"]["relative3d.mp4"]["path"]
        reference = Image.open(reference_path).convert("RGB")
        width, height = reference.size

        for scale in SCALES:
            foreign_process_check(preflight["gpu"]["uuid"])
            name = condition_name(scale)
            condition_root = output_root / name
            condition_root.mkdir(exist_ok=False)
            manifest.update(status="running", active_condition=name)
            write_json(output_root / "run_manifest.json", manifest)
            begin = time.monotonic()
            with (
                (condition_root / "stdout.log").open("x") as stdout,
                (condition_root / "stderr.log").open("x") as stderr,
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                control_frames = geoedit_inference.load_video_or_image(
                    control_path, height, width, 21
                )
                mask_frames = geoedit_inference.load_video_or_image(
                    alpha_path, height, width, 21
                )
                video = pipe(
                    prompt=case["prompt"],
                    negative_prompt=case["negative_prompt"],
                    height=height,
                    width=width,
                    num_frames=21,
                    num_inference_steps=20,
                    vace_video=control_frames,
                    vace_video_mask=mask_frames,
                    vace_reference_image=reference,
                    vace_scale=scale,
                    enable_ttm=False,
                    seed=1,
                    tiled=True,
                )
                geoedit_inference.save_video(
                    video, str(condition_root / "raw.mp4"), fps=8, quality=5
                )
                del video, control_frames, mask_frames
            projected = project_outputs(
                condition_root / "raw.mp4",
                reference_path,
                alpha_path,
                condition_root,
                21,
                8,
                [0, 3, 6, 10, 15, 20],
            )
            foreign_process_check(preflight["gpu"]["uuid"])
            record = {
                "condition": name,
                "vace_scale": scale,
                "status": "complete_requires_action_and_photo_review",
                "seed": 1,
                "frames": 21,
                "steps": 20,
                "lora_enabled": False,
                "ttm_enabled": False,
                "wall_time_seconds": round(time.monotonic() - begin, 3),
                "outside_support_max_pixel_difference": projected[
                    "outside_support_max_pixel_difference"
                ],
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

        manifest["status"] = "complete_requires_matched_baseline_review"
        (output_root / "COMPLETE").write_text(
            datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
        )
        code = 0
    except Exception as exc:
        manifest.update(status="technical_failure_preserved", error=repr(exc))
        (output_root / "failure.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        (output_root / "FAILED").write_text(
            datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
        )
        code = 3

    manifest.pop("active_condition", None)
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
