#!/usr/bin/env python3
"""Run one frozen LoRA-off VACE pilot for the Day 17 high-lift control."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


METHOD = "relative3d_high_lift_udon_vace_lora_off_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_file(path: Path, record: dict, *, hash_file: bool = True) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_size = record.get("size_bytes", record.get("bytes"))
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ValueError(f"Size mismatch: {path}")
    if hash_file and sha256(path) != record["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")


def run_text(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip()


def resource_preflight(config: dict, output_root: Path, report_path: Path, gpu_index: int) -> dict:
    query = run_text([
        "nvidia-smi", "-i", str(gpu_index),
        "--query-gpu=uuid,name,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    fields = [item.strip() for item in query.split(",")]
    if len(fields) != 5:
        raise RuntimeError(f"Unexpected nvidia-smi output: {query}")
    uuid, name, total, free, utilization = fields
    applications = run_text([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    processes = []
    for line in applications.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) >= 4 and parts[0] == uuid:
            owner = subprocess.run(
                ["ps", "-o", "user=,pid=,cmd=", "-p", parts[1]], check=False,
                text=True, capture_output=True,
            ).stdout.strip()
            processes.append({"pid": parts[1], "process": parts[2],
                              "used_memory_mib": parts[3], "owner_record": owner})
    available_kib = 0
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            available_kib = int(line.split()[1])
            break
    gate = config["resource_gate"]
    checks = {
        "output_root_absent": not output_root.exists(),
        "preflight_report_absent": not report_path.exists(),
        "gpu_name": name == gate["required_gpu_name"],
        "gpu_free_memory": int(free) >= gate["min_free_memory_mib"],
        "gpu_utilization": int(utilization) <= gate["max_utilization_percent"],
        "zero_compute_processes": len(processes) == 0,
        "host_available_memory": available_kib // 1024 >= gate["min_available_system_memory_mib"],
    }
    report = {
        "schema_version": "foodstateedit.high_lift_vace_preflight.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": {"physical_index": gpu_index, "uuid": uuid, "name": name,
                "total_memory_mib": int(total), "free_memory_mib": int(free),
                "utilization_percent": int(utilization), "compute_processes": processes},
        "host_available_memory_mib": available_kib // 1024,
        "checks": checks,
        "passed": all(checks.values()),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if not report["passed"]:
        raise RuntimeError(f"Resource preflight failed: {checks}")
    return report


def decode_video(path: Path, expected_frames: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) != expected_frames:
        raise RuntimeError(f"{path}: decoded {len(frames)}, expected {expected_frames}")
    return frames


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise OSError(f"Could not create {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def make_sheet(frames: list[np.ndarray], indices: list[int]) -> np.ndarray:
    height, width = frames[0].shape[:2]
    sheet = np.zeros((height * 2, width * 3, 3), dtype=np.uint8)
    for slot, frame_index in enumerate(indices):
        row, column = divmod(slot, 3)
        sheet[row * height:(row + 1) * height, column * width:(column + 1) * width] = frames[frame_index]
    return sheet


def project_outputs(raw_video: Path, reference_path: Path, alpha_path: Path,
                    output_root: Path, expected_frames: int, fps: int,
                    review_indices: list[int]) -> dict:
    raw_frames = decode_video(raw_video, expected_frames)
    reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    alpha = cv2.imread(str(alpha_path), cv2.IMREAD_GRAYSCALE)
    if reference is None or alpha is None:
        raise RuntimeError("Could not read reference or edit alpha")
    height, width = reference.shape[:2]
    weight = alpha.astype(np.float32)[..., None] / 255.0
    protected = alpha == 0
    projected = []
    outside_max = 0
    for raw in raw_frames:
        if raw.shape[:2] != (height, width):
            raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LANCZOS4)
        frame = np.rint(raw.astype(np.float32) * weight + reference.astype(np.float32) * (1.0 - weight))
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        difference = np.abs(frame.astype(np.int16) - reference.astype(np.int16))
        outside_max = max(outside_max, int(difference[protected].max(initial=0)))
        projected.append(frame)
    if outside_max != 0:
        raise AssertionError("Projection changed protected pixels")
    projected_video = output_root / "projected.mp4"
    review = output_root / "projected_review.png"
    final_hold = output_root / "projected_final_hold.png"
    write_video(projected_video, projected, fps)
    if not cv2.imwrite(str(review), make_sheet(projected, review_indices)):
        raise OSError("Could not write review sheet")
    if not cv2.imwrite(str(final_hold), projected[review_indices[-1]]):
        raise OSError("Could not write final hold")
    return {"decoded_frames": len(projected), "outside_support_max_pixel_difference": outside_max,
            "projected_video": projected_video, "review": review, "final_hold": final_hold}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("method") != METHOD or not config.get("execution_allowed"):
        raise ValueError("Unexpected or disabled high-lift config")
    if str(args.dataset_root.resolve()) != config["dataset"]["remote_root"]:
        raise ValueError("Dataset root differs from frozen config")
    if str(args.output_root.resolve()) != config["output_root"]:
        raise ValueError("Output root differs from frozen config")
    if sha256(Path(__file__)) != config["implementation"]["runner_sha256"]:
        raise ValueError("Runner hash mismatch")
    preflight = resource_preflight(config, args.output_root, args.preflight_report, args.gpu_index)

    dataset_root = args.dataset_root.resolve()
    validate_file(dataset_root / "dataset_manifest.json", config["dataset"]["manifest"])
    for record in config["dataset"]["files"].values():
        validate_file(dataset_root / record["path"], record)
    runtime = config["runtime"]
    audit_path = Path(runtime["model_hash_audit"]["path"])
    validate_file(audit_path, runtime["model_hash_audit"])
    audit_text = audit_path.read_text(encoding="utf-8")
    for relative, record in runtime["model_files"].items():
        model_path = Path(runtime["model_root"]) / relative
        validate_file(model_path, record, hash_file=False)
        if record["sha256"] not in audit_text or relative not in audit_text:
            raise ValueError(f"Model audit does not bind {relative}")
    for relative, expected_hash in runtime["geoedit_files"].items():
        if sha256(Path(runtime["geoedit_root"]) / relative) != expected_hash:
            raise ValueError(f"GeoEdit runtime mismatch: {relative}")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    started_at = datetime.now(timezone.utc)
    manifest = {
        "schema_version": "foodstateedit.high_lift_vace_run.v1",
        "method": METHOD,
        "status": "running",
        "scientific_status": config["scientific_status"],
        "claim_limit": config["claim_limit"],
        "host": socket.gethostname(),
        "physical_gpu": args.gpu_index,
        "started_at": started_at.isoformat(),
        "config_sha256": sha256(args.config),
        "runner_sha256": sha256(Path(__file__)),
        "preflight_sha256": sha256(args.preflight_report),
        "pipeline_load_count": 0,
        "new_vace_inference_count": 1,
        "lora_enabled": False,
    }
    write_json(output_root / "run_manifest.json", manifest)
    (output_root / "RUNNING").write_text(started_at.isoformat() + "\n", encoding="utf-8")
    inference_config = config["inference"]
    environment = os.environ.copy()
    environment.update(runtime["offline_environment"])
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu_index),
        "DIFFSYNTH_MODEL_BASE_PATH": str(Path(runtime["model_root"]).parent.parent),
        "HF_HOME": runtime["cache_root"],
        "TRANSFORMERS_CACHE": runtime["cache_root"],
        "PYTHONPATH": runtime["geoedit_root"],
    })
    os.environ.update(environment)
    Path(runtime["cache_root"]).mkdir(parents=True, exist_ok=True)
    try:
        geoedit_root = Path(runtime["geoedit_root"]).resolve()
        sys.path.insert(0, str(geoedit_root))
        os.chdir(geoedit_root)
        from geoedit import inference

        reference_path = dataset_root / config["dataset"]["files"]["reference"]["path"]
        alpha_path = dataset_root / config["dataset"]["files"]["edit_alpha"]["path"]
        control_path = dataset_root / config["dataset"]["files"]["relative3d_control"]["path"]
        with Image.open(reference_path) as image:
            reference_image = image.convert("RGB")
            width, height = reference_image.size
        stdout_path = output_root / "stdout.log"
        stderr_path = output_root / "stderr.log"
        raw_video = output_root / "raw.mp4"
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            pipe = inference.load_pipeline(inference.resolve_vram_limit(None))
            manifest["pipeline_load_count"] = 1
            control = inference.load_video_or_image(control_path, height, width, inference_config["num_frames"])
            edit_mask = inference.load_video_or_image(alpha_path, height, width, inference_config["num_frames"])
            video = pipe(
                prompt=inference_config["prompt"], negative_prompt=inference_config["negative_prompt"],
                height=height, width=width, num_frames=inference_config["num_frames"],
                num_inference_steps=inference_config["num_inference_steps"],
                vace_video=control, vace_video_mask=edit_mask,
                vace_reference_image=reference_image, vace_scale=inference_config["vace_scale"],
                enable_ttm=inference_config["enable_ttm"], seed=inference_config["seed"], tiled=True,
            )
            inference.save_video(video, str(raw_video), fps=inference_config["fps"], quality=5)
            del video
        projection = project_outputs(
            raw_video, reference_path, alpha_path, output_root,
            inference_config["num_frames"], inference_config["fps"],
            inference_config["review_frame_indices"],
        )
        artifacts = {name: {"path": str(path), "size_bytes": path.stat().st_size,
                            "sha256": sha256(path)}
                     for name, path in {
                         "raw_video": raw_video, "projected_video": projection["projected_video"],
                         "review": projection["review"], "final_hold": projection["final_hold"],
                         "stdout": stdout_path, "stderr": stderr_path,
                     }.items()}
        manifest.update({"status": "complete_requires_action_and_photo_review",
                         "decoded_frames": projection["decoded_frames"],
                         "outside_support_max_pixel_difference": projection["outside_support_max_pixel_difference"],
                         "artifacts": artifacts})
        (output_root / "RUNNING").unlink()
        (output_root / "COMPLETE").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
        return_code = 0
    except Exception as error:
        failure = output_root / "failure.txt"
        failure.write_text(traceback.format_exc(), encoding="utf-8")
        manifest.update({"status": "technical_failure_preserved",
                         "error": f"{type(error).__name__}: {error}",
                         "failure_sha256": sha256(failure)})
        (output_root / "RUNNING").unlink(missing_ok=True)
        (output_root / "FAILED").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
        return_code = 3
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["wall_time_seconds"] = round(time.monotonic() - started, 3)
    write_json(output_root / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
