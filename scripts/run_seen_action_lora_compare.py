#!/usr/bin/env python3
"""Run one resident-pipeline LoRA-off/on comparison on a seen action family."""

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

from run_vace_fork_3d_compare import sha256_file, verify_file, write_json
from run_vace_fork_action_lora_eval_v2 import install_runtime_injection


METHOD = "vace_seen_action_lora_same_seed_compare_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    return parser.parse_args()


def decode_select_project(video_path: Path, reference_path: Path, alpha_path: Path, target_path: Path, output_root: Path, condition: str, selected_index: int, expected_frames: int) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) != expected_frames:
        raise RuntimeError(f"{condition}: decoded {len(frames)} frames, expected {expected_frames}")
    reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
    alpha = cv2.imread(str(alpha_path), cv2.IMREAD_GRAYSCALE)
    if reference is None or target is None or alpha is None:
        raise RuntimeError("Failed to read reference, target, or alpha")
    height, width = reference.shape[:2]
    raw = frames[selected_index]
    if raw.shape[:2] != (height, width):
        raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LANCZOS4)
    weight = alpha.astype(np.float32)[..., None] / 255.0
    projected = np.rint(raw.astype(np.float32) * weight + reference.astype(np.float32) * (1.0 - weight))
    projected = np.clip(projected, 0, 255).astype(np.uint8)
    raw_path = output_root / f"{condition}_raw_selected_frame.png"
    projected_path = output_root / f"{condition}_edited_2d.png"
    if not cv2.imwrite(str(raw_path), raw) or not cv2.imwrite(str(projected_path), projected):
        raise RuntimeError(f"Failed to write {condition} selected images")
    protected = alpha == 0
    support = alpha > 0
    outside = np.abs(projected.astype(np.int16) - reference.astype(np.int16))[protected]
    if int(outside.max(initial=0)) != 0:
        raise AssertionError(f"{condition}: protected pixels changed")
    target_difference = np.abs(projected.astype(np.int16) - target.astype(np.int16))
    return {
        "decoded_frames": len(frames),
        "raw_selected_frame": raw_path,
        "edited_2d": projected_path,
        "outside_support_max_pixel_difference": int(outside.max(initial=0)),
        "target_rgb_mae_global": float(target_difference.mean()),
        "target_rgb_mae_inside_support": float(target_difference[support].mean()),
        "target_changed_pixel_fraction_inside_support": float(np.any(target_difference[support] > 0, axis=1).mean()),
    }


def injected_count(module) -> int:
    return sum(1 for item in module.modules() if item.__class__.__name__ == "LoRAInjectedLinear")


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    preflight_report = args.preflight_report.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    samples = {sample["sample_id"]: sample for sample in config["samples"]}
    sample = samples.get(args.sample_id)
    if config.get("method") != METHOD or sample is None:
        raise ValueError("Unexpected method or sample")
    if output_root != Path(config["output_base"]) / args.sample_id:
        raise ValueError("Output root differs from frozen config")
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse output root: {output_root}")

    preflight_script = Path(__file__).resolve().with_name("preflight_seen_action_lora_eval.py")
    command = [sys.executable, str(preflight_script), "--config", str(config_path), "--dataset-root", str(dataset_root), "--sample-id", args.sample_id, "--output-root", str(output_root), "--gpu-index", str(args.gpu_index), "--report", str(preflight_report)]
    preflight = subprocess.run(command, check=False)
    if preflight.returncode != 0:
        print("Preflight blocked inference; no output directory was created.", file=sys.stderr)
        return preflight.returncode

    verified_inputs = {key: verify_file(dataset_root / record["path"], record, key) for key, record in sample["files"].items()}
    inference_config = config["inference"]
    if hashlib.sha256(sample["prompt"].encode("utf-8")).hexdigest() != sample["prompt_sha256_utf8"]:
        raise ValueError("Prompt hash mismatch")
    runtime = config["runtime"]
    environment = os.environ.copy()
    environment.update(runtime["offline_environment"])
    environment.update({"CUDA_VISIBLE_DEVICES": str(args.gpu_index), "DIFFSYNTH_MODEL_BASE_PATH": str(Path(runtime["model_root"]).parent.parent), "HF_HOME": runtime["cache_root"], "TRANSFORMERS_CACHE": runtime["cache_root"], "PYTHONPATH": runtime["geoedit_root"]})
    os.environ.update(environment)
    Path(runtime["cache_root"]).mkdir(parents=True, exist_ok=True)

    output_root.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    manifest: dict[str, object] = {
        "schema_version": "foodstateedit.seen_action_lora_compare_run.v1",
        "method": METHOD,
        "scientific_status": config["scientific_status"],
        "claim_limit": config["claim_limit"],
        "status": "running",
        "sample_id": args.sample_id,
        "family": sample["family"],
        "host": socket.gethostname(),
        "selected_physical_gpu": args.gpu_index,
        "started_at": started_at.isoformat(),
        "config_sha256": sha256_file(config_path),
        "preflight_report_sha256": sha256_file(preflight_report),
        "worker_sha256": sha256_file(Path(__file__).resolve()),
        "verified_inputs": verified_inputs,
        "pipeline_load_count": 0,
        "conditions": {},
        "outputs": [],
    }
    write_json(output_root / "run_manifest.json", manifest)
    write_json(output_root / "command.json", {"sample_id": args.sample_id, "seed": inference_config["seed"], "num_frames": inference_config["num_frames"], "num_inference_steps": inference_config["num_inference_steps"], "vace_scale": inference_config["vace_scale"], "enable_ttm": inference_config["enable_ttm"], "selected_frame_index": inference_config["selected_frame_index"], "lora_alpha": config["adapter"]["alpha"], "cuda_visible_devices": str(args.gpu_index)})
    (output_root / "RUNNING").write_text(started_at.isoformat() + "\n", encoding="utf-8")

    try:
        geoedit_root = Path(runtime["geoedit_root"]).resolve()
        sys.path.insert(0, str(geoedit_root))
        os.chdir(geoedit_root)
        from geoedit import inference

        reference_path = dataset_root / sample["files"]["reference"]["path"]
        control_path = dataset_root / sample["files"]["control"]["path"]
        alpha_path = dataset_root / sample["files"]["alpha"]["path"]
        target_path = dataset_root / sample["files"]["target"]["path"]
        with Image.open(reference_path) as image:
            reference_image = image.convert("RGB")
            width, height = reference_image.size
        if [width, height] != sample["dimensions"]:
            raise ValueError(f"Dimensions changed: {(width, height)}")
        motion_video = inference.load_video_or_image(control_path, height, width, inference_config["num_frames"])
        edit_mask = inference.load_video_or_image(alpha_path, height, width, inference_config["num_frames"])

        stdout_path = output_root / "stdout.log"
        stderr_path = output_root / "stderr.log"
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            pipe = inference.load_pipeline(inference.resolve_vram_limit(None))
            manifest["pipeline_load_count"] = 1
            for condition in ("lora_off", "lora_on"):
                injection = None
                if condition == "lora_on":
                    injection = install_runtime_injection(pipe, Path(config["adapter"]["checkpoint"]["path"]), config["adapter"]["alpha"])
                    if injection["injected_linear_count"] != 80 or injected_count(pipe.vace) != 80 or injected_count(pipe.vace2) != 0:
                        raise RuntimeError("High/low-noise injection isolation failed")
                result_video = output_root / f"{condition}.mp4"
                video = pipe(prompt=sample["prompt"], negative_prompt=inference_config["negative_prompt"], height=height, width=width, num_frames=inference_config["num_frames"], num_inference_steps=inference_config["num_inference_steps"], vace_video=motion_video, vace_video_mask=edit_mask, vace_reference_image=reference_image, vace_scale=inference_config["vace_scale"], enable_ttm=inference_config["enable_ttm"], seed=inference_config["seed"], tiled=True)
                inference.save_video(video, str(result_video), fps=inference_config["fps"], quality=5)
                projection = decode_select_project(result_video, reference_path, alpha_path, target_path, output_root, condition, inference_config["selected_frame_index"], inference_config["num_frames"])
                manifest["conditions"][condition] = {
                    "lora_enabled": condition == "lora_on",
                    "runtime_injection": injection,
                    "decoded_frames": projection["decoded_frames"],
                    "outside_support_max_pixel_difference": projection["outside_support_max_pixel_difference"],
                    "target_rgb_mae_global": projection["target_rgb_mae_global"],
                    "target_rgb_mae_inside_support": projection["target_rgb_mae_inside_support"],
                    "target_changed_pixel_fraction_inside_support": projection["target_changed_pixel_fraction_inside_support"],
                }

        off_mae = manifest["conditions"]["lora_off"]["target_rgb_mae_inside_support"]
        on_mae = manifest["conditions"]["lora_on"]["target_rgb_mae_inside_support"]
        manifest["comparison"] = {"target_inside_support_mae_delta_on_minus_off": on_mae - off_mae, "lora_on_closer_to_target": on_mae < off_mae}
        output_paths = [output_root / name for name in ("lora_off.mp4", "lora_on.mp4", "lora_off_raw_selected_frame.png", "lora_on_raw_selected_frame.png", "lora_off_edited_2d.png", "lora_on_edited_2d.png", "stdout.log", "stderr.log", "command.json")]
        manifest["outputs"] = [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in output_paths]
        manifest["status"] = "complete_requires_seen_family_action_and_photo_review"
        return_code = 0
        (output_root / "RUNNING").unlink()
        (output_root / "COMPLETE").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
    except Exception as error:
        failure_path = output_root / "failure.txt"
        failure_path.write_text(traceback.format_exc(), encoding="utf-8")
        manifest["status"] = "technical_failure_preserved"
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["outputs"] = [{"path": str(failure_path), "size_bytes": failure_path.stat().st_size, "sha256": sha256_file(failure_path)}]
        return_code = 3
        (output_root / "RUNNING").unlink(missing_ok=True)
        (output_root / "FAILED").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["wall_time_seconds"] = round(time.monotonic() - started, 3)
    write_json(output_root / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
