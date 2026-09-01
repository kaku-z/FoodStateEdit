#!/usr/bin/env python3
"""Run one resident-pipeline five-condition seen phase-action checkpoint sweep."""

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
from run_vace_fork_action_lora_eval_v2 import install_runtime_injection, resolve_parent


METHOD = "vace_phase_action_checkpoint_sweep_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    return parser.parse_args()


def decode_video(path: Path, expected_frames: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) != expected_frames:
        raise RuntimeError(f"{path}: decoded {len(frames)} frames, expected {expected_frames}")
    return frames


def make_contact_sheet(frames: list[np.ndarray], indices: list[int]) -> np.ndarray:
    selected = [frames[index] for index in indices]
    height, width = selected[0].shape[:2]
    sheet = np.zeros((height * 2, width * 3, 3), dtype=np.uint8)
    for slot, frame in enumerate(selected):
        row, column = divmod(slot, 3)
        sheet[row * height : (row + 1) * height, column * width : (column + 1) * width] = frame
    return sheet


def project_and_score(
    result_video: Path,
    target_video: Path,
    reference_path: Path,
    alpha_path: Path,
    output_root: Path,
    condition: str,
    phase_schedule: list[dict[str, object]],
    review_indices: list[int],
    expected_frames: int,
) -> dict[str, object]:
    generated = decode_video(result_video, expected_frames)
    targets = decode_video(target_video, expected_frames)
    reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    alpha = cv2.imread(str(alpha_path), cv2.IMREAD_GRAYSCALE)
    if reference is None or alpha is None:
        raise RuntimeError("Failed to read reference or alpha")
    height, width = reference.shape[:2]
    weight = alpha.astype(np.float32)[..., None] / 255.0
    protected = alpha == 0
    support = alpha > 0
    projected_frames: list[np.ndarray] = []
    per_frame_mae: list[float] = []
    per_phase: dict[str, list[float]] = {}
    outside_max = 0
    for index, (raw, target) in enumerate(zip(generated, targets, strict=True)):
        if raw.shape[:2] != (height, width):
            raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LANCZOS4)
        if target.shape[:2] != (height, width):
            target = cv2.resize(target, (width, height), interpolation=cv2.INTER_LANCZOS4)
        projected = np.rint(raw.astype(np.float32) * weight + reference.astype(np.float32) * (1.0 - weight))
        projected = np.clip(projected, 0, 255).astype(np.uint8)
        outside = np.abs(projected.astype(np.int16) - reference.astype(np.int16))[protected]
        outside_max = max(outside_max, int(outside.max(initial=0)))
        difference = np.abs(projected.astype(np.int16) - target.astype(np.int16))
        mae = float(difference[support].mean())
        per_frame_mae.append(mae)
        phase = str(phase_schedule[index]["phase"])
        per_phase.setdefault(phase, []).append(mae)
        projected_frames.append(projected)
    if outside_max != 0:
        raise AssertionError(f"{condition}: protected pixels changed")
    contact_sheet_path = output_root / f"{condition}_projected_review.png"
    selected_path = output_root / f"{condition}_projected_final_hold.png"
    if not cv2.imwrite(str(contact_sheet_path), make_contact_sheet(projected_frames, review_indices)):
        raise RuntimeError(f"Failed to write {contact_sheet_path}")
    if not cv2.imwrite(str(selected_path), projected_frames[review_indices[-1]]):
        raise RuntimeError(f"Failed to write {selected_path}")
    return {
        "decoded_frames": len(generated),
        "outside_support_max_pixel_difference": outside_max,
        "target_rgb_mae_inside_support_mean": float(np.mean(per_frame_mae)),
        "target_rgb_mae_inside_support_per_frame": per_frame_mae,
        "target_rgb_mae_inside_support_by_phase": {phase: float(np.mean(values)) for phase, values in per_phase.items()},
        "review_frame_mae": {str(index): per_frame_mae[index] for index in review_indices},
        "contact_sheet": contact_sheet_path,
        "final_hold": selected_path,
    }


def injected_count(module) -> int:
    return sum(1 for item in module.modules() if item.__class__.__name__ == "LoRAInjectedLinear")


def remove_runtime_injection(pipe, targets: list[str]) -> None:
    for target in targets:
        parent, child_name = resolve_parent(pipe.vace, target)
        wrapped = getattr(parent, child_name)
        if wrapped.__class__.__name__ != "LoRAInjectedLinear" or not hasattr(wrapped, "base"):
            raise TypeError(f"Target {target} is not a removable LoRAInjectedLinear")
        setattr(parent, child_name, wrapped.base)
    if injected_count(pipe.vace) != 0:
        raise RuntimeError("Previous LoRA injection was not fully removed")


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

    preflight_script = Path(__file__).resolve().with_name("preflight_phase_action_checkpoint_sweep.py")
    command = [sys.executable, str(preflight_script), "--config", str(config_path), "--dataset-root", str(dataset_root), "--sample-id", args.sample_id, "--output-root", str(output_root), "--gpu-index", str(args.gpu_index), "--report", str(preflight_report)]
    preflight = subprocess.run(command, check=False)
    if preflight.returncode != 0:
        print("Preflight blocked inference; no output directory was created.", file=sys.stderr)
        return preflight.returncode

    verified_inputs = {key: verify_file(dataset_root / record["path"], record, key) for key, record in sample["files"].items()}
    inference_config = config["inference"]
    if hashlib.sha256(sample["prompt"].encode("utf-8")).hexdigest() != sample["prompt_sha256_utf8"]:
        raise ValueError("Prompt hash mismatch")
    phase_path = dataset_root / config["dataset"]["phase_schedule"]["path"]
    phase_schedule = json.loads(phase_path.read_text(encoding="utf-8"))
    if [item["frame"] for item in phase_schedule] != list(range(inference_config["num_frames"])):
        raise ValueError("Phase schedule does not cover every frame")

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
        "schema_version": "foodstateedit.phase_action_checkpoint_sweep_run.v1",
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
        "blind_fork_evaluation_allowed": False,
    }
    write_json(output_root / "run_manifest.json", manifest)
    write_json(output_root / "command.json", {"sample_id": args.sample_id, "conditions": [item["name"] for item in config["adapters"]["conditions"]], "seed": inference_config["seed"], "num_frames": inference_config["num_frames"], "num_inference_steps": inference_config["num_inference_steps"], "vace_scale": inference_config["vace_scale"], "enable_ttm": inference_config["enable_ttm"], "review_frame_indices": inference_config["review_frame_indices"], "cuda_visible_devices": str(args.gpu_index)})
    (output_root / "RUNNING").write_text(started_at.isoformat() + "\n", encoding="utf-8")

    try:
        geoedit_root = Path(runtime["geoedit_root"]).resolve()
        sys.path.insert(0, str(geoedit_root))
        os.chdir(geoedit_root)
        from geoedit import inference

        reference_path = dataset_root / sample["files"]["reference"]["path"]
        control_path = dataset_root / sample["files"]["control"]["path"]
        alpha_path = dataset_root / sample["files"]["alpha"]["path"]
        target_video_path = dataset_root / sample["files"]["target_video"]["path"]
        with Image.open(reference_path) as image:
            reference_image = image.convert("RGB")
            width, height = reference_image.size
        if [width, height] != sample["dimensions"]:
            raise ValueError(f"Dimensions changed: {(width, height)}")
        motion_video = inference.load_video_or_image(control_path, height, width, inference_config["num_frames"])
        edit_mask = inference.load_video_or_image(alpha_path, height, width, inference_config["num_frames"])

        stdout_path = output_root / "stdout.log"
        stderr_path = output_root / "stderr.log"
        previous_targets: list[str] = []
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            pipe = inference.load_pipeline(inference.resolve_vram_limit(None))
            manifest["pipeline_load_count"] = 1
            for condition in config["adapters"]["conditions"]:
                if previous_targets:
                    remove_runtime_injection(pipe, previous_targets)
                    previous_targets = []
                injection = None
                checkpoint = condition["checkpoint"]
                if checkpoint is not None:
                    injection = install_runtime_injection(pipe, Path(checkpoint["path"]), config["adapters"]["alpha"])
                    previous_targets = list(injection["targets"])
                    if injection["injected_linear_count"] != 80 or injected_count(pipe.vace) != 80 or injected_count(pipe.vace2) != 0:
                        raise RuntimeError("High/low-noise injection isolation failed")
                result_video = output_root / f"{condition['name']}.mp4"
                video = pipe(prompt=sample["prompt"], negative_prompt=inference_config["negative_prompt"], height=height, width=width, num_frames=inference_config["num_frames"], num_inference_steps=inference_config["num_inference_steps"], vace_video=motion_video, vace_video_mask=edit_mask, vace_reference_image=reference_image, vace_scale=inference_config["vace_scale"], enable_ttm=inference_config["enable_ttm"], seed=inference_config["seed"], tiled=True)
                inference.save_video(video, str(result_video), fps=inference_config["fps"], quality=5)
                del video
                projection = project_and_score(result_video, target_video_path, reference_path, alpha_path, output_root, condition["name"], phase_schedule, inference_config["review_frame_indices"], inference_config["num_frames"])
                manifest["conditions"][condition["name"]] = {
                    "step": condition["step"],
                    "lora_enabled": checkpoint is not None,
                    "checkpoint": checkpoint,
                    "runtime_injection": injection,
                    "decoded_frames": projection["decoded_frames"],
                    "outside_support_max_pixel_difference": projection["outside_support_max_pixel_difference"],
                    "target_rgb_mae_inside_support_mean": projection["target_rgb_mae_inside_support_mean"],
                    "target_rgb_mae_inside_support_per_frame": projection["target_rgb_mae_inside_support_per_frame"],
                    "target_rgb_mae_inside_support_by_phase": projection["target_rgb_mae_inside_support_by_phase"],
                    "review_frame_mae": projection["review_frame_mae"],
                }

        off_mae = manifest["conditions"]["lora_off"]["target_rgb_mae_inside_support_mean"]
        comparisons = {}
        for condition in config["adapters"]["conditions"][1:]:
            name = condition["name"]
            mae = manifest["conditions"][name]["target_rgb_mae_inside_support_mean"]
            comparisons[name] = {"target_inside_support_mae_delta_vs_lora_off": mae - off_mae, "numerically_closer_to_target": mae < off_mae}
        manifest["comparisons"] = comparisons
        manifest["best_condition_by_target_support_mae"] = min(manifest["conditions"], key=lambda name: manifest["conditions"][name]["target_rgb_mae_inside_support_mean"])
        output_paths = [output_root / "command.json", stdout_path, stderr_path]
        for condition in config["adapters"]["conditions"]:
            name = condition["name"]
            output_paths.extend([output_root / f"{name}.mp4", output_root / f"{name}_projected_review.png", output_root / f"{name}_projected_final_hold.png"])
        manifest["outputs"] = [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in output_paths]
        manifest["status"] = "complete_requires_seen_phase_action_and_photo_review"
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
