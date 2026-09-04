#!/usr/bin/env python3
"""Run the frozen resident-pipeline Day 13 five-condition evaluation."""

from __future__ import annotations

import argparse
import contextlib
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

from run_vace_fork_3d_compare import sha256_file, write_json
from run_vace_fork_action_lora_eval_v2 import install_runtime_injection, resolve_parent


SCHEMA_VERSION = "foodstateedit.flexible_completion_evaluation.v1"
METHOD = "relative3d_topology_weighted_flexible_completion_eval_v1"
MASK_KEYS = (
    "flexible_strand_mask_video",
    "pinch_contact_mask_video",
    "source_connection_mask_video",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
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


def resize_frame(frame: np.ndarray, width: int, height: int, *, mask: bool = False) -> np.ndarray:
    if frame.shape[:2] == (height, width):
        return frame
    interpolation = cv2.INTER_NEAREST if mask else cv2.INTER_LANCZOS4
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def make_contact_sheet(frames: list[np.ndarray], indices: list[int]) -> np.ndarray:
    selected = [frames[index] for index in indices]
    height, width = selected[0].shape[:2]
    sheet = np.zeros((height * 2, width * 3, 3), dtype=np.uint8)
    for slot, frame in enumerate(selected):
        row, column = divmod(slot, 3)
        sheet[row * height : (row + 1) * height, column * width : (column + 1) * width] = frame
    return sheet


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise OSError(f"Could not create {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def safe_masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    return float(values[mask].mean())


def project_and_score(
    result_video: Path,
    target_video: Path,
    reference_path: Path,
    alpha_path: Path,
    mask_paths: dict[str, Path],
    output_root: Path,
    condition: str,
    phase_schedule: list[dict[str, object]],
    review_indices: list[int],
    expected_frames: int,
    fps: int,
) -> dict[str, object]:
    generated = decode_video(result_video, expected_frames)
    targets = decode_video(target_video, expected_frames)
    mask_frames = {key: decode_video(path, expected_frames) for key, path in mask_paths.items()}
    reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    alpha = cv2.imread(str(alpha_path), cv2.IMREAD_GRAYSCALE)
    if reference is None or alpha is None:
        raise RuntimeError("Failed to read frozen reference or alpha")
    height, width = reference.shape[:2]
    alpha_weight = alpha.astype(np.float32)[..., None] / 255.0
    protected = alpha == 0
    support = alpha > 0
    projected_frames: list[np.ndarray] = []
    support_mae: list[float] = []
    support_by_phase: dict[str, list[float]] = {}
    topology_numerator = 0.0
    topology_denominator = 0.0
    pinch_values: list[np.ndarray] = []
    connection_values: list[np.ndarray] = []
    topology_per_frame: list[float | None] = []
    pinch_per_frame: list[float | None] = []
    connection_per_frame: list[float | None] = []
    outside_max = 0
    for index, (raw, target) in enumerate(zip(generated, targets, strict=True)):
        raw = resize_frame(raw, width, height)
        target = resize_frame(target, width, height)
        projected = np.rint(raw.astype(np.float32) * alpha_weight + reference.astype(np.float32) * (1.0 - alpha_weight))
        projected = np.clip(projected, 0, 255).astype(np.uint8)
        outside = np.abs(projected.astype(np.int16) - reference.astype(np.int16))[protected]
        outside_max = max(outside_max, int(outside.max(initial=0)))
        difference = np.abs(projected.astype(np.int16) - target.astype(np.int16)).astype(np.float32)
        frame_support_mae = float(difference[support].mean())
        support_mae.append(frame_support_mae)
        phase = str(phase_schedule[index]["phase"])
        support_by_phase.setdefault(phase, []).append(frame_support_mae)
        binary_masks = {
            key: resize_frame(mask_frames[key][index], width, height, mask=True)[..., :3].mean(axis=2) > 127
            for key in MASK_KEYS
        }
        focus_weight = (
            3.0 * binary_masks[MASK_KEYS[0]].astype(np.float32)
            + 7.0 * binary_masks[MASK_KEYS[1]].astype(np.float32)
            + 5.0 * binary_masks[MASK_KEYS[2]].astype(np.float32)
        )
        focus_active = focus_weight > 0
        if np.any(focus_active):
            channel_weight = focus_weight[..., None]
            numerator = float((difference * channel_weight).sum())
            denominator = float(channel_weight.sum() * difference.shape[2])
            value = numerator / denominator
            topology_numerator += numerator
            topology_denominator += denominator
            topology_per_frame.append(value)
        else:
            topology_per_frame.append(None)
        pinch = binary_masks[MASK_KEYS[1]]
        connection = binary_masks[MASK_KEYS[2]]
        pinch_value = safe_masked_mean(difference, pinch)
        connection_value = safe_masked_mean(difference, connection)
        pinch_per_frame.append(pinch_value)
        connection_per_frame.append(connection_value)
        if pinch_value is not None:
            pinch_values.append(difference[pinch])
        if connection_value is not None:
            connection_values.append(difference[connection])
        projected_frames.append(projected)
    if outside_max != 0:
        raise AssertionError(f"{condition}: projected output changed protected pixels")
    projected_video = output_root / f"{condition}_projected.mp4"
    review_path = output_root / f"{condition}_projected_review.png"
    final_path = output_root / f"{condition}_projected_final_hold.png"
    write_video(projected_video, projected_frames, fps)
    if not cv2.imwrite(str(review_path), make_contact_sheet(projected_frames, review_indices)):
        raise OSError(f"Could not create {review_path}")
    if not cv2.imwrite(str(final_path), projected_frames[review_indices[-1]]):
        raise OSError(f"Could not create {final_path}")
    return {
        "decoded_frames": len(generated),
        "outside_support_max_pixel_difference": outside_max,
        "target_rgb_mae_inside_support_mean": float(np.mean(support_mae)),
        "target_rgb_mae_inside_support_per_frame": support_mae,
        "target_rgb_mae_inside_support_by_phase": {phase: float(np.mean(values)) for phase, values in support_by_phase.items()},
        "target_rgb_mae_inside_3d_topology_weight_volume": topology_numerator / topology_denominator,
        "target_rgb_mae_inside_3d_topology_weight_per_frame": topology_per_frame,
        "target_rgb_mae_inside_pinch_contact_roi": float(np.concatenate(pinch_values).mean()),
        "target_rgb_mae_inside_pinch_contact_roi_per_frame": pinch_per_frame,
        "target_rgb_mae_inside_source_connection_roi": float(np.concatenate(connection_values).mean()),
        "target_rgb_mae_inside_source_connection_roi_per_frame": connection_per_frame,
        "projected_video": projected_video,
        "review": review_path,
        "final_hold": final_path,
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


def percent_improvement(baseline: float, proposed: float) -> float:
    if baseline <= 0:
        raise ValueError("Metric baseline must be positive")
    return 100.0 * (baseline - proposed) / baseline


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    preflight_report = args.preflight_report.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION or config.get("method") != METHOD:
        raise ValueError("Unexpected Day 13 evaluation config")
    if str(dataset_root) != config["dataset"]["remote_root"] or str(output_root) != config["output_root"]:
        raise ValueError("Dataset or output root differs from frozen config")
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse output root: {output_root}")
    repo_root = Path(__file__).resolve().parents[1]
    preflight_script = repo_root / config["implementation"]["evaluation_preflight"]["path"]
    command = [
        sys.executable,
        str(preflight_script),
        "--config",
        str(config_path),
        "--dataset-root",
        str(dataset_root),
        "--output-root",
        str(output_root),
        "--gpu-index",
        str(args.gpu_index),
        "--report",
        str(preflight_report),
    ]
    preflight = subprocess.run(command, check=False)
    if preflight.returncode != 0:
        print("Preflight blocked evaluation; no output directory was created.", file=sys.stderr)
        return preflight.returncode

    dataset = config["dataset"]
    inference_config = config["inference"]
    phase_path = dataset_root / dataset["files"]["phase_schedule"]["path"]
    phase_schedule = json.loads(phase_path.read_text(encoding="utf-8"))
    if [item["frame"] for item in phase_schedule] != list(range(inference_config["num_frames"])):
        raise ValueError("Frozen phase schedule does not cover every frame")
    runtime = config["runtime"]
    environment = os.environ.copy()
    environment.update(runtime["offline_environment"])
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu_index),
            "DIFFSYNTH_MODEL_BASE_PATH": str(Path(runtime["model_root"]).parent.parent),
            "HF_HOME": runtime["cache_root"],
            "TRANSFORMERS_CACHE": runtime["cache_root"],
            "PYTHONPATH": runtime["geoedit_root"],
        }
    )
    os.environ.update(environment)
    Path(runtime["cache_root"]).mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    manifest: dict[str, object] = {
        "schema_version": "foodstateedit.flexible_completion_evaluation_run.v1",
        "method": METHOD,
        "scientific_status": config["scientific_status"],
        "claim_limit": config["claim_limit"],
        "status": "running",
        "host": socket.gethostname(),
        "selected_physical_gpu": args.gpu_index,
        "started_at": started_at.isoformat(),
        "config_sha256": sha256_file(config_path),
        "preflight_report_sha256": sha256_file(preflight_report),
        "worker_sha256": sha256_file(Path(__file__).resolve()),
        "pipeline_load_count": 0,
        "conditions": {},
        "blind_fork_evaluation_allowed": False,
    }
    write_json(output_root / "run_manifest.json", manifest)
    write_json(
        output_root / "command.json",
        {
            "conditions": [item["name"] for item in config["conditions"]],
            "seed": inference_config["seed"],
            "num_frames": inference_config["num_frames"],
            "num_inference_steps": inference_config["num_inference_steps"],
            "vace_scale": inference_config["vace_scale"],
            "enable_ttm": inference_config["enable_ttm"],
            "review_frame_indices": inference_config["review_frame_indices"],
            "cuda_visible_devices": str(args.gpu_index),
        },
    )
    (output_root / "RUNNING").write_text(started_at.isoformat() + "\n", encoding="utf-8")
    try:
        geoedit_root = Path(runtime["geoedit_root"]).resolve()
        sys.path.insert(0, str(geoedit_root))
        os.chdir(geoedit_root)
        from geoedit import inference

        reference_path = dataset_root / dataset["files"]["reference_image"]["path"]
        alpha_path = dataset_root / dataset["files"]["edit_alpha"]["path"]
        target_path = dataset_root / dataset["files"]["target_video"]["path"]
        mask_paths = {key: dataset_root / dataset["files"][key]["path"] for key in MASK_KEYS}
        with Image.open(reference_path) as image:
            reference_image = image.convert("RGB")
            width, height = reference_image.size
        edit_mask = inference.load_video_or_image(alpha_path, height, width, inference_config["num_frames"])
        control_cache = {
            name: inference.load_video_or_image(dataset_root / record["path"], height, width, inference_config["num_frames"])
            for name, record in config["controls"].items()
        }
        stdout_path = output_root / "stdout.log"
        stderr_path = output_root / "stderr.log"
        previous_targets: list[str] = []
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            pipe = inference.load_pipeline(inference.resolve_vram_limit(None))
            manifest["pipeline_load_count"] = 1
            for condition in config["conditions"]:
                if previous_targets:
                    remove_runtime_injection(pipe, previous_targets)
                    previous_targets = []
                injection = None
                checkpoint = condition["checkpoint"]
                if checkpoint is not None:
                    injection = install_runtime_injection(pipe, Path(checkpoint["path"]), config["adapter_alpha"])
                    previous_targets = list(injection["targets"])
                    if (
                        injection["injected_linear_count"] != 80
                        or injected_count(pipe.vace) != 80
                        or injected_count(pipe.vace2) != 0
                    ):
                        raise RuntimeError("High/low-noise VACE injection isolation failed")
                result_path = output_root / f"{condition['name']}.mp4"
                video = pipe(
                    prompt=config["prompt"],
                    negative_prompt=inference_config["negative_prompt"],
                    height=height,
                    width=width,
                    num_frames=inference_config["num_frames"],
                    num_inference_steps=inference_config["num_inference_steps"],
                    vace_video=control_cache[condition["control"]],
                    vace_video_mask=edit_mask,
                    vace_reference_image=reference_image,
                    vace_scale=inference_config["vace_scale"],
                    enable_ttm=inference_config["enable_ttm"],
                    seed=inference_config["seed"],
                    tiled=True,
                )
                inference.save_video(video, str(result_path), fps=inference_config["fps"], quality=5)
                del video
                metrics = project_and_score(
                    result_path,
                    target_path,
                    reference_path,
                    alpha_path,
                    mask_paths,
                    output_root,
                    condition["name"],
                    phase_schedule,
                    inference_config["review_frame_indices"],
                    inference_config["num_frames"],
                    inference_config["fps"],
                )
                artifacts = {
                    name: {
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for name, path in {
                        "raw_video": result_path,
                        "projected_video": metrics.pop("projected_video"),
                        "review": metrics.pop("review"),
                        "final_hold": metrics.pop("final_hold"),
                    }.items()
                }
                manifest["conditions"][condition["name"]] = {
                    "control": condition["control"],
                    "step": condition["step"],
                    "lora_enabled": checkpoint is not None,
                    "checkpoint": checkpoint,
                    "runtime_injection": injection,
                    **metrics,
                    "artifacts": artifacts,
                }

        uniform = manifest["conditions"]["relative3d_uniform_step32"]
        weighted = manifest["conditions"]["relative3d_topology_weighted_step32"]
        topology_improvement = percent_improvement(
            uniform["target_rgb_mae_inside_3d_topology_weight_volume"],
            weighted["target_rgb_mae_inside_3d_topology_weight_volume"],
        )
        pinch_improvement = percent_improvement(
            uniform["target_rgb_mae_inside_pinch_contact_roi"],
            weighted["target_rgb_mae_inside_pinch_contact_roi"],
        )
        manifest["primary_comparison"] = {
            "baseline": "relative3d_uniform_step32",
            "proposed": "relative3d_topology_weighted_step32",
            "topology_weight_volume_mae_improvement_percent": topology_improvement,
            "pinch_contact_roi_mae_improvement_percent": pinch_improvement,
            "numeric_gate_at_least_5_percent_each": topology_improvement >= 5.0 and pinch_improvement >= 5.0,
        }
        manifest["contextual_geometry_comparison"] = {
            "baseline": "planar_uniform_step32",
            "relative3d": "relative3d_uniform_step32",
            "support_mae_improvement_percent": percent_improvement(
                manifest["conditions"]["planar_uniform_step32"]["target_rgb_mae_inside_support_mean"],
                uniform["target_rgb_mae_inside_support_mean"],
            ),
        }
        manifest["all_outside_support_exact"] = all(
            item["outside_support_max_pixel_difference"] == 0
            for item in manifest["conditions"].values()
        )
        manifest["status"] = "complete_requires_two_blinded_semantic_and_photo_reviewers"
        return_code = 0
        (output_root / "RUNNING").unlink()
        (output_root / "COMPLETE").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
    except Exception as error:
        failure_path = output_root / "failure.txt"
        failure_path.write_text(traceback.format_exc(), encoding="utf-8")
        manifest["status"] = "technical_failure_preserved"
        manifest["error"] = f"{type(error).__name__}: {error}"
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
