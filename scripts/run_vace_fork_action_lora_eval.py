#!/usr/bin/env python3
"""Run the frozen Day 9 blind fork evaluation with the action LoRA enabled."""

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

from PIL import Image

from run_vace_fork_3d_compare import extract_selected_and_project, sha256_file, verify_file, write_json


METHOD = "vace_fork_action_lora_blind_eval_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    preflight_report = args.preflight_report.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("method") != METHOD:
        raise ValueError(f"Unexpected method: {config.get('method')}")
    if str(input_root) != config["control"]["remote_root"] or str(output_root) != config["output_root"]:
        raise ValueError("Input or output root differs from the frozen config")
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse output root: {output_root}")

    preflight_script = Path(__file__).resolve().with_name("preflight_vace_fork_action_lora_eval.py")
    preflight_command = [
        sys.executable, str(preflight_script),
        "--config", str(config_path),
        "--input-root", str(input_root),
        "--output-root", str(output_root),
        "--report", str(preflight_report),
    ]
    if args.gpu_index is not None:
        preflight_command.extend(["--gpu-index", str(args.gpu_index)])
    preflight = subprocess.run(preflight_command, check=False)
    if preflight.returncode != 0:
        print("Preflight blocked inference; no output directory was created.", file=sys.stderr)
        return preflight.returncode
    preflight_data = json.loads(preflight_report.read_text(encoding="utf-8"))
    selected_gpu = preflight_data.get("selected_gpu")
    if not preflight_data.get("ready") or selected_gpu is None:
        raise RuntimeError("Passing preflight did not select a GPU")

    verified_inputs = [verify_file(input_root / name, expected, f"control file {name}") for name, expected in config["control"]["files"].items()]
    inference_config = config["inference"]
    if hashlib.sha256(inference_config["prompt"].encode("utf-8")).hexdigest() != inference_config["prompt_sha256_utf8"]:
        raise ValueError("Prompt hash mismatch")
    if hashlib.sha256(inference_config["negative_prompt"].encode("utf-8")).hexdigest() != inference_config["negative_prompt_sha256_utf8"]:
        raise ValueError("Negative-prompt hash mismatch")

    adapter = config["adapter"]
    checkpoint = Path(adapter["checkpoint"]["path"])
    verified_checkpoint = verify_file(checkpoint, adapter["checkpoint"], "action LoRA checkpoint")
    runtime = config["runtime"]
    environment = os.environ.copy()
    environment.update(runtime["offline_environment"])
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(selected_gpu),
        "DIFFSYNTH_MODEL_BASE_PATH": str(Path(runtime["model_root"]).parent.parent),
        "HF_HOME": runtime["cache_root"],
        "TRANSFORMERS_CACHE": runtime["cache_root"],
        "PYTHONPATH": runtime["geoedit_root"],
    })
    os.environ.update(environment)
    Path(runtime["cache_root"]).mkdir(parents=True, exist_ok=True)

    output_root.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    worker_path = Path(__file__).resolve()
    manifest: dict[str, object] = {
        "schema_version": "foodstateedit.vace_fork_action_lora_eval_run.v1",
        "method": METHOD,
        "scientific_status": config["scientific_status"],
        "claim_limit": config["claim_limit"],
        "status": "running",
        "host": socket.gethostname(),
        "selected_physical_gpu": selected_gpu,
        "started_at": started_at.isoformat(),
        "config_sha256": sha256_file(config_path),
        "preflight_report_sha256": sha256_file(preflight_report),
        "worker_sha256": sha256_file(worker_path),
        "control_manifest_sha256": sha256_file(input_root / "run_manifest.json"),
        "verified_inputs": verified_inputs,
        "adapter": {
            "checkpoint": verified_checkpoint,
            "alpha": adapter["alpha"],
            "pipeline_load_count": 0,
            "lora_load_count": 0,
            "training_manifest_sha256": adapter["training_manifest"]["sha256"],
            "validation_report_sha256": adapter["validation"]["sha256"],
            "held_out_family": adapter["held_out"]["family"],
            "training_occurrences": adapter["held_out"]["training_occurrences"],
        },
        "day8_lora_off_baseline": config["day8_lora_off_baseline"],
        "inference": {
            "seed": inference_config["seed"],
            "num_frames": inference_config["num_frames"],
            "num_inference_steps": inference_config["num_inference_steps"],
            "vace_scale": inference_config["vace_scale"],
            "enable_ttm": inference_config["enable_ttm"],
            "selected_frame_index": inference_config["selected_frame_index"],
            "selection_policy": inference_config["selection_policy"],
            "projection_alpha": inference_config["projection_alpha"],
        },
        "outputs": [],
    }
    write_json(output_root / "run_manifest.json", manifest)
    write_json(output_root / "command.json", {
        "call": "WanVideoPipeline.__call__",
        "vace_video": str(input_root / "dynamic_control.mp4"),
        "vace_video_mask": str(input_root / "dynamic_mask.mp4"),
        "vace_reference_image": str(input_root / "first_frame.png"),
        "prompt_sha256_utf8": inference_config["prompt_sha256_utf8"],
        "negative_prompt_sha256_utf8": inference_config["negative_prompt_sha256_utf8"],
        "seed": inference_config["seed"],
        "num_frames": inference_config["num_frames"],
        "num_inference_steps": inference_config["num_inference_steps"],
        "vace_scale": inference_config["vace_scale"],
        "enable_ttm": inference_config["enable_ttm"],
        "selected_frame_index": inference_config["selected_frame_index"],
        "lora_checkpoint_sha256": adapter["checkpoint"]["sha256"],
        "lora_alpha": adapter["alpha"],
        "cuda_visible_devices": str(selected_gpu),
    })
    (output_root / "RUNNING").write_text(started_at.isoformat() + "\n", encoding="utf-8")

    try:
        geoedit_root = Path(runtime["geoedit_root"]).resolve()
        sys.path.insert(0, str(geoedit_root))
        os.chdir(geoedit_root)
        from geoedit import inference

        stdout_path = output_root / "stdout.log"
        stderr_path = output_root / "stderr.log"
        result_video = output_root / "result.mp4"
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            pipe = inference.load_pipeline(inference.resolve_vram_limit(None))
            manifest["adapter"]["pipeline_load_count"] = 1
            pipe.load_lora(pipe.vace, str(checkpoint), alpha=adapter["alpha"])
            manifest["adapter"]["lora_load_count"] = 1
            with Image.open(input_root / "first_frame.png") as image:
                reference_image = image.convert("RGB")
                width, height = reference_image.size
            if (width, height) != (inference_config["width"], inference_config["height"]):
                raise ValueError(f"Input dimensions changed: {(width, height)}")
            motion_video = inference.load_video_or_image(input_root / "dynamic_control.mp4", height, width, inference_config["num_frames"])
            edit_mask = inference.load_video_or_image(input_root / "dynamic_mask.mp4", height, width, inference_config["num_frames"])
            video = pipe(
                prompt=inference_config["prompt"],
                negative_prompt=inference_config["negative_prompt"],
                height=height,
                width=width,
                num_frames=inference_config["num_frames"],
                num_inference_steps=inference_config["num_inference_steps"],
                vace_video=motion_video,
                vace_video_mask=edit_mask,
                vace_reference_image=reference_image,
                vace_scale=inference_config["vace_scale"],
                enable_ttm=inference_config["enable_ttm"],
                seed=inference_config["seed"],
                tiled=True,
            )
            inference.save_video(video, str(result_video), fps=inference_config["fps"], quality=5)

        projection = extract_selected_and_project(result_video, input_root, output_root, inference_config["selected_frame_index"], inference_config["num_frames"])
        manifest["inference"].update({
            "decoded_frames": projection["decoded_frames"],
            "outside_motion_support_max_pixel_difference": projection["outside_motion_support_max_pixel_difference"],
            "outside_motion_support_rgb_mae": projection["outside_motion_support_rgb_mae"],
            "feather_band_rgb_mae": projection["feather_band_rgb_mae"],
        })
        output_paths = [result_video, projection["raw_selected_frame"], projection["edited_2d"], stdout_path, stderr_path, output_root / "command.json"]
        manifest["outputs"] = [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in output_paths]
        manifest["status"] = "complete_requires_blind_action_and_photo_review"
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
