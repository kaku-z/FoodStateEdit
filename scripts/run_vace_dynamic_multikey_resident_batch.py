#!/usr/bin/env python3
"""Run the frozen VACE-direct deterministic dynamic-multikey v1 pilot."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from PIL import Image

from run_geoedit_anchor import (
    sha256_file,
    verify_hash,
    verify_model_audit,
    write_json,
)


EXPECTED_GEOEDIT_FILES = {
    "geoedit/inference.py":
        "d093ae7e2245884ccb7fc664b78a919ea5443088e77aa1a53c6a033f1e8c8bf1",
    "diffsynth/pipelines/wan_video.py":
        "50bb12685ef9ccb6b89d6600c27ad0cd2f12b5c67c5301fc51ea3f0eff02bfb5",
    "diffsynth/utils/data/__init__.py":
        "fe056b4a675a345cf02d6c76d327e8434103ccf2a4066ff208ce41368771e360",
    "tests/test_masks.py":
        "fc6400e66c94a22e5048fbf7fc37058e6bac099c8c2024086b7ea408e640f421",
}
METHOD = "vace_direct_dynamic_multikey"
SEED = 1
NUM_FRAMES = 21
NUM_INFERENCE_STEPS = 20
VACE_SCALE = 1.0
SELECTED_FRAME_INDEX = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--anchor", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--geoedit-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-hash-audit", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--vace-scale", type=float, default=VACE_SCALE)
    parser.add_argument("--selected-frame-index", type=int, default=SELECTED_FRAME_INDEX)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def now_jst() -> str:
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds")


def read_noncomment_text(path: Path) -> str:
    return "\n".join(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def verify_dynamic_proxy(proxy_dir: Path) -> dict[str, object]:
    manifest_path = proxy_dir / "proxy_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "foodstateedit.dynamic_multikey_proxy.v1":
        raise ValueError(f"Unexpected dynamic proxy schema: {manifest_path}")
    if manifest.get("anchor_id") != proxy_dir.name:
        raise ValueError(f"Dynamic proxy identity mismatch: {manifest_path}")
    if manifest.get("method") != METHOD or not manifest.get("deterministic"):
        raise ValueError(f"Dynamic proxy method mismatch: {manifest_path}")
    if manifest.get("frame_count") != NUM_FRAMES:
        raise ValueError(f"Dynamic proxy frame count mismatch: {manifest_path}")
    if manifest.get("selected_frame_index") != SELECTED_FRAME_INDEX:
        raise ValueError(f"Dynamic proxy selected frame mismatch: {manifest_path}")
    if manifest.get("keyframe_policy") != "deterministic_geometry_only_no_image_generation":
        raise ValueError(f"Dynamic proxy contains an ineligible keyframe policy: {manifest_path}")
    for name, record in manifest["files"].items():
        verify_hash(proxy_dir / name, record["sha256"], f"dynamic proxy file {name}")
    return manifest


def extract_selected_and_project(
    video_path: Path,
    proxy_dir: Path,
    run_dir: Path,
    selected_frame_index: int,
) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) <= selected_frame_index:
        raise RuntimeError(
            f"Video has {len(frames)} frames; cannot select {selected_frame_index}"
        )
    raw = frames[selected_frame_index]
    source = cv2.imread(str(proxy_dir / "first_frame.png"), cv2.IMREAD_COLOR)
    alpha = cv2.imread(str(proxy_dir / "edit_alpha.png"), cv2.IMREAD_GRAYSCALE)
    if source is None or alpha is None:
        raise RuntimeError("Failed to read first_frame.png or edit_alpha.png")
    height, width = source.shape[:2]
    if raw.shape[:2] != (height, width):
        raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LANCZOS4)
    if alpha.shape != (height, width):
        alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)

    raw_path = run_dir / "raw_selected_frame.png"
    edited_path = run_dir / "edited_2d.png"
    if not cv2.imwrite(str(raw_path), raw):
        raise RuntimeError(f"Failed to write {raw_path}")
    weight = alpha.astype(np.float32)[..., None] / 255.0
    edited = np.rint(
        raw.astype(np.float32) * weight
        + source.astype(np.float32) * (1.0 - weight)
    )
    edited = np.clip(edited, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(edited_path), edited):
        raise RuntimeError(f"Failed to write {edited_path}")

    difference = np.abs(edited.astype(np.int16) - source.astype(np.int16))
    protected = alpha == 0
    outside_max = int(difference[protected].max(initial=0))
    if outside_max != 0:
        raise AssertionError(f"Exact protection failed: {outside_max}")
    feather = np.logical_and(alpha > 0, alpha < 255)
    return {
        "decoded_frames": len(frames),
        "selected_frame_index": selected_frame_index,
        "raw_selected_frame": raw_path,
        "edited_2d": edited_path,
        "outside_edit_alpha_max_pixel_difference": outside_max,
        "outside_edit_alpha_rgb_mae": float(difference[protected].mean()),
        "feather_band_rgb_mae": (
            float(difference[feather].mean()) if feather.any() else None
        ),
    }


def build_manifest(
    args: argparse.Namespace,
    anchor_id: str,
    proxy_dir: Path,
    model_files: list[dict[str, object]],
    code_files: list[dict[str, object]],
    worker_record: dict[str, object],
) -> dict[str, object]:
    with Image.open(proxy_dir / "first_frame.png") as image:
        width, height = image.size
    return {
        "schema_version": "foodstateedit.run.v1",
        "run_id": f"{anchor_id}__{METHOD}__seed_{args.seed}",
        "case_id": anchor_id,
        "method": METHOD,
        "seed": args.seed,
        "model": {
            "name": "Wan2.2-VACE-Fun-A14B",
            "root": str(args.model_root.resolve()),
            "files": model_files,
            "verified_before_worker_start": True,
            "full_file_hash_audit": str(args.model_hash_audit.resolve()),
            "full_file_hash_audit_sha256": sha256_file(args.model_hash_audit.resolve()),
        },
        "inference": {
            "width": width,
            "height": height,
            "frames": args.num_frames,
            "steps": args.steps,
            "vace_scale": args.vace_scale,
            "vace_video": "dynamic_control.mp4",
            "vace_video_mask": "dynamic_mask.mp4",
            "vace_reference_image": "first_frame.png",
            "enable_ttm": False,
            "selected_frame_index": args.selected_frame_index,
            "selection_policy": "fixed_before_inference",
            "dynamic_proxy_manifest_sha256": sha256_file(
                proxy_dir / "proxy_manifest.json"
            ),
            "exact_protection_projection": True,
            "resident_pipeline": True,
        },
        "environment": {
            "host": socket.gethostname(),
            "gpu": f"NVIDIA RTX A6000; CUDA_VISIBLE_DEVICES={args.gpu}",
            "python": str(Path(sys.executable).resolve()),
            "code_commit": args.code_commit,
            "created_at": now_jst(),
            "geoedit_files": code_files,
            "worker_file": worker_record,
        },
        "status": "running",
        "outputs": [],
    }


def main() -> None:
    args = parse_args()
    if len(args.anchor) != len(set(args.anchor)):
        raise ValueError("Duplicate --anchor values are not allowed")
    if (
        args.seed != SEED
        or args.num_frames != NUM_FRAMES
        or args.steps != NUM_INFERENCE_STEPS
        or args.vace_scale != VACE_SCALE
        or args.selected_frame_index != SELECTED_FRAME_INDEX
    ):
        raise ValueError("Worker arguments differ from frozen dynamic-multikey v1")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse output root: {output_root}")
    proxy_root = args.proxy_root.resolve()
    proxies = {}
    for anchor_id in args.anchor:
        proxy_dir = proxy_root / anchor_id
        proxy = verify_dynamic_proxy(proxy_dir)
        if proxy["anchor_id"] != anchor_id:
            raise ValueError(f"Proxy identity mismatch: {proxy_dir}")
        proxies[anchor_id] = proxy_dir

    model_files = verify_model_audit(
        args.model_root.resolve(), args.model_hash_audit.resolve()
    )
    geoedit_root = args.geoedit_root.resolve()
    code_files = [
        verify_hash(geoedit_root / relative, expected, f"GeoEdit v2 file {relative}")
        for relative, expected in EXPECTED_GEOEDIT_FILES.items()
    ]
    worker_path = Path(__file__).resolve()
    worker_record = {
        "path": str(worker_path),
        "size_bytes": worker_path.stat().st_size,
        "sha256": sha256_file(worker_path),
    }

    os.environ.update({
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "DIFFSYNTH_SKIP_DOWNLOAD": "true",
        "DIFFSYNTH_MODEL_BASE_PATH": str(args.model_root.parent.parent.resolve()),
        "HF_HOME": str(args.cache_root.resolve()),
        "TOKENIZERS_PARALLELISM": "false",
    })
    args.cache_root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(geoedit_root))
    os.chdir(geoedit_root)
    from geoedit import inference

    original_load_pipeline = inference.load_pipeline
    pipeline_cache: dict[str, object] = {}
    pipeline_load_count = 0

    def resident_load_pipeline(vram_limit: float):
        nonlocal pipeline_load_count
        if "pipeline" not in pipeline_cache:
            pipeline_cache["pipeline"] = original_load_pipeline(vram_limit)
            pipeline_load_count += 1
        return pipeline_cache["pipeline"]

    output_root.mkdir(parents=True, exist_ok=False)
    results = []
    for anchor_id in args.anchor:
        proxy_dir = proxies[anchor_id]
        run_dir = output_root / anchor_id / f"seed_{args.seed}"
        run_dir.mkdir(parents=True, exist_ok=False)
        output_video = run_dir / "result.mp4"
        manifest = build_manifest(
            args, anchor_id, proxy_dir, model_files, code_files, worker_record
        )
        write_json(run_dir / "run_manifest.json", manifest)
        write_json(run_dir / "command.json", {
            "call": "WanVideoPipeline.__call__",
            "vace_video": str((proxy_dir / "dynamic_control.mp4").resolve()),
            "vace_video_mask": str((proxy_dir / "dynamic_mask.mp4").resolve()),
            "vace_reference_image": str((proxy_dir / "first_frame.png").resolve()),
            "enable_ttm": False,
            "vace_scale": args.vace_scale,
            "num_frames": args.num_frames,
            "num_inference_steps": args.steps,
            "seed": args.seed,
            "selected_frame_index": args.selected_frame_index,
        })
        (run_dir / "RUNNING").write_text(now_jst() + "\n", encoding="utf-8")
        started = time.monotonic()
        try:
            pipe = resident_load_pipeline(inference.resolve_vram_limit(None))
            with Image.open(proxy_dir / "first_frame.png") as image:
                reference_image = image.convert("RGB")
                width, height = reference_image.size
            motion_video = inference.load_video_or_image(
                proxy_dir / "dynamic_control.mp4", height, width, args.num_frames
            )
            edit_mask = inference.load_video_or_image(
                proxy_dir / "dynamic_mask.mp4", height, width, args.num_frames
            )
            prompt = read_noncomment_text(proxy_dir / "prompt.txt")
            negative_prompt = read_noncomment_text(proxy_dir / "negative_prompt.txt")
            with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout, \
                 (run_dir / "stderr.log").open("w", encoding="utf-8") as stderr, \
                 contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                video = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_frames=args.num_frames,
                    num_inference_steps=args.steps,
                    vace_video=motion_video,
                    vace_video_mask=edit_mask,
                    vace_reference_image=reference_image,
                    vace_scale=args.vace_scale,
                    enable_ttm=False,
                    seed=args.seed,
                    tiled=True,
                )
                inference.save_video(video, str(output_video), fps=15, quality=5)
            projected = extract_selected_and_project(
                output_video, proxy_dir, run_dir, args.selected_frame_index
            )
            manifest["inference"].update({
                "decoded_frames": projected["decoded_frames"],
                "outside_edit_alpha_max_pixel_difference": projected[
                    "outside_edit_alpha_max_pixel_difference"
                ],
                "outside_edit_alpha_rgb_mae": projected["outside_edit_alpha_rgb_mae"],
                "feather_band_rgb_mae": projected["feather_band_rgb_mae"],
                "pipeline_load_count_at_completion": pipeline_load_count,
                "wall_time_seconds": round(time.monotonic() - started, 3),
            })
            output_paths = [
                output_video,
                projected["raw_selected_frame"],
                projected["edited_2d"],
            ]
            manifest["outputs"] = [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in output_paths
            ]
            manifest["status"] = "complete"
            write_json(run_dir / "run_manifest.json", manifest)
            (run_dir / "RUNNING").unlink()
            (run_dir / "COMPLETE").write_text(now_jst() + "\n", encoding="utf-8")
        except Exception as error:
            manifest["status"] = "technical_failure"
            manifest["outputs"] = []
            manifest["inference"]["pipeline_load_count_at_failure"] = pipeline_load_count
            manifest["inference"]["wall_time_seconds"] = round(
                time.monotonic() - started, 3
            )
            manifest["environment"]["error"] = f"{type(error).__name__}: {error}"
            write_json(run_dir / "run_manifest.json", manifest)
            (run_dir / "RUNNING").unlink(missing_ok=True)
            (run_dir / "FAILED").write_text(
                f"{type(error).__name__}: {error}\n", encoding="utf-8"
            )
        results.append({
            "anchor_id": anchor_id,
            "status": manifest["status"],
            "run_manifest": str((run_dir / "run_manifest.json").resolve()),
            "wall_time_seconds": manifest["inference"]["wall_time_seconds"],
        })

    summary = {
        "schema_version": 1,
        "method": METHOD,
        "seed": args.seed,
        "selected_frame_index": args.selected_frame_index,
        "worker_gpu": args.gpu,
        "case_count": len(results),
        "complete_count": sum(item["status"] == "complete" for item in results),
        "technical_failure_count": sum(
            item["status"] == "technical_failure" for item in results
        ),
        "pipeline_load_count": pipeline_load_count,
        "resident_contract_passed": pipeline_load_count == 1,
        "runs": results,
    }
    write_json(output_root / "worker_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if summary["technical_failure_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
