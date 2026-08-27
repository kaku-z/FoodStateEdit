#!/usr/bin/env python3
"""Run one frozen GeoEdit anchor smoke test without downloading any model.

The launcher is intentionally single-run so several anchors can be queued on
different GPUs without sharing writable run state.  It verifies the frozen
proxy, model, and GeoEdit override hashes before starting, refuses to overwrite
an existing run, and retains a technical-failure manifest and logs on error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np


MODEL_FILES = {
    "high_noise_model/diffusion_pytorch_model.safetensors":
        "66c61b736c5674deeeef17861e494d3652cc9b1463a9656bf18c2c72d2c5f007",
    "low_noise_model/diffusion_pytorch_model.safetensors":
        "0bf791adfb8330d451d2f5c03577b2a8fb780453f8ef05e23a8fa91f27a2134d",
    "models_t5_umt5-xxl-enc-bf16.pth":
        "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d",
    "Wan2.1_VAE.pth":
        "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981",
}

MODEL_SIZES = {
    "high_noise_model/diffusion_pytorch_model.safetensors": 34_675_325_000,
    "low_noise_model/diffusion_pytorch_model.safetensors": 34_675_325_000,
    "models_t5_umt5-xxl-enc-bf16.pth": 11_361_920_418,
    "Wan2.1_VAE.pth": 507_609_880,
}

GEOEDIT_FILES = {
    "geoedit/inference.py":
        "f78488216536e70744e839c0294319f334d541adc36f9e76136c14e7ffd328fa",
    "diffsynth/pipelines/wan_video.py":
        "28a1b5e3939983139e2dcd4a7f19d085fcec0017434793bac163767b7ae29c00",
    "diffsynth/utils/data/__init__.py":
        "fe056b4a675a345cf02d6c76d327e8434103ccf2a4066ff208ce41368771e360",
    "tests/test_masks.py":
        "57176ea2595f236846e1d38a79f58e3d586a99e67ce7ef0df4747af768a982da",
}

METHODS = {
    "vanilla_geoedit": {
        "mask": "mask.png",
        "replace_mode": "non_hole",
        "warm_start": True,
        "schedule": {"rigid": 15, "contact": 15, "material": 15, "hole": 15},
    },
    "geoedit_unified_action_mask": {
        "mask": "edit_alpha.png",
        "replace_mode": "mask_new",
        "warm_start": False,
        "schedule": {"rigid": 15, "contact": 15, "material": 15, "hole": 15},
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path: Path, expected: str, label: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {path}: {actual} != {expected}")
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": actual}


def verify_proxy(proxy_dir: Path) -> dict[str, object]:
    manifest_path = proxy_dir / "proxy_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "foodstateedit.common_proxy.v1":
        raise ValueError(f"Unexpected proxy schema: {manifest_path}")
    if manifest.get("status") not in {
        "provisional_geometry_proxy_requires_visual_confirmation",
        "frozen_geometry_proxy_visual_confirmation_passed",
    }:
        raise ValueError(f"Proxy is not eligible for smoke inference: {manifest_path}")
    for name, item in manifest["files"].items():
        local_path = proxy_dir / name
        verify_hash(local_path, item["sha256"], f"proxy file {name}")
    return manifest


def verify_model_audit(model_root: Path, audit_path: Path) -> list[dict[str, object]]:
    """Reuse the one full-file hash audit instead of rereading 81 GiB per run."""
    if not audit_path.is_file():
        raise FileNotFoundError(f"Missing model hash audit: {audit_path}")
    audited: dict[str, str] = {}
    for line in audit_path.read_text(encoding="utf-8").splitlines():
        digest, path_text = line.split(maxsplit=1)
        audited[str(Path(path_text).resolve())] = digest
    files = []
    for relative, expected in MODEL_FILES.items():
        path = (model_root / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing model file: {path}")
        if path.stat().st_size != MODEL_SIZES[relative]:
            raise ValueError(f"Model size changed after hash audit: {path}")
        if audited.get(str(path)) != expected:
            raise ValueError(f"Model is absent or mismatched in frozen hash audit: {path}")
        files.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": expected})
    return files


def extract_and_project(video_path: Path, proxy_dir: Path, run_dir: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"No decodable frames in {video_path}")

    raw = frames[-1]
    source = cv2.imread(str(proxy_dir / "first_frame.png"), cv2.IMREAD_COLOR)
    alpha = cv2.imread(str(proxy_dir / "edit_alpha.png"), cv2.IMREAD_GRAYSCALE)
    if source is None or alpha is None:
        raise RuntimeError("Failed to read first_frame.png or edit_alpha.png")
    height, width = source.shape[:2]
    if raw.shape[:2] != (height, width):
        raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LANCZOS4)
    if alpha.shape != (height, width):
        alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)

    raw_path = run_dir / "raw_last_frame.png"
    edited_path = run_dir / "edited_2d.png"
    if not cv2.imwrite(str(raw_path), raw):
        raise RuntimeError(f"Failed to write {raw_path}")
    weight = alpha.astype(np.float32)[..., None] / 255.0
    edited = np.rint(raw.astype(np.float32) * weight + source.astype(np.float32) * (1.0 - weight))
    edited = np.clip(edited, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(edited_path), edited):
        raise RuntimeError(f"Failed to write {edited_path}")

    protected = alpha == 0
    outside_max = int(np.abs(edited.astype(np.int16) - source.astype(np.int16))[protected].max())
    if outside_max != 0:
        raise AssertionError(f"Exact protection projection failed: max difference {outside_max}")
    return {
        "decoded_frames": len(frames),
        "raw_last_frame": raw_path,
        "edited_2d": edited_path,
        "outside_edit_alpha_max_pixel_difference": outside_max,
    }


def write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--proxy-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--geoedit-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-hash-audit", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=21)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--tweak-index", type=int, default=3)
    parser.add_argument("--tstrong-index", type=int, default=15)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = METHODS[args.method]
    proxy_dir = args.proxy_dir.resolve()
    proxy = verify_proxy(proxy_dir)
    anchor_id = str(proxy["anchor_id"])
    model_files = verify_model_audit(args.model_root.resolve(), args.model_hash_audit.resolve())
    code_files = [
        verify_hash(args.geoedit_root / relative, expected, f"GeoEdit file {relative}")
        for relative, expected in GEOEDIT_FILES.items()
    ]
    run_dir = args.output_root.resolve() / anchor_id / f"seed_{args.seed}"
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    source = cv2.imread(str(proxy_dir / "first_frame.png"), cv2.IMREAD_COLOR)
    if source is None:
        raise RuntimeError(f"Cannot read proxy first frame: {proxy_dir}")
    height, width = source.shape[:2]

    output_video = run_dir / "result.mp4"
    command = [
        str(args.python), "-m", "geoedit.inference",
        "--input-dir", str(proxy_dir),
        "--output", str(output_video),
        "--motion-signal-mask", str(proxy_dir / config["mask"]),
        "--tweak-index", str(args.tweak_index),
        "--tstrong-index", str(args.tstrong_index),
        "--replace-mode", str(config["replace_mode"]),
        "--num-frames", str(args.num_frames),
        "--num-inference-steps", str(args.steps),
        "--seed", str(args.seed),
    ]
    if config["replace_mode"] == "non_hole":
        command.extend(["--mask-old", str(proxy_dir / "mask_old.png")])
    if not config["warm_start"]:
        command.append("--no-warm-start")

    now = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds")
    run_id = f"{anchor_id}__{args.method}__seed_{args.seed}"
    manifest: dict[str, object] = {
        "schema_version": "foodstateedit.run.v1",
        "run_id": run_id,
        "case_id": anchor_id,
        "method": args.method,
        "seed": args.seed,
        "model": {
            "name": "Wan2.2-VACE-Fun-A14B",
            "root": str(args.model_root.resolve()),
            "files": model_files,
            "verified_before_run": True,
            "full_file_hash_audit": str(args.model_hash_audit.resolve()),
            "full_file_hash_audit_sha256": sha256_file(args.model_hash_audit.resolve()),
        },
        "inference": {
            "width": width,
            "height": height,
            "frames": args.num_frames,
            "steps": args.steps,
            "warm_start": bool(config["warm_start"]),
            "schedule": config["schedule"],
            "tweak_index": args.tweak_index,
            "tstrong_index": args.tstrong_index,
            "replace_mode": config["replace_mode"],
            "motion_signal_mask": config["mask"],
            "common_proxy_manifest_sha256": sha256_file(proxy_dir / "proxy_manifest.json"),
            "exact_protection_projection": True,
        },
        "environment": {
            "host": socket.gethostname(),
            "gpu": f"NVIDIA RTX A6000; CUDA_VISIBLE_DEVICES={args.gpu}",
            "python": str(args.python.resolve()),
            "code_commit": args.code_commit,
            "created_at": now,
            "geoedit_files": code_files,
        },
        "status": "running",
        "outputs": [],
    }
    write_json(run_dir / "run_manifest.json", manifest)
    write_json(run_dir / "command.json", {"argv": command})
    (run_dir / "RUNNING").write_text(now + "\n", encoding="utf-8")

    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "DIFFSYNTH_SKIP_DOWNLOAD": "true",
        "DIFFSYNTH_MODEL_BASE_PATH": str(args.model_root.parent.parent.resolve()),
        "HF_HOME": str(args.cache_root.resolve()),
        "TRANSFORMERS_CACHE": str((args.cache_root / "transformers").resolve()),
        "TOKENIZERS_PARALLELISM": "false",
    })
    args.cache_root.mkdir(parents=True, exist_ok=True)
    (args.cache_root / "transformers").mkdir(parents=True, exist_ok=True)

    try:
        with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout, \
             (run_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                command,
                cwd=args.geoedit_root,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"GeoEdit exited with status {completed.returncode}")
        projected = extract_and_project(output_video, proxy_dir, run_dir)
        manifest["inference"]["decoded_frames"] = projected["decoded_frames"]
        manifest["inference"]["outside_edit_alpha_max_pixel_difference"] = projected[
            "outside_edit_alpha_max_pixel_difference"
        ]
        output_paths = [output_video, projected["raw_last_frame"], projected["edited_2d"]]
        manifest["outputs"] = [
            {"path": str(path), "sha256": sha256_file(path)} for path in output_paths
        ]
        manifest["status"] = "complete"
        write_json(run_dir / "run_manifest.json", manifest)
        (run_dir / "RUNNING").unlink()
        (run_dir / "COMPLETE").write_text(
            datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds") + "\n",
            encoding="utf-8",
        )
        print(f"COMPLETE {run_id}", flush=True)
    except Exception as error:
        manifest["status"] = "technical_failure"
        manifest["outputs"] = []
        manifest["environment"]["error"] = f"{type(error).__name__}: {error}"
        write_json(run_dir / "run_manifest.json", manifest)
        (run_dir / "RUNNING").unlink(missing_ok=True)
        (run_dir / "FAILED").write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
