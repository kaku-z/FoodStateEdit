#!/usr/bin/env python3
"""Run one frozen Qwen-Image-Edit comparison on the Day10 soup input."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from run_qwen_image_edit_direct_baseline import (
    foreign_process_check,
    gate_snapshot,
    gpu_snapshot,
    support_locked_composite,
)


SCHEMA = "foodstateedit.day34_single_qwen_soup_baseline.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA or not config.get("execution_allowed"):
        raise ValueError("Unexpected or disabled config")
    if sha256_file(Path(__file__)) != config["implementation"]["runner_sha256"]:
        raise ValueError("Runner hash mismatch")
    if args.output_root.resolve() != Path(config["output_root"]).resolve():
        raise ValueError("Output root differs from frozen config")
    if args.preflight.resolve() != Path(config["preflight_path"]).resolve():
        raise ValueError("Preflight path differs from frozen config")
    if args.output_root.exists() or args.preflight.exists():
        raise FileExistsError("Refusing to overwrite output or preflight evidence")

    sample = config["sample"]
    backend = config["backend"]
    input_path = Path(sample["input_image"])
    mask_path = Path(sample["allowed_edit_region"])
    model_root = Path(backend["model_root"])
    checks = {
        "input": input_path.is_file() and sha256_file(input_path) == sample["input_sha256"],
        "allowed_edit_region": mask_path.is_file() and sha256_file(mask_path) == sample["allowed_edit_region_sha256"],
        "model_index": (model_root / "model_index.json").is_file(),
    }
    for relative, expected in backend["weight_files"].items():
        path = model_root / relative
        checks[f"weight:{relative}"] = (
            path.is_file()
            and path.stat().st_size == expected["size_bytes"]
            and sha256_file(path) == expected["sha256"]
        )

    snapshot = gpu_snapshot(args.physical_gpu)
    checks.update(gate_snapshot(snapshot, config["resource_gate"]))
    preflight = {
        "schema_version": "foodstateedit.day34_single_qwen_soup_preflight.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "physical_gpu": args.physical_gpu,
        "gpu": snapshot,
        "config_sha256": sha256_file(args.config),
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.preflight.parent.mkdir(parents=True, exist_ok=True)
    with args.preflight.open("x", encoding="utf-8") as stream:
        json.dump(preflight, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if not preflight["passed"]:
        raise RuntimeError("Preflight failed; no output directory created")

    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.physical_gpu),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        TORCH_COMPILE_DISABLE="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    import torch
    from diffusers import QwenImageEditPlusPipeline

    args.output_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {
        "schema_version": "foodstateedit.day34_single_qwen_soup_run.v1",
        "status": "loading_pipeline",
        "host": socket.gethostname(),
        "physical_gpu": args.physical_gpu,
        "gpu_uuid": snapshot["uuid"],
        "config_sha256": sha256_file(args.config),
        "runner_sha256": sha256_file(Path(__file__)),
        "pipeline_load_count": 0,
        "claim_limit": config["claim_limit"],
    }
    write_json(args.output_root / "run_manifest.json", manifest)

    try:
        foreign_process_check(snapshot["uuid"])
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            str(model_root), torch_dtype=torch.bfloat16, local_files_only=True
        )
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=False)
        manifest["pipeline_load_count"] = 1
        write_json(args.output_root / "run_manifest.json", manifest)

        base = Image.open(input_path).convert("RGB")
        allowed = Image.open(mask_path).convert("L")
        seed = int(config["seed"])
        inference = config["inference"]
        foreign_process_check(snapshot["uuid"])
        generator = torch.Generator(device="cuda").manual_seed(seed)
        result = pipe(
            image=base,
            prompt=sample["instruction"],
            negative_prompt=inference["negative_prompt"],
            num_inference_steps=int(inference["num_inference_steps"]),
            true_cfg_scale=float(inference["true_cfg_scale"]),
            guidance_scale=float(inference["guidance_scale"]),
            generator=generator,
        ).images[0]
        result.save(args.output_root / "raw_qwen.png")
        final, outside_max = support_locked_composite(base, result, allowed)
        base.save(args.output_root / "input.png")
        allowed.resize(base.size, Image.Resampling.NEAREST).save(args.output_root / "allowed_edit_region.png")
        final.save(args.output_root / "final.png")
        manifest.update(
            status="complete_requires_review",
            seed=seed,
            outside_support_max_pixel_difference=outside_max,
            files={
                path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in args.output_root.iterdir()
                if path.is_file() and path.name != "run_manifest.json"
            },
        )
        (args.output_root / "COMPLETE").write_text(
            datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
        )
        code = 0
    except Exception as exc:
        manifest.update(status="technical_failure_preserved", error=repr(exc))
        (args.output_root / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        (args.output_root / "FAILED").write_text(
            datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
        )
        code = 3

    manifest["wall_time_seconds"] = round(time.perf_counter() - started, 3)
    write_json(args.output_root / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
