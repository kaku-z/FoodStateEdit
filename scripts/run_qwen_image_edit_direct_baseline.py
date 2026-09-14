#!/usr/bin/env python3
"""Run the frozen Qwen-Image-Edit-2511 direct-input development baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SCHEMA = "foodstateedit.qwen_image_edit_direct_baseline.v1"
RUN_SCHEMA = "foodstateedit.qwen_image_edit_direct_baseline_run.v1"
CASE_ORDER = ("ramen", "soup", "rice", "cake")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def command_lines(command: list[str]) -> list[str]:
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return [line for line in completed.stdout.splitlines() if line.strip()]


def gpu_snapshot(physical_gpu: int) -> dict[str, Any]:
    rows = command_lines([
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    parsed: dict[int, dict[str, Any]] = {}
    for row in rows:
        fields = [item.strip() for item in row.split(",")]
        index = int(fields[0])
        parsed[index] = {
            "index": index,
            "name": fields[1],
            "uuid": fields[2],
            "memory_total_mib": int(fields[3]),
            "memory_free_mib": int(fields[4]),
            "utilization_percent": int(fields[5]),
        }
    selected = parsed[physical_gpu]
    process_rows = command_lines([
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ])
    selected["compute_processes"] = [
        row for row in process_rows if row.split(",", 1)[0].strip() == selected["uuid"]
    ]
    available_kib = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
        if line.startswith("MemAvailable:")
    )
    selected["host_mem_available_mib"] = available_kib // 1024
    return selected


def gate_snapshot(snapshot: dict[str, Any], gate: dict[str, Any]) -> dict[str, bool]:
    return {
        "gpu_name": snapshot["name"] == gate["required_gpu_name"],
        "gpu_free_memory": snapshot["memory_free_mib"] >= gate["min_free_memory_mib"],
        "gpu_utilization": snapshot["utilization_percent"] <= gate["max_utilization_percent"],
        "zero_compute_processes": not snapshot["compute_processes"],
        "host_available_memory": snapshot["host_mem_available_mib"] >= gate["min_available_system_memory_mib"],
    }


def foreign_process_check(gpu_uuid: str) -> None:
    rows = command_lines([
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ])
    foreign = []
    for row in rows:
        fields = [item.strip() for item in row.split(",")]
        if len(fields) >= 2 and fields[0] == gpu_uuid and int(fields[1]) != os.getpid():
            foreign.append(row)
    if foreign:
        raise RuntimeError(
            "Foreign compute process appeared; preserve failure without preemption: "
            + repr(foreign)
        )


def support_locked_composite(
    base: Image.Image,
    edited: Image.Image,
    allowed: Image.Image,
) -> tuple[Image.Image, int]:
    base_arr = np.asarray(base.convert("RGB"), dtype=np.uint8)
    candidate = np.asarray(
        edited.convert("RGB").resize(base.size, Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    alpha_u8 = np.asarray(
        allowed.convert("L").resize(base.size, Image.Resampling.NEAREST),
        dtype=np.uint8,
    )
    alpha = alpha_u8.astype(np.float32) / 255.0
    mixed = np.rint(candidate * alpha[..., None] + base_arr.astype(np.float32) * (1.0 - alpha[..., None]))
    final = np.clip(mixed, 0, 255).astype(np.uint8)
    outside = alpha_u8 == 0
    maximum = int(
        np.abs(final.astype(np.int16) - base_arr.astype(np.int16))[outside].max(initial=0)
    )
    if maximum != 0:
        raise AssertionError("Qwen baseline changed protected pixels")
    return Image.fromarray(final), maximum


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA or not config.get("execution_allowed"):
        raise ValueError("Unexpected or disabled Qwen baseline config")
    if tuple(config["case_order"]) != CASE_ORDER:
        raise ValueError("Frozen case order changed")
    if sha256_file(Path(__file__)) != config["implementation"]["runner_sha256"]:
        raise ValueError("Runner hash mismatch")
    if args.output_root.resolve() != Path(config["output_root"]).resolve():
        raise ValueError("Output root differs from frozen path")
    if args.preflight.resolve() != Path(config["preflight_path"]).resolve():
        raise ValueError("Preflight path differs from frozen path")
    if args.output_root.exists() or args.preflight.exists():
        raise FileExistsError("Refusing to overwrite output or preflight evidence")

    backend = config["backend"]
    model_root = Path(backend["model_root"])
    path_checks: dict[str, bool] = {
        "model_root": model_root.is_dir(),
        "model_index": (model_root / "model_index.json").is_file(),
    }
    for case_id in CASE_ORDER:
        case = config["cases"][case_id]
        input_path = Path(case["input_image"])
        allowed_path = Path(case["allowed_edit_region"])
        path_checks[f"{case_id}_input"] = input_path.is_file() and sha256_file(input_path) == case["input_sha256"]
        path_checks[f"{case_id}_allowed"] = allowed_path.is_file() and sha256_file(allowed_path) == case["allowed_edit_region_sha256"]
    weight_checks: dict[str, bool] = {}
    for relative, expected in backend["weight_files"].items():
        path = model_root / relative
        weight_checks[relative] = (
            path.is_file()
            and path.stat().st_size == expected["size_bytes"]
            and sha256_file(path) == expected["sha256"]
        )

    snapshot = gpu_snapshot(args.physical_gpu)
    resource_checks = gate_snapshot(snapshot, config["resource_gate"])
    preflight = {
        "schema_version": "foodstateedit.qwen_image_edit_direct_baseline_preflight.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "physical_gpu": args.physical_gpu,
        "gpu": snapshot,
        "config_sha256": sha256_file(args.config),
        "checks": {**resource_checks, **path_checks, **weight_checks},
    }
    preflight["passed"] = all(preflight["checks"].values())
    args.preflight.parent.mkdir(parents=True, exist_ok=True)
    with args.preflight.open("x", encoding="utf-8") as stream:
        json.dump(preflight, stream, indent=2)
        stream.write("\n")
    if not preflight["passed"]:
        raise RuntimeError("Preflight failed; no sample output was created")

    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.physical_gpu),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        TORCH_COMPILE_DISABLE="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    import torch  # noqa: E402
    from diffusers import QwenImageEditPlusPipeline  # noqa: E402

    args.output_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest: dict[str, Any] = {
        "schema_version": RUN_SCHEMA,
        "status": "loading_pipeline",
        "host": socket.gethostname(),
        "physical_gpu": args.physical_gpu,
        "gpu_uuid": snapshot["uuid"],
        "config_sha256": sha256_file(args.config),
        "runner_sha256": sha256_file(Path(__file__)),
        "pipeline_load_count": 0,
        "expected_cases": list(CASE_ORDER),
        "expected_seeds": config["seeds"],
        "completed_outputs": [],
        "claim_limit": config["claim_limit"],
    }
    write_json(args.output_root / "run_manifest.json", manifest)

    try:
        foreign_process_check(snapshot["uuid"])
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            str(model_root),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=False)
        manifest["pipeline_load_count"] = 1
        write_json(args.output_root / "run_manifest.json", manifest)

        inference = config["inference"]
        for case_id in CASE_ORDER:
            case = config["cases"][case_id]
            base = Image.open(case["input_image"]).convert("RGB")
            allowed = Image.open(case["allowed_edit_region"]).convert("L")
            case_root = args.output_root / case_id
            case_root.mkdir(parents=True, exist_ok=False)
            for seed in config["seeds"]:
                foreign_process_check(snapshot["uuid"])
                seed_root = case_root / f"seed_{seed}"
                seed_root.mkdir(parents=True, exist_ok=False)
                begin = time.perf_counter()
                generator = torch.Generator(device="cuda").manual_seed(int(seed))
                result = pipe(
                    image=base,
                    prompt=case["instruction"],
                    negative_prompt=inference["negative_prompt"],
                    num_inference_steps=int(inference["num_inference_steps"]),
                    true_cfg_scale=float(inference["true_cfg_scale"]),
                    guidance_scale=float(inference["guidance_scale"]),
                    generator=generator,
                ).images[0]
                result.save(seed_root / "raw_qwen.png")
                final, outside_max = support_locked_composite(base, result, allowed)
                base.save(seed_root / "input.png")
                allowed.resize(base.size, Image.Resampling.NEAREST).save(seed_root / "allowed_edit_region.png")
                final.save(seed_root / "final.png")
                record = {
                    "case_id": case_id,
                    "seed": int(seed),
                    "status": "complete_requires_blind_review",
                    "outside_support_max_pixel_difference": outside_max,
                    "wall_time_seconds": round(time.perf_counter() - begin, 3),
                    "files": {
                        path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                        for path in seed_root.iterdir()
                        if path.is_file()
                    },
                }
                write_json(seed_root / "condition_manifest.json", record)
                manifest["status"] = "running"
                manifest["completed_outputs"].append(record)
                write_json(args.output_root / "run_manifest.json", manifest)

        manifest["status"] = "complete_requires_blind_review"
        (args.output_root / "COMPLETE").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
        code = 0
    except Exception as exc:
        manifest.update(status="technical_failure_preserved", error=repr(exc))
        (args.output_root / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        (args.output_root / "FAILED").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
        code = 3

    manifest["wall_time_seconds"] = round(time.perf_counter() - started, 3)
    write_json(args.output_root / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
