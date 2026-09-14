#!/usr/bin/env python3
"""Run a frozen same-input ChordEdit direct-edit baseline for one food case."""
from __future__ import annotations

import argparse
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
from typing import Any

import numpy as np
from PIL import Image


SCHEMA = "foodstateedit.chordedit_direct_baseline.v1"
RUN_SCHEMA = "foodstateedit.chordedit_direct_baseline_run.v1"
CASE_ORDER = ("ramen", "soup", "rice", "cake")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


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


def square_crop_box(mask: Image.Image, padding: int) -> tuple[int, int, int, int]:
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("Allowed edit region is empty")
    width, height = mask.size
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0 - padding), max(0, y0 - padding)
    x1, y1 = min(width, x1 + padding), min(height, y1 + padding)
    side = min(max(x1 - x0, y1 - y0), width, height)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    sx0 = max(0, min(int(round(cx - side / 2.0)), width - side))
    sy0 = max(0, min(int(round(cy - side / 2.0)), height - side))
    return sx0, sy0, sx0 + side, sy0 + side


def support_locked_composite(
    base: Image.Image,
    edited_crop: Image.Image,
    crop_box: tuple[int, int, int, int],
    allowed: Image.Image,
) -> tuple[Image.Image, int]:
    base_arr = np.asarray(base.convert("RGB"), dtype=np.uint8)
    candidate = base_arr.copy()
    x0, y0, x1, y1 = crop_box
    edited = np.asarray(
        edited_crop.convert("RGB").resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    source = base_arr[y0:y1, x0:x1].astype(np.float32)
    alpha_full = np.asarray(allowed.convert("L"), dtype=np.uint8)
    alpha = alpha_full[y0:y1, x0:x1].astype(np.float32) / 255.0
    mixed = np.rint(edited * alpha[..., None] + source * (1.0 - alpha[..., None]))
    candidate[y0:y1, x0:x1] = np.clip(mixed, 0, 255).astype(np.uint8)
    outside = alpha_full == 0
    maximum = int(
        np.abs(candidate.astype(np.int16) - base_arr.astype(np.int16))[outside].max(initial=0)
    )
    if maximum != 0:
        raise AssertionError("Direct baseline changed protected pixels")
    return Image.fromarray(candidate), maximum


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case-id", choices=CASE_ORDER, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA or not config.get("execution_allowed"):
        raise ValueError("Unexpected or disabled direct-baseline config")
    if tuple(config["case_order"]) != CASE_ORDER:
        raise ValueError("Frozen case order changed")
    if sha256_file(Path(__file__)) != config["implementation"]["runner_sha256"]:
        raise ValueError("Runner hash mismatch")
    expected_output = Path(config["output_root_template"].format(case_id=args.case_id))
    expected_preflight = Path(config["preflight_template"].format(case_id=args.case_id))
    if args.output_root.resolve() != expected_output.resolve():
        raise ValueError("Output root differs from frozen path")
    if args.preflight.resolve() != expected_preflight.resolve():
        raise ValueError("Preflight path differs from frozen path")
    if args.output_root.exists() or args.preflight.exists():
        raise FileExistsError("Refusing to overwrite output or preflight evidence")

    backend = config["backend"]
    case = config["cases"][args.case_id]
    paths = {
        "input": Path(case["input_image"]),
        "allowed": Path(case["allowed_edit_region"]),
        "pipeline": Path(backend["project_root"]) / "pipeline_chord.py",
        "model_index": Path(backend["model_root"]) / "model_index.json",
    }
    path_checks = {name: path.is_file() for name, path in paths.items()}
    path_checks["input_sha256"] = (
        paths["input"].is_file()
        and sha256_file(paths["input"]) == case["input_sha256"]
    )
    path_checks["allowed_edit_region_sha256"] = (
        paths["allowed"].is_file()
        and sha256_file(paths["allowed"]) == case["allowed_edit_region_sha256"]
    )
    weight_checks = {}
    for relative, expected in backend["weight_files"].items():
        path = Path(backend["model_root"]) / relative
        weight_checks[relative] = (
            path.is_file()
            and path.stat().st_size == expected["size_bytes"]
            and sha256_file(path) == expected["sha256"]
        )
    snapshot = gpu_snapshot(args.physical_gpu)
    resource_checks = gate_snapshot(snapshot, config["resource_gate"])
    preflight = {
        "schema_version": "foodstateedit.chordedit_direct_baseline_preflight.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "physical_gpu": args.physical_gpu,
        "gpu": snapshot,
        "config_sha256": sha256_file(args.config),
        "checks": {**resource_checks, **path_checks, **weight_checks},
    }
    preflight["passed"] = all(preflight["checks"].values())
    args.preflight.parent.mkdir(parents=True, exist_ok=True)
    with args.preflight.open("x", encoding="utf-8") as handle:
        json.dump(preflight, handle, indent=2)
        handle.write("\n")
    if not preflight["passed"]:
        raise RuntimeError("Preflight failed; no sample output was created")

    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.physical_gpu),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    sys.path.insert(0, backend["project_root"])
    import torch  # noqa: E402
    from pipeline_chord import ChordEditPipeline  # noqa: E402

    args.output_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {
        "schema_version": RUN_SCHEMA,
        "status": "loading_pipeline",
        "case_id": args.case_id,
        "host": socket.gethostname(),
        "physical_gpu": args.physical_gpu,
        "gpu_uuid": snapshot["uuid"],
        "config_sha256": sha256_file(args.config),
        "runner_sha256": sha256_file(Path(__file__)),
        "pipeline_load_count": 0,
        "expected_seeds": config["seeds"],
        "completed_seeds": [],
        "claim_limit": config["claim_limit"],
    }
    write_json(args.output_root / "run_manifest.json", manifest)

    try:
        foreign_process_check(snapshot["uuid"])
        model_root = Path(backend["model_root"])
        component_paths = {
            "unet_path": str(model_root / "unet"),
            "scheduler_path": str(model_root / "scheduler"),
            "text_encoder_path": str(model_root / "text_encoder"),
            "tokenizer_path": str(model_root / "tokenizer"),
            "vae_path": str(model_root / "vae"),
        }
        policy = config["inference"]
        edit_config = {
            key: policy[key]
            for key in ("noise_samples", "n_steps", "t_start", "t_end", "t_delta", "step_scale")
        }
        edit_config["cleanup"] = True
        pipe = ChordEditPipeline.from_local_weights(
            component_paths=component_paths,
            default_edit_config=edit_config,
            device="cuda",
            torch_dtype=torch.float16,
            compute_dtype=torch.float16,
            image_size=int(policy["image_size"]),
            use_center_crop=False,
            use_attention_mask=False,
            use_safety_checker=False,
        )
        manifest["pipeline_load_count"] = 1
        base = Image.open(paths["input"]).convert("RGB")
        allowed = Image.open(paths["allowed"]).convert("L").resize(base.size, Image.Resampling.NEAREST)
        box = square_crop_box(allowed, int(policy["crop_padding_px"]))
        input_crop = base.crop(box).resize(
            (policy["image_size"], policy["image_size"]), Image.Resampling.LANCZOS
        )

        for seed in config["seeds"]:
            foreign_process_check(snapshot["uuid"])
            seed_root = args.output_root / f"seed_{seed}"
            seed_root.mkdir(parents=True, exist_ok=False)
            begin = time.perf_counter()
            raw = pipe(
                image=input_crop,
                source_prompt=case["source_prompt"],
                target_prompt=case["target_prompt"],
                seed=int(seed),
                edit_config=edit_config,
            ).images[0]
            raw.save(seed_root / "raw_editor_crop.png")
            final, outside_max = support_locked_composite(base, raw, box, allowed)
            base.save(seed_root / "input.png")
            allowed.save(seed_root / "allowed_edit_region.png")
            final.save(seed_root / "final.png")
            record = {
                "seed": seed,
                "status": "complete_requires_blind_review",
                "crop_box_xyxy": list(box),
                "outside_support_max_pixel_difference": outside_max,
                "wall_time_seconds": round(time.perf_counter() - begin, 3),
                "files": {
                    path.name: {
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for path in seed_root.iterdir()
                    if path.is_file()
                },
            }
            write_json(seed_root / "condition_manifest.json", record)
            manifest["status"] = "running"
            manifest["completed_seeds"].append(record)
            write_json(args.output_root / "run_manifest.json", manifest)

        manifest["status"] = "complete_requires_blind_review"
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
    print(json.dumps(manifest, indent=2), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
