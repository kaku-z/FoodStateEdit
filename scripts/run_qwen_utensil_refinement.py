#!/usr/bin/env python3
"""Run a frozen Qwen utensil-only appearance-refinement sweep.

Qwen proposes a crop-local edit.  The proposal is registered back to the
pre-refinement crop and composited only inside a frozen utensil support mask.
The final output remains a conservative rollback until semantic review is
completed.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
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
from PIL import Image, ImageDraw, ImageFilter


RUNTIME_ROOT = Path(__file__).resolve().parent
LOCAL_DEPENDENCY_ROOT = Path(__file__).resolve().parents[1] / "foodstateedit"
for dependency_root in (RUNTIME_ROOT, LOCAL_DEPENDENCY_ROOT):
    if str(dependency_root) not in sys.path:
        sys.path.insert(0, str(dependency_root))

from appearance_refinement import (  # noqa: E402
    REQUIRED_SEMANTIC_CHECKS,
    apply_decision,
    audit_refinement_candidate,
    strict_local_composite,
)


SCHEMA = "foodstateedit.qwen_utensil_refinement.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


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
    if physical_gpu not in parsed:
        raise ValueError(f"physical GPU {physical_gpu} was not reported")
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
        raise RuntimeError("Foreign compute process appeared; no preemption: " + repr(foreign))


def draw_support(size: tuple[int, int], spec: dict[str, Any]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for item in spec["lines"]:
        draw.line([tuple(point) for point in item["points"]], fill=255, width=int(item["width"]), joint="curve")
    for points in spec["polygons"]:
        draw.polygon([tuple(point) for point in points], fill=255)
    for item in spec.get("ellipses", []):
        draw.ellipse(tuple(item["bbox"]), outline=255, width=int(item["width"]))
    dilation = int(spec.get("dilation_px", 0))
    if dilation:
        mask = mask.filter(ImageFilter.MaxFilter(2 * dilation + 1))
    if mask.getbbox() is None:
        raise ValueError("frozen utensil support is empty")
    return mask


def square_crop_box(mask: Image.Image, padding: int) -> tuple[int, int, int, int]:
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("cannot crop empty support")
    width, height = mask.size
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0 - padding), max(0, y0 - padding)
    x1, y1 = min(width, x1 + padding), min(height, y1 + padding)
    side = min(max(x1 - x0, y1 - y0), width, height)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    sx0 = max(0, min(int(round(cx - side / 2.0)), width - side))
    sy0 = max(0, min(int(round(cy - side / 2.0)), height - side))
    return sx0, sy0, sx0 + side, sy0 + side


def align_crop(
    base: Image.Image,
    edited: Image.Image,
    support: Image.Image,
    *,
    max_translation_px: int,
    search_downsample: int,
) -> tuple[Image.Image, dict[str, Any]]:
    """Register a proposal by exhaustive protected-region translation search."""
    base_rgb = np.asarray(base.convert("RGB"), dtype=np.uint8)
    edit_rgb = np.asarray(edited.convert("RGB").resize(base.size, Image.Resampling.LANCZOS), dtype=np.uint8)
    factor = max(1, int(search_downsample))
    small_size = (max(1, base.width // factor), max(1, base.height // factor))
    base_gray = np.asarray(base.convert("L").resize(small_size, Image.Resampling.BILINEAR), dtype=np.float32)
    edit_gray = np.asarray(edited.convert("L").resize(small_size, Image.Resampling.BILINEAR), dtype=np.float32)
    protected = np.asarray(support.convert("L").resize(small_size, Image.Resampling.NEAREST), dtype=np.uint8) == 0
    radius = max(0, int(round(max_translation_px / factor)))
    best: tuple[float, int, int, int] | None = None
    height, width = base_gray.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out_x0, out_x1 = max(0, dx), min(width, width + dx)
            out_y0, out_y1 = max(0, dy), min(height, height + dy)
            src_x0, src_x1 = max(0, -dx), min(width, width - dx)
            src_y0, src_y1 = max(0, -dy), min(height, height - dy)
            valid = protected[out_y0:out_y1, out_x0:out_x1]
            count = int(valid.sum())
            if count < int(protected.sum() * 0.70):
                continue
            difference = np.abs(
                base_gray[out_y0:out_y1, out_x0:out_x1]
                - edit_gray[src_y0:src_y1, src_x0:src_x1]
            )
            score = float(difference[valid].mean())
            candidate = (score, abs(dx) + abs(dy), dx, dy)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise RuntimeError("translation registration found no valid overlap")
    mae, _, dx_small, dy_small = best
    dx, dy = dx_small * factor, dy_small * factor
    aligned = base_rgb.copy()
    out_x0, out_x1 = max(0, dx), min(base.width, base.width + dx)
    out_y0, out_y1 = max(0, dy), min(base.height, base.height + dy)
    src_x0, src_x1 = max(0, -dx), min(base.width, base.width - dx)
    src_y0, src_y1 = max(0, -dy), min(base.height, base.height - dy)
    aligned[out_y0:out_y1, out_x0:out_x1] = edit_rgb[src_y0:src_y1, src_x0:src_x1]
    return Image.fromarray(aligned, mode="RGB"), {
        "protected_region_mae_255": mae,
        "protected_region_similarity": float(1.0 - mae / 255.0),
        "translation_xy_pixels": [dx, dy],
        "search_downsample": factor,
    }


def structure_mask(base: Image.Image, hard: Image.Image) -> Image.Image:
    rgb = np.asarray(base.convert("RGB"), dtype=np.float32)
    gray = rgb.mean(axis=2)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = gray[:, 1:] - gray[:, :-1]
    gy[1:, :] = gray[1:, :] - gray[:-1, :]
    edge = np.sqrt(gx * gx + gy * gy)
    hard_arr = np.asarray(hard.convert("L"), dtype=np.uint8) > 0
    cutoff = float(np.percentile(edge[hard_arr], 55.0))
    return Image.fromarray((hard_arr & (edge >= cutoff)).astype(np.uint8) * 255, mode="L")


def run_case(
    *,
    case_id: str,
    case: dict[str, Any],
    config: dict[str, Any],
    pipe: Any,
    torch: Any,
    output_root: Path,
    gpu_uuid: str,
) -> dict[str, Any]:
    case_started = time.perf_counter()
    case_root = output_root / case_id
    case_root.mkdir(parents=True, exist_ok=False)
    base = Image.open(case["source_image"]).convert("RGB")
    hard = draw_support(base.size, case["utensil_support"])
    soft = hard.filter(ImageFilter.GaussianBlur(int(config["composition"]["soft_feather_px"])))
    structure = structure_mask(base, hard)
    box = square_crop_box(hard, int(case["crop_padding_px"]))
    crop_size = int(case["crop_size"])
    base_crop = base.crop(box).resize((crop_size, crop_size), Image.Resampling.LANCZOS)
    hard_crop = hard.crop(box).resize((crop_size, crop_size), Image.Resampling.NEAREST)

    base.save(case_root / "00_pre_refinement.png")
    hard.save(case_root / "01_utensil_hard_support.png")
    soft.save(case_root / "02_utensil_soft_mask.png")
    structure.save(case_root / "03_structure_mask.png")
    base_crop.save(case_root / "04_input_crop.png")

    foreign_process_check(gpu_uuid)
    inference = config["inference"]
    call_kwargs: dict[str, Any] = {
        "image": base_crop,
        "prompt": case["instruction"],
        "negative_prompt": inference["negative_prompt"],
        "num_inference_steps": int(inference["num_inference_steps"]),
        "true_cfg_scale": float(inference["true_cfg_scale"]),
        "guidance_scale": float(inference["guidance_scale"]),
        "generator": torch.Generator(device="cuda").manual_seed(int(case["seed"])),
    }
    signature = inspect.signature(pipe.__call__)
    if "height" in signature.parameters and "width" in signature.parameters:
        call_kwargs.update(height=crop_size, width=crop_size)
    raw = pipe(**call_kwargs).images[0].convert("RGB")
    raw.save(case_root / "05_raw_qwen_crop.png")

    aligned_crop, registration = align_crop(
        base_crop,
        raw,
        hard_crop,
        max_translation_px=int(config["alignment"]["max_translation_px"]),
        search_downsample=int(config["alignment"]["search_downsample"]),
    )
    aligned_crop.save(case_root / "06_aligned_qwen_crop.png")
    aligned_full = base.copy()
    aligned_full.paste(
        aligned_crop.resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS),
        box,
    )
    candidate = strict_local_composite(
        base,
        aligned_full,
        soft,
        hard,
        alpha=float(config["composition"]["blend_alpha"]),
    )
    candidate.save(case_root / "07_candidate.png")

    manual = {name: None for name in REQUIRED_SEMANTIC_CHECKS}
    audit = audit_refinement_candidate(
        base,
        candidate,
        hard,
        structure,
        config["acceptance_gate"],
        manual,
    )
    audit["registration"] = registration
    audit["automatic_checks"]["protected_registration_similarity"] = (
        registration["protected_region_similarity"]
        >= float(config["alignment"]["min_protected_similarity"])
    )
    audit["accepted"] = (
        all(audit["automatic_checks"].values())
        and audit["manual_complete"]
        and all(value is True for value in audit["manual_checks"].values())
    )
    audit["disposition"] = (
        "use_candidate" if audit["accepted"] else "rollback_to_pre_refinement_frame"
    )
    write_json(case_root / "08_automatic_gate.json", audit, exclusive=True)
    write_json(case_root / "09_manual_review.json", {
        "status": "pending_visual_review",
        "checks": manual,
        "uncertain_is_failure": True,
    }, exclusive=True)
    apply_decision(base, candidate, audit).save(case_root / "10_final.png")
    record = {
        "case_id": case_id,
        "seed": int(case["seed"]),
        "status": "candidate_generated_manual_review_pending_final_is_conservative_rollback",
        "crop_box_xyxy": list(box),
        "audit": audit,
        "wall_time_seconds": round(time.perf_counter() - case_started, 3),
        "files": {
            path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in case_root.iterdir()
            if path.is_file()
        },
    }
    write_json(case_root / "condition_manifest.json", record, exclusive=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA or not config.get("execution_allowed"):
        raise ValueError("unexpected or disabled config")
    if sha256_file(Path(__file__)) != config["implementation"]["runner_sha256"]:
        raise ValueError("runner hash mismatch")
    if args.output_root.resolve() != Path(config["output_root"]).resolve():
        raise ValueError("output root differs from frozen path")
    if args.preflight.resolve() != Path(config["preflight_path"]).resolve():
        raise ValueError("preflight path differs from frozen path")
    if args.output_root.exists() or args.preflight.exists():
        raise FileExistsError("refusing to overwrite output or preflight evidence")

    case_order = tuple(config["case_order"])
    if not case_order or set(case_order) != set(config["cases"]):
        raise ValueError("case order and case mapping differ")
    backend = config["backend"]
    model_root = Path(backend["model_root"])
    dependency = Path(__file__).with_name("appearance_refinement.py")
    path_checks = {
        "model_root": model_root.is_dir(),
        "model_index": (model_root / "model_index.json").is_file(),
        "appearance_refinement_dependency": dependency.is_file() and sha256_file(dependency) == config["implementation"]["appearance_refinement_sha256"],
    }
    for case_id in case_order:
        source = Path(config["cases"][case_id]["source_image"])
        path_checks[f"{case_id}_source_image"] = (
            source.is_file()
            and sha256_file(source) == config["cases"][case_id]["source_sha256"]
        )
    weight_checks = {}
    for relative, expected in backend["weight_files"].items():
        path = model_root / relative
        weight_checks[relative] = path.is_file() and path.stat().st_size == expected["size_bytes"] and sha256_file(path) == expected["sha256"]
    snapshot = gpu_snapshot(args.physical_gpu)
    checks = {**gate_snapshot(snapshot, config["resource_gate"]), **path_checks, **weight_checks}
    preflight = {
        "schema_version": "foodstateedit.qwen_utensil_refinement_preflight.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "physical_gpu": args.physical_gpu,
        "gpu": snapshot,
        "config_sha256": sha256_file(args.config),
        "checks": checks,
        "passed": all(checks.values()),
    }
    write_json(args.preflight, preflight, exclusive=True)
    if not preflight["passed"]:
        raise RuntimeError("preflight failed; no sample output was created")

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
        "schema_version": "foodstateedit.qwen_utensil_refinement_run.v1",
        "status": "loading_pipeline",
        "host": socket.gethostname(),
        "physical_gpu": args.physical_gpu,
        "gpu_uuid": snapshot["uuid"],
        "config_sha256": sha256_file(args.config),
        "runner_sha256": sha256_file(Path(__file__)),
        "pipeline_load_count": 0,
        "candidate_count": 0,
        "expected_cases": list(case_order),
        "completed_cases": [],
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
        for case_id in case_order:
            record = run_case(
                case_id=case_id,
                case=config["cases"][case_id],
                config=config,
                pipe=pipe,
                torch=torch,
                output_root=args.output_root,
                gpu_uuid=snapshot["uuid"],
            )
            manifest["candidate_count"] += 1
            manifest["completed_cases"].append(record)
            manifest["status"] = "running"
            write_json(args.output_root / "run_manifest.json", manifest)
        manifest.update(
            status="complete_candidates_generated_manual_review_pending_finals_are_conservative_rollbacks",
            wall_time_seconds=round(time.perf_counter() - started, 3),
        )
        write_json(args.output_root / "run_manifest.json", manifest)
        (args.output_root / "COMPLETE").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2), flush=True)
        return 0
    except Exception as exc:
        manifest.update(
            status="technical_failure_preserved",
            error=repr(exc),
            wall_time_seconds=round(time.perf_counter() - started, 3),
        )
        write_json(args.output_root / "run_manifest.json", manifest)
        (args.output_root / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        (args.output_root / "FAILED").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
