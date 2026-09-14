#!/usr/bin/env python3
"""Run one frozen, support-locked ChordEdit appearance-refinement candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foodstateedit.appearance_refinement import (  # noqa: E402
    REQUIRED_SEMANTIC_CHECKS,
    apply_decision,
    audit_refinement_candidate,
    strict_local_composite,
    validate_refinement_config,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
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
    selected["compute_processes"] = [row for row in process_rows if row.split(",", 1)[0].strip() == selected["uuid"]]
    mem_available_kib = None
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            mem_available_kib = int(line.split()[1])
            break
    selected["host_mem_available_mib"] = None if mem_available_kib is None else mem_available_kib // 1024
    return selected


def gate_snapshot(snapshot: dict[str, Any], gate: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"id": "gpu_name", "passed": snapshot["name"] == gate["required_gpu_name"], "observed": snapshot["name"]},
        {"id": "gpu_free_memory", "passed": snapshot["memory_free_mib"] >= gate["min_free_memory_mib"], "observed": snapshot["memory_free_mib"]},
        {"id": "gpu_utilization", "passed": snapshot["utilization_percent"] <= gate["max_utilization_percent"], "observed": snapshot["utilization_percent"]},
        {"id": "zero_compute_processes", "passed": not snapshot["compute_processes"], "observed": snapshot["compute_processes"]},
        {"id": "host_available_memory", "passed": snapshot["host_mem_available_mib"] is not None and snapshot["host_mem_available_mib"] >= gate["min_available_system_memory_mib"], "observed": snapshot["host_mem_available_mib"]},
    ]


def derive_masks(
    base: Image.Image,
    reference: Image.Image,
    allowed: Image.Image,
    threshold: int,
    dilation: int,
    feather: int,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    base_arr = np.asarray(base.convert("RGB"), dtype=np.int16)
    reference_arr = np.asarray(reference.convert("RGB").resize(base.size, Image.Resampling.LANCZOS), dtype=np.int16)
    allowed_arr = np.asarray(allowed.convert("L").resize(base.size, Image.Resampling.NEAREST), dtype=np.uint8) > 0
    difference = np.max(np.abs(base_arr - reference_arr), axis=2) >= threshold
    initial = Image.fromarray((difference & allowed_arr).astype(np.uint8) * 255, mode="L")
    hard = initial.filter(ImageFilter.MaxFilter(2 * dilation + 1))
    hard_arr = (np.asarray(hard) > 0) & allowed_arr
    if not hard_arr.any():
        raise ValueError("derived repair support is empty")
    hard = Image.fromarray(hard_arr.astype(np.uint8) * 255, mode="L")
    soft = hard.filter(ImageFilter.GaussianBlur(feather))

    gray = np.asarray(base.convert("L"), dtype=np.float32)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    edge = np.sqrt(gx * gx + gy * gy)
    cutoff = float(np.percentile(edge[hard_arr], 60.0))
    structure_arr = hard_arr & (edge >= cutoff)
    structure = Image.fromarray(structure_arr.astype(np.uint8) * 255, mode="L")
    return hard, soft, structure


def square_crop_box(mask: Image.Image, padding: int) -> tuple[int, int, int, int]:
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("cannot crop an empty mask")
    width, height = mask.size
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0 - padding), max(0, y0 - padding)
    x1, y1 = min(width, x1 + padding), min(height, y1 + padding)
    side = min(max(x1 - x0, y1 - y0), width, height)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    sx0 = max(0, min(int(round(cx - side / 2.0)), width - side))
    sy0 = max(0, min(int(round(cy - side / 2.0)), height - side))
    return sx0, sy0, sx0 + side, sy0 + side


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.preflight.exists():
        raise FileExistsError(args.preflight)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_refinement_config(config)

    pilot = config["pilot"]
    backend = config["backend"]
    required_paths = [
        args.config,
        Path(pilot["source_image"]),
        Path(pilot["reference_image"]),
        Path(pilot["allowed_edit_region"]),
        Path(backend["project_root"]) / "pipeline_chord.py",
        Path(backend["model_root"]) / "model_index.json",
    ] + [Path(backend["model_root"]) / name for name in ("unet", "vae", "text_encoder", "tokenizer", "scheduler")]
    path_checks = [{"path": str(path), "exists": path.exists()} for path in required_paths]
    weight_checks = []
    for relative, expected in backend["weight_files"].items():
        weight_path = Path(backend["model_root"]) / relative
        exists = weight_path.is_file()
        observed_size = weight_path.stat().st_size if exists else None
        observed_sha256 = sha256_file(weight_path) if exists and observed_size == expected["size_bytes"] else None
        weight_checks.append({
            "id": f"weight:{relative}",
            "passed": exists and observed_size == expected["size_bytes"] and observed_sha256 == expected["sha256"],
            "observed_size_bytes": observed_size,
            "observed_sha256": observed_sha256,
        })
    snapshot = gpu_snapshot(args.physical_gpu)
    checks = gate_snapshot(snapshot, config["resource_gate"])
    checks.extend({"id": f"path:{row['path']}", "passed": row["exists"]} for row in path_checks)
    checks.extend(weight_checks)
    ready = all(check["passed"] for check in checks)
    preflight = {
        "schema_version": "foodstateedit.geometry_locked_refinement_preflight.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "physical_gpu": args.physical_gpu,
        "gpu": snapshot,
        "config_sha256": sha256_file(args.config),
        "checks": checks,
        "ready": ready,
    }
    write_json_exclusive(args.preflight, preflight)
    if not ready:
        raise RuntimeError("preflight failed; no sample output directory was created")

    second = gpu_snapshot(args.physical_gpu)
    if not all(check["passed"] for check in gate_snapshot(second, config["resource_gate"])):
        raise RuntimeError("GPU changed after preflight; no sample output directory was created")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, backend["project_root"])
    import torch  # noqa: E402
    from pipeline_chord import ChordEditPipeline  # noqa: E402

    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    base = Image.open(pilot["source_image"]).convert("RGB")
    reference = Image.open(pilot["reference_image"]).convert("RGB")
    allowed = Image.open(pilot["allowed_edit_region"]).convert("L")
    hard, soft, structure = derive_masks(
        base,
        reference,
        allowed,
        int(pilot["support_difference_threshold_255"]),
        int(pilot["support_dilation_px"]),
        int(config["composition"]["soft_feather_px"]),
    )
    base.save(args.output_dir / "00_pre_refinement.png")
    hard.save(args.output_dir / "01_hard_support.png")
    soft.save(args.output_dir / "02_soft_mask.png")
    structure.save(args.output_dir / "structure_mask.png")

    box = square_crop_box(hard, int(pilot["crop_padding_px"]))
    crop = base.crop(box).resize((pilot["image_size"], pilot["image_size"]), Image.Resampling.LANCZOS)
    crop.save(args.output_dir / "03_input_crop.png")
    model_root = Path(backend["model_root"])
    component_paths = {
        "unet_path": str(model_root / "unet"),
        "scheduler_path": str(model_root / "scheduler"),
        "text_encoder_path": str(model_root / "text_encoder"),
        "tokenizer_path": str(model_root / "tokenizer"),
        "vae_path": str(model_root / "vae"),
    }
    policy = config["candidate_policy"]
    edit_config = {key: policy[key] for key in ("noise_samples", "n_steps", "t_start", "t_end", "t_delta", "step_scale")}
    edit_config["cleanup"] = True
    pipe = ChordEditPipeline.from_local_weights(
        component_paths=component_paths,
        default_edit_config=edit_config,
        device="cuda",
        torch_dtype=torch.float16,
        compute_dtype=torch.float16,
        image_size=int(pilot["image_size"]),
        use_center_crop=False,
        use_attention_mask=False,
        use_safety_checker=False,
    )
    raw = pipe(
        image=crop,
        source_prompt=pilot["source_prompt"],
        target_prompt=pilot["target_prompt"],
        seed=int(pilot["seed"]),
        edit_config=edit_config,
    ).images[0]
    raw.save(args.output_dir / "04_raw_editor_crop.png")
    raw_full = base.copy()
    raw_full.paste(raw.resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS), box)
    candidate = strict_local_composite(base, raw_full, soft, hard, alpha=float(policy["blend_alpha"]))
    candidate.save(args.output_dir / "05_candidate.png")

    manual_template = {name: None for name in REQUIRED_SEMANTIC_CHECKS}
    audit = audit_refinement_candidate(base, candidate, hard, structure, config["acceptance_gate"], manual_template)
    write_json_exclusive(args.output_dir / "06_automatic_gate.json", audit)
    write_json_exclusive(args.output_dir / "07_manual_review.json", {
        "status": "pending_independent_review",
        "checks": manual_template,
        "uncertain_is_failure": True,
    })
    apply_decision(base, candidate, audit).save(args.output_dir / "08_final.png")
    manifest = {
        "schema_version": "foodstateedit.geometry_locked_refinement_run.v1",
        "status": "candidate_generated_manual_review_pending_final_is_conservative_rollback",
        "host": socket.gethostname(),
        "physical_gpu": args.physical_gpu,
        "gpu_uuid": snapshot["uuid"],
        "config_sha256": sha256_file(args.config),
        "preflight_sha256": sha256_file(args.preflight),
        "crop_box_xyxy": list(box),
        "pipeline_load_count": 1,
        "candidate_count": 1,
        "audit": audit,
        "wall_time_seconds": round(time.perf_counter() - started, 3),
    }
    write_json_exclusive(args.output_dir / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
