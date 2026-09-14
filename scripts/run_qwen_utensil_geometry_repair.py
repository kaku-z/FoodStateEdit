#!/usr/bin/env python3
"""Run a frozen, support-local Qwen utensil geometry repair.

The learned editor proposes one crop-local repair.  The proposal is registered
to the source crop, composited only inside the frozen hard support, and written
as the final development output without a metric-based accept/rollback gate.
Diagnostics are still recorded for later human review.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for dependency_root in (SCRIPT_ROOT, PROJECT_ROOT / "foodstateedit"):
    if str(dependency_root) not in sys.path:
        sys.path.insert(0, str(dependency_root))

import run_qwen_utensil_refinement as base_runner  # noqa: E402
from appearance_refinement import audit_refinement_candidate, strict_local_composite  # noqa: E402


SCHEMA = "foodstateedit.qwen_utensil_geometry_repair.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


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
    started = time.perf_counter()
    case_root = output_root / case_id
    case_root.mkdir(parents=True, exist_ok=False)
    source = Image.open(case["source_image"]).convert("RGB")
    hard = base_runner.draw_support(source.size, case["utensil_support"])
    soft = hard.filter(ImageFilter.GaussianBlur(int(config["composition"]["soft_feather_px"])))
    structure = base_runner.structure_mask(source, hard)
    crop_box = base_runner.square_crop_box(hard, int(case["crop_padding_px"]))
    crop_size = int(case["crop_size"])
    source_crop = source.crop(crop_box).resize((crop_size, crop_size), Image.Resampling.LANCZOS)
    hard_crop = hard.crop(crop_box).resize((crop_size, crop_size), Image.Resampling.NEAREST)

    source.save(case_root / "00_pre_repair.png")
    hard.save(case_root / "01_hard_support.png")
    soft.save(case_root / "02_soft_mask.png")
    structure.save(case_root / "03_structure_mask.png")
    source_crop.save(case_root / "04_input_crop.png")

    base_runner.foreign_process_check(gpu_uuid)
    inference = config["inference"]
    call_kwargs: dict[str, Any] = {
        "image": source_crop,
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

    aligned_crop, registration = base_runner.align_crop(
        source_crop,
        raw,
        hard_crop,
        max_translation_px=int(config["alignment"]["max_translation_px"]),
        search_downsample=int(config["alignment"]["search_downsample"]),
    )
    aligned_crop.save(case_root / "06_aligned_qwen_crop.png")
    aligned_full = source.copy()
    aligned_full.paste(
        aligned_crop.resize(
            (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]),
            Image.Resampling.LANCZOS,
        ),
        crop_box,
    )
    candidate = strict_local_composite(
        source,
        aligned_full,
        soft,
        hard,
        alpha=float(config["composition"]["blend_alpha"]),
    )
    candidate.save(case_root / "07_candidate.png")

    diagnostics = audit_refinement_candidate(
        source,
        candidate,
        hard,
        structure,
        config["diagnostic_thresholds"],
        {},
    )
    diagnostics.update(
        registration=registration,
        output_policy="direct_candidate_output_pending_human_review",
        automatic_metrics_are_not_a_promotion_gate=True,
        accepted=None,
        disposition="candidate_written_as_development_final_without_automatic_gate",
    )
    write_json(case_root / "08_diagnostics.json", diagnostics, exclusive=True)
    write_json(
        case_root / "09_manual_review.json",
        {
            "status": "pending_unblinded_visual_review",
            "questions": config["manual_review_questions"],
            "answers": {question: None for question in config["manual_review_questions"]},
            "note": "Human review is descriptive and does not trigger an automatic rollback.",
        },
        exclusive=True,
    )
    candidate.save(case_root / "10_final_direct_candidate.png")
    record = {
        "case_id": case_id,
        "seed": int(case["seed"]),
        "status": "complete_direct_candidate_pending_human_review",
        "crop_box_xyxy": list(crop_box),
        "diagnostics": diagnostics,
        "wall_time_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(case_root / "condition_manifest.json", record, exclusive=True)
    record["files"] = {
        path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in case_root.iterdir()
        if path.is_file() and path.name != "condition_manifest.json"
    }
    write_json(case_root / "condition_manifest.json", record)
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
    implementation = config["implementation"]
    if sha256_file(Path(__file__)) != implementation["runner_sha256"]:
        raise ValueError("runner hash mismatch")
    if sha256_file(SCRIPT_ROOT / "run_qwen_utensil_refinement.py") != implementation["base_runner_sha256"]:
        raise ValueError("base runner hash mismatch")
    if sha256_file(PROJECT_ROOT / "foodstateedit" / "appearance_refinement.py") != implementation["appearance_refinement_sha256"]:
        raise ValueError("appearance refinement hash mismatch")
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
    path_checks = {
        "model_root": model_root.is_dir(),
        "model_index": (model_root / "model_index.json").is_file(),
    }
    for case_id in case_order:
        source = Path(config["cases"][case_id]["source_image"])
        path_checks[f"{case_id}_source_image"] = (
            source.is_file() and sha256_file(source) == config["cases"][case_id]["source_sha256"]
        )
    weight_checks = {}
    for relative, expected in backend["weight_files"].items():
        path = model_root / relative
        weight_checks[relative] = (
            path.is_file()
            and path.stat().st_size == expected["size_bytes"]
            and sha256_file(path) == expected["sha256"]
        )
    snapshot = base_runner.gpu_snapshot(args.physical_gpu)
    checks = {
        **base_runner.gate_snapshot(snapshot, config["resource_gate"]),
        **path_checks,
        **weight_checks,
    }
    preflight = {
        "schema_version": "foodstateedit.qwen_utensil_geometry_repair_preflight.v1",
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
        "schema_version": "foodstateedit.qwen_utensil_geometry_repair_run.v1",
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
        base_runner.foreign_process_check(snapshot["uuid"])
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
            status="complete_direct_candidates_pending_human_review",
            wall_time_seconds=round(time.perf_counter() - started, 3),
        )
        write_json(args.output_root / "run_manifest.json", manifest)
        (args.output_root / "COMPLETE").write_text(
            datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
        )
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
        (args.output_root / "FAILED").write_text(
            datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
