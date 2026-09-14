#!/usr/bin/env python3
"""Compute same-input soup preservation metrics for the Day 34 presentation case.

Pixel metrics quantify similarity to the input, not whether the requested
utensil action is semantically correct.  Semantic pass/fail fields are an
explicit internal, non-blind review of one selected synthetic sample.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
METRICS_MODULE = ROOT / "scripts" / "compute_day33_traditional_metrics.py"
INPUT = ROOT / "artifacts" / "day9_action_pseudo_dataset_v3" / "vace_reference_image" / "clear_broth_spoon_imagegen_pseudo_v1.png"
MASK = ROOT / "artifacts" / "day9_action_pseudo_dataset_v3" / "edit_alpha" / "clear_broth_spoon_imagegen_pseudo_v1.png"
OUTPUTS = {
    "ChordEdit": ROOT / "artifacts" / "day34_presentation_soup_same_input_baselines_v1" / "chordedit_soup" / "seed_1" / "final.png",
    "Qwen-Image-Edit (raw)": ROOT / "artifacts" / "day34_presentation_soup_same_input_baselines_v1" / "qwen" / "raw_qwen.png",
    "FoodStateEdit controlled VACE": ROOT / "ppt_projects" / "foodstateedit_nakamizo_20260913" / "images" / "p10_success_soup_output.png",
}
REVIEW = {
    "ChordEdit": {
        "action_success": False,
        "photo_success": False,
        "preservation_success": True,
        "diagnosis": "Malformed/truncated utensil fragment; no complete spoon retaining broth.",
    },
    "Qwen-Image-Edit (raw)": {
        "action_success": True,
        "photo_success": True,
        "preservation_success": False,
        "diagnosis": "Photorealistic spoon retaining broth, but the raw editor globally reframes/reconstructs the scene.",
    },
    "FoodStateEdit controlled VACE": {
        "action_success": True,
        "photo_success": True,
        "preservation_success": True,
        "diagnosis": "One coherent spoon retaining broth; exact protected-region preservation is imposed by projection.",
    },
}
JSON_PATH = ROOT / "results" / "day34_presentation_soup_same_input_metrics_v1.json"
MD_PATH = ROOT / "results" / "DAY34_PRESENTATION_SOUP_SAME_INPUT_METRICS_20260913.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: float):
    return None if not np.isfinite(value) else float(value)


def main() -> None:
    spec = importlib.util.spec_from_file_location("day33_metrics", METRICS_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {METRICS_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with Image.open(INPUT) as image:
        size = image.size
    reference, _ = module.load_rgb(INPUT)
    with Image.open(MASK) as image:
        mask = np.asarray(image.convert("L").resize(size, Image.Resampling.NEAREST)) > 0

    records = []
    for method, path in OUTPUTS.items():
        output, resized = module.load_rgb(path, size)
        values = module.metrics(reference, output, mask)
        values = {key: finite(value) for key, value in values.items()}
        review = REVIEW[method]
        records.append({
            "method": method,
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(path),
            "resized_to_input_for_metrics": resized,
            "metrics": values,
            "internal_non_blind_review": {
                **review,
                "strict_end_to_end_success": bool(
                    review["action_success"]
                    and review["photo_success"]
                    and review["preservation_success"]
                ),
            },
        })

    payload = {
        "schema_version": "foodstateedit.day34_presentation_soup_same_input_metrics.v1",
        "date": "2026-09-13",
        "sample": {
            "case_id": "clear_broth_spoon_imagegen_pseudo_v1",
            "source_type": "selected synthetic pseudo input",
            "seed": 1,
            "input_path": str(INPUT.relative_to(ROOT)).replace("\\", "/"),
            "input_sha256": sha256(INPUT),
            "mask_path": str(MASK.relative_to(ROOT)).replace("\\", "/"),
            "mask_sha256": sha256(MASK),
        },
        "records": records,
        "metric_scope": {
            "psnr_ssim_mae": "Input-fidelity/preservation diagnostics only; they do not measure action correctness.",
            "outside_support": "Pixels outside the frozen edit-alpha support; SSIM excludes an 11-pixel dilated boundary.",
            "qwen_raw": "The direct baseline is evaluated as its raw global output; its diagnostic masked composite is not credited as generator preservation.",
            "foodstateedit_projection": "Exact outside-support preservation is imposed by compositing and is not evidence of native generator preservation.",
        },
        "claim_limit": "One selected synthetic sample, one seed, and non-blind internal review. No statistical superiority, generalization, physical correctness, or human-preference claim is supported.",
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def value(record: dict, key: str, digits: int = 3) -> str:
        item = record["metrics"][key]
        return "∞" if item is None else f"{item:.{digits}f}"

    lines = [
        "# Day 34 same-input soup presentation metrics",
        "",
        "All three methods use the same selected synthetic soup input and seed 1.",
        "Pixel metrics measure input fidelity, not whether the requested action is correct.",
        "",
        "| Method | Full PSNR ↑ | Full SSIM ↑ | Outside SSIM ↑ | Outside MAE ↓ | Outside exact ↑ | Action | Photo | Preservation | Strict |",
        "|---|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|",
    ]
    for record in records:
        review = record["internal_non_blind_review"]
        mark = lambda flag: "✓" if flag else "✗"
        lines.append(
            f"| {record['method']} | {value(record, 'full_psnr_db', 2)} | "
            f"{value(record, 'full_ssim')} | {value(record, 'outside_ssim')} | "
            f"{value(record, 'outside_rgb_mae')} | "
            f"{100.0 * record['metrics']['outside_exact_pixel_fraction']:.1f}% | "
            f"{mark(review['action_success'])} | {mark(review['photo_success'])} | "
            f"{mark(review['preservation_success'])} | {mark(review['strict_end_to_end_success'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- ChordEdit obtains exact protected-pixel preservation through local compositing but does not produce the requested action.",
        "- Raw Qwen-Image-Edit produces the strongest photographic action result in this single case, while globally changing the framing/background.",
        "- FoodStateEdit is the only strict pass under this internal case-level rubric because it combines a successful static spoon state with exact projected background preservation.",
        "- This is one selected synthetic example and cannot support a statistical superiority or generalization claim.",
    ]
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
