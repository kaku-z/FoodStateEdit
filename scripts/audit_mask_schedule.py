#!/usr/bin/env python3
"""Measure semantic-mask overlap and its effect on staged projection windows."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image


LAYERS = ("rigid", "contact", "material", "hole")
ENDPOINTS = {"rigid": 18, "contact": 15, "material": 8, "hole": 8}
SEGMENTS = (
    (3, 8, LAYERS),
    (8, 15, ("rigid", "contact")),
    (15, 18, ("rigid",)),
    (18, 20, ()),
)


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 127


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def audit_case(case_dir: Path) -> dict[str, object]:
    masks = {
        layer: load_mask(case_dir / f"mask_{layer}.png")
        for layer in LAYERS
    }
    shapes = {mask.shape for mask in masks.values()}
    if len(shapes) != 1:
        raise ValueError(f"Mask shape mismatch: {case_dir}")

    counts = {layer: int(mask.sum()) for layer, mask in masks.items()}
    overlaps = []
    for left, right in combinations(LAYERS, 2):
        count = int(np.logical_and(masks[left], masks[right]).sum())
        overlaps.append({
            "layers": [left, right],
            "pixel_count": count,
            "fraction_of_left": ratio(count, counts[left]),
            "fraction_of_right": ratio(count, counts[right]),
        })

    shadowing = []
    for layer in LAYERS:
        later = [
            other for other in LAYERS
            if ENDPOINTS[other] > ENDPOINTS[layer]
        ]
        later_union = np.logical_or.reduce([masks[name] for name in later]) if later else np.zeros_like(masks[layer])
        count = int(np.logical_and(masks[layer], later_union).sum())
        shadowing.append({
            "layer": layer,
            "endpoint": ENDPOINTS[layer],
            "later_endpoint_layers": later,
            "shadowed_pixel_count": count,
            "shadowed_fraction": ratio(count, counts[layer]),
        })

    active_segments = []
    for start, end, layers in SEGMENTS:
        active = np.logical_or.reduce([masks[name] for name in layers]) if layers else np.zeros_like(next(iter(masks.values())))
        active_segments.append({
            "step_window": [start, end],
            "active_layers": list(layers),
            "active_union_pixel_count": int(active.sum()),
        })

    full_union = np.logical_or.reduce(list(masks.values()))
    rigid_equals_full_union = bool(np.array_equal(masks["rigid"], full_union))
    return {
        "anchor_id": case_dir.name,
        "image_height": int(next(iter(masks.values())).shape[0]),
        "image_width": int(next(iter(masks.values())).shape[1]),
        "layer_pixel_counts": counts,
        "full_union_pixel_count": int(full_union.sum()),
        "rigid_equals_full_union": rigid_equals_full_union,
        "pairwise_overlaps": overlaps,
        "later_endpoint_shadowing": shadowing,
        "active_segments": active_segments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {args.output}")

    case_dirs = sorted(
        path for path in args.proxy_root.iterdir()
        if path.is_dir() and (path / "proxy_manifest.json").is_file()
    )
    if len(case_dirs) != 4:
        raise ValueError(f"Expected four proxy cases, found {len(case_dirs)}")
    cases = [audit_case(path) for path in case_dirs]
    audit = {
        "schema_version": "foodstateedit.mask_schedule_audit.v1",
        "schedule": ENDPOINTS,
        "window_semantics": "half-open [tweak_index, endpoint)",
        "case_count": len(cases),
        "any_semantic_overlap": any(
            item["pixel_count"] > 0
            for case in cases for item in case["pairwise_overlaps"]
        ),
        "any_shorter_window_shadowed": any(
            item["shadowed_pixel_count"] > 0
            for case in cases for item in case["later_endpoint_shadowing"]
        ),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
