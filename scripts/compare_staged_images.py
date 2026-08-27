#!/usr/bin/env python3
"""Compare same-seed staged outputs inside and outside the editable support."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ANCHORS = (
    "soup_spoon_001",
    "fried_rice_spatula_001",
    "ramen_chopsticks_001",
    "pasta_fork_001",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_outputs(roots: list[Path]) -> dict[str, Path]:
    outputs = {}
    for root in roots:
        for path in root.glob("*/seed_1/edited_2d.png"):
            anchor = path.parent.parent.name
            if anchor in outputs:
                raise ValueError(f"Duplicate edited image for {anchor}")
            outputs[anchor] = path
    if set(outputs) != set(ANCHORS):
        raise ValueError(f"Output anchor mismatch: {sorted(outputs)}")
    return outputs


def main() -> None:
    import cv2
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-root", action="append", type=Path, required=True)
    parser.add_argument("--right-root", action="append", type=Path, required=True)
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--left-method", required=True)
    parser.add_argument("--right-method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite comparison: {args.output}")

    left = find_outputs(args.left_root)
    right = find_outputs(args.right_root)
    cases = []
    for anchor in ANCHORS:
        left_image = cv2.imread(str(left[anchor]), cv2.IMREAD_COLOR)
        right_image = cv2.imread(str(right[anchor]), cv2.IMREAD_COLOR)
        alpha = cv2.imread(
            str(args.proxy_root / anchor / "edit_alpha.png"), cv2.IMREAD_GRAYSCALE
        )
        if left_image is None or right_image is None or alpha is None:
            raise RuntimeError(f"Cannot read comparison inputs: {anchor}")
        if left_image.shape != right_image.shape or left_image.shape[:2] != alpha.shape:
            raise ValueError(f"Comparison shape mismatch: {anchor}")

        difference = np.abs(
            left_image.astype(np.int16) - right_image.astype(np.int16)
        )
        changed = np.any(difference > 0, axis=2)
        edit = alpha > 0
        outside = np.logical_not(edit)
        mse = float(np.mean(np.square(difference.astype(np.float64))))
        cases.append({
            "anchor_id": anchor,
            "left_sha256": sha256_file(left[anchor]),
            "right_sha256": sha256_file(right[anchor]),
            "full_rgb_mae": round(float(difference.mean()), 6),
            "full_max_pixel_difference": int(difference.max(initial=0)),
            "full_changed_pixel_fraction": round(float(changed.mean()), 6),
            "full_psnr_db": None if mse == 0 else round(20 * math.log10(255.0 / math.sqrt(mse)), 6),
            "edit_support_pixel_count": int(edit.sum()),
            "edit_support_rgb_mae": round(float(difference[edit].mean()), 6),
            "edit_support_max_pixel_difference": int(difference[edit].max(initial=0)),
            "edit_support_changed_pixel_fraction": round(float(changed[edit].mean()), 6),
            "outside_support_rgb_mae": round(float(difference[outside].mean()), 6),
            "outside_support_max_pixel_difference": int(difference[outside].max(initial=0)),
        })

    report = {
        "schema_version": "foodstateedit.same_seed_image_comparison.v1",
        "seed": 1,
        "left_method": args.left_method,
        "right_method": args.right_method,
        "case_count": len(cases),
        "all_hashes_different": all(
            case["left_sha256"] != case["right_sha256"] for case in cases
        ),
        "all_outside_support_exact": all(
            case["outside_support_max_pixel_difference"] == 0 for case in cases
        ),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
