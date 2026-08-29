#!/usr/bin/env python3
"""Project frozen raw VACE frames through the pre-existing motion union.

This is a deterministic post-hoc failure diagnostic, not a formal method result.
It never runs a learned model and never selects cases, seeds, or frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ANCHOR_GPU = {
    "soup_spoon_001": "gpu0",
    "fried_rice_spatula_001": "gpu0",
    "ramen_chopsticks_001": "gpu1",
    "pasta_fork_001": "gpu1",
}
METHOD = "motion_union_projection_diagnostic"
SOURCE_METHOD = "vace_direct_dynamic_multikey"
SOURCE_SEED = 1
SOURCE_SELECTED_FRAME_INDEX = 18


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_image(path: Path, flags: int) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def verify_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {actual} != {expected}: {path}")


def output_hash(run_manifest: dict[str, object], filename: str) -> str:
    matches = [
        record["sha256"]
        for record in run_manifest["outputs"]
        if Path(record["path"]).name == filename
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {filename} output record")
    return str(matches[0])


def compose_case(
    anchor: str,
    gpu: str,
    vace_root: Path,
    proxy_root: Path,
    proxy_manifest_root: Path,
    config_path: Path,
    output_dir: Path,
    code_commit: str,
) -> dict[str, object]:
    run_dir = vace_root / gpu / anchor / f"seed_{SOURCE_SEED}"
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = load_json(run_manifest_path)
    if (
        run_manifest.get("case_id") != anchor
        or run_manifest.get("method") != SOURCE_METHOD
        or run_manifest.get("seed") != SOURCE_SEED
        or run_manifest.get("status") != "complete"
        or run_manifest["inference"].get("selected_frame_index")
        != SOURCE_SELECTED_FRAME_INDEX
    ):
        raise ValueError(f"Source run identity mismatch: {run_manifest_path}")

    proxy_dir = proxy_root / anchor
    proxy_manifest_path = proxy_manifest_root / anchor / "proxy_manifest.json"
    proxy_manifest = load_json(proxy_manifest_path)
    if (
        proxy_manifest.get("schema_version")
        != "foodstateedit.dynamic_multikey_proxy.v1"
        or proxy_manifest.get("anchor_id") != anchor
        or proxy_manifest.get("selected_frame_index")
        != SOURCE_SELECTED_FRAME_INDEX
    ):
        raise ValueError(f"Dynamic proxy identity mismatch: {proxy_manifest_path}")

    raw_path = run_dir / "raw_selected_frame.png"
    old_projection_path = run_dir / "edited_2d.png"
    source_path = proxy_dir / "first_frame.png"
    final_alpha_path = proxy_dir / "edit_alpha.png"
    union_alpha_path = proxy_dir / "motion_union_alpha.png"
    verify_hash(raw_path, output_hash(run_manifest, raw_path.name), "raw frame")
    verify_hash(
        old_projection_path,
        output_hash(run_manifest, old_projection_path.name),
        "reported projection",
    )
    for name, path in (
        ("first_frame.png", source_path),
        ("edit_alpha.png", final_alpha_path),
        ("motion_union_alpha.png", union_alpha_path),
    ):
        verify_hash(path, proxy_manifest["files"][name]["sha256"], name)

    source = read_image(source_path, cv2.IMREAD_COLOR)
    raw = read_image(raw_path, cv2.IMREAD_COLOR)
    old_projection = read_image(old_projection_path, cv2.IMREAD_COLOR)
    final_alpha = read_image(final_alpha_path, cv2.IMREAD_GRAYSCALE)
    union_alpha = read_image(union_alpha_path, cv2.IMREAD_GRAYSCALE)
    if (
        raw.shape != source.shape
        or old_projection.shape != source.shape
        or final_alpha.shape != source.shape[:2]
        or union_alpha.shape != source.shape[:2]
    ):
        raise ValueError(f"Input shape mismatch for {anchor}")

    weight = union_alpha.astype(np.float32)[..., None] / 255.0
    result = np.rint(
        raw.astype(np.float32) * weight
        + source.astype(np.float32) * (1.0 - weight)
    )
    result = np.clip(result, 0, 255).astype(np.uint8)
    outside_union = union_alpha == 0
    outside_difference = np.abs(
        result.astype(np.int16) - source.astype(np.int16)
    )[outside_union]
    outside_max = int(outside_difference.max(initial=0))
    if outside_max != 0:
        raise AssertionError(f"Motion-union exact protection failed for {anchor}")

    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "edited_2d_motion_union.png"
    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError(f"Cannot write {output_path}")
    review = np.concatenate(
        [
            np.concatenate([source, old_projection], axis=1),
            np.concatenate([raw, result], axis=1),
        ],
        axis=0,
    )
    review_path = output_dir / "review_board.png"
    if not cv2.imwrite(str(review_path), review):
        raise RuntimeError(f"Cannot write {review_path}")

    extra_support = np.logical_and(union_alpha > 0, final_alpha == 0)
    recovered_change = np.logical_and(
        extra_support,
        np.any(result != source, axis=2),
    )
    manifest = {
        "schema_version": "foodstateedit.motion_union_projection_run.v1",
        "anchor_id": anchor,
        "method": METHOD,
        "scientific_status": "post_hoc_exploratory_not_preregistered",
        "deterministic": True,
        "stochastic_rerun": False,
        "source_method": SOURCE_METHOD,
        "source_seed": SOURCE_SEED,
        "source_selected_frame_index": SOURCE_SELECTED_FRAME_INDEX,
        "code_commit": code_commit,
        "config_sha256": sha256_file(config_path),
        "source_run_manifest_sha256": sha256_file(run_manifest_path),
        "dynamic_proxy_manifest_sha256": sha256_file(proxy_manifest_path),
        "invariants": {
            "outside_motion_union_alpha_max_pixel_difference": outside_max,
            "motion_union_support_pixel_count": int((union_alpha > 0).sum()),
            "final_edit_support_pixel_count": int((final_alpha > 0).sum()),
            "extra_motion_support_pixel_count": int(extra_support.sum()),
            "recovered_changed_pixel_count_outside_final_support": int(
                recovered_change.sum()
            ),
        },
        "inputs": {
            "raw_selected_frame_sha256": sha256_file(raw_path),
            "reported_final_projection_sha256": sha256_file(old_projection_path),
            "source_sha256": sha256_file(source_path),
            "final_edit_alpha_sha256": sha256_file(final_alpha_path),
            "motion_union_alpha_sha256": sha256_file(union_alpha_path),
        },
        "outputs": {
            "edited_2d_motion_union.png": sha256_file(output_path),
            "review_board.png": sha256_file(review_path),
        },
        "status": "complete_post_hoc_diagnostic_not_formal",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vace-root", type=Path, required=True)
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--proxy-manifest-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to reuse output root: {args.output_root}")
    config_path = args.config.resolve()
    config = load_json(config_path)
    if (
        config.get("schema_version")
        != "foodstateedit.motion_union_projection_diagnostic.v1"
        or config.get("method") != METHOD
        or config.get("scientific_status")
        != "post_hoc_exploratory_not_preregistered"
        or config.get("stochastic_rerun") is not False
    ):
        raise ValueError(f"Diagnostic config mismatch: {config_path}")

    args.output_root.mkdir(parents=True, exist_ok=False)
    cases = [
        compose_case(
            anchor,
            gpu,
            args.vace_root.resolve(),
            args.proxy_root.resolve(),
            args.proxy_manifest_root.resolve(),
            config_path,
            args.output_root / anchor,
            args.code_commit,
        )
        for anchor, gpu in ANCHOR_GPU.items()
    ]
    summary = {
        "schema_version": "foodstateedit.motion_union_projection_summary.v1",
        "method": METHOD,
        "scientific_status": "post_hoc_exploratory_not_preregistered",
        "case_count": len(cases),
        "complete_count": sum(case["status"].startswith("complete") for case in cases),
        "stochastic_rerun": False,
        "all_outside_motion_union_exact": all(
            case["invariants"]["outside_motion_union_alpha_max_pixel_difference"]
            == 0
            for case in cases
        ),
        "cases": cases,
    }
    (args.output_root / "motion_union_projection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "case_count": len(cases),
        "complete_count": summary["complete_count"],
        "all_outside_motion_union_exact": summary["all_outside_motion_union_exact"],
    }, indent=2))


if __name__ == "__main__":
    main()
