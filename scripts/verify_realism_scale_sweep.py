#!/usr/bin/env python3
"""Verify Day 21 pullback, compute diagnostics, and compose fixed-frame A/B sheets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CASES = ("soup", "rice", "cake")
CONDITIONS = ("vace_scale_0p6", "vace_scale_0p8")
BASELINES = {
    "soup": ROOT / "artifacts/day19_multimaterial_gp40_recovered_20260909_v1/soup__relative3d/projected_final_hold.png",
    "rice": ROOT / "artifacts/day19_multimaterial_gp40_recovered_20260909_v1/rice__relative3d/projected_final_hold.png",
    "cake": ROOT / "artifacts/day20_cake_recovery_gp40_20260909_v1/cake__relative3d/projected_final_hold.png",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def parse_case_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        case, raw_path = value.split("=", 1)
        if case not in CASES or case in result:
            raise ValueError(f"Invalid or duplicate case path: {case}")
        result[case] = Path(raw_path).resolve()
    if set(result) != set(CASES):
        raise ValueError(f"Expected paths for {CASES}, got {sorted(result)}")
    return result


def verify_listing(root: Path, listing: Path) -> int:
    records = {}
    for line in listing.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(None, 1)
        relative = relative.strip().removeprefix("./")
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"Unsafe path in listing: {relative}")
        if digest(path) != expected:
            raise ValueError(f"Remote/local SHA-256 mismatch: {path}")
        records[relative] = expected
    local = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if set(records) != local:
        raise ValueError("Remote/local file set differs")
    return len(records)


def grayscale_gradient(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def diagnostics(source: np.ndarray, image: np.ndarray, alpha: np.ndarray) -> dict:
    support = alpha > 0
    hard = alpha == 255
    kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(support.astype(np.uint8), kernel) > 0
    eroded = cv2.erode(support.astype(np.uint8), kernel) > 0
    boundary = dilated & ~eroded
    outside = ~support
    absolute = np.abs(image.astype(np.int16) - source.astype(np.int16))
    laplacian = cv2.Laplacian(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_32F
    )
    return {
        "outside_support_max_pixel_difference": int(absolute[outside].max(initial=0)),
        "inside_support_mae": float(absolute[support].mean()),
        "hard_support_changed_fraction": float(np.any(absolute > 3, axis=2)[hard].mean()),
        "boundary_gradient_mean": float(grayscale_gradient(image)[boundary].mean()),
        "support_laplacian_variance": float(laplacian[support].var()),
        "metric_note": "Pixel/edge statistics diagnose preservation and texture only; they do not prove action, conservation, contact, or photo realism.",
    }


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)


def compose_case(case: str, images: list[tuple[str, Path]], output: Path) -> None:
    opened = [(label, Image.open(path).convert("RGB")) for label, path in images]
    width, height = opened[0][1].size
    margin, gap, label_height = 20, 15, 45
    canvas = Image.new(
        "RGB",
        (margin * 2 + len(opened) * width + (len(opened) - 1) * gap, height + 110),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 10), f"{case.upper()} | fixed final frame 20", font=font(28), fill="black")
    for column, (label, image) in enumerate(opened):
        x = margin + column * (width + gap)
        draw.text((x, 55), label, font=font(22), fill="black")
        canvas.paste(image, (x, 55 + label_height))
    canvas.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="case=local run root")
    parser.add_argument("--remote-hashes", action="append", required=True, help="case=listing")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    runs = parse_case_paths(args.run)
    listings = parse_case_paths(args.remote_hashes)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = ROOT / "artifacts/day19_multimaterial_dataset_v3"
    config = json.loads((ROOT / "configs/day21_realism_scale_sweep_v1.json").read_text(encoding="utf-8"))

    report = {
        "schema_version": "foodstateedit.realism_scale_sweep_verification.v1",
        "status": "technically_verified_requires_manual_action_and_photo_review",
        "cases": {},
        "claim_limit": config["claim_limit"],
    }
    figure_inputs = []
    for case in CASES:
        root = runs[case]
        verified_count = verify_listing(root, listings[case])
        manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "complete_requires_matched_baseline_review":
            raise ValueError(f"Incomplete {case} run: {manifest['status']}")
        if manifest["pipeline_load_count"] != 1:
            raise ValueError(f"Unexpected pipeline load count for {case}")
        if [item["condition"] for item in manifest["completed_conditions"]] != list(CONDITIONS):
            raise ValueError(f"Unexpected condition order for {case}")
        source = cv2.imread(str(dataset / case / "reference.png"))
        alpha = cv2.imread(str(dataset / case / "edit_alpha.png"), cv2.IMREAD_GRAYSCALE)
        baseline = BASELINES[case]
        if digest(baseline) != config["existing_baseline"]["projected_final_hold_sha256"][case]:
            raise ValueError(f"Baseline hash mismatch for {case}")
        rows = {"scale_1p0_existing_baseline": diagnostics(source, cv2.imread(str(baseline)), alpha)}
        paths = [("Input", dataset / case / "reference.png"), ("scale 1.0 baseline", baseline)]
        for condition in CONDITIONS[::-1]:
            condition_root = root / condition
            record = json.loads((condition_root / "condition_manifest.json").read_text(encoding="utf-8"))
            if record["outside_support_max_pixel_difference"] != 0 or record["frames"] != 21:
                raise ValueError(f"Technical invariant failure: {case}/{condition}")
            for item in record["files"].values():
                path = root / item["path"]
                if path.stat().st_size != item["size_bytes"] or digest(path) != item["sha256"]:
                    raise ValueError(f"Manifest output mismatch: {path}")
            final = condition_root / "projected_final_hold.png"
            rows[condition] = diagnostics(source, cv2.imread(str(final)), alpha)
            paths.append((condition.replace("vace_scale_", "scale ").replace("p", "."), final))
        figure = output / f"{case}_scale_comparison.png"
        compose_case(case, paths, figure)
        figure_inputs.append((case, figure))
        report["cases"][case] = {
            "run_manifest_sha256": digest(root / "run_manifest.json"),
            "verified_remote_files": verified_count,
            "pipeline_load_count": 1,
            "diagnostics": rows,
            "comparison_figure_sha256": digest(figure),
            "manual_review": None,
        }

    (output / "verification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "cases": list(report["cases"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
