#!/usr/bin/env python3
"""Build hash-verified Qwen baseline and masked-composite diagnostic grids."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CASES = ("ramen", "soup", "rice", "cake")
SEEDS = (1, 2, 3)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def fit_panel(path: Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (width, height), (244, 244, 244))
    panel.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return panel


def raw_preservation_metrics(seed_root: Path) -> dict[str, object]:
    with Image.open(seed_root / "input.png") as source:
        base = source.convert("RGB")
    with Image.open(seed_root / "raw_qwen.png") as source:
        raw = source.convert("RGB")
        raw_size = list(raw.size)
    with Image.open(seed_root / "allowed_edit_region.png") as source:
        allowed = source.convert("L").resize(base.size, Image.Resampling.NEAREST)
    base_arr = np.asarray(base, dtype=np.int16)
    raw_arr = np.asarray(raw.resize(base.size, Image.Resampling.LANCZOS), dtype=np.int16)
    outside = np.asarray(allowed, dtype=np.uint8) == 0
    absolute = np.abs(raw_arr - base_arr)
    maximum_per_pixel = absolute.max(axis=2)
    return {
        "input_size": list(base.size),
        "raw_size": raw_size,
        "comparison": "raw output resized to input dimensions with LANCZOS; no registration",
        "outside_support_max_pixel_difference": int(maximum_per_pixel[outside].max(initial=0)),
        "outside_support_mean_absolute_difference": round(float(absolute[outside].mean()), 6),
        "outside_support_fraction_pixels_max_channel_gt_10": round(
            float((maximum_per_pixel[outside] > 10).mean()), 9
        ),
    }


def build_grid(
    artifact_root: Path,
    output: Path,
    filename: str,
    title: str,
    subtitle: str,
    expected: dict[tuple[str, int, str], str],
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite review figure: {output}")
    panel_width, panel_height = 326, 248
    row_label_width, title_height, header_height, gap = 150, 78, 48, 8
    columns = (("input", "Input"), ("seed_1", "Seed 1"), ("seed_2", "Seed 2"), ("seed_3", "Seed 3"))
    width = row_label_width + gap + len(columns) * (panel_width + gap)
    height = title_height + header_height + len(CASES) * (panel_height + gap) + 42
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), title, font=font(28, True), fill=(22, 28, 40))
    draw.text((14, 47), subtitle, font=font(17), fill=(76, 82, 94))
    for index, (_, heading) in enumerate(columns):
        x = row_label_width + gap + index * (panel_width + gap)
        bbox = draw.textbbox((0, 0), heading, font=font(18, True))
        text_width = bbox[2] - bbox[0]
        draw.text(
            (x + (panel_width - text_width) / 2, title_height + 12),
            heading,
            font=font(18, True),
            fill=(22, 28, 40),
        )

    records: list[dict[str, object]] = []
    for row, case_id in enumerate(CASES):
        paths = [artifact_root / case_id / "seed_1" / "input.png"]
        paths.extend(artifact_root / case_id / f"seed_{seed}" / filename for seed in SEEDS)
        for index, path in enumerate(paths):
            seed = 1 if index == 0 else index
            current_filename = "input.png" if index == 0 else filename
            actual = sha256_file(path)
            if actual != expected[(case_id, seed, current_filename)]:
                raise ValueError(f"Hash mismatch: {path}")
            x = row_label_width + gap + index * (panel_width + gap)
            y = title_height + header_height + row * (panel_height + gap)
            canvas.paste(fit_panel(path, panel_width, panel_height), (x, y))
            records.append(
                {
                    "case_id": case_id,
                    "column": columns[index][0],
                    "path": str(path.relative_to(ROOT)),
                    "sha256": actual,
                }
            )
        y = title_height + header_height + row * (panel_height + gap)
        draw.text((14, y + 105), case_id.upper(), font=font(20, True), fill=(22, 28, 40))

    draw.text(
        (14, height - 30),
        "Development evidence only; cake is synthetic; independent blind review remains required.",
        font=font(16),
        fill=(122, 40, 40),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    return {
        "path": str(output.relative_to(ROOT)),
        "sha256": sha256_file(output),
        "width": width,
        "height": height,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=ROOT / "artifacts/day27_qwen_image_edit_direct_baseline_v1",
    )
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    run_manifest_path = artifact_root / "run_manifest.json"
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete_requires_blind_review":
        raise ValueError(f"Unexpected run state: {manifest.get('status')}")
    if manifest.get("pipeline_load_count") != 1:
        raise ValueError("Expected exactly one pipeline load")
    completed = manifest.get("completed_outputs", [])
    if len(completed) != len(CASES) * len(SEEDS):
        raise ValueError(f"Expected 12 outputs, found {len(completed)}")

    expected = {
        (record["case_id"], int(record["seed"]), filename): data["sha256"]
        for record in completed
        for filename, data in record["files"].items()
    }
    raw_grid = build_grid(
        artifact_root,
        artifact_root / "review_grid_raw_qwen.png",
        "raw_qwen.png",
        "Day 27: Qwen-Image-Edit direct-input baseline",
        "Original model output | same input | three fixed seeds | internal non-blind review",
        expected,
    )
    composite_grid = build_grid(
        artifact_root,
        artifact_root / "review_grid_masked_composite_diagnostic.png",
        "final.png",
        "Day 27 diagnostic: support-locked composite",
        "Hard source-space mask applied after global Qwen edit; not the Qwen baseline endpoint",
        expected,
    )
    review_manifest = {
        "schema_version": "foodstateedit.day27_qwen_review.v1",
        "status": "complete_hash_verified_requires_human_review",
        "baseline_endpoint": "raw_qwen.png",
        "diagnostic_endpoint": "final.png",
        "raw_qwen_grid": raw_grid,
        "masked_composite_grid": composite_grid,
        "raw_output_preservation_diagnostics": [
            {
                "case_id": case_id,
                "seed": seed,
                **raw_preservation_metrics(artifact_root / case_id / f"seed_{seed}"),
            }
            for case_id in CASES
            for seed in SEEDS
        ],
        "interpretation_boundary": (
            "The raw Qwen output is the external-editor baseline. The support-locked composite "
            "is retained only to diagnose source-space mask misregistration after a global edit."
        ),
    }
    output = artifact_root / "review_grid_manifest.json"
    output.write_text(json.dumps(review_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output), "raw_grid": raw_grid["path"], "diagnostic_grid": composite_grid["path"]}, indent=2))


if __name__ == "__main__":
    main()
