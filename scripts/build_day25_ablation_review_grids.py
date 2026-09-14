#!/usr/bin/env python3
"""Build fixed 3-seed x 5-condition Day 25 visual review grids."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


CONDITIONS = (
    "native_scale_1p0",
    "planar_scale_1p0",
    "relative3d_scale_1p0",
    "relative3d_scale_0p6",
    "relative3d_scale_0p8",
)


LEGACY_SEED1 = {
    ("soup", "native_scale_1p0"): "artifacts/day22_native_input_vace_pilot_v2_soup/native_input_text_mask_only",
    ("soup", "planar_scale_1p0"): "artifacts/day19_multimaterial_gp40_recovered_20260909_v1/soup__planar",
    ("soup", "relative3d_scale_1p0"): "artifacts/day19_multimaterial_gp40_recovered_20260909_v1/soup__relative3d",
    ("soup", "relative3d_scale_0p6"): "artifacts/day21_realism_scale_sweep_v1/soup/vace_scale_0p6",
    ("soup", "relative3d_scale_0p8"): "artifacts/day21_realism_scale_sweep_v1/soup/vace_scale_0p8",
    ("rice", "planar_scale_1p0"): "artifacts/day19_multimaterial_gp40_recovered_20260909_v1/rice__planar",
    ("rice", "relative3d_scale_1p0"): "artifacts/day19_multimaterial_gp40_recovered_20260909_v1/rice__relative3d",
    ("rice", "relative3d_scale_0p6"): "artifacts/day21_realism_scale_sweep_v1/rice/vace_scale_0p6",
    ("rice", "relative3d_scale_0p8"): "artifacts/day21_realism_scale_sweep_v1/rice/vace_scale_0p8",
    ("cake", "planar_scale_1p0"): "artifacts/day19_multimaterial_gp40_recovered_20260909_v1/cake__planar",
    ("cake", "relative3d_scale_1p0"): "artifacts/day20_cake_recovery_gp40_20260909_v1/cake__relative3d",
    ("cake", "relative3d_scale_0p6"): "artifacts/day21_realism_scale_sweep_v1/cake/vace_scale_0p6",
    ("cake", "relative3d_scale_0p8"): "artifacts/day21_realism_scale_sweep_v1/cake/vace_scale_0p8",
}


def resolve_artifact(repo: Path, collection: Path, case: str, seed: int, condition: str, name: str) -> Path:
    legacy = LEGACY_SEED1.get((case, condition)) if seed == 1 else None
    if legacy:
        return repo / legacy / name
    return collection / case / f"seed_{seed}" / condition / name


def fit(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        return ImageOps.pad(source.convert("RGB"), size, color="white", method=Image.Resampling.LANCZOS)


def build_grid(repo: Path, collection: Path, case: str, artifact: str, destination: Path) -> None:
    panel = (420, 300) if artifact == "projected_review.png" else (360, 360)
    top = 72
    left = 86
    canvas = Image.new("RGB", (left + panel[0] * len(CONDITIONS), top + panel[1] * 3), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, condition in enumerate(CONDITIONS):
        draw.text((left + column * panel[0] + 8, 16), condition, fill="black", font=font)
    for row, seed in enumerate((1, 2, 3)):
        draw.text((12, top + row * panel[1] + 12), f"seed {seed}", fill="black", font=font)
        for column, condition in enumerate(CONDITIONS):
            path = resolve_artifact(repo, collection, case, seed, condition, artifact)
            if not path.is_file():
                raise FileNotFoundError(path)
            canvas.paste(fit(path, panel), (left + column * panel[0], top + row * panel[1]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty review root: {args.output_root}")
    for case in args.case:
        build_grid(args.repo.resolve(), args.collection_root.resolve(), case, "projected_final_hold.png", args.output_root / f"{case}_final_grid.png")
        build_grid(args.repo.resolve(), args.collection_root.resolve(), case, "projected_review.png", args.output_root / f"{case}_process_grid.png")
    print(f"Built review grids for {len(args.case)} cases in {args.output_root}")


if __name__ == "__main__":
    main()
