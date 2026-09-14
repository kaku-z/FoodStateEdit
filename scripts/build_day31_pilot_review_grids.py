#!/usr/bin/env python3
"""Build family-level reference/planar/relative3D review grids for Day31."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


TILE = (320, 240)
HEADER = 32
COLUMNS = ("reference.png", "planar_final.png", "relative3d_final.png")
LABELS = ("Reference", "Planar control", "Relative3D control")


def tile(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), TILE, Image.Resampling.LANCZOS)


def build_grid(root: Path, cases: list[dict], family: str) -> Image.Image:
    width = TILE[0] * len(COLUMNS)
    height = HEADER + len(cases) * (TILE[1] + HEADER)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(LABELS):
        draw.text((column * TILE[0] + 8, 9), label, fill="black")
    for row, case in enumerate(cases):
        y = HEADER + row * (TILE[1] + HEADER)
        draw.rectangle((0, y, width, y + HEADER), fill=(238, 241, 245))
        draw.text((8, y + 9), f"{case['case_id']} / {family} / deterministic pilot template", fill="black")
        for column, filename in enumerate(COLUMNS):
            image = tile(root / case["case_id"] / filename)
            x = column * TILE[0] + (TILE[0] - image.width) // 2
            image_y = y + HEADER + (TILE[1] - image.height) // 2
            canvas.paste(image, (x, image_y))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_root}")
    manifest = json.loads((args.input_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True)
    outputs = []
    for family in sorted(manifest["family_counts"]):
        cases = [case for case in manifest["cases"] if case["family"] == family]
        output = args.output_root / f"{family}_final_controls.png"
        build_grid(args.input_root, cases, family).save(output, optimize=True)
        outputs.append(str(output))
    print(json.dumps({"outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
