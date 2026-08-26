#!/usr/bin/env python3
"""Render contact sheets for the provisional 15-case family splits."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed-inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def make_board(output: Path, family: str, rows: list[dict[str, str]]) -> None:
    columns = 5
    thumb_width, thumb_height = 280, 210
    label_height = 56
    gap = 12
    row_count = (len(rows) + columns - 1) // columns
    board = Image.new(
        "RGB",
        (gap + columns * (thumb_width + gap), 54 + row_count * (thumb_height + label_height + gap)),
        "#eeeeee",
    )
    draw = ImageDraw.Draw(board)
    font = ImageFont.load_default()
    draw.text((gap, 16), f"FoodStateEdit provisional split: {family} (5 pilot + 10 test)", fill="black", font=font)
    for index, row in enumerate(rows):
        x = gap + (index % columns) * (thumb_width + gap)
        y = 54 + (index // columns) * (thumb_height + label_height + gap)
        with Image.open(row["source_path"]) as image:
            thumb = ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), (thumb_width, thumb_height))
        tile = Image.new("RGB", (thumb_width, thumb_height), "white")
        tile.paste(thumb, ((thumb_width - thumb.width) // 2, (thumb_height - thumb.height) // 2))
        board.paste(tile, (x, y))
        draw.multiline_text(
            (x, y + thumb_height + 4),
            f"{row['candidate_id']} | {row['split']}\nclass {row['class_id']}: {row['class_name']}\n{row['dataset_relative_image']}",
            fill="black",
            font=font,
            spacing=2,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    board.save(output, optimize=True)


def main() -> None:
    args = parse_args()
    with args.reviewed_inventory.open("r", encoding="utf-8-sig", newline="") as stream:
        frozen = [row for row in csv.DictReader(stream) if row["freeze_status"] == "provisional_frozen"]
    by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in frozen:
        by_family[row["family"]].append(row)
    if len(frozen) != 60 or set(by_family) != {"liquid", "granular", "strand", "strand_contact"}:
        raise ValueError("Expected exactly 60 provisional cases across four families")
    for family, rows in by_family.items():
        rows.sort(key=lambda row: (0 if row["split"] == "pilot" else 1, row["candidate_id"]))
        if sum(row["split"] == "pilot" for row in rows) != 5 or sum(row["split"] == "test" for row in rows) != 10:
            raise ValueError(f"Bad split counts for {family}")
        make_board(args.output_dir / f"frozen_{family}.png", family, rows)


if __name__ == "__main__":
    main()
