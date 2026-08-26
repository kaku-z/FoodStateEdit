#!/usr/bin/env python3
"""Build a deterministic, review-first UECFOOD256 candidate inventory.

The script never copies source images. It writes a lightweight CSV containing
paths, hashes, dimensions, perceptual hashes, and explicit manual-review fields,
plus contact sheets for screening utensil/action eligibility.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps


@dataclass(frozen=True)
class FamilySpec:
    dish: str
    utensil: str
    action: str
    class_ids: tuple[int, ...]


FAMILIES = {
    "liquid": FamilySpec(
        dish="soup",
        utensil="spoon",
        action="scoop_liquid",
        class_ids=(36, 37, 89, 90, 91, 105, 128, 129, 135, 156, 168, 173, 176, 183, 186, 201, 205, 221, 240, 252, 256),
    ),
    "granular": FamilySpec(
        dish="fried_rice",
        utensil="spoon_or_spatula",
        action="scoop_payload",
        # Exclude class 82 (omelet with fried rice): the egg layer hides the
        # granular source material and makes scoop-state evaluation ambiguous.
        class_ids=(9, 227),
    ),
    "strand": FamilySpec(
        dish="ramen_or_udon",
        utensil="chopsticks",
        action="lift_strand",
        class_ids=(20, 21, 23, 24, 25, 96, 102, 111, 115),
    ),
    "strand_contact": FamilySpec(
        dish="spaghetti",
        utensil="fork",
        action="twirl_and_lift",
        class_ids=(27, 84),
    ),
}

# These images were used during method development and may only enter a pilot
# split. Pinning them makes the leakage constraint explicit and reproducible.
PILOT_ONLY_PINS = {
    "strand": ("20/1897.jpg", "23/11555.jpg"),
}

CSV_FIELDS = (
    "candidate_id",
    "family",
    "dish",
    "utensil",
    "action",
    "dataset",
    "class_id",
    "class_name",
    "dataset_relative_image",
    "source_path",
    "source_sha256",
    "width",
    "height",
    "format",
    "dhash64",
    "exact_duplicate_group",
    "near_duplicate_review",
    "license_status",
    "pilot_only",
    "has_target_utensil",
    "has_target_action",
    "food_visible",
    "container_visible",
    "edit_space",
    "manual_status",
    "exclusion_reason",
    "freeze_status",
    "split",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(seed: int, relative_path: str) -> str:
    return hashlib.sha256(f"{seed}:{relative_path}".encode("utf-8")).hexdigest()


def load_categories(path: Path) -> dict[int, str]:
    categories: dict[int, str] = {}
    with path.open("r", encoding="utf-8-sig") as stream:
        next(stream)
        for line in stream:
            raw_id, name = line.rstrip("\n").split("\t", 1)
            categories[int(raw_id)] = name
    return categories


def image_files(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )


def select_balanced(
    dataset_root: Path,
    family: str,
    spec: FamilySpec,
    count: int,
    seed: int,
) -> list[Path]:
    pools: dict[int, list[Path]] = {}
    for class_id in spec.class_ids:
        candidates = image_files(dataset_root / str(class_id))
        pools[class_id] = sorted(
            candidates,
            key=lambda path: stable_rank(seed, path.relative_to(dataset_root).as_posix()),
        )

    selected: list[Path] = []
    seen: set[Path] = set()
    for relative in PILOT_ONLY_PINS.get(family, ()):
        path = dataset_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Pinned pilot source is missing: {path}")
        selected.append(path)
        seen.add(path)

    cursors = {class_id: 0 for class_id in spec.class_ids}
    while len(selected) < count:
        progressed = False
        for class_id in spec.class_ids:
            pool = pools[class_id]
            cursor = cursors[class_id]
            while cursor < len(pool) and pool[cursor] in seen:
                cursor += 1
            cursors[class_id] = cursor
            if cursor >= len(pool):
                continue
            path = pool[cursor]
            cursors[class_id] += 1
            selected.append(path)
            seen.add(path)
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise RuntimeError(f"Only found {len(selected)} candidates for {family}; requested {count}")
    return selected


def dhash64(image: Image.Image) -> str:
    grayscale = ImageOps.exif_transpose(image).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(grayscale.getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return f"{value:016x}"


def hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def build_rows(dataset_root: Path, count: int, seed: int) -> list[dict[str, object]]:
    categories = load_categories(dataset_root / "category.txt")
    rows: list[dict[str, object]] = []
    prefixes = {"liquid": "soup", "granular": "rice", "strand": "noodle", "strand_contact": "pasta"}

    for family, spec in FAMILIES.items():
        selected = select_balanced(dataset_root, family, spec, count, seed)
        pins = set(PILOT_ONLY_PINS.get(family, ()))
        for index, path in enumerate(selected, start=1):
            relative = path.relative_to(dataset_root).as_posix()
            class_id = int(relative.split("/", 1)[0])
            with Image.open(path) as image:
                oriented = ImageOps.exif_transpose(image)
                width, height = oriented.size
                image_format = image.format or path.suffix.lstrip(".").upper()
                perceptual_hash = dhash64(image)
            rows.append(
                {
                    "candidate_id": f"{prefixes[family]}_{index:03d}",
                    "family": family,
                    "dish": spec.dish,
                    "utensil": spec.utensil,
                    "action": spec.action,
                    "dataset": "UECFOOD256",
                    "class_id": class_id,
                    "class_name": categories[class_id],
                    "dataset_relative_image": relative,
                    "source_path": str(path),
                    "source_sha256": sha256_file(path),
                    "width": width,
                    "height": height,
                    "format": image_format,
                    "dhash64": perceptual_hash,
                    "exact_duplicate_group": "",
                    "near_duplicate_review": "",
                    "license_status": "noncommercial_research_only_do_not_redistribute",
                    "pilot_only": str(relative in pins).lower(),
                    "has_target_utensil": "unreviewed",
                    "has_target_action": "unreviewed",
                    "food_visible": "unreviewed",
                    "container_visible": "unreviewed",
                    "edit_space": "unreviewed",
                    "manual_status": "unreviewed",
                    "exclusion_reason": "",
                    "freeze_status": "candidate",
                    "split": "pilot" if relative in pins else "unassigned",
                }
            )
    return rows


def annotate_duplicates(rows: list[dict[str, object]], near_threshold: int) -> list[dict[str, object]]:
    by_sha: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_sha[str(row["source_sha256"])].append(row)

    group_number = 0
    for duplicate_rows in by_sha.values():
        if len(duplicate_rows) < 2:
            continue
        group_number += 1
        group_id = f"exact_{group_number:03d}"
        for row in duplicate_rows:
            row["exact_duplicate_group"] = group_id

    near_pairs: list[dict[str, object]] = []
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            if left["source_sha256"] == right["source_sha256"]:
                continue
            distance = hamming_hex(str(left["dhash64"]), str(right["dhash64"]))
            if distance <= near_threshold:
                pair_id = f"{left['candidate_id']}|{right['candidate_id']}"
                left["near_duplicate_review"] = ";".join(filter(None, [str(left["near_duplicate_review"]), pair_id]))
                right["near_duplicate_review"] = ";".join(filter(None, [str(right["near_duplicate_review"]), pair_id]))
                near_pairs.append(
                    {
                        "left": left["candidate_id"],
                        "right": right["candidate_id"],
                        "hamming_distance": distance,
                    }
                )
    return near_pairs


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def make_board(dataset_root: Path, output: Path, family: str, rows: list[dict[str, object]]) -> None:
    columns = 4
    thumb_width, thumb_height = 300, 220
    label_height = 58
    gap = 12
    rows_count = (len(rows) + columns - 1) // columns
    canvas_width = gap + columns * (thumb_width + gap)
    canvas_height = 54 + gap + rows_count * (thumb_height + label_height + gap)
    board = Image.new("RGB", (canvas_width, canvas_height), "#eeeeee")
    draw = ImageDraw.Draw(board)
    font = ImageFont.load_default()
    draw.text((gap, 16), f"FoodStateEdit Day 2 review: {family} ({len(rows)} candidates)", fill="black", font=font)

    for index, row in enumerate(rows):
        grid_x = index % columns
        grid_y = index // columns
        x = gap + grid_x * (thumb_width + gap)
        y = 54 + grid_y * (thumb_height + label_height + gap)
        source = dataset_root / str(row["dataset_relative_image"])
        with Image.open(source) as image:
            thumb = ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), (thumb_width, thumb_height))
        tile = Image.new("RGB", (thumb_width, thumb_height), "white")
        offset = ((thumb_width - thumb.width) // 2, (thumb_height - thumb.height) // 2)
        tile.paste(thumb, offset)
        board.paste(tile, (x, y))
        label = (
            f"{row['candidate_id']} | class {row['class_id']}\n"
            f"{row['class_name']}\n{row['dataset_relative_image']}"
        )
        draw.multiline_text((x, y + thumb_height + 4), label, fill="black", font=font, spacing=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    board.save(output, optimize=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count-per-family", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--near-duplicate-threshold", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not (dataset_root / "category.txt").is_file():
        raise FileNotFoundError(f"Not a UECFOOD256 root: {dataset_root}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(dataset_root, args.count_per_family, args.seed)
    near_pairs = annotate_duplicates(rows, args.near_duplicate_threshold)
    write_csv(output_dir / "candidate_inventory_v0.csv", rows)

    for family in FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        make_board(dataset_root, output_dir / f"review_{family}.png", family, family_rows)

    summary = {
        "schema_version": "foodstateedit.candidate_inventory_summary.v0",
        "dataset": "UECFOOD256",
        "dataset_root": str(dataset_root),
        "dataset_readme_sha256": sha256_file(dataset_root / "README.txt"),
        "category_file_sha256": sha256_file(dataset_root / "category.txt"),
        "selection_seed": args.seed,
        "count_per_family": args.count_per_family,
        "total_candidates": len(rows),
        "family_counts": dict(Counter(str(row["family"]) for row in rows)),
        "class_counts": dict(Counter(f"{row['class_id']}:{row['class_name']}" for row in rows)),
        "exact_duplicate_groups": len({row["exact_duplicate_group"] for row in rows if row["exact_duplicate_group"]}),
        "near_duplicate_threshold": args.near_duplicate_threshold,
        "near_duplicate_pairs": near_pairs,
        "manual_review_required": True,
        "redistribute_source_images": False,
    }
    (output_dir / "inventory_summary_v0.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
