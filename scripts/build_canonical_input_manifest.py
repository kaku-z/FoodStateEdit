#!/usr/bin/env python3
"""Audit an existing x4 preprocessing tree for the frozen benchmark.

The frozen UECFOOD256 manifest remains the source-of-truth for case identity.
This script does not copy or modify images.  It records the exact derived image
that every editing method must consume and verifies traceability back to the
frozen source image.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from PIL import Image, ImageChops, ImageOps, ImageStat


OUTPUT_FIELDS = (
    "case_id",
    "family",
    "split",
    "dataset_relative_image",
    "source_path",
    "source_sha256",
    "source_width",
    "source_height",
    "source_dhash64",
    "canonical_input_path",
    "canonical_input_resolved_path",
    "canonical_input_sha256",
    "canonical_width",
    "canonical_height",
    "nominal_x4_width",
    "nominal_x4_height",
    "alignment_trim_width",
    "alignment_trim_height",
    "scale_x",
    "scale_y",
    "canonical_dhash64",
    "dhash_distance",
    "bicubic_rgb_mae",
    "preprocess_method",
    "validation_status",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash64(image: Image.Image) -> str:
    grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(grayscale.getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return f"{value:016x}"


def hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def image_audit(source_path: Path, canonical_path: Path) -> dict[str, object]:
    with Image.open(source_path) as source_file, Image.open(canonical_path) as canonical_file:
        source = ImageOps.exif_transpose(source_file).convert("RGB")
        canonical = ImageOps.exif_transpose(canonical_file).convert("RGB")
        source_width, source_height = source.size
        canonical_width, canonical_height = canonical.size

        nominal_width = source_width * 4
        nominal_height = source_height * 4
        expected_width = nominal_width - nominal_width % 8
        expected_height = nominal_height - nominal_height % 8
        if canonical_width != expected_width or canonical_height != expected_height:
            raise ValueError(
                f"Expected x4 dimensions aligned down to multiples of 8 for {canonical_path}: "
                f"source={source.size}, expected={(expected_width, expected_height)}, "
                f"canonical={canonical.size}"
            )

        source_hash = dhash64(source)
        canonical_hash = dhash64(canonical)
        bicubic = source.resize(canonical.size, Image.Resampling.BICUBIC)
        channel_means = ImageStat.Stat(ImageChops.difference(bicubic, canonical)).mean
        bicubic_rgb_mae = sum(channel_means) / len(channel_means)

    return {
        "source_width": source_width,
        "source_height": source_height,
        "source_dhash64": source_hash,
        "canonical_width": canonical_width,
        "canonical_height": canonical_height,
        "nominal_x4_width": nominal_width,
        "nominal_x4_height": nominal_height,
        "alignment_trim_width": nominal_width - canonical_width,
        "alignment_trim_height": nominal_height - canonical_height,
        "scale_x": canonical_width / source_width,
        "scale_y": canonical_height / source_height,
        "canonical_dhash64": canonical_hash,
        "dhash_distance": hamming_hex(source_hash, canonical_hash),
        "bicubic_rgb_mae": bicubic_rgb_mae,
    }


def build_rows(
    frozen_manifest: Path,
    inventory: Path,
    canonical_root: Path,
    max_dhash_distance: int,
) -> list[dict[str, object]]:
    frozen_rows = read_csv(frozen_manifest)
    inventory_rows = {row["candidate_id"]: row for row in read_csv(inventory)}
    if len(frozen_rows) != 60:
        raise ValueError(f"Expected 60 frozen cases, found {len(frozen_rows)}")

    rows: list[dict[str, object]] = []
    for frozen in frozen_rows:
        case_id = frozen["case_id"]
        if frozen["freeze_status"] != "frozen":
            raise ValueError(f"Case {case_id} is not frozen")
        if case_id not in inventory_rows:
            raise KeyError(f"Case {case_id} is missing from inventory")

        candidate = inventory_rows[case_id]
        source_path = Path(frozen["source_path"])
        relative_path = Path(candidate["dataset_relative_image"])
        canonical_path = canonical_root / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing frozen source image: {source_path}")
        if not canonical_path.is_file():
            raise FileNotFoundError(f"Missing canonical x4 input: {canonical_path}")

        source_sha256 = sha256_file(source_path)
        if source_sha256 != frozen["source_sha256"]:
            raise ValueError(
                f"Frozen source hash mismatch for {case_id}: "
                f"expected {frozen['source_sha256']}, got {source_sha256}"
            )
        if candidate["source_sha256"] != source_sha256:
            raise ValueError(f"Inventory source hash mismatch for {case_id}")

        metrics = image_audit(source_path, canonical_path)
        if int(metrics["dhash_distance"]) > max_dhash_distance:
            raise ValueError(
                f"Canonical input changed coarse structure for {case_id}: "
                f"dHash distance {metrics['dhash_distance']} > {max_dhash_distance}"
            )
        if int(candidate["width"]) != metrics["source_width"] or int(candidate["height"]) != metrics["source_height"]:
            raise ValueError(f"Inventory dimensions mismatch for {case_id}")

        rows.append(
            {
                "case_id": case_id,
                "family": frozen["family"],
                "split": frozen["split"],
                "dataset_relative_image": relative_path.as_posix(),
                "source_path": str(source_path),
                "source_sha256": source_sha256,
                **metrics,
                "canonical_input_path": str(canonical_path),
                "canonical_input_resolved_path": str(canonical_path.resolve()),
                "canonical_input_sha256": sha256_file(canonical_path),
                "preprocess_method": "existing_precomputed_osediff_x4",
                "validation_status": f"validated_x4_aligned_to_8_dhash_le_{max_dhash_distance}",
            }
        )

    canonical_hashes = [str(row["canonical_input_sha256"]) for row in rows]
    if len(set(canonical_hashes)) != len(canonical_hashes):
        duplicates = [value for value, count in Counter(canonical_hashes).items() if count > 1]
        raise ValueError(f"Duplicate canonical input hashes: {duplicates}")
    return rows


def write_outputs(
    rows: list[dict[str, object]],
    output: Path,
    summary_path: Path,
    canonical_root: Path,
    max_dhash_distance: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (str(item["family"]), str(item["split"]), str(item["case_id"]))):
            formatted = dict(row)
            formatted["scale_x"] = f"{float(row['scale_x']):.1f}"
            formatted["scale_y"] = f"{float(row['scale_y']):.1f}"
            formatted["bicubic_rgb_mae"] = f"{float(row['bicubic_rgb_mae']):.4f}"
            writer.writerow(formatted)

    distances = [int(row["dhash_distance"]) for row in rows]
    maes = [float(row["bicubic_rgb_mae"]) for row in rows]
    summary = {
        "schema_version": 1,
        "case_count": len(rows),
        "family_counts": dict(sorted(Counter(str(row["family"]) for row in rows).items())),
        "split_counts": dict(sorted(Counter(str(row["split"]) for row in rows).items())),
        "canonical_root": str(canonical_root),
        "canonical_root_resolved": str(canonical_root.resolve()),
        "preprocess_method": "existing_precomputed_osediff_x4",
        "dimension_rule": "floor((source_dimension * 4) / 8) * 8",
        "all_dimensions_x4_aligned_to_multiple_of_8": True,
        "maximum_alignment_trim_pixels": max(
            max(int(row["alignment_trim_width"]), int(row["alignment_trim_height"])) for row in rows
        ),
        "all_source_hashes_match_frozen_manifest": True,
        "all_canonical_hashes_unique": True,
        "maximum_allowed_dhash_distance": max_dhash_distance,
        "all_dhash_distances_within_limit": max(distances) <= max_dhash_distance,
        "minimum_canonical_width": min(int(row["canonical_width"]) for row in rows),
        "minimum_canonical_height": min(int(row["canonical_height"]) for row in rows),
        "dhash_distance": {
            "minimum": min(distances),
            "median": statistics.median(distances),
            "maximum": max(distances),
        },
        "bicubic_rgb_mae": {
            "minimum": round(min(maes), 4),
            "mean": round(statistics.fmean(maes), 4),
            "maximum": round(max(maes), 4),
        },
        "policy": (
            "Every method consumes the same hash-locked canonical x4-aligned input. "
            "Frozen source images remain the identity and provenance layer."
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--max-dhash-distance", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(args.frozen_manifest, args.inventory, args.canonical_root, args.max_dhash_distance)
    write_outputs(rows, args.output, args.summary, args.canonical_root, args.max_dhash_distance)
    print(f"Validated {len(rows)} canonical inputs -> {args.output}")


if __name__ == "__main__":
    main()
