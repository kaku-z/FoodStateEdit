#!/usr/bin/env python3
"""Render the frozen vector anchor specs into five-layer raster annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


LAYER_NAMES = ("rigid", "contact", "material", "hole")
PREVIEW_COLORS = {
    "rigid": (255, 64, 64, 145),
    "contact": (255, 220, 40, 170),
    "material": (50, 210, 80, 145),
    "hole": (60, 130, 255, 145),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def to_pixel(point: list[float], width: int, height: int) -> tuple[int, int]:
    return round(point[0] * (width - 1)), round(point[1] * (height - 1))


def render_primitives(primitives: list[dict[str, object]], width: int, height: int) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    min_dimension = min(width, height)
    for primitive in primitives:
        primitive_type = primitive["type"]
        if primitive_type in {"line", "polyline"}:
            points = [to_pixel(point, width, height) for point in primitive["points"]]
            line_width = max(1, round(float(primitive["width"]) * min_dimension))
            draw.line(points, fill=255, width=line_width, joint="curve")
            radius = line_width // 2
            for x, y in (points[0], points[-1]):
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
        elif primitive_type == "ellipse":
            center_x, center_y = to_pixel(primitive["center"], width, height)
            radius_x = round(float(primitive["radii"][0]) * width)
            radius_y = round(float(primitive["radii"][1]) * height)
            draw.ellipse(
                (center_x - radius_x, center_y - radius_y, center_x + radius_x, center_y + radius_y),
                fill=255,
            )
        elif primitive_type == "polygon":
            points = [to_pixel(point, width, height) for point in primitive["points"]]
            draw.polygon(points, fill=255)
        else:
            raise ValueError(f"Unsupported primitive type: {primitive_type}")
    return mask


def coverage(mask: Image.Image) -> int:
    histogram = mask.histogram()
    return sum(histogram[1:])


def intersection_pixels(left: Image.Image, right: Image.Image) -> int:
    return coverage(Image.frombytes("L", left.size, bytes(a & b for a, b in zip(left.tobytes(), right.tobytes()))))


def polyline_length(points: list[list[float]], width: int, height: int) -> float:
    pixels = [to_pixel(point, width, height) for point in points]
    return sum(math.hypot(right[0] - left[0], right[1] - left[1]) for left, right in zip(pixels, pixels[1:]))


def make_case_manifest(
    spec: dict[str, object],
    canonical: dict[str, str],
    relative_dir: Path,
    builder_commit: str,
    created_at: str,
) -> dict[str, object]:
    width = int(canonical["canonical_width"])
    height = int(canonical["canonical_height"])
    action = spec["action"]
    contact_anchors = [
        list(to_pixel(point, width, height)) for point in action["contact_anchors_normalized"]
    ]
    return {
        "schema_version": "foodstateedit.case.v1",
        "case_id": spec["anchor_id"],
        "split": "pilot",
        "source": {
            "path": canonical["canonical_input_path"],
            "sha256": canonical["canonical_input_sha256"],
            "kind": "real",
            "license": "UECFOOD256_noncommercial_research_only",
            "has_target_utensil": False,
            "width": width,
            "height": height,
        },
        "scene": {
            "family": spec["family"],
            "dish": spec["dish"],
            "container": spec["container"],
            "food_materials": spec["food_materials"],
        },
        "action": {
            "type": action["type"],
            "utensil": action["utensil"],
            "payload_relation": action["payload_relation"],
            "amount": action["amount"],
            "amount_unit": action["amount_unit"],
        },
        "controls": {
            "contact_anchors": contact_anchors,
            "target_direction": action["target_direction_normalized"],
            "manual_correction": True,
            "annotation_seconds": 0,
        },
        "layers": {
            "rigid": str(relative_dir / "mask_rigid.png"),
            "contact": str(relative_dir / "mask_contact.png"),
            "material": str(relative_dir / "mask_material.png"),
            "hole": str(relative_dir / "mask_hole.png"),
            "protect": str(relative_dir / "mask_protect.png"),
            "edit_alpha": str(relative_dir / "edit_alpha.png"),
        },
        "constraints": [
            {"id": "benchmark_source_case", "category": "provenance", "criterion": spec["case_id"]},
            *[
                {"id": item["id"], "category": "hard_action", "criterion": item["criterion"]}
                for item in spec["hard_constraints"]
            ],
            {
                "id": "exact_protection",
                "category": "preservation",
                "criterion": "max_absolute_pixel_difference_where_mask_protect_is_255 == 0",
            },
        ],
        "provenance": {
            "builder": "scripts/build_anchor_annotations.py",
            "builder_commit": builder_commit,
            "created_at": created_at,
        },
    }


def build_anchor(
    spec: dict[str, object],
    canonical: dict[str, str],
    output_root: Path,
    preview_root: Path,
    feather_radius_norm: float,
    builder_commit: str,
    created_at: str,
) -> dict[str, object]:
    width = int(canonical["canonical_width"])
    height = int(canonical["canonical_height"])
    output_dir = output_root / str(spec["anchor_id"])
    output_dir.mkdir(parents=True, exist_ok=True)

    masks = {
        name: render_primitives(spec["layers"][name]["primitives"], width, height)
        for name in LAYER_NAMES
    }
    for name, mask in masks.items():
        if coverage(mask) == 0:
            raise ValueError(f"{spec['anchor_id']} has an empty {name} mask")
        mask.save(output_dir / f"mask_{name}.png", optimize=True)

    if intersection_pixels(masks["contact"], masks["rigid"]) == 0:
        raise ValueError(f"{spec['anchor_id']} contact mask does not touch rigid mask")
    if intersection_pixels(masks["contact"], masks["material"]) == 0:
        raise ValueError(f"{spec['anchor_id']} contact mask does not touch material mask")

    hard_union = Image.new("L", (width, height), 0)
    for mask in masks.values():
        hard_union = Image.frombytes(
            "L", hard_union.size, bytes(max(a, b) for a, b in zip(hard_union.tobytes(), mask.tobytes()))
        )
    feather_radius = max(1, round(feather_radius_norm * min(width, height)))
    edit_alpha = hard_union.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    edit_alpha.save(output_dir / "edit_alpha.png", optimize=True)
    protect = edit_alpha.point(lambda value: 255 if value == 0 else 0)
    protect.save(output_dir / "mask_protect.png", optimize=True)
    if coverage(protect) == 0:
        raise ValueError(f"{spec['anchor_id']} has no protected pixels")

    length_ratio = None
    if spec["family"] == "strand":
        target_points = spec["layers"]["material"]["primitives"][0]["points"]
        source_points = spec["layers"]["hole"]["primitives"][0]["points"]
        length_ratio = polyline_length(target_points, width, height) / polyline_length(source_points, width, height)
        if not 0.95 <= length_ratio <= 1.05:
            raise ValueError(f"{spec['anchor_id']} violates centerline conservation: ratio={length_ratio:.6f}")

    relative_dir = Path("benchmark") / "anchors_v1" / str(spec["anchor_id"])
    manifest = make_case_manifest(spec, canonical, relative_dir, builder_commit, created_at)
    (output_dir / "case_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with Image.open(canonical["canonical_input_path"]) as source_file:
        preview = source_file.convert("RGBA")
    for name, color in PREVIEW_COLORS.items():
        color_layer = Image.new("RGBA", preview.size, color)
        preview = Image.composite(color_layer, preview, masks[name])
    preview_root.mkdir(parents=True, exist_ok=True)
    preview.save(preview_root / f"{spec['anchor_id']}_overlay.png", optimize=True)

    pixels = width * height
    return {
        "anchor_id": spec["anchor_id"],
        "case_id": spec["case_id"],
        "family": spec["family"],
        "canonical_width": width,
        "canonical_height": height,
        "canonical_input_sha256": canonical["canonical_input_sha256"],
        "coverage_fraction": {
            name: round(coverage(mask) / pixels, 6) for name, mask in masks.items()
        },
        "edit_alpha_nonzero_fraction": round(coverage(edit_alpha) / pixels, 6),
        "protect_fraction": round(coverage(protect) / pixels, 6),
        "contact_rigid_intersection_pixels": intersection_pixels(masks["contact"], masks["rigid"]),
        "contact_material_intersection_pixels": intersection_pixels(masks["contact"], masks["material"]),
        "target_source_centerline_length_ratio": round(length_ratio, 6) if length_ratio is not None else None,
        "status": "five_layer_annotation_complete",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preview-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--builder-commit", required=True)
    parser.add_argument("--created-at", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = json.loads(args.specs.read_text(encoding="utf-8"))
    canonical_rows = {row["case_id"]: row for row in read_csv(args.canonical_manifest)}
    if specs["five_layer_order"] != ["rigid", "contact", "material", "hole", "protect"]:
        raise ValueError("Unexpected five-layer order")
    results = []
    for spec in specs["anchors"]:
        canonical = canonical_rows[spec["case_id"]]
        if canonical["split"] != "pilot" or canonical["family"] != spec["family"]:
            raise ValueError(f"Anchor/source mismatch for {spec['anchor_id']}")
        results.append(
            build_anchor(
                spec,
                canonical,
                args.output_root,
                args.preview_root,
                float(specs["edit_alpha_policy"]["feather_radius_normalized_min_dimension"]),
                args.builder_commit,
                args.created_at,
            )
        )
    summary = {
        "schema_version": 1,
        "anchor_count": len(results),
        "five_layer_order": specs["five_layer_order"],
        "anchors": results,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Built {len(results)} five-layer anchor annotations -> {args.output_root}")


if __name__ == "__main__":
    main()
