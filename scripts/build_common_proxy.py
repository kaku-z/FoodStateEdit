#!/usr/bin/env python3
"""Build one deterministic, same-control proxy package for all four anchors.

This is a geometry proxy, not a target photograph.  It reuses source texture,
procedural utensil shading, explicit relative layer depth, and the frozen five
semantic masks.  It never loads or downloads a learned model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


MASK_FILES = {
    "rigid": "mask_rigid.png",
    "contact": "mask_contact.png",
    "material": "mask_material.png",
    "hole": "mask_hole.png",
    "protect": "mask_protect.png",
    "edit_alpha": "edit_alpha.png",
}

PROMPTS = {
    "soup_spoon_001": (
        "A realistic close-up food photograph of one stainless-steel spoon scooping potage. "
        "The spoon contains visible soup; the liquid surface remains level with no crater. "
        "Preserve the handled white soup crock, garnish, table, lighting, and background."
    ),
    "fried_rice_spatula_001": (
        "A realistic close-up food photograph of one serving spatula lifting a coherent scoop "
        "of fried rice. Distinct rice grains rest on the blade and one shallow source reduction "
        "is visible. Preserve the decorated plate, ingredients, lighting, and background."
    ),
    "ramen_chopsticks_001": (
        "A realistic close-up food photograph of exactly two separate wooden chopsticks lifting "
        "one continuous glossy udon strand. The noodle passes between both tips, remains connected "
        "to the bowl at both hanging ends, and obeys rear-chopstick, noodle, front-chopstick order. "
        "Preserve the bowl, broth, scallions, tray, table, and background."
    ),
    "pasta_fork_001": (
        "A realistic close-up food photograph of exactly one stainless-steel fork twirling and "
        "lifting several continuous spaghetti strands. Pasta visibly wraps the four tines and at "
        "least one strand remains connected to the plate. Preserve the sauce, eggplant, parsley, "
        "decorated plate, lighting, and background."
    ),
}

NEGATIVE_PROMPT = (
    "hand, arm, person, fingers, duplicate utensil, extra utensil, malformed utensil, fused "
    "chopsticks, third chopstick, disconnected food, duplicated food, floating payload, spill, "
    "source crater, background change, plate change, bowl change, blur, ghosting, text, watermark"
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


def working_size(width: int, height: int, long_side: int = 736) -> tuple[int, int]:
    scale = long_side / max(width, height)
    target_width = max(16, round(width * scale / 16) * 16)
    target_height = max(16, round(height * scale / 16) * 16)
    return target_width, target_height


def to_pixel(point: list[float], width: int, height: int) -> tuple[int, int]:
    return round(point[0] * (width - 1)), round(point[1] * (height - 1))


def render_primitive(primitive: dict[str, object], width: int, height: int) -> np.ndarray:
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    min_dimension = min(width, height)
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
        draw.ellipse((center_x - radius_x, center_y - radius_y, center_x + radius_x, center_y + radius_y), fill=255)
    elif primitive_type == "polygon":
        draw.polygon([to_pixel(point, width, height) for point in primitive["points"]], fill=255)
    else:
        raise ValueError(f"Unsupported primitive type: {primitive_type}")
    return np.asarray(image)


def load_mask(path: Path, size: tuple[int, int], resample: Image.Resampling) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L").resize(size, resample=resample))


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        raise ValueError("Cannot compute a bounding box for an empty mask")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def source_texture(source: np.ndarray, source_mask: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    sx0, sy0, sx1, sy1 = bbox(source_mask)
    tx0, ty0, tx1, ty1 = bbox(target_mask)
    crop = Image.fromarray(source[sy0:sy1, sx0:sx1]).resize((tx1 - tx0, ty1 - ty0), Image.Resampling.BICUBIC)
    texture = np.zeros_like(source)
    texture[ty0:ty1, tx0:tx1] = np.asarray(crop)
    return texture


def utensil_texture(width: int, height: int, utensil: str) -> np.ndarray:
    yy, xx = np.indices((height, width))
    if utensil == "chopsticks":
        grain = 12.0 * np.sin((xx + 0.35 * yy) / 13.0)
        red = np.clip(158 + grain, 0, 255)
        green = np.clip(102 + 0.65 * grain, 0, 255)
        blue = np.clip(52 + 0.35 * grain, 0, 255)
    else:
        shine = 48.0 * (0.5 + 0.5 * np.sin((xx + yy) / 11.0))
        red = np.clip(150 + shine, 0, 255)
        green = np.clip(155 + shine, 0, 255)
        blue = np.clip(162 + shine, 0, 255)
    return np.stack([red, green, blue], axis=-1).astype(np.uint8)


def composite(base: np.ndarray, overlay: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = base.copy()
    active = mask > 127
    result[active] = overlay[active]
    return result


def render_proxy(
    source: np.ndarray,
    masks: dict[str, np.ndarray],
    anchor_spec: dict[str, object],
) -> np.ndarray:
    hole_binary = (masks["hole"] > 127).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    inpaint_mask = cv2.dilate(hole_binary, kernel, iterations=1)
    repaired_bgr = cv2.inpaint(cv2.cvtColor(source, cv2.COLOR_RGB2BGR), inpaint_mask, 5.0, cv2.INPAINT_TELEA)
    proxy = cv2.cvtColor(repaired_bgr, cv2.COLOR_BGR2RGB)

    contact_blur = cv2.GaussianBlur(masks["contact"], (0, 0), sigmaX=4.0).astype(np.float32) / 255.0
    proxy = np.clip(proxy.astype(np.float32) * (1.0 - 0.22 * contact_blur[..., None]), 0, 255).astype(np.uint8)

    material_texture = source_texture(source, masks["hole"], masks["material"])
    utensil = anchor_spec["action"]["utensil"]
    rigid_texture = utensil_texture(source.shape[1], source.shape[0], utensil)
    rigid_outline = cv2.dilate((masks["rigid"] > 127).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    rigid_inner = masks["rigid"] > 127
    outline_only = np.logical_and(rigid_outline, np.logical_not(rigid_inner))
    proxy[outline_only] = (58, 48, 40) if utensil == "chopsticks" else (78, 82, 88)

    if utensil == "chopsticks":
        primitives = anchor_spec["layers"]["rigid"]["primitives"]
        if len(primitives) != 2:
            raise ValueError("Chopstick proxy requires exactly two rigid primitives")
        rear = render_primitive(primitives[0], source.shape[1], source.shape[0])
        front = render_primitive(primitives[1], source.shape[1], source.shape[0])
        proxy = composite(proxy, rigid_texture, rear)
        proxy = composite(proxy, material_texture, masks["material"])
        proxy = composite(proxy, rigid_texture, front)
    else:
        proxy = composite(proxy, rigid_texture, masks["rigid"])
        proxy = composite(proxy, material_texture, masks["material"])

    edit_support = masks["edit_alpha"] > 0
    proxy[np.logical_not(edit_support)] = source[np.logical_not(edit_support)]
    return proxy


def relative_depth(source: np.ndarray, masks: dict[str, np.ndarray], utensil: str) -> np.ndarray:
    depth = np.full(source.shape[:2], 176, dtype=np.uint8)
    depth[masks["hole"] > 127] = 150
    depth[masks["material"] > 127] = 88
    depth[masks["rigid"] > 127] = 76
    depth[masks["contact"] > 127] = 82
    if utensil == "chopsticks":
        depth[masks["material"] > 127] = 80
    return cv2.GaussianBlur(depth, (0, 0), sigmaX=2.0)


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask.astype(np.uint8), mode="L").save(path, optimize=True)


def make_preview(source: np.ndarray, proxy: np.ndarray, depth: np.ndarray, masks: dict[str, np.ndarray]) -> Image.Image:
    overlay = source.copy()
    colors = {"rigid": (255, 64, 64), "contact": (255, 220, 40), "material": (50, 210, 80), "hole": (60, 130, 255)}
    for name, color in colors.items():
        active = masks[name] > 127
        overlay[active] = np.asarray(color, dtype=np.uint8)
    panels = [Image.fromarray(source), Image.fromarray(proxy), Image.fromarray(depth).convert("RGB"), Image.fromarray(overlay)]
    canvas = Image.new("RGB", (source.shape[1] * 2, source.shape[0] * 2), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % 2) * source.shape[1], (index // 2) * source.shape[0]))
    return canvas


def build_case(
    anchor: dict[str, str],
    canonical: dict[str, str],
    spec: dict[str, object],
    anchor_root: Path,
    output_root: Path,
    source_root: Path | None = None,
) -> dict[str, object]:
    anchor_id = anchor["anchor_id"]
    source_path = (
        source_root / f"{anchor['case_id']}.jpg"
        if source_root is not None
        else Path(canonical["canonical_input_path"])
    )
    if sha256_file(source_path) != canonical["canonical_input_sha256"]:
        raise ValueError(f"Canonical source hash mismatch: {anchor_id}")
    with Image.open(source_path) as source_file:
        source_image = ImageOps.exif_transpose(source_file).convert("RGB")
    size = working_size(*source_image.size)
    source_image = source_image.resize(size, Image.Resampling.LANCZOS)
    source = np.asarray(source_image)

    input_dir = anchor_root / anchor_id
    masks = {
        name: load_mask(input_dir / filename, size, Image.Resampling.BILINEAR if name == "edit_alpha" else Image.Resampling.NEAREST)
        for name, filename in MASK_FILES.items()
    }
    masks["protect"] = np.where(masks["edit_alpha"] == 0, 255, 0).astype(np.uint8)
    for name in ("rigid", "contact", "material", "hole"):
        if not np.any(masks[name] > 127):
            raise ValueError(f"Empty resized {name} mask for {anchor_id}")

    proxy = render_proxy(source, masks, spec)
    depth = relative_depth(source, masks, spec["action"]["utensil"])
    support = masks["edit_alpha"] > 0
    outside_max_diff = int(np.abs(proxy.astype(np.int16) - source.astype(np.int16))[np.logical_not(support)].max(initial=0))
    if outside_max_diff != 0:
        raise ValueError(f"Proxy changed protected pixels for {anchor_id}: {outside_max_diff}")

    output_dir = output_root / anchor_id
    output_dir.mkdir(parents=True, exist_ok=False)
    Image.fromarray(source).save(output_dir / "first_frame.png", optimize=True)
    Image.fromarray(proxy).save(output_dir / "motion_signal.png", optimize=True)
    Image.fromarray(depth, mode="L").save(output_dir / "depth.png", optimize=True)
    save_mask(np.maximum.reduce([masks["rigid"], masks["contact"], masks["material"]]), output_dir / "mask.png")
    save_mask(masks["hole"], output_dir / "mask_old.png")
    for name in ("rigid", "contact", "material", "hole", "protect", "edit_alpha"):
        save_mask(masks[name], output_dir / f"mask_{name}.png" if name != "edit_alpha" else output_dir / "edit_alpha.png")
    (output_dir / "prompt.txt").write_text(PROMPTS[anchor_id] + "\n", encoding="utf-8")
    (output_dir / "negative_prompt.txt").write_text(NEGATIVE_PROMPT + "\n", encoding="utf-8")
    make_preview(source, proxy, depth, masks).save(output_dir / "proxy_review.png", optimize=True)

    files = [
        "first_frame.png", "motion_signal.png", "depth.png", "mask.png", "mask_old.png",
        "mask_rigid.png", "mask_contact.png", "mask_material.png", "mask_hole.png",
        "mask_protect.png", "edit_alpha.png", "prompt.txt", "negative_prompt.txt",
    ]
    manifest = {
        "schema_version": "foodstateedit.common_proxy.v1",
        "anchor_id": anchor_id,
        "source_case_id": anchor["case_id"],
        "canonical_input_path": str(source_path),
        "canonical_input_sha256": canonical["canonical_input_sha256"],
        "working_width": size[0],
        "working_height": size[1],
        "depth_mode": "deterministic_relative_layer_depth_no_learned_model",
        "texture_mode": "source_hole_texture_warp_with_procedural_utensil_shading",
        "outside_edit_alpha_max_pixel_difference": outside_max_diff,
        "changed_fraction": round(float(np.any(proxy != source, axis=2).mean()), 6),
        "files": {name: {"path": str(output_dir / name), "sha256": sha256_file(output_dir / name)} for name in files},
        "status": "provisional_geometry_proxy_requires_visual_confirmation",
    }
    (output_dir / "proxy_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--anchor-specs", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        help=(
            "Optional portable mirror containing <case_id>.jpg files. Each file "
            "must still match canonical_input_sha256."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output root: {args.output_root}")
    canonical = {row["case_id"]: row for row in read_csv(args.canonical_manifest)}
    anchors = read_csv(args.anchor_manifest)
    specs_json = json.loads(args.anchor_specs.read_text(encoding="utf-8"))
    specs = {spec["anchor_id"]: spec for spec in specs_json["anchors"]}
    if len(anchors) != 4 or set(specs) != {row["anchor_id"] for row in anchors}:
        raise ValueError("Expected the same four anchors in manifest and specs")
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = [
        build_case(
            anchor,
            canonical[anchor["case_id"]],
            specs[anchor["anchor_id"]],
            args.anchor_root,
            args.output_root,
            args.source_root,
        )
        for anchor in anchors
    ]
    summary = {
        "schema_version": 1,
        "proxy_version": "common_proxy_v1",
        "case_count": len(results),
        "all_protected_pixels_exact": all(item["outside_edit_alpha_max_pixel_difference"] == 0 for item in results),
        "all_require_visual_confirmation": all(item["status"] == "provisional_geometry_proxy_requires_visual_confirmation" for item in results),
        "cases": results,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Built {len(results)} common proxy packages -> {args.output_root}")


if __name__ == "__main__":
    main()
