#!/usr/bin/env python3
"""Build a controlled GeoEdit liquid fill-level case from a source/oracle pair.

The oracle is used only to construct and audit a target liquid proxy. Pixels
outside the detected target liquid surface are copied bit-exactly from the
source, so the downstream experiment cannot hide background drift from the
image-pair generator.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--width", type=int, default=736)
    parser.add_argument("--height", type=int, default=592)
    return parser.parse_args()


def require_new_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {path}")
    path.mkdir(parents=True)


def fit_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        fitted = ImageOps.fit(
            image.convert("RGB"), size, method=Image.Resampling.LANCZOS
        )
    return np.asarray(fitted, dtype=np.uint8)


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        raise ValueError("segmentation produced no foreground component")
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == label


def central_roi(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    return (
        ((xx - 0.5 * width) / (0.37 * width)) ** 2
        + ((yy - 0.55 * height) / (0.46 * height)) ** 2
        <= 1.0
    )


def bowl_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    white = (hsv[..., 1] < 62) & (hsv[..., 2] > 105) & central_roi(rgb.shape[:2])
    white = cv2.morphologyEx(
        white.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    )
    component = largest_component(white)
    contours, _ = cv2.findContours(
        component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    points = np.concatenate(contours, axis=0)
    hull = cv2.convexHull(points)
    support = np.zeros_like(white, dtype=np.uint8)
    cv2.fillConvexPoly(support, hull, 1)
    return support.astype(bool)


def bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def align_oracle(source: np.ndarray, oracle: np.ndarray) -> tuple[np.ndarray, list[float]]:
    source_box = bounding_box(bowl_mask(source))
    oracle_box = bounding_box(bowl_mask(oracle))
    sx = (source_box[2] - source_box[0]) / max(oracle_box[2] - oracle_box[0], 1)
    sy = (source_box[3] - source_box[1]) / max(oracle_box[3] - oracle_box[1], 1)
    scale = float(np.clip(0.5 * (sx + sy), 0.94, 1.06))
    source_center = np.asarray(
        [(source_box[0] + source_box[2]) / 2, (source_box[1] + source_box[3]) / 2]
    )
    oracle_center = np.asarray(
        [(oracle_box[0] + oracle_box[2]) / 2, (oracle_box[1] + oracle_box[3]) / 2]
    )
    translation = source_center - scale * oracle_center
    transform = np.asarray(
        [[scale, 0.0, translation[0]], [0.0, scale, translation[1]]],
        dtype=np.float32,
    )
    aligned = cv2.warpAffine(
        oracle,
        transform,
        (source.shape[1], source.shape[0]),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return aligned, [float(value) for value in transform.ravel()]


def liquid_mask(rgb: np.ndarray, bowl: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    golden = (
        (hsv[..., 0] >= 4)
        & (hsv[..., 0] <= 38)
        & (hsv[..., 1] >= 75)
        & (hsv[..., 2] >= 70)
        & bowl
    )
    golden = cv2.morphologyEx(
        golden.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
    )
    component = largest_component(golden)
    contours, _ = cv2.findContours(
        component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros_like(golden)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    filled = cv2.morphologyEx(
        filled,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
    )
    return filled.astype(bool)


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def labeled_panel(image: Image.Image, label: str) -> Image.Image:
    panel = image.convert("RGB").copy()
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, min(panel.width, 310), 30), fill=(0, 0, 0))
    draw.text((8, 8), label, fill=(255, 255, 255))
    return panel


def main() -> None:
    args = parse_args()
    if args.width % 16 or args.height % 16:
        raise ValueError("width and height must be divisible by 16")
    output_dir = args.output_dir.resolve()
    require_new_directory(output_dir)
    size = (args.width, args.height)
    source = fit_rgb(args.source.resolve(), size)
    oracle_raw = fit_rgb(args.oracle.resolve(), size)
    oracle, transform = align_oracle(source, oracle_raw)

    source_bowl = bowl_mask(source)
    target_bowl = bowl_mask(oracle)
    old_liquid = liquid_mask(source, source_bowl)
    new_liquid = liquid_mask(oracle, target_bowl)
    old_area = int(old_liquid.sum())
    new_area = int(new_liquid.sum())
    if new_area <= 1.15 * old_area:
        raise ValueError(
            f"oracle liquid area did not rise enough: old={old_area}, new={new_area}"
        )

    # Keep the edit localized and feather only at the new liquid boundary.
    alpha_image = Image.fromarray(new_liquid.astype(np.uint8) * 255, mode="L").filter(
        ImageFilter.GaussianBlur(radius=1.5)
    )
    alpha = np.asarray(alpha_image, dtype=np.float32)[..., None] / 255.0
    motion = np.rint(oracle.astype(np.float32) * alpha + source * (1.0 - alpha)).astype(
        np.uint8
    )
    outside = ~cv2.dilate(
        new_liquid.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    ).astype(bool)
    # Make the background lock an exact invariant after feathering.
    motion[outside] = source[outside]

    depth = np.full(source.shape[:2], 176, dtype=np.uint8)
    depth[source_bowl] = 142
    depth[new_liquid] = 112
    depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=3.0)

    Image.fromarray(source, mode="RGB").save(output_dir / "first_frame.png")
    Image.fromarray(source, mode="RGB").save(output_dir / "source_original.png")
    Image.fromarray(oracle, mode="RGB").save(output_dir / "oracle_aligned.png")
    Image.fromarray(motion, mode="RGB").save(output_dir / "motion_signal.png")
    Image.fromarray(depth, mode="L").save(output_dir / "depth.png")
    save_mask(output_dir / "mask_old.png", old_liquid)
    save_mask(output_dir / "mask.png", new_liquid)
    save_mask(output_dir / "mask_target_only.png", new_liquid & ~old_liquid)
    alpha_image.save(output_dir / "edit_alpha.png")
    shutil.copy2(args.source.resolve(), output_dir / "raw_source_imagegen.png")
    shutil.copy2(args.oracle.resolve(), output_dir / "raw_oracle_imagegen.png")

    prompt = (
        "A photorealistic white ceramic bowl filled to a high safe serving level "
        "with clear golden chicken broth, exactly three scallion rings floating "
        "on the surface, unchanged dark stone tabletop, unchanged camera and lighting."
    )
    negative = (
        "overflow, splash, steam, spoon, chopsticks, ladle, hands, people, extra bowl, "
        "extra garnish, text, logo, watermark, duplicate objects, deformed bowl, blur"
    )
    (output_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    (output_dir / "negative_prompt.txt").write_text(
        negative + "\n", encoding="utf-8"
    )

    mask_rgb = np.zeros_like(source)
    mask_rgb[..., 1] = new_liquid.astype(np.uint8) * 255
    mask_rgb[..., 0] = old_liquid.astype(np.uint8) * 255
    panels = [
        labeled_panel(Image.fromarray(source), "source: low fill"),
        labeled_panel(Image.fromarray(motion), "locked target proxy: high fill"),
        labeled_panel(Image.fromarray(mask_rgb), "red=old, green=new liquid"),
    ]
    sheet = Image.new("RGB", (size[0] * 3, size[1]), (0, 0, 0))
    for index, panel in enumerate(panels):
        sheet.paste(panel, (index * size[0], 0))
    sheet.save(output_dir / "conditioning_contact_sheet.png")

    source_pixels = source.astype(np.int16)
    motion_pixels = motion.astype(np.int16)
    manifest = {
        "schema_version": "foodstateedit.liquid_case.v0.1",
        "action": "increase_fill_level",
        "material": "liquid",
        "source_image": str(args.source.resolve()),
        "oracle_image": str(args.oracle.resolve()),
        "inference_size": list(size),
        "oracle_alignment_affine": transform,
        "old_liquid_area_px": old_area,
        "new_liquid_area_px": new_area,
        "projected_surface_area_ratio": new_area / old_area,
        "target_only_area_px": int((new_liquid & ~old_liquid).sum()),
        "outside_edit_max_difference": int(
            np.abs(source_pixels[outside] - motion_pixels[outside]).max()
        ),
        "geoedit": {
            "recommended_replace_mode": "mask_new",
            "reason": "fill increase has no old-minus-new removal hole",
            "new_mask": "mask.png",
            "old_mask": "mask_old.png",
        },
        "provenance": {
            "source": "built-in image generation, controlled low-fill input",
            "oracle": "built-in precise image edit, used only as target proxy",
            "background_lock": "source pixels copied outside target liquid surface",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
