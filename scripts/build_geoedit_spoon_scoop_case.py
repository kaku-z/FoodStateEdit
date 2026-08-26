#!/usr/bin/env python3
"""Build a controlled GeoEdit case for inserting a spoon that scoops soup.

The ImageGen edit is an audited target proxy only.  The GeoEdit reference is
always the utensil-free source, and pixels outside the explicit action mask are
copied bit-exactly from that source.
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
    parser.add_argument(
        "--mask-version",
        choices=("broad_v0_1", "tight_v0_2"),
        default="broad_v0_1",
    )
    parser.add_argument(
        "--preprocess-moved-garnish",
        action="store_true",
        help="Remove only the source garnish object that moves into the spoon.",
    )
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


def white_bowl_left_anchor(rgb: np.ndarray) -> tuple[int, int, int, int]:
    """Measure an oracle/source alignment anchor that excludes the spoon."""
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = (hsv[..., 1] < 50) & (hsv[..., 2] > 150)
    mask[: int(0.13 * height)] = False
    mask[int(0.85 * height) :] = False
    mask[:, : int(0.12 * width)] = False
    mask[:, int(0.46 * width) :] = False
    component = largest_component(mask)
    ys, xs = np.nonzero(component)
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def green_components(rgb: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    height, width = rgb.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    # The target spoon and all three garnish objects stay within this support.
    # Restricting the detector prevents greenish tabletop texture from being
    # counted as food.
    support = (
        ((xx - 0.50 * width) / (0.34 * width)) ** 2
        + ((yy - 0.50 * height) / (0.36 * height)) ** 2
        <= 1.0
    )
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    green = (
        (hsv[..., 0] >= 25)
        & (hsv[..., 0] <= 105)
        & (hsv[..., 1] >= 55)
        & (hsv[..., 2] >= 20)
        & support
    ).astype(np.uint8)
    green = cv2.morphologyEx(
        green,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 7)),
    )
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        green, connectivity=8
    )
    items: list[tuple[int, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if 100 <= area <= 1400:
            items.append((label, area))
    items.sort(key=lambda item: item[1], reverse=True)
    if len(items) != 3:
        raise ValueError(
            f"expected exactly three green garnish objects, found areas="
            f"{[item[1] for item in items]}"
        )
    selected = np.isin(labels, [label for label, _ in items])
    records = [
        {
            "area_px": area,
            "centroid_xy": [
                round(float(centroids[label, 0]), 2),
                round(float(centroids[label, 1]), 2),
            ],
        }
        for label, area in items
    ]
    return selected, records


def liquid_mask(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    support = (
        ((xx - 0.49 * width) / (0.30 * width)) ** 2
        + ((yy - 0.53 * height) / (0.27 * height)) ** 2
        <= 1.0
    )
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    golden = (
        (hsv[..., 0] >= 4)
        & (hsv[..., 0] <= 38)
        & (hsv[..., 1] >= 65)
        & (hsv[..., 2] >= 65)
        & support
    ).astype(np.uint8)
    golden = cv2.morphologyEx(
        golden,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    )
    return largest_component(golden)


def ellipse_mask(
    shape: tuple[int, int], center: tuple[float, float], axes: tuple[float, float], angle: float
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (round(center[0] * width), round(center[1] * height)),
        (round(axes[0] * width), round(axes[1] * height)),
        angle,
        0,
        360,
        1,
        thickness=cv2.FILLED,
    )
    return mask.astype(bool)


def build_action_masks(
    shape: tuple[int, int],
    version: str,
    moved_garnish_center: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    """Normalized annotations for this controlled oracle composition."""
    height, width = shape
    if version == "tight_v0_2":
        spoon_bowl = ellipse_mask(shape, (0.598, 0.456), (0.120, 0.082), -4.0)
        spoon_inner = ellipse_mask(shape, (0.598, 0.456), (0.092, 0.057), -4.0)
        centerline = (
            (0.704, 0.438),
            (0.775, 0.344),
            (0.875, 0.250),
            (1.000, 0.165),
        )
        handle_thickness = max(14, round(0.025 * width))
        handle_dilation = (5, 5)
    else:
        spoon_bowl = ellipse_mask(shape, (0.586, 0.445), (0.136, 0.086), -4.0)
        spoon_inner = ellipse_mask(shape, (0.586, 0.445), (0.105, 0.061), -4.0)
        centerline = (
            (0.665, 0.425),
            (0.755, 0.330),
            (0.870, 0.245),
            (1.000, 0.165),
        )
        handle_thickness = max(20, round(0.035 * width))
        handle_dilation = (9, 9)

    handle = np.zeros((height, width), dtype=np.uint8)
    points = np.asarray(
        [
            (min(round(x * width), width - 1), round(y * height))
            for x, y in centerline
        ],
        dtype=np.int32,
    )
    cv2.polylines(
        handle,
        [points],
        isClosed=False,
        color=1,
        thickness=handle_thickness,
        lineType=cv2.LINE_AA,
    )
    handle = cv2.dilate(
        handle,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, handle_dilation),
    ).astype(bool)

    if version == "tight_v0_2":
        if moved_garnish_center is None:
            raise ValueError("tight_v0_2 requires moved_garnish_center")
        garnish_transfer = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(
            garnish_transfer,
            (round(moved_garnish_center[0]), round(moved_garnish_center[1])),
            max(18, round(0.028 * width)),
            1,
            thickness=cv2.FILLED,
        )
        garnish_transfer = garnish_transfer.astype(bool)
        contact_response = np.zeros((height, width), dtype=bool)
        close_size = (9, 9)
    else:
        garnish_transfer = ellipse_mask(shape, (0.493, 0.548), (0.095, 0.095), 0.0)
        contact_response = ellipse_mask(shape, (0.585, 0.470), (0.155, 0.105), -4.0)
        close_size = (21, 21)
    utensil = spoon_bowl | handle
    action = utensil | garnish_transfer | contact_response
    action = cv2.morphologyEx(
        action.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_size),
    ).astype(bool)
    return {
        "utensil": utensil,
        "spoon_bowl": spoon_bowl,
        "spoon_inner": spoon_inner,
        "handle": handle,
        "garnish_transfer": garnish_transfer,
        "contact_response": contact_response,
        "action": action,
    }


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def labeled_panel(image: Image.Image, label: str) -> Image.Image:
    panel = image.convert("RGB").copy()
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, min(panel.width, 400), 30), fill=(0, 0, 0))
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
    oracle = fit_rgb(args.oracle.resolve(), size)

    source_anchor = white_bowl_left_anchor(source)
    oracle_anchor = white_bowl_left_anchor(oracle)
    anchor_delta = [abs(a - b) for a, b in zip(source_anchor, oracle_anchor)]
    if max(anchor_delta) > 4:
        raise ValueError(
            "source/oracle bowl anchors are not aligned closely enough: "
            f"source={source_anchor}, oracle={oracle_anchor}"
        )

    source_garnish, source_garnish_items = green_components(source)
    target_garnish, target_garnish_items = green_components(oracle)
    moved_item = min(
        source_garnish_items, key=lambda item: float(item["centroid_xy"][1])
    )
    moved_center = tuple(float(value) for value in moved_item["centroid_xy"])
    masks = build_action_masks(
        source.shape[:2], args.mask_version, moved_garnish_center=moved_center
    )
    action = masks["action"]
    moved_garnish = source_garnish & masks["garnish_transfer"]
    if args.mask_version == "broad_v0_1":
        if not np.all(source_garnish <= action):
            raise ValueError("source garnish transfer is not fully covered by action mask")
        if not np.all(target_garnish <= action):
            raise ValueError("target garnish transfer is not fully covered by action mask")
    else:
        if int(moved_garnish.sum()) < 100:
            raise ValueError("tight mask did not cover the moved source garnish")
        target_spoon_item = min(
            target_garnish_items, key=lambda item: float(item["centroid_xy"][1])
        )
        target_spoon_x, target_spoon_y = (
            int(round(float(value))) for value in target_spoon_item["centroid_xy"]
        )
        if not masks["spoon_bowl"][target_spoon_y, target_spoon_x]:
            raise ValueError("target spoon garnish centroid is outside the spoon bowl")

    reference = source.copy()
    source_garnish_preprocess_mask = np.zeros(source.shape[:2], dtype=bool)
    if args.preprocess_moved_garnish:
        source_garnish_preprocess_mask = cv2.dilate(
            moved_garnish.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        ).astype(bool)
        replacement = cv2.medianBlur(source, 41)
        preprocess_alpha = np.asarray(
            Image.fromarray(
                source_garnish_preprocess_mask.astype(np.uint8) * 255, mode="L"
            ).filter(ImageFilter.GaussianBlur(radius=2.0)),
            dtype=np.float32,
        )[..., None] / 255.0
        reference = np.rint(
            replacement.astype(np.float32) * preprocess_alpha
            + source.astype(np.float32) * (1.0 - preprocess_alpha)
        ).astype(np.uint8)
        reference[~action] = source[~action]

    alpha_image = Image.fromarray(action.astype(np.uint8) * 255, mode="L").filter(
        ImageFilter.GaussianBlur(radius=2.0)
    )
    alpha = np.asarray(alpha_image, dtype=np.float32)[..., None] / 255.0
    motion = np.rint(
        oracle.astype(np.float32) * alpha + reference.astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)
    outside = alpha[..., 0] <= 0.0
    motion[outside] = reference[outside]

    target_hsv = cv2.cvtColor(oracle, cv2.COLOR_RGB2HSV)
    payload = (
        masks["spoon_inner"]
        & (target_hsv[..., 0] >= 4)
        & (target_hsv[..., 0] <= 38)
        & (target_hsv[..., 1] >= 45)
        & (target_hsv[..., 2] >= 55)
    )
    if int(payload.sum()) < 800:
        raise ValueError(f"spoon payload segmentation is too small: {int(payload.sum())}")

    source_liquid = liquid_mask(reference)
    depth = np.full(source.shape[:2], 176, dtype=np.uint8)
    depth[source_liquid] = 112
    depth[masks["utensil"]] = 78
    depth[payload] = 84
    depth[target_garnish] = 82
    depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=2.2)

    Image.fromarray(reference, mode="RGB").save(output_dir / "first_frame.png")
    Image.fromarray(source, mode="RGB").save(output_dir / "source_original.png")
    Image.fromarray(reference, mode="RGB").save(
        output_dir / "source_reference_preprocessed.png"
    )
    Image.fromarray(oracle, mode="RGB").save(output_dir / "oracle_aligned.png")
    Image.fromarray(motion, mode="RGB").save(output_dir / "motion_signal.png")
    Image.fromarray(depth, mode="L").save(output_dir / "depth.png")
    save_mask(output_dir / "mask.png", action)
    save_mask(output_dir / "mask_old.png", np.zeros_like(action))
    save_mask(output_dir / "mask_utensil.png", masks["utensil"])
    save_mask(output_dir / "mask_spoon_bowl.png", masks["spoon_bowl"])
    save_mask(output_dir / "mask_spoon_payload.png", payload)
    save_mask(output_dir / "mask_garnish_source.png", source_garnish)
    save_mask(output_dir / "mask_garnish_target.png", target_garnish)
    save_mask(
        output_dir / "mask_moved_garnish_removal.png",
        source_garnish_preprocess_mask,
    )
    alpha_image.save(output_dir / "edit_alpha.png")
    shutil.copy2(args.source.resolve(), output_dir / "raw_source_imagegen.png")
    shutil.copy2(args.oracle.resolve(), output_dir / "raw_oracle_imagegen.png")

    prompt = (
        "A photorealistic white ceramic bowl of clear golden broth on a dark stone "
        "tabletop. Exactly one stainless-steel soup spoon enters from the upper right "
        "and is lifted just above the soup, visibly holding golden broth and exactly "
        "one scallion ring. Exactly two scallion rings remain in the bowl, exactly "
        "three scallion rings total. Correct spoon, rim, liquid, and occlusion geometry."
    )
    negative = (
        "hand, arm, person, second spoon, duplicate spoon, fork, chopsticks, ladle, "
        "empty spoon, dry spoon, spilling, splash, overflow, extra garnish, missing "
        "garnish, deformed bowl, changed camera, changed background, text, watermark"
    )
    (output_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    (output_dir / "negative_prompt.txt").write_text(
        negative + "\n", encoding="utf-8"
    )

    semantic = np.zeros_like(source)
    semantic[..., 0] = masks["utensil"].astype(np.uint8) * 255
    semantic[..., 1] = payload.astype(np.uint8) * 255
    semantic[..., 2] = masks["garnish_transfer"].astype(np.uint8) * 255
    mask_preview = source.copy()
    overlay = np.zeros_like(source)
    overlay[..., 1] = action.astype(np.uint8) * 255
    mask_preview = np.rint(0.55 * mask_preview + 0.45 * overlay).astype(np.uint8)
    panels = [
        labeled_panel(Image.fromarray(source), "source: no utensil"),
        labeled_panel(Image.fromarray(oracle), "ImageGen oracle only"),
        labeled_panel(Image.fromarray(motion), "locked target proxy"),
        labeled_panel(Image.fromarray(mask_preview), "green: GeoEdit action mask"),
        labeled_panel(Image.fromarray(semantic), "red=spoon green=payload blue=transfer"),
        labeled_panel(Image.fromarray(depth).convert("RGB"), "depth control"),
    ]
    sheet = Image.new("RGB", (3 * args.width, 2 * args.height), (0, 0, 0))
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 3) * args.width, (index // 3) * args.height))
    sheet.save(output_dir / "conditioning_contact_sheet.png")

    source_pixels = source.astype(np.int16)
    motion_pixels = motion.astype(np.int16)
    manifest = {
        "schema_version": "foodstateedit.spoon_scoop_case.v0.1",
        "action": "insert_spoon_and_scoop_liquid",
        "source_has_utensil": False,
        "target_utensil_type": "stainless_steel_spoon",
        "target_utensil_count": 1,
        "payload_material": "clear_golden_broth",
        "payload_relation": "contained_by_and_supported_by_spoon_bowl",
        "hand_count": 0,
        "source_garnish_count": 3,
        "target_garnish_count": 3,
        "target_garnish_distribution": {"bowl": 2, "spoon": 1},
        "source_garnish_components": source_garnish_items,
        "target_garnish_components": target_garnish_items,
        "source_bowl_left_anchor": source_anchor,
        "oracle_bowl_left_anchor": oracle_anchor,
        "anchor_absolute_delta": anchor_delta,
        "mask_version": args.mask_version,
        "moved_source_garnish_centroid_xy": list(moved_center),
        "moved_source_garnish_preprocessed": args.preprocess_moved_garnish,
        "moved_source_garnish_preprocess_area_px": int(
            source_garnish_preprocess_mask.sum()
        ),
        "oracle_alignment_affine": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "inference_size": list(size),
        "action_mask_area_px": int(action.sum()),
        "action_mask_fraction": float(action.mean()),
        "utensil_annotation_area_px": int(masks["utensil"].sum()),
        "payload_proxy_area_px": int(payload.sum()),
        "outside_edit_max_difference": int(
            np.abs(source_pixels[outside] - motion_pixels[outside]).max()
        ),
        "geoedit": {
            "replace_mode": "mask_new",
            "warm_start": False,
            "new_mask": "mask.png",
            "old_mask": "mask_old.png",
        },
        "provenance": {
            "source": "controlled utensil-free image",
            "oracle": "built-in ImageGen precise edit, target proxy only",
            "background_lock": "source pixels copied exactly outside edit alpha",
        },
        "acceptance_checks": [
            "exactly_one_spoon",
            "no_hand_or_other_utensil",
            "spoon_bowl_contains_visible_broth",
            "spoon_rim_and_liquid_occlusion_is_physical",
            "exactly_three_scallion_rings_total",
            "two_rings_in_bowl_and_one_ring_in_spoon",
            "outside_edit_pixels_are_bit_exact",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
