#!/usr/bin/env python3
"""Project, select, and evaluate a GeoEdit spoon-scooping result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def require_new_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {path}")
    path.mkdir(parents=True)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        fitted = image.convert("L").resize(
            (shape[1], shape[0]), Image.Resampling.NEAREST
        )
    return np.asarray(fitted, dtype=np.uint8) > 127


def read_video(path: Path, size: tuple[int, int]) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4)
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValueError("video contains no readable frames")
    return frames, fps if fps > 0 else 15.0


def gray_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    weights = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
    first_gray = first.astype(np.float32) @ weights
    second_gray = second.astype(np.float32) @ weights
    if float(first_gray.std()) < 1e-6 or float(second_gray.std()) < 1e-6:
        return None
    return float(np.corrcoef(first_gray, second_gray)[0, 1])


def region_proxy_metrics(
    edited: np.ndarray, proxy: np.ndarray, support: np.ndarray
) -> dict[str, float | None]:
    edited_values = edited[support].astype(np.float32)
    proxy_values = proxy[support].astype(np.float32)
    edited_gray = cv2.cvtColor(edited, cv2.COLOR_RGB2GRAY)
    proxy_gray = cv2.cvtColor(proxy, cv2.COLOR_RGB2GRAY)
    edited_edges = cv2.Canny(edited_gray, 80, 160).astype(bool) & support
    proxy_edges = cv2.Canny(proxy_gray, 80, 160).astype(bool) & support
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edited_dilated = cv2.dilate(edited_edges.astype(np.uint8), kernel).astype(bool)
    proxy_dilated = cv2.dilate(proxy_edges.astype(np.uint8), kernel).astype(bool)
    precision = float(proxy_dilated[edited_edges].mean()) if edited_edges.any() else 0.0
    recall = float(edited_dilated[proxy_edges].mean()) if proxy_edges.any() else 0.0
    edge_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    laplacian = cv2.Laplacian(edited_gray, cv2.CV_32F)
    return {
        "area_px": int(support.sum()),
        "rgb_mae_from_proxy": float(np.mean(np.abs(edited_values - proxy_values))),
        "gray_correlation_with_proxy": gray_correlation(
            edited_values, proxy_values
        ),
        "tolerant_edge_precision_with_proxy": precision,
        "tolerant_edge_recall_with_proxy": recall,
        "tolerant_edge_f1_with_proxy": edge_f1,
        "edited_laplacian_variance": float(laplacian[support].var()),
    }


def green_binary(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    support = (
        ((xx - 0.50 * width) / (0.34 * width)) ** 2
        + ((yy - 0.50 * height) / (0.36 * height)) ** 2
        <= 1.0
    )
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (
        (hsv[..., 0] >= 25)
        & (hsv[..., 0] <= 105)
        & (hsv[..., 1] >= 55)
        & (hsv[..., 2] >= 20)
        & support
    )


def green_components(rgb: np.ndarray) -> list[dict[str, object]]:
    green = green_binary(rgb).astype(np.uint8)
    green = cv2.morphologyEx(
        green,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 7)),
    )
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        green, connectivity=8
    )
    components = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if 80 <= area <= 1800:
            components.append(
                {
                    "area_px": area,
                    "centroid_xy": [
                        round(float(centroids[label, 0]), 2),
                        round(float(centroids[label, 1]), 2),
                    ],
                }
            )
    components.sort(key=lambda item: int(item["area_px"]), reverse=True)
    return components


def payload_golden_fraction(rgb: np.ndarray, support: np.ndarray) -> float:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    golden = (
        (hsv[..., 0] >= 4)
        & (hsv[..., 0] <= 38)
        & (hsv[..., 1] >= 38)
        & (hsv[..., 2] >= 55)
    )
    return float(golden[support].mean())


def spoon_connectivity(
    rgb: np.ndarray, utensil: np.ndarray, spoon_bowl: np.ndarray
) -> dict[str, object]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    metal = (
        (hsv[..., 1] < 72)
        & (hsv[..., 2] > 92)
        & utensil
    ).astype(np.uint8)
    metal = cv2.morphologyEx(
        metal,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(metal, connectivity=8)
    candidates = sorted(
        [
            (label, int(stats[label, cv2.CC_STAT_AREA]))
            for label in range(1, count)
            if int(stats[label, cv2.CC_STAT_AREA]) >= 250
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    if not candidates:
        return {
            "largest_metal_component_area_px": 0,
            "touches_right_edge": False,
            "intersects_spoon_bowl": False,
            "connected_handle_and_bowl": False,
        }
    label, area = candidates[0]
    component = labels == label
    touches_right = bool(component[:, -12:].any())
    intersects_bowl = bool(np.logical_and(component, spoon_bowl).any())
    return {
        "largest_metal_component_area_px": area,
        "touches_right_edge": touches_right,
        "intersects_spoon_bowl": intersects_bowl,
        "connected_handle_and_bowl": touches_right and intersects_bowl,
    }


def panel(image: np.ndarray, label: str) -> Image.Image:
    result = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, min(result.width, 430), 30), fill=(0, 0, 0))
    draw.text((8, 8), label, fill=(255, 255, 255))
    return result


def main() -> None:
    args = parse_args()
    case_dir = args.case_dir.resolve()
    output_dir = args.output_dir.resolve()
    require_new_directory(output_dir)
    source = load_rgb(case_dir / "source_original.png")
    proxy = load_rgb(case_dir / "motion_signal.png")
    oracle = load_rgb(case_dir / "oracle_aligned.png")
    height, width = source.shape[:2]
    action = load_mask(case_dir / "mask.png", (height, width))
    utensil = load_mask(case_dir / "mask_utensil.png", (height, width))
    spoon_bowl = load_mask(case_dir / "mask_spoon_bowl.png", (height, width))
    payload = load_mask(case_dir / "mask_spoon_payload.png", (height, width))
    material_path = case_dir / "mask_material.png"
    if material_path.exists():
        material = load_mask(material_path, (height, width))
    else:
        material = (
            payload
            | load_mask(case_dir / "mask_garnish_target.png", (height, width))
            | load_mask(
                case_dir / "mask_moved_garnish_removal.png", (height, width)
            )
        )
    source_garnish_mask = load_mask(
        case_dir / "mask_garnish_source.png", (height, width)
    )
    with Image.open(case_dir / "edit_alpha.png") as image:
        alpha = np.asarray(
            image.convert("L").resize((width, height), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )[..., None] / 255.0

    frames, fps = read_video(args.video.resolve(), (width, height))
    selection_pool = frames[1:] if len(frames) > 1 else frames
    proxy_values = proxy[action].astype(np.float32)
    frame_mae = [
        float(np.mean(np.abs(frame[action].astype(np.float32) - proxy_values)))
        for frame in selection_pool
    ]
    best_pool_index = int(np.argmin(frame_mae))
    best_index = best_pool_index + (1 if len(frames) > 1 else 0)

    outside = alpha[..., 0] <= 0.0

    def composite(frame: np.ndarray) -> np.ndarray:
        merged = np.rint(
            frame.astype(np.float32) * alpha
            + source.astype(np.float32) * (1.0 - alpha)
        ).astype(np.uint8)
        merged[outside] = source[outside]
        return merged

    edited_frames = [composite(frame) for frame in frames]
    edited = edited_frames[best_index]
    edited_last = edited_frames[-1]
    Image.fromarray(edited, mode="RGB").save(output_dir / "edited_2d.png")
    Image.fromarray(edited_last, mode="RGB").save(output_dir / "edited_2d_last.png")

    writer = cv2.VideoWriter(
        str(output_dir / "edited_video.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise OSError("could not open edited_video.mp4 for writing")
    for frame in edited_frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    garnish = green_components(edited)
    count, labels, _, centroids = cv2.connectedComponentsWithStats(
        source_garnish_mask.astype(np.uint8), connectivity=8
    )
    source_labels = list(range(1, count))
    moved_source_label = min(source_labels, key=lambda label: centroids[label, 1])
    moved_source_site = cv2.dilate(
        (labels == moved_source_label).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
    ).astype(bool)
    residual_green_pixels = int(
        (green_binary(edited) & moved_source_site & ~spoon_bowl).sum()
    )
    spoon_garnish_count = 0
    for item in garnish:
        x, y = item["centroid_xy"]
        xi = min(max(int(round(x)), 0), width - 1)
        yi = min(max(int(round(y)), 0), height - 1)
        if spoon_bowl[yi, xi]:
            spoon_garnish_count += 1
    bowl_garnish_count = len(garnish) - spoon_garnish_count
    outside_difference = np.abs(
        edited[outside].astype(np.int16) - source[outside].astype(np.int16)
    )
    selected_values = edited[action].astype(np.float32)
    connectivity = spoon_connectivity(edited, utensil, spoon_bowl)
    golden_fraction = payload_golden_fraction(edited, payload)
    automatic_gate_pass = bool(
        int(outside_difference.max()) == 0
        and len(garnish) == 3
        and spoon_garnish_count == 1
        and bowl_garnish_count == 2
        and residual_green_pixels <= 5
        and golden_fraction >= 0.35
        and connectivity["connected_handle_and_bowl"]
    )
    metrics = {
        "schema_version": "foodstateedit.spoon_scoop_result.v0.1",
        "frame_count": len(frames),
        "selected_frame_index": best_index,
        "last_frame_index": len(frames) - 1,
        "selection": "minimum action-mask RGB MAE from audited target proxy",
        "action_mask_rgb_mae_from_proxy": float(
            np.mean(np.abs(selected_values - proxy_values))
        ),
        "action_mask_gray_correlation_with_proxy": gray_correlation(
            selected_values, proxy_values
        ),
        "region_proxy_metrics": {
            "action": region_proxy_metrics(edited, proxy, action),
            "rigid_utensil": region_proxy_metrics(edited, proxy, utensil),
            "material": region_proxy_metrics(edited, proxy, material),
        },
        "outside_alpha_max_difference": int(outside_difference.max()),
        "outside_alpha_rgb_mae": float(outside_difference.mean()),
        "spoon_payload_golden_fraction": golden_fraction,
        "garnish_components": garnish,
        "detected_garnish_count_total": len(garnish),
        "detected_garnish_count_in_spoon": spoon_garnish_count,
        "detected_garnish_count_in_bowl": bowl_garnish_count,
        "residual_green_pixels_at_moved_source_site": residual_green_pixels,
        "spoon_connectivity": connectivity,
        "automatic_gate_pass": automatic_gate_pass,
        "manual_checks_required": [
            "exactly_one_spoon",
            "no_hand_or_other_utensil",
            "spoon_visibly_contains_broth",
            "spoon_handle_and_bowl_are_not_deformed",
            "rim_and_liquid_occlusion_is_physical",
            "no_splash_or_overflow",
            "photorealistic_reflections_and_local_shadow",
        ],
    }
    (output_dir / "spoon_scoop_result_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    absolute_difference = np.abs(
        edited.astype(np.int16) - source.astype(np.int16)
    ).astype(np.uint8)
    difference_display = np.clip(absolute_difference * 4, 0, 255).astype(np.uint8)
    images = [
        panel(source, "source: no utensil"),
        panel(oracle, "ImageGen oracle only"),
        panel(proxy, "locked GeoEdit target proxy"),
        panel(edited, f"GeoEdit selected frame {best_index}"),
        panel(edited_last, f"GeoEdit last frame {len(frames) - 1}"),
        panel(difference_display, "4x absolute change from source"),
    ]
    sheet = Image.new("RGB", (3 * width, 2 * height), (0, 0, 0))
    for index, image in enumerate(images):
        sheet.paste(image, ((index % 3) * width, (index // 3) * height))
    sheet.save(output_dir / "spoon_scoop_result_contact_sheet.png")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
