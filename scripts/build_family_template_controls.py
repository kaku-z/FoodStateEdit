#!/usr/bin/env python3
"""Build fixed family-template planar/relative-3D controls for pilot or test cases.

The templates are deterministic geometry signals, not target photographs or
independent annotations.  Test execution must remain disabled until the method
rule and this builder are frozen from pilot-only evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


FRAME_COUNT = 21
REVIEW_INDICES = (0, 3, 6, 10, 15, 20)
FAMILY_COUNTS = {"pilot": 5, "test": 10}
FAMILY_ORDER = ("liquid", "granular", "strand", "strand_contact")
NEGATIVE = (
    "hand, arm, person, extra utensil, duplicate utensil, malformed utensil, "
    "floating unsupported food, duplicate payload, altered plate, altered bowl, "
    "background change, text, watermark, severe blur, ghosting"
)
PROMPTS = {
    "liquid": "A realistic photograph of the same soup. Exactly one stainless-steel spoon scoops and lifts a visible spoonful while the liquid remains naturally contained. Preserve the bowl, table, garnish, lighting and background. No hand, spill, crater or duplicate utensil.",
    "granular": "A realistic photograph of the same fried rice. Exactly one stainless-steel serving spatula lifts a coherent scoop with distinct grains and a matching shallow reduction in the source mound. Preserve the plate, ingredients, lighting and background. No hand or duplicate payload.",
    "strand": "A realistic photograph of the same noodle dish. Exactly two separate wooden chopsticks grip and lift one continuous noodle strand high above the bowl while at least one end remains connected to the source. Preserve the bowl, soup, garnish, lighting and background. No hand, third chopstick or duplicated noodle.",
    "strand_contact": "A realistic photograph of the same pasta. Exactly one four-tined stainless-steel fork twirls and lifts continuous spaghetti strands while at least one strand remains connected to the plate. Preserve the plate, sauce, garnish, lighting and background. No hand, second fork or duplicate bundle.",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def working_size(width: int, height: int, long_side: int = 688) -> tuple[int, int]:
    scale = long_side / max(width, height)
    return max(16, round(width * scale / 16) * 16), max(16, round(height * scale / 16) * 16)


def smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def phase(index: int) -> str:
    return "source" if index == 0 else "approach" if index < 6 else "contact" if index < 9 else "lift" if index < 16 else "hold"


def pixel(point: tuple[float, float], width: int, height: int) -> np.ndarray:
    return np.asarray([point[0] * width, point[1] * height], dtype=np.float64)


def ellipse_mask(shape: tuple[int, int], center: np.ndarray, radii: np.ndarray) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, tuple(np.rint(center).astype(int)), tuple(np.maximum(2, np.rint(radii)).astype(int)), 0, 0, 360, 255, -1, cv2.LINE_AA)
    return mask


def sampled_color(source: np.ndarray, center: np.ndarray, radii: np.ndarray, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    mask = ellipse_mask(source.shape[:2], center, radii) > 127
    values = source[mask]
    if not len(values):
        return fallback
    return tuple(int(value) for value in np.median(values, axis=0))


def draw_metal_line(frame: np.ndarray, start: np.ndarray, end: np.ndarray, width: int) -> None:
    cv2.line(frame, tuple(np.rint(start + [2, 3]).astype(int)), tuple(np.rint(end + [2, 3]).astype(int)), (42, 45, 48), width + 5, cv2.LINE_AA)
    cv2.line(frame, tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int)), (178, 186, 196), width, cv2.LINE_AA)
    cv2.line(frame, tuple(np.rint(start - [1, 1]).astype(int)), tuple(np.rint(end - [1, 1]).astype(int)), (242, 244, 246), max(2, width // 5), cv2.LINE_AA)


def transform_points(points: np.ndarray, source_anchor: np.ndarray, target_anchor: np.ndarray, amount: float, mode: str, depth: float = 1.0) -> np.ndarray:
    translated = points + (target_anchor - source_anchor) * amount
    if mode == "planar" or amount <= 0:
        return translated
    scale = 1.0 / (1.0 - 0.12 * amount * depth)
    yaw = 0.08 * amount
    relative = translated - target_anchor
    rotated = np.stack([relative[:, 0] * np.cos(yaw) - relative[:, 1] * np.sin(yaw), relative[:, 0] * np.sin(yaw) + relative[:, 1] * np.cos(yaw)], axis=1)
    return target_anchor + rotated * scale


def lift_state(index: int, contact: np.ndarray, final: np.ndarray) -> tuple[np.ndarray, float]:
    amount = smoothstep((index - 8) / 7.0)
    return contact * (1.0 - amount) + final * amount, amount


def render_solid(source: np.ndarray, family: str, mode: str) -> tuple[list[np.ndarray], dict[str, Any]]:
    height, width = source.shape[:2]
    if family == "liquid":
        contact = pixel((0.55, 0.61), width, height); final = pixel((0.46, 0.28), width, height)
        radii = np.asarray([0.075 * width, 0.045 * height]); handle = pixel((-0.30, -0.30), width, height)
    else:
        contact = pixel((0.56, 0.61), width, height); final = pixel((0.65, 0.28), width, height)
        radii = np.asarray([0.095 * width, 0.062 * height]); handle = pixel((0.31, -0.25), width, height)
    payload_mask = ellipse_mask((height, width), contact, radii)
    softened = cv2.GaussianBlur(source, (0, 0), sigmaX=max(2.0, min(width, height) * 0.018))
    repaired = source.copy()
    repaired[payload_mask > 32] = softened[payload_mask > 32]
    frames: list[np.ndarray] = []
    trace = []
    for index in range(FRAME_COUNT):
        if index == 0:
            frames.append(source.copy()); continue
        target, amount = lift_state(index, contact, final)
        approach = smoothstep((index - 1) / 4.0)
        utensil_anchor = contact + (target - contact if index >= 6 else handle * (1.0 - approach))
        frame = repaired.copy() if index >= 9 else source.copy()
        if family == "liquid":
            bowl_center = utensil_anchor
            bowl_radii = radii * (1.0 + (0.12 * amount if mode == "relative3d" else 0.0))
            cv2.ellipse(frame, tuple(np.rint(bowl_center).astype(int)), tuple(np.rint(bowl_radii * [1.15, 1.25]).astype(int)), 0, 0, 360, (184, 191, 200), -1, cv2.LINE_AA)
            cv2.ellipse(frame, tuple(np.rint(bowl_center).astype(int)), tuple(np.rint(bowl_radii * [1.15, 1.25]).astype(int)), 0, 0, 360, (70, 76, 82), 3, cv2.LINE_AA)
            color = sampled_color(source, contact, radii * 0.75, (184, 150, 112))
            cv2.ellipse(frame, tuple(np.rint(bowl_center).astype(int)), tuple(np.rint(bowl_radii * [0.92, 0.72]).astype(int)), 0, 0, 360, color, -1, cv2.LINE_AA)
            draw_metal_line(frame, bowl_center, bowl_center + handle, max(7, round(0.012 * min(width, height))))
        else:
            blade = np.asarray([[-1.1, -0.55], [1.05, -0.55], [1.18, 1.05], [-1.1, 1.05]]) * radii + utensil_anchor
            cv2.fillPoly(frame, [np.rint(blade).astype(np.int32)], (170, 181, 193), cv2.LINE_AA)
            cv2.polylines(frame, [np.rint(blade).astype(np.int32)], True, (70, 76, 82), 3, cv2.LINE_AA)
            draw_metal_line(frame, utensil_anchor, utensil_anchor + handle, max(8, round(0.014 * min(width, height))))
            color = sampled_color(source, contact, radii * 0.8, (174, 145, 92))
            cv2.ellipse(frame, tuple(np.rint(utensil_anchor - [0, 0.20 * radii[1]]).astype(int)), tuple(np.rint(radii * [0.90, 0.72]).astype(int)), 0, 0, 360, color, -1, cv2.LINE_AA)
        trace.append({"frame": index, "phase": phase(index), "amount": amount, "anchor_uv": target.tolist()})
        frames.append(frame)
    return frames, {"family": family, "mode": mode, "contact_uv": contact.tolist(), "final_uv": final.tolist(), "trace": trace, "geometry_kind": "fixed_family_template_relative_depth_not_reconstruction"}


def render_strands(source: np.ndarray, family: str, mode: str) -> tuple[list[np.ndarray], dict[str, Any]]:
    height, width = source.shape[:2]
    contact = pixel((0.62, 0.56), width, height)
    final = pixel((0.62, 0.23 if family == "strand" else 0.29), width, height)
    handle = pixel((0.30, -0.24), width, height)
    roots = [pixel((x, 0.66), width, height) for x in ((0.48, 0.70) if family == "strand" else (0.45, 0.54, 0.64, 0.73))]
    fallback = (211, 193, 148) if family == "strand" else (184, 139, 96)
    color = sampled_color(source, pixel((0.58, 0.62), width, height), np.asarray([0.14 * width, 0.10 * height]), fallback)
    frames: list[np.ndarray] = []
    trace = []
    for index in range(FRAME_COUNT):
        frame = source.copy()
        if index == 0:
            frames.append(frame); continue
        target, amount = lift_state(index, contact, final)
        approach = smoothstep((index - 1) / 4.0)
        utensil_anchor = contact + (target - contact if index >= 6 else handle * (1.0 - approach))
        curves = []
        for offset, root in enumerate(roots):
            bend = np.asarray([(-0.03 + 0.02 * offset) * width, -0.14 * height * amount])
            points = np.stack([root, (root + contact) / 2 + bend, contact])
            points = transform_points(points, contact, target, amount, mode, 0.9 + 0.05 * offset)
            points[0] = root
            curves.append(points)
        strand_width = max(5, round(0.012 * min(width, height)))
        for points in curves:
            cv2.polylines(frame, [np.rint(points).astype(np.int32)], False, tuple(int(v * 0.62) for v in color), strand_width + 5, cv2.LINE_AA)
        if family == "strand":
            separation = 0.022 * height
            for side in (-1, 1):
                tip = utensil_anchor + [0, side * separation]
                draw_metal_line(frame, tip, tip + handle + [0, side * 0.018 * height], max(6, round(0.010 * min(width, height))))
        else:
            draw_metal_line(frame, utensil_anchor, utensil_anchor + handle, max(7, round(0.012 * min(width, height))))
            axis = handle / max(np.linalg.norm(handle), 1.0); across = np.asarray([-axis[1], axis[0]])
            for tine in (-1.5, -0.5, 0.5, 1.5):
                start = utensil_anchor + across * tine * 0.012 * min(width, height)
                cv2.line(frame, tuple(np.rint(start).astype(int)), tuple(np.rint(start - axis * 0.075 * min(width, height)).astype(int)), (185, 194, 204), max(3, round(0.005 * min(width, height))), cv2.LINE_AA)
        for points in curves:
            cv2.polylines(frame, [np.rint(points).astype(np.int32)], False, color, strand_width, cv2.LINE_AA)
            cv2.polylines(frame, [np.rint(points - [1, 1]).astype(np.int32)], False, tuple(min(255, v + 38) for v in color), max(2, strand_width // 3), cv2.LINE_AA)
        trace.append({"frame": index, "phase": phase(index), "amount": amount, "anchor_uv": target.tolist()})
        frames.append(frame)
    return frames, {"family": family, "mode": mode, "contact_uv": contact.tolist(), "final_uv": final.tolist(), "root_uv": [root.tolist() for root in roots], "trace": trace, "geometry_kind": "fixed_family_template_relative_depth_not_reconstruction"}


def write_video(path: Path, frames: list[np.ndarray], fps: int = 8) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise OSError(f"Cannot open video writer: {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def review_sheet(frames: list[np.ndarray], label: str) -> Image.Image:
    height, width = frames[0].shape[:2]
    canvas = Image.new("RGB", (width * 3, (height + 28) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for slot, index in enumerate(REVIEW_INDICES):
        x = (slot % 3) * width; y = (slot // 3) * (height + 28)
        canvas.paste(Image.fromarray(frames[index]), (x, y + 28))
        draw.text((x + 5, y + 6), f"{label} / f{index} / {phase(index)} / CONTROL", fill="black")
    return canvas


def feather_union(frames_by_mode: dict[str, list[np.ndarray]], source: np.ndarray) -> np.ndarray:
    union = np.logical_or.reduce([np.any(frame != source, axis=2) for frames in frames_by_mode.values() for frame in frames])
    hard = np.asarray(Image.fromarray((union * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(17)))
    alpha = hard.copy()
    for radius in range(9, 17):
        expanded = np.asarray(Image.fromarray((union * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(2 * radius + 1))) > 0
        alpha = np.maximum(alpha, expanded.astype(np.uint8) * round(255 * (17 - radius) / 9))
    if np.any(alpha[union] != 255):
        raise AssertionError("Changed proxy pixels are not fully inside hard support")
    return alpha


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("pilot", "test"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_root}")
    data_rows = [row for row in read_csv(args.data_manifest) if row["split"] == args.split]
    canonical = {row["case_id"]: row for row in read_csv(args.canonical_manifest)}
    expected = FAMILY_COUNTS[args.split]
    if Counter(row["family"] for row in data_rows) != Counter({family: expected for family in FAMILY_ORDER}):
        raise ValueError("Split is not balanced")
    if args.split == "test" and any(row["annotation_status"] != "pending" for row in data_rows):
        raise ValueError("Unexpected pre-existing test annotation state")
    args.output_root.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema_version": "foodstateedit.family_template_controls.v1",
        "split": args.split,
        "scientific_status": "pilot_template_controls" if args.split == "pilot" else "frozen_heldout_template_controls_not_ground_truth",
        "builder_sha256": sha256_file(Path(__file__)),
        "case_count": len(data_rows),
        "family_counts": dict(sorted(Counter(row["family"] for row in data_rows).items())),
        "frame_count": FRAME_COUNT,
        "review_indices": list(REVIEW_INDICES),
        "per_image_parameter_tuning": False,
        "cases": [],
        "claim_limit": "Fixed template controls are intervention inputs, not target photographs, independent labels or evidence of physical correctness.",
    }
    for row in sorted(data_rows, key=lambda item: (item["family"], item["case_id"])):
        can = canonical[row["case_id"]]
        source_path = Path(can["canonical_input_path"])
        if sha256_file(source_path) != can["canonical_input_sha256"]:
            raise ValueError(f"Canonical source hash mismatch: {row['case_id']}")
        with Image.open(source_path) as image_file:
            source_image = ImageOps.exif_transpose(image_file).convert("RGB")
        source_image = source_image.resize(working_size(*source_image.size), Image.Resampling.LANCZOS)
        source = np.asarray(source_image)
        renderer = render_solid if row["family"] in ("liquid", "granular") else render_strands
        frames_by_mode: dict[str, list[np.ndarray]] = {}
        geometry = {}
        for mode in ("planar", "relative3d"):
            frames_by_mode[mode], geometry[mode] = renderer(source, row["family"], mode)
            if len(frames_by_mode[mode]) != FRAME_COUNT or not np.array_equal(frames_by_mode[mode][0], source):
                raise AssertionError(f"Invalid frame contract: {row['case_id']} {mode}")
        alpha = feather_union(frames_by_mode, source)
        case_root = args.output_root / row["case_id"]
        case_root.mkdir()
        source_image.save(case_root / "reference.png", optimize=True)
        Image.fromarray(alpha, mode="L").save(case_root / "edit_alpha.png", optimize=True)
        for mode, frames in frames_by_mode.items():
            for frame in frames:
                if np.any(frame[alpha == 0] != source[alpha == 0]):
                    raise AssertionError(f"Protected pixels changed: {row['case_id']} {mode}")
            write_video(case_root / f"{mode}.mp4", frames)
            Image.fromarray(frames[-1]).save(case_root / f"{mode}_final.png", optimize=True)
            review_sheet(frames, f"{row['case_id']} / {mode}").save(case_root / f"{mode}_review.png", optimize=True)
        (case_root / "geometry.json").write_text(json.dumps(geometry, indent=2) + "\n", encoding="utf-8")
        files = {path.name: file_record(path, args.output_root) for path in sorted(case_root.iterdir()) if path.is_file()}
        manifest["cases"].append({
            "case_id": row["case_id"], "family": row["family"], "dish": row["dish"], "utensil": row["utensil"], "action": row["action"],
            "source_kind": row["source_kind"], "license": row["license"], "canonical_input_sha256": can["canonical_input_sha256"],
            "width": source.shape[1], "height": source.shape[0], "prompt": PROMPTS[row["family"]], "negative_prompt": NEGATIVE,
            "support_fraction": float((alpha > 0).mean()), "files": files,
        })
    manifest_path = args.output_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(args.output_root), "manifest_sha256": sha256_file(manifest_path), "case_count": len(manifest["cases"]), "family_counts": manifest["family_counts"]}, indent=2))


if __name__ == "__main__":
    main()
