#!/usr/bin/env python3
"""Build deterministic source/approach/contact/lift/final VACE controls.

The builder consumes the frozen common proxy and anchor geometry. It does not
load a learned model. Geometry is dense in time, the first frame is an exact
source anchor, and the four final-hold controls are exact copies of the frozen
static proxy. Image generation is never used to choose or paint keyframes.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from build_common_proxy import (
    NEGATIVE_PROMPT,
    composite,
    render_primitive,
    sha256_file,
    source_texture,
    utensil_texture,
)


ANCHORS = (
    "soup_spoon_001",
    "fried_rice_spatula_001",
    "ramen_chopsticks_001",
    "pasta_fork_001",
)
METHOD = "vace_direct_dynamic_multikey"
FRAME_COUNT = 21
FPS = 15.0
SOURCE_FRAME = 0
APPROACH_START = 1
APPROACH_END = 5
CONTACT_START = 6
CONTACT_END = 8
LIFT_START = 9
LIFT_END = 16
FINAL_START = 17
FINAL_END = 20
SELECTED_FRAME_INDEX = 18


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def phase_for_frame(index: int) -> str:
    if index == SOURCE_FRAME:
        return "source_anchor"
    if APPROACH_START <= index <= APPROACH_END:
        return "approach"
    if CONTACT_START <= index <= CONTACT_END:
        return "contact_hold"
    if LIFT_START <= index <= LIFT_END:
        return "lift"
    if FINAL_START <= index <= FINAL_END:
        return "final_hold"
    raise ValueError(f"Frame {index} is outside the frozen schedule")


def read_image(path: Path, flags: int) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def verify_static_proxy(proxy_dir: Path) -> dict[str, object]:
    manifest_path = proxy_dir / "proxy_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "foodstateedit.common_proxy.v1":
        raise ValueError(f"Unexpected common proxy schema: {manifest_path}")
    if manifest.get("anchor_id") != proxy_dir.name:
        raise ValueError(f"Common proxy identity mismatch: {manifest_path}")
    for name, record in manifest["files"].items():
        path = proxy_dir / name
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Common proxy hash mismatch: {path}")
    return manifest


def mask_centroid_normalized(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 127)
    if not len(xs):
        raise ValueError("Cannot find centroid of an empty mask")
    height, width = mask.shape
    return np.asarray(
        [float(xs.mean()) / (width - 1), float(ys.mean()) / (height - 1)],
        dtype=np.float32,
    )


def translate_primitive(
    primitive: dict[str, object], delta: np.ndarray
) -> dict[str, object]:
    result = copy.deepcopy(primitive)
    if "points" in result:
        result["points"] = [
            [float(point[0] + delta[0]), float(point[1] + delta[1])]
            for point in result["points"]
        ]
    if "center" in result:
        result["center"] = [
            float(result["center"][0] + delta[0]),
            float(result["center"][1] + delta[1]),
        ]
    return result


def deform_material_primitive(
    primitive: dict[str, object],
    source_delta: np.ndarray,
    final_contact: np.ndarray,
    lift: float,
) -> dict[str, object]:
    result = copy.deepcopy(primitive)
    remaining = 1.0 - lift
    if "center" in result:
        result["center"] = [
            float(result["center"][0] + source_delta[0] * remaining),
            float(result["center"][1] + source_delta[1] * remaining),
        ]
    if "points" in result:
        points = []
        for point in result["points"]:
            point_array = np.asarray(point, dtype=np.float32)
            distance = float(np.linalg.norm(point_array - final_contact))
            # Keep bowl/plate roots fixed while the gripped segment follows the tool.
            weight = math.exp(-(distance * distance) / (2.0 * 0.20 * 0.20))
            moved = point_array + source_delta * remaining * weight
            points.append([float(moved[0]), float(moved[1])])
        result["points"] = points
    return result


def render_primitives(
    primitives: list[dict[str, object]], width: int, height: int
) -> list[np.ndarray]:
    return [render_primitive(primitive, width, height) for primitive in primitives]


def union_masks(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    if not masks:
        return np.zeros(shape, dtype=np.uint8)
    return np.maximum.reduce(masks).astype(np.uint8)


def approach_offset(spec: dict[str, object], final_contact: np.ndarray) -> np.ndarray:
    first_rigid = spec["layers"]["rigid"]["primitives"][0]
    handle = np.asarray(first_rigid["points"][0], dtype=np.float32)
    direction = handle - final_contact
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        raise ValueError(f"Cannot infer approach direction for {spec['anchor_id']}")
    return 0.30 * direction / norm


def render_dynamic_frame(
    source_rgb: np.ndarray,
    repaired_rgb: np.ndarray,
    spec: dict[str, object],
    static_masks: dict[str, np.ndarray],
    index: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    height, width = source_rgb.shape[:2]
    utensil = str(spec["action"]["utensil"])
    phase = phase_for_frame(index)
    final_contact = np.asarray(
        spec["action"]["contact_anchors_normalized"][0], dtype=np.float32
    )
    source_contact = mask_centroid_normalized(static_masks["hole"])
    source_delta = source_contact - final_contact
    outside_delta = approach_offset(spec, final_contact)

    if phase == "source_anchor":
        empty = np.zeros((height, width), dtype=np.uint8)
        return source_rgb.copy(), {
            "rigid": empty,
            "contact": empty,
            "material": empty,
            "hole": empty,
        }
    if phase == "approach":
        progress = smoothstep(
            (index - APPROACH_START) / (APPROACH_END - APPROACH_START)
        )
        rigid_delta = source_delta + outside_delta * (1.0 - progress)
        lift = 0.0
    elif phase == "contact_hold":
        rigid_delta = source_delta
        lift = 0.0
    elif phase == "lift":
        lift = smoothstep((index - LIFT_START) / (LIFT_END - LIFT_START))
        rigid_delta = source_delta * (1.0 - lift)
    else:
        raise AssertionError("Final-hold frames are copied before dynamic rendering")

    rigid_primitives = [
        translate_primitive(primitive, rigid_delta)
        for primitive in spec["layers"]["rigid"]["primitives"]
    ]
    contact_primitives = [
        translate_primitive(primitive, rigid_delta)
        for primitive in spec["layers"]["contact"]["primitives"]
    ]
    rigid_parts = render_primitives(rigid_primitives, width, height)
    rigid_mask = union_masks(rigid_parts, (height, width))
    contact_mask = union_masks(
        render_primitives(contact_primitives, width, height), (height, width)
    )

    canvas = source_rgb.copy()
    material_mask = np.zeros((height, width), dtype=np.uint8)
    hole_mask = np.zeros((height, width), dtype=np.uint8)
    if phase == "lift":
        material_primitives = [
            deform_material_primitive(
                primitive, source_delta, final_contact, lift
            )
            for primitive in spec["layers"]["material"]["primitives"]
        ]
        material_mask = union_masks(
            render_primitives(material_primitives, width, height),
            (height, width),
        )
        hole_mask = static_masks["hole"]
        hole_alpha = (hole_mask.astype(np.float32) / 255.0) * lift
        canvas = np.rint(
            source_rgb.astype(np.float32) * (1.0 - hole_alpha[..., None])
            + repaired_rgb.astype(np.float32) * hole_alpha[..., None]
        ).astype(np.uint8)

    contact_blur = cv2.GaussianBlur(
        contact_mask, (0, 0), sigmaX=4.0
    ).astype(np.float32) / 255.0
    canvas = np.clip(
        canvas.astype(np.float32) * (1.0 - 0.16 * contact_blur[..., None]),
        0,
        255,
    ).astype(np.uint8)

    rigid_texture = utensil_texture(width, height, utensil)
    if np.any(material_mask):
        material_texture = source_texture(
            source_rgb, static_masks["hole"], material_mask
        )
    else:
        material_texture = np.zeros_like(source_rgb)

    if utensil == "chopsticks":
        if len(rigid_parts) != 2:
            raise ValueError("Chopstick trajectory requires exactly two rigid parts")
        canvas = composite(canvas, rigid_texture, rigid_parts[0])
        if np.any(material_mask):
            canvas = composite(canvas, material_texture, material_mask)
        canvas = composite(canvas, rigid_texture, rigid_parts[1])
    else:
        canvas = composite(canvas, rigid_texture, rigid_mask)
        if np.any(material_mask):
            canvas = composite(canvas, material_texture, material_mask)

    return canvas, {
        "rigid": rigid_mask,
        "contact": contact_mask,
        "material": material_mask,
        "hole": hole_mask,
    }


def write_video(path: Path, frames_rgb: list[np.ndarray], fps: float) -> None:
    height, width = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {path}")
    for frame in frames_rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def make_review(frames: list[np.ndarray], indices: list[int]) -> Image.Image:
    height, width = frames[0].shape[:2]
    canvas = Image.new("RGB", (width * len(indices), height), "black")
    for column, index in enumerate(indices):
        canvas.paste(Image.fromarray(frames[index]), (column * width, 0))
    return canvas


def build_case(
    static_proxy_dir: Path,
    spec: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    static_manifest = verify_static_proxy(static_proxy_dir)
    source_bgr = read_image(static_proxy_dir / "first_frame.png", cv2.IMREAD_COLOR)
    final_bgr = read_image(static_proxy_dir / "motion_signal.png", cv2.IMREAD_COLOR)
    source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
    height, width = source_rgb.shape[:2]
    if final_rgb.shape != source_rgb.shape:
        raise ValueError(f"Static proxy shape mismatch: {static_proxy_dir}")
    static_masks = {
        name: read_image(
            static_proxy_dir / filename, cv2.IMREAD_GRAYSCALE
        )
        for name, filename in {
            "rigid": "mask_rigid.png",
            "contact": "mask_contact.png",
            "material": "mask_material.png",
            "hole": "mask_hole.png",
            "edit_alpha": "edit_alpha.png",
        }.items()
    }
    inpaint_mask = cv2.dilate(
        (static_masks["hole"] > 127).astype(np.uint8) * 255,
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    )
    repaired_bgr = cv2.inpaint(source_bgr, inpaint_mask, 5.0, cv2.INPAINT_TELEA)
    repaired_rgb = cv2.cvtColor(repaired_bgr, cv2.COLOR_BGR2RGB)

    frames: list[np.ndarray] = []
    layer_records: list[dict[str, int | str]] = []
    changed_union = np.zeros((height, width), dtype=bool)
    for index in range(FRAME_COUNT):
        phase = phase_for_frame(index)
        if phase == "final_hold":
            frame = final_rgb.copy()
            layer_masks = {
                name: static_masks[name]
                for name in ("rigid", "contact", "material", "hole")
            }
        else:
            frame, layer_masks = render_dynamic_frame(
                source_rgb, repaired_rgb, spec, static_masks, index
            )
        frames.append(frame)
        changed = np.any(frame != source_rgb, axis=2)
        changed_union |= changed
        layer_records.append({
            "index": index,
            "phase": phase,
            "changed_pixels": int(changed.sum()),
            "rigid_pixels": int((layer_masks["rigid"] > 127).sum()),
            "contact_pixels": int((layer_masks["contact"] > 127).sum()),
            "material_pixels": int((layer_masks["material"] > 127).sum()),
            "hole_pixels": int((layer_masks["hole"] > 127).sum()),
        })

    if not np.array_equal(frames[SOURCE_FRAME], source_rgb):
        raise AssertionError("Source-anchor control changed the canonical source")
    for index in range(FINAL_START, FINAL_END + 1):
        if not np.array_equal(frames[index], final_rgb):
            raise AssertionError(f"Final-hold control {index} changed the static proxy")

    changed_union |= static_masks["edit_alpha"] > 0
    radius = max(8, int(round(0.014 * min(height, width))))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    support_binary = cv2.dilate(changed_union.astype(np.uint8), kernel) > 0
    support_alpha = cv2.GaussianBlur(
        support_binary.astype(np.uint8) * 255,
        (0, 0),
        sigmaX=max(2.0, radius / 3.0),
    )
    support_alpha[support_binary] = 255
    mask_frames_rgb = []
    for index in range(FRAME_COUNT):
        mask = (
            np.zeros_like(support_alpha)
            if index == SOURCE_FRAME
            else support_alpha
        )
        mask_frames_rgb.append(np.repeat(mask[..., None], 3, axis=2))

    output_dir.mkdir(parents=True, exist_ok=False)
    Image.fromarray(source_rgb).save(output_dir / "first_frame.png", optimize=True)
    Image.fromarray(final_rgb).save(output_dir / "final_proxy.png", optimize=True)
    Image.fromarray(static_masks["edit_alpha"]).save(
        output_dir / "edit_alpha.png", optimize=True
    )
    Image.fromarray(support_alpha).save(
        output_dir / "motion_union_alpha.png", optimize=True
    )
    write_video(output_dir / "dynamic_control.mp4", frames, FPS)
    write_video(output_dir / "dynamic_mask.mp4", mask_frames_rgb, FPS)
    review_indices = [0, 3, 5, 8, 12, 16, SELECTED_FRAME_INDEX]
    make_review(frames, review_indices).save(
        output_dir / "control_review.png", optimize=True
    )
    (output_dir / "prompt.txt").write_text(
        (static_proxy_dir / "prompt.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    negative = (static_proxy_dir / "negative_prompt.txt").read_text(
        encoding="utf-8"
    )
    if negative.strip() != NEGATIVE_PROMPT:
        raise ValueError(f"Frozen negative prompt mismatch: {static_proxy_dir}")
    (output_dir / "negative_prompt.txt").write_text(negative, encoding="utf-8")

    files = (
        "first_frame.png",
        "final_proxy.png",
        "edit_alpha.png",
        "motion_union_alpha.png",
        "dynamic_control.mp4",
        "dynamic_mask.mp4",
        "control_review.png",
        "prompt.txt",
        "negative_prompt.txt",
    )
    outside_final = static_masks["edit_alpha"] == 0
    final_outside_difference = np.abs(
        frames[SELECTED_FRAME_INDEX].astype(np.int16)
        - source_rgb.astype(np.int16)
    )[outside_final]
    manifest = {
        "schema_version": "foodstateedit.dynamic_multikey_proxy.v1",
        "anchor_id": static_proxy_dir.name,
        "method": METHOD,
        "deterministic": True,
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "selected_frame_index": SELECTED_FRAME_INDEX,
        "phase_schedule": {
            "source_anchor": [SOURCE_FRAME, SOURCE_FRAME],
            "approach": [APPROACH_START, APPROACH_END],
            "contact_hold": [CONTACT_START, CONTACT_END],
            "lift": [LIFT_START, LIFT_END],
            "final_hold": [FINAL_START, FINAL_END],
        },
        "geometry_policy": (
            "translated rigid approach; source-contact hold; root-weighted deformable "
            "lift; exact static-proxy final hold"
        ),
        "keyframe_policy": "deterministic_geometry_only_no_image_generation",
        "z_order": (
            "rear_chopstick -> deformable payload -> front_chopstick"
            if spec["action"]["utensil"] == "chopsticks"
            else "rigid utensil -> supported payload"
        ),
        "motion_support_fraction": round(float((support_alpha > 0).mean()), 6),
        "first_frame_vs_source_max_pixel_difference": 0,
        "final_hold_vs_static_proxy_max_pixel_difference": 0,
        "selected_control_outside_final_edit_alpha_max_pixel_difference": int(
            final_outside_difference.max(initial=0)
        ),
        "static_proxy_manifest_sha256": sha256_file(
            static_proxy_dir / "proxy_manifest.json"
        ),
        "frames": layer_records,
        "files": {
            name: {"path": str(output_dir / name), "sha256": sha256_file(output_dir / name)}
            for name in files
        },
        "status": "frozen_deterministic_control_requires_visual_confirmation",
    }
    (output_dir / "proxy_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-proxy-root", type=Path, required=True)
    parser.add_argument("--anchor-specs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to reuse output root: {args.output_root}")
    specs_json = json.loads(args.anchor_specs.read_text(encoding="utf-8"))
    specs = {item["anchor_id"]: item for item in specs_json["anchors"]}
    if set(specs) != set(ANCHORS):
        raise ValueError(f"Expected exactly the frozen anchors: {sorted(specs)}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    cases = [
        build_case(
            args.static_proxy_root / anchor,
            specs[anchor],
            args.output_root / anchor,
        )
        for anchor in ANCHORS
    ]
    summary = {
        "schema_version": "foodstateedit.dynamic_multikey_proxy_summary.v1",
        "method": METHOD,
        "case_count": len(cases),
        "all_deterministic": all(case["deterministic"] for case in cases),
        "all_source_anchors_exact": all(
            case["first_frame_vs_source_max_pixel_difference"] == 0 for case in cases
        ),
        "all_final_holds_exact": all(
            case["final_hold_vs_static_proxy_max_pixel_difference"] == 0 for case in cases
        ),
        "all_selected_controls_exact_outside_final_edit_alpha": all(
            case["selected_control_outside_final_edit_alpha_max_pixel_difference"] == 0
            for case in cases
        ),
        "cases": cases,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "method": METHOD,
        "case_count": len(cases),
        "all_source_anchors_exact": summary["all_source_anchors_exact"],
        "all_final_holds_exact": summary["all_final_holds_exact"],
    }, indent=2))


if __name__ == "__main__":
    main()
