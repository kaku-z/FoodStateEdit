#!/usr/bin/env python3
"""Build a deterministic relative-3D fork/spaghetti motion and project it to 2-D."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foodstateedit.projection3d import (
    PinholeCamera,
    cubic_bezier,
    helix_about_axis,
    interpolate_points,
    polyline_length,
    smoothstep,
    zbuffer_splat,
)


METHOD = "fork_relative_3d_projection_v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--anchor-specs", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--final-proxy", type=Path, required=True)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L").resize(size, Image.Resampling.NEAREST)) > 127


def normalized_to_pixels(point: list[float], width: int, height: int) -> np.ndarray:
    return np.asarray([point[0] * width, point[1] * height], dtype=np.float64)


def phase_state(index: int, timeline: dict[str, object]) -> tuple[str, float]:
    if index == 0:
        return "source_anchor", 0.0
    approach_start, approach_end = timeline["approach"]
    contact_start, contact_end = timeline["contact_twirl"]
    lift_start, lift_end = timeline["lift"]
    final_start, final_end = timeline["final_hold"]
    if approach_start <= index <= approach_end:
        return "approach", (index - approach_start) / max(approach_end - approach_start, 1)
    if contact_start <= index <= contact_end:
        return "contact_twirl", (index - contact_start + 1) / (contact_end - contact_start + 1)
    if lift_start <= index <= lift_end:
        return "lift", (index - lift_start) / max(lift_end - lift_start, 1)
    if final_start <= index <= final_end:
        return "final_hold", 1.0
    raise ValueError(f"Frame {index} is outside the frozen phase schedule")


def draw_curve_segments(
    canvas: np.ndarray,
    camera: PinholeCamera,
    curves: list[np.ndarray],
    fork_depth: float,
    front: bool,
    line_width: int,
    colors: list[tuple[int, int, int]],
    mask: np.ndarray,
) -> None:
    for curve_index, curve in enumerate(curves):
        uv, depth = camera.project(curve)
        color = colors[curve_index % len(colors)]
        for index in range(len(curve) - 1):
            segment_depth = 0.5 * float(depth[index] + depth[index + 1])
            is_front = segment_depth < fork_depth
            if is_front != front:
                continue
            start = tuple(np.rint(uv[index]).astype(int))
            end = tuple(np.rint(uv[index + 1]).astype(int))
            cv2.line(canvas, start, end, color, line_width, cv2.LINE_AA)
            cv2.line(mask, start, end, 255, line_width, cv2.LINE_AA)


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise OSError(f"Could not create video: {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def make_review(frames: list[np.ndarray], indices: list[int]) -> Image.Image:
    height, width = frames[0].shape[:2]
    canvas = Image.new("RGB", (width * len(indices), height), "black")
    for column, index in enumerate(indices):
        panel = Image.fromarray(frames[index])
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, 155, 26), fill=(0, 0, 0))
        draw.text((7, 7), f"frame {index}", fill=(255, 255, 255))
        canvas.paste(panel, (column * width, 0))
    return canvas


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output_dir}")

    config_path = args.config.resolve()
    specs_path = args.anchor_specs.resolve()
    source_path = args.source.resolve()
    final_proxy_path = args.final_proxy.resolve()
    annotation_dir = args.annotation_dir.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    specs = json.loads(specs_path.read_text(encoding="utf-8"))
    anchor = next(item for item in specs["anchors"] if item["anchor_id"] == config["anchor_id"])
    if anchor["action"]["type"] != "twirl_and_lift" or anchor["action"]["utensil"] != "fork":
        raise ValueError("Frozen pilot requires one fork twirl-and-lift anchor")

    source = load_rgb(source_path)
    final_proxy = load_rgb(final_proxy_path)
    if source.shape != final_proxy.shape:
        raise ValueError("Source and final proxy shapes differ")
    height, width = source.shape[:2]
    rigid_mask = load_mask(annotation_dir / "mask_rigid.png", (width, height))
    hole_mask = load_mask(annotation_dir / "mask_hole.png", (width, height))
    if not np.any(rigid_mask) or not np.any(hole_mask):
        raise ValueError("Rigid and hole masks must be nonempty")

    camera = PinholeCamera.normalized_relative(width, height, config["camera"]["focal_scale"])
    depths = config["relative_depth"]
    action = anchor["action"]
    contact_final_uv = normalized_to_pixels(action["contact_anchors_normalized"][0], width, height)
    direction_uv = np.asarray(
        [action["target_direction_normalized"][0] * width, action["target_direction_normalized"][1] * height],
        dtype=np.float64,
    )
    contact_source_uv = contact_final_uv - direction_uv
    handle_final_uv = normalized_to_pixels(anchor["layers"]["rigid"]["primitives"][0]["points"][0], width, height)

    rigid_y, rigid_x = np.nonzero(rigid_mask)
    rigid_final_uv = np.column_stack([rigid_x, rigid_y]).astype(np.float64)
    rigid_colors = final_proxy[rigid_y, rigid_x]
    rigid_contact_uv = rigid_final_uv - direction_uv[None, :]
    rigid_start_uv = rigid_final_uv - config["fork"]["approach_direction_scale"] * direction_uv[None, :]
    rigid_final_3d = camera.backproject(rigid_final_uv, depths["fork_final"])
    rigid_contact_3d = camera.backproject(rigid_contact_uv, depths["source_contact"])
    rigid_start_3d = camera.backproject(rigid_start_uv, depths["approach_start"])
    rigid_final_reprojection, _ = camera.project(rigid_final_3d)
    reprojection_error = np.linalg.norm(rigid_final_reprojection - rigid_final_uv, axis=1)

    inpainted = cv2.cvtColor(
        cv2.inpaint(
            cv2.cvtColor(source, cv2.COLOR_RGB2BGR),
            hole_mask.astype(np.uint8) * 255,
            5.0,
            cv2.INPAINT_TELEA,
        ),
        cv2.COLOR_BGR2RGB,
    )
    timeline = config["timeline"]
    spaghetti = config["spaghetti"]
    tail_anchors_uv = [normalized_to_pixels(point, width, height) for point in spaghetti["tail_anchor_normalized"]]
    tail_anchors_3d = [camera.backproject(point[None, :], depths["plate_anchor"])[0] for point in tail_anchors_uv]
    colors = [(239, 177, 73), (219, 126, 43), (246, 194, 94), (198, 89, 30)]

    frames: list[np.ndarray] = []
    frame_masks: list[np.ndarray] = []
    changed_union = np.zeros((height, width), dtype=bool)
    records: list[dict[str, object]] = []
    selected_geometry: dict[str, np.ndarray] = {}

    for index in range(timeline["frame_count"]):
        phase, phase_amount = phase_state(index, timeline)
        if phase == "source_anchor":
            frame = source.copy()
            action_mask = np.zeros((height, width), dtype=np.uint8)
            frames.append(frame)
            frame_masks.append(action_mask)
            records.append({"index": index, "phase": phase, "changed_pixels": 0})
            continue

        if phase == "approach":
            rigid_points = interpolate_points(rigid_start_3d, rigid_contact_3d, phase_amount)
            lift_amount = 0.0
            twirl_amount = 0.0
        elif phase == "contact_twirl":
            rigid_points = rigid_contact_3d
            lift_amount = 0.0
            twirl_amount = smoothstep(phase_amount)
        else:
            lift_amount = smoothstep(phase_amount)
            rigid_points = interpolate_points(rigid_contact_3d, rigid_final_3d, lift_amount)
            twirl_amount = 1.0

        fork_uv, fork_z = camera.project(rigid_points)
        fork_depth = float(np.median(fork_z))
        repair_weight = lift_amount if phase in {"lift", "final_hold"} else 0.0
        frame = source.copy()
        if repair_weight > 0:
            blended = np.rint(
                source.astype(np.float32) * (1.0 - repair_weight)
                + inpainted.astype(np.float32) * repair_weight
            ).astype(np.uint8)
            frame[hole_mask] = blended[hole_mask]

        action_mask = np.zeros((height, width), dtype=np.uint8)
        curves: list[np.ndarray] = []
        helix_curves: list[np.ndarray] = []
        tail_curves: list[np.ndarray] = []
        if twirl_amount > 0:
            center_source = camera.backproject(contact_source_uv[None, :], depths["source_contact"])[0]
            center_final = camera.backproject(contact_final_uv[None, :], depths["fork_final"])[0]
            center = interpolate_points(center_source[None, :], center_final[None, :], lift_amount)[0]
            handle_source_uv = handle_final_uv - direction_uv
            handle_current_uv = handle_source_uv * (1.0 - lift_amount) + handle_final_uv * lift_amount
            handle = camera.backproject(handle_current_uv[None, :], fork_depth)[0]
            axis = handle - center
            turns = max(0.55, spaghetti["helix_turns"] * twirl_amount)
            for strand_index in range(spaghetti["strand_count"]):
                phase_offset = 2.0 * np.pi * strand_index / spaghetti["strand_count"]
                helix = helix_about_axis(
                    center,
                    axis,
                    spaghetti["helix_radius"] * (0.86 + 0.06 * strand_index),
                    spaghetti["helix_pitch"],
                    turns,
                    spaghetti["helix_samples"],
                    phase_offset,
                )
                tail_end = tail_anchors_3d[strand_index]
                gravity = np.asarray([0.0, 0.055 + 0.006 * strand_index, 0.018])
                control_a = helix[-1] + gravity
                control_b = tail_end + np.asarray([0.0, -0.045, -0.015])
                tail = cubic_bezier(
                    helix[-1], control_a, control_b, tail_end, spaghetti["tail_samples"]
                )
                helix_curves.append(helix)
                tail_curves.append(tail)
            curves = [*helix_curves, *tail_curves]
            draw_curve_segments(
                frame,
                camera,
                curves,
                fork_depth,
                False,
                spaghetti["line_width_px"],
                colors,
                action_mask,
            )

        frame, visible_rigid = zbuffer_splat(frame, fork_uv, fork_z, rigid_colors)
        fork_pixels = np.rint(fork_uv).astype(int)
        valid_fork = (
            (fork_pixels[:, 0] >= 0)
            & (fork_pixels[:, 0] < width)
            & (fork_pixels[:, 1] >= 0)
            & (fork_pixels[:, 1] < height)
        )
        action_mask[fork_pixels[valid_fork, 1], fork_pixels[valid_fork, 0]] = 255
        if curves:
            draw_curve_segments(
                frame,
                camera,
                curves,
                fork_depth,
                True,
                spaghetti["line_width_px"],
                colors,
                action_mask,
            )
        action_mask[hole_mask & (repair_weight > 0)] = 255
        changed = np.any(frame != source, axis=2)
        changed_union |= changed
        frames.append(frame)
        frame_masks.append(action_mask)
        record: dict[str, object] = {
            "index": index,
            "phase": phase,
            "phase_amount": round(float(phase_amount), 6),
            "lift_amount": round(float(lift_amount), 6),
            "twirl_amount": round(float(twirl_amount), 6),
            "fork_depth_median": fork_depth,
            "fork_visible_splat_fraction": float(visible_rigid.mean()),
            "changed_pixels": int(changed.sum()),
        }
        if helix_curves:
            all_helix = np.concatenate(helix_curves)
            record["helix_depth_min"] = float(all_helix[:, 2].min())
            record["helix_depth_max"] = float(all_helix[:, 2].max())
            record["depth_crosses_fork"] = bool(
                all_helix[:, 2].min() < fork_depth < all_helix[:, 2].max()
            )
            record["strand_lengths_3d"] = [
                polyline_length(helix) + polyline_length(tail)
                for helix, tail in zip(helix_curves, tail_curves)
            ]
        records.append(record)
        if index == timeline["selected_frame_index"]:
            selected_geometry = {
                "fork_points": rigid_points.astype(np.float32),
                "fork_uv": fork_uv.astype(np.float32),
                "helix_points": np.stack(helix_curves).astype(np.float32),
                "tail_points": np.stack(tail_curves).astype(np.float32),
            }

    if not np.array_equal(frames[0], source):
        raise AssertionError("Frame zero is not the exact source")
    selected_record = records[timeline["selected_frame_index"]]
    if config["projection"]["depth_crossing_required"] and not selected_record.get("depth_crosses_fork"):
        raise AssertionError("Selected 3-D helix does not cross the fork depth")

    radius = max(8, int(round(config["projection"]["motion_support_dilation_fraction_min_dimension"] * min(height, width))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    support_binary = cv2.dilate(changed_union.astype(np.uint8), kernel) > 0
    support_alpha = cv2.GaussianBlur(
        support_binary.astype(np.uint8) * 255,
        (0, 0),
        sigmaX=max(2.0, radius / 3.0),
    )
    support_alpha[support_binary] = 255
    mask_frames = [
        np.repeat((np.zeros_like(support_alpha) if index == 0 else support_alpha)[..., None], 3, axis=2)
        for index in range(timeline["frame_count"])
    ]
    outside_support = support_alpha == 0
    outside_difference = np.abs(
        frames[timeline["selected_frame_index"]][outside_support].astype(np.int16)
        - source[outside_support].astype(np.int16)
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    Image.fromarray(source).save(output_dir / "first_frame.png")
    Image.fromarray(frames[timeline["selected_frame_index"]]).save(output_dir / "projected_frame_18.png")
    Image.fromarray(support_alpha).save(output_dir / "motion_union_alpha.png")
    write_video(output_dir / "dynamic_control.mp4", frames, timeline["fps"])
    write_video(output_dir / "dynamic_mask.mp4", mask_frames, timeline["fps"])
    make_review(frames, [0, 3, 5, 6, 8, 12, 16, 18]).save(output_dir / "control_review.png")
    np.savez_compressed(
        output_dir / "geometry_3d.npz",
        intrinsic=camera.intrinsic.astype(np.float32),
        contact_source_uv=contact_source_uv.astype(np.float32),
        contact_final_uv=contact_final_uv.astype(np.float32),
        **selected_geometry,
    )

    files = [
        "first_frame.png",
        "projected_frame_18.png",
        "motion_union_alpha.png",
        "dynamic_control.mp4",
        "dynamic_mask.mp4",
        "control_review.png",
        "geometry_3d.npz",
    ]
    manifest = {
        "schema_version": "foodstateedit.fork_3d_projection_run.v0",
        "method": METHOD,
        "scientific_status": config["scientific_status"],
        "claim_limit": config["claim_limit"],
        "anchor_id": anchor["anchor_id"],
        "geometry_source": camera.geometry_source,
        "camera_intrinsic": camera.intrinsic.tolist(),
        "frame_count": timeline["frame_count"],
        "fps": timeline["fps"],
        "selected_frame_index": timeline["selected_frame_index"],
        "fork_final_reprojection_max_error_px": float(reprojection_error.max()),
        "fork_final_reprojection_mean_error_px": float(reprojection_error.mean()),
        "selected_depth_crosses_fork": bool(selected_record["depth_crosses_fork"]),
        "selected_helix_depth_range": [
            selected_record["helix_depth_min"],
            selected_record["helix_depth_max"],
        ],
        "selected_fork_depth": selected_record["fork_depth_median"],
        "selected_strand_lengths_3d": selected_record["strand_lengths_3d"],
        "outside_motion_support_max_pixel_difference": int(outside_difference.max(initial=0)),
        "motion_support_fraction": float((support_alpha > 0).mean()),
        "source_sha256": sha256_file(source_path),
        "final_proxy_sha256": sha256_file(final_proxy_path),
        "config_sha256": sha256_file(config_path),
        "anchor_specs_sha256": sha256_file(specs_path),
        "frames": records,
        "files": {
            name: {"size_bytes": (output_dir / name).stat().st_size, "sha256": sha256_file(output_dir / name)}
            for name in files
        },
        "status": "complete_relative_3d_projection_pilot_requires_visual_and_learned_render_review",
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
