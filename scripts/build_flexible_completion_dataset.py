#!/usr/bin/env python3
"""Build the immutable Day 13 udon relative-3D training package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foodstateedit.projection3d import PinholeCamera, cubic_bezier, smoothstep


SCHEMA_VERSION = "foodstateedit.flexible_completion_dataset.v1"
DESIGN_SCHEMA = "foodstateedit.flexible_completion_mechanism_pilot.v0"
GEOMETRY_SCHEMA = "foodstateedit.flexible_completion_udon_relative3d_geometry.v1"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design-config",
        type=Path,
        default=root / "configs" / "flexible_completion_mechanism_pilot_v0.json",
    )
    parser.add_argument(
        "--geometry-config",
        type=Path,
        default=root / "configs" / "flexible_completion_udon_relative3d_geometry_v1.json",
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def file_record(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def checked_copy(source: Path, target: Path, expected_sha256: str) -> None:
    if not source.is_file() or sha256_file(source) != expected_sha256:
        raise ValueError(f"Frozen source hash mismatch: {source}")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if sha256_file(target) != expected_sha256:
        raise IOError(f"Byte-exact copy verification failed: {target}")


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def normalized_uv(point: list[float], width: int, height: int) -> np.ndarray:
    return np.asarray([point[0] * width, point[1] * height], dtype=np.float64)


def phase_amount(frame_index: int, timeline: dict[str, list[int]]) -> tuple[str, float]:
    for phase in ("source", "approach", "contact", "lift", "final_hold"):
        start, end = timeline[phase]
        if start <= frame_index <= end:
            if start == end:
                return phase, 0.0 if phase == "source" else 1.0
            return phase, (frame_index - start) / (end - start)
    raise ValueError(f"Frame {frame_index} is outside the frozen timeline")


def draw_polyline(
    canvas: np.ndarray,
    uv: np.ndarray,
    color: tuple[int, int, int],
    width: int,
    support: np.ndarray,
) -> None:
    layer = Image.fromarray(canvas)
    draw = ImageDraw.Draw(layer)
    points = [tuple(point) for point in np.rint(uv).astype(int)]
    draw.line(points, fill=color, width=width, joint="curve")
    layer_array = np.asarray(layer)
    canvas[support] = layer_array[support]


def draw_segment(
    canvas: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[int, int, int],
    width: int,
    support: np.ndarray,
) -> None:
    layer = Image.fromarray(canvas)
    draw = ImageDraw.Draw(layer)
    draw.line(
        [tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int))],
        fill=color,
        width=width,
    )
    layer_array = np.asarray(layer)
    canvas[support] = layer_array[support]


def draw_mask_polyline(mask: np.ndarray, uv: np.ndarray, width: int) -> None:
    image = Image.fromarray(mask)
    draw = ImageDraw.Draw(image)
    points = [tuple(point) for point in np.rint(uv).astype(int)]
    draw.line(points, fill=255, width=width, joint="curve")
    mask[:] = np.asarray(image)


def draw_mask_circle(mask: np.ndarray, center: np.ndarray, radius: int) -> None:
    image = Image.fromarray(mask)
    draw = ImageDraw.Draw(image)
    x, y = np.rint(center).astype(int)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    mask[:] = np.asarray(image)


def render_strand(
    frame: np.ndarray,
    strand_uv: np.ndarray,
    strand_z: np.ndarray,
    chopstick_depths: tuple[float, float],
    support: np.ndarray,
    strand_config: dict[str, object],
    front: bool,
) -> None:
    colors = {
        "shadow": tuple(strand_config["shadow_rgb"]),
        "body": tuple(strand_config["color_rgb"]),
        "highlight": tuple(strand_config["highlight_rgb"]),
    }
    split_depth = sum(chopstick_depths) / 2.0
    segment_front = 0.5 * (strand_z[:-1] + strand_z[1:]) < split_depth
    selected = segment_front == front
    starts = np.flatnonzero(selected & np.r_[True, ~selected[:-1]])
    ends = np.flatnonzero(selected & np.r_[~selected[1:], True])
    for start_index, end_index in zip(starts, ends):
        segment = strand_uv[start_index : end_index + 2]
        draw_polyline(
            frame,
            segment + np.asarray([2.0, 3.0]),
            colors["shadow"],
            int(strand_config["shadow_width_px"]),
            support,
        )
        draw_polyline(
            frame,
            segment,
            colors["body"],
            int(strand_config["line_width_px"]),
            support,
        )
        draw_polyline(
            frame,
            segment + np.asarray([-1.0, -1.0]),
            colors["highlight"],
            max(2, int(strand_config["line_width_px"]) // 4),
            support,
        )


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    height, width = frames[0].shape[:2]
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "12",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "1",
        "-map_metadata",
        "-1",
        "-fflags",
        "+bitexact",
        "-y",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise OSError(f"Could not create video: {path} (ffmpeg {return_code})")
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"Video write failed: {path}")


def make_review(frames: list[np.ndarray], indices: list[int], label: str) -> Image.Image:
    height, width = frames[0].shape[:2]
    canvas = Image.new("RGB", (width * 3, height * 2), "black")
    for panel_index, frame_index in enumerate(indices):
        panel = Image.fromarray(frames[frame_index])
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, 205, 28), fill=(0, 0, 0))
        draw.text((7, 7), f"{label} frame {frame_index}", fill=(255, 255, 255))
        canvas.paste(panel, ((panel_index % 3) * width, (panel_index // 3) * height))
    return canvas


def write_metadata(
    path: Path,
    *,
    video_path: str,
    control_path: str,
    reference_path: str,
    prompt: str,
    include_topology: bool,
) -> None:
    fieldnames = [
        "video",
        "vace_video",
        "vace_reference_image",
        "prompt",
        "training_sample_id",
        "training_row_id",
    ]
    if include_topology:
        fieldnames[3:3] = [
            "flexible_strand_mask_video",
            "pinch_contact_mask_video",
            "source_connection_mask_video",
        ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row_id in range(2):
            row: dict[str, object] = {
                "video": video_path,
                "vace_video": control_path,
                "vace_reference_image": reference_path,
                "prompt": prompt,
                "training_sample_id": "udon_chopsticks_imagegen_pseudo_v1",
                "training_row_id": row_id,
            }
            if include_topology:
                row.update(
                    {
                        "flexible_strand_mask_video": "topology_masks/flexible_strand.mp4",
                        "pinch_contact_mask_video": "topology_masks/pinch_contact.mp4",
                        "source_connection_mask_video": "topology_masks/source_connection.mp4",
                    }
                )
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    design_path = args.design_config.resolve()
    geometry_path = args.geometry_config.resolve()
    design = json.loads(design_path.read_text(encoding="utf-8"))
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    if design.get("schema_version") != DESIGN_SCHEMA or design.get("execution_allowed"):
        raise ValueError("Expected the frozen non-executable Day 13 design")
    if geometry.get("schema_version") != GEOMETRY_SCHEMA:
        raise ValueError("Unexpected relative-3D geometry schema")
    sample = design["sample"]
    if geometry["sample_id"] != sample["sample_id"]:
        raise ValueError("Geometry and design sample ids differ")

    source_root = (
        args.source_root.resolve()
        if args.source_root is not None
        else (root / sample["source_dataset_root"]).resolve()
    )
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse output root: {output_root}")
    source_manifest_path = source_root / "dataset_manifest.json"
    if sha256_file(source_manifest_path) != sample["source_dataset_manifest_sha256"]:
        raise ValueError("Frozen Day 12 source manifest hash mismatch")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    selected = next(
        item
        for item in source_manifest["samples"]
        if item["sample_id"] == sample["sample_id"]
    )

    output_root.mkdir(parents=True)
    copy_map = {
        "video": ("target_video", "video/udon_chopsticks_imagegen_pseudo_v1.mp4"),
        "vace_reference_image": (
            "reference_image",
            "vace_reference_image/udon_chopsticks_imagegen_pseudo_v1.png",
        ),
        "target_keyframe": (
            "target_keyframe",
            "target_keyframe/udon_chopsticks_imagegen_pseudo_v1.png",
        ),
        "edit_alpha": (
            "edit_alpha",
            "edit_alpha/udon_chopsticks_imagegen_pseudo_v1.png",
        ),
        "phase_schedule": ("phase_schedule", "phase_schedule.json"),
        "planar_vace_control": (
            "planar_vace_control",
            "vace_video_planar/udon_chopsticks_imagegen_pseudo_v1.mp4",
        ),
    }
    copied_records: dict[str, dict[str, object]] = {}
    for logical_name, (design_name, destination_relative) in copy_map.items():
        frozen = sample["files"][design_name]
        source = source_root / frozen["path"]
        destination = output_root / destination_relative
        checked_copy(source, destination, frozen["sha256"])
        copied_records[logical_name] = file_record(output_root, destination)

    source_image = read_rgb(
        output_root / copied_records["vace_reference_image"]["path"]
    )
    alpha = read_gray(output_root / copied_records["edit_alpha"]["path"])
    height, width = source_image.shape[:2]
    support = alpha > 0
    if not np.any(support):
        raise ValueError("Frozen edit support is empty")
    camera = PinholeCamera.normalized_relative(
        width, height, geometry["camera"]["focal_scale"]
    )
    anchors = geometry["anchors_normalized"]
    depths = geometry["relative_depth"]
    chopsticks = geometry["chopsticks"]
    strand_config = geometry["strand"]
    masks_config = geometry["masks"]
    timeline = geometry["timeline"]
    approach_uv = normalized_uv(anchors["approach_pinch_uv"], width, height)
    contact_uv = normalized_uv(anchors["contact_pinch_uv"], width, height)
    final_uv = normalized_uv(anchors["final_pinch_uv"], width, height)
    bowl_uv = normalized_uv(anchors["fixed_bowl_connection_uv"], width, height)

    frames: list[np.ndarray] = []
    strand_masks: list[np.ndarray] = []
    contact_masks: list[np.ndarray] = []
    connection_masks: list[np.ndarray] = []
    frame_records: list[dict[str, object]] = []
    strand_xyz_frames = np.zeros(
        (geometry["frame_count"], strand_config["samples"], 3), dtype=np.float32
    )
    pinch_xyz_frames = np.zeros((geometry["frame_count"], 3), dtype=np.float32)
    changed_union = np.zeros((height, width), dtype=bool)
    observed_depth_crossing = False

    for frame_index in range(geometry["frame_count"]):
        phase, amount = phase_amount(frame_index, timeline)
        frame = source_image.copy()
        strand_mask = np.zeros((height, width), dtype=np.uint8)
        contact_mask = np.zeros_like(strand_mask)
        connection_mask = np.zeros_like(strand_mask)
        if phase == "source":
            frames.append(frame)
            strand_masks.append(strand_mask)
            contact_masks.append(contact_mask)
            connection_masks.append(connection_mask)
            frame_records.append(
                {"frame": frame_index, "phase": phase, "changed_pixels": 0}
            )
            continue

        if phase == "approach":
            pinch_uv = approach_uv * (1.0 - smoothstep(amount)) + contact_uv * smoothstep(amount)
            lift_amount = 0.0
        elif phase == "contact":
            pinch_uv = contact_uv
            lift_amount = 0.0
        else:
            lift_amount = 1.0 if phase == "final_hold" else smoothstep(amount)
            pinch_uv = contact_uv * (1.0 - lift_amount) + final_uv * lift_amount

        pinch_xyz = camera.backproject(pinch_uv[None, :], depths["pinch"])[0]
        pinch_xyz_frames[frame_index] = pinch_xyz.astype(np.float32)
        offset = np.asarray(
            [
                chopsticks["handle_offset_normalized"][0] * width,
                chopsticks["handle_offset_normalized"][1] * height,
            ]
        )
        separation = chopsticks["tip_separation_normalized"] * height
        handle_separation = chopsticks["handle_separation_normalized"] * height
        chopstick_lines: list[tuple[np.ndarray, np.ndarray, float, tuple[int, int, int]]] = []
        for side, depth_key, color_key in (
            (-1.0, "near_chopstick", "near_color_rgb"),
            (1.0, "far_chopstick", "far_color_rgb"),
        ):
            tip_uv = pinch_uv + np.asarray([0.0, side * separation / 2.0])
            handle_uv = pinch_uv + offset + np.asarray([0.0, side * handle_separation / 2.0])
            tip_xyz = camera.backproject(tip_uv[None, :], depths[depth_key])[0]
            handle_xyz = camera.backproject(handle_uv[None, :], depths[depth_key])[0]
            projected, _ = camera.project(np.stack([tip_xyz, handle_xyz]))
            chopstick_lines.append(
                (projected[0], projected[1], float(depths[depth_key]), tuple(chopsticks[color_key]))
            )

        strand_uv: np.ndarray | None = None
        strand_z: np.ndarray | None = None
        if phase in {"contact", "lift", "final_hold"}:
            bowl_xyz = camera.backproject(bowl_uv[None, :], depths["bowl_connection"])[0]
            control_a_uv = bowl_uv + np.asarray(
                [
                    strand_config["control_a_offset_normalized"][0] * width,
                    strand_config["control_a_offset_normalized"][1] * height,
                ]
            )
            midpoint_uv = bowl_uv * 0.42 + pinch_uv * 0.58
            control_b_uv = midpoint_uv + np.asarray(
                [
                    strand_config["control_b_offset_normalized"][0] * width,
                    strand_config["control_b_offset_normalized"][1] * height,
                ]
            )
            control_a_xyz = camera.backproject(
                control_a_uv[None, :], depths["strand_control_a"]
            )[0]
            control_b_xyz = camera.backproject(
                control_b_uv[None, :], depths["strand_control_b"]
            )[0]
            strand_xyz = cubic_bezier(
                bowl_xyz,
                control_a_xyz,
                control_b_xyz,
                pinch_xyz,
                strand_config["samples"],
            )
            strand_uv, strand_z = camera.project(strand_xyz)
            strand_xyz_frames[frame_index] = strand_xyz.astype(np.float32)
            observed_depth_crossing |= bool(
                strand_z.min() < depths["near_chopstick"]
                and strand_z.max() > depths["far_chopstick"]
            )
            render_strand(
                frame,
                strand_uv,
                strand_z,
                (depths["near_chopstick"], depths["far_chopstick"]),
                support,
                strand_config,
                front=False,
            )

        for start, end, _, color in sorted(chopstick_lines, key=lambda item: item[2], reverse=True):
            draw_segment(
                frame,
                start + np.asarray([2.0, 3.0]),
                end + np.asarray([2.0, 3.0]),
                (40, 24, 17),
                chopsticks["shadow_width_px"],
                support,
            )
            draw_segment(frame, start, end, color, chopsticks["line_width_px"], support)
            draw_segment(
                frame,
                start + np.asarray([-1.0, -1.0]),
                end + np.asarray([-1.0, -1.0]),
                tuple(chopsticks["highlight_rgb"]),
                max(2, chopsticks["line_width_px"] // 4),
                support,
            )

        if strand_uv is not None and strand_z is not None:
            render_strand(
                frame,
                strand_uv,
                strand_z,
                (depths["near_chopstick"], depths["far_chopstick"]),
                support,
                strand_config,
                front=True,
            )
            if phase == "contact":
                local_count = max(
                    2,
                    int(round(len(strand_uv) * strand_config["contact_local_fraction"])),
                )
                active_uv = strand_uv[-local_count:]
            else:
                active_uv = strand_uv
            draw_mask_polyline(strand_mask, active_uv, masks_config["strand_width_px"])
            draw_mask_circle(
                contact_mask, pinch_uv, masks_config["pinch_radius_px"]
            )
            if phase in {"lift", "final_hold"}:
                draw_mask_circle(
                    connection_mask,
                    bowl_uv,
                    masks_config["source_connection_radius_px"],
                )
            strand_mask[~support] = 0
            contact_mask[~support] = 0
            connection_mask[~support] = 0

        changed = np.any(frame != source_image, axis=2)
        if np.any(changed & ~support):
            raise AssertionError("Relative-3D render changed pixels outside frozen edit support")
        changed_union |= changed
        frames.append(frame)
        strand_masks.append(strand_mask)
        contact_masks.append(contact_mask)
        connection_masks.append(connection_mask)
        frame_records.append(
            {
                "frame": frame_index,
                "phase": phase,
                "phase_amount": round(float(amount), 6),
                "lift_amount": round(float(lift_amount), 6),
                "pinch_uv": pinch_uv.tolist(),
                "strand_mask_pixels": int((strand_mask > 0).sum()),
                "pinch_contact_mask_pixels": int((contact_mask > 0).sum()),
                "source_connection_mask_pixels": int((connection_mask > 0).sum()),
                "changed_pixels": int(changed.sum()),
            }
        )

    if not np.array_equal(frames[0], source_image):
        raise AssertionError("Frame zero must equal the source before video encoding")
    if not observed_depth_crossing:
        raise AssertionError("Flexible strand did not cross both chopstick depths")
    for index, record in enumerate(frame_records):
        phase = record["phase"]
        mask_nonempty = any(
            np.any(mask_list[index])
            for mask_list in (strand_masks, contact_masks, connection_masks)
        )
        if phase in {"source", "approach"} and mask_nonempty:
            raise AssertionError(f"Topology mask activates before contact at frame {index}")
        if phase == "contact" and (
            not np.any(strand_masks[index]) or not np.any(contact_masks[index])
        ):
            raise AssertionError(f"Contact masks are missing at frame {index}")
        if phase in {"lift", "final_hold"} and not all(
            np.any(mask_list[index])
            for mask_list in (strand_masks, contact_masks, connection_masks)
        ):
            raise AssertionError(f"Full topology masks are missing at frame {index}")

    relative_control_path = (
        output_root
        / "vace_video_relative3d"
        / "udon_chopsticks_imagegen_pseudo_v1.mp4"
    )
    relative_control_path.parent.mkdir(parents=True)
    write_video(relative_control_path, frames, geometry["fps"])
    topology_dir = output_root / "topology_masks"
    topology_dir.mkdir(parents=True)
    mask_sets = {
        "flexible_strand.mp4": strand_masks,
        "pinch_contact.mp4": contact_masks,
        "source_connection.mp4": connection_masks,
    }
    for name, masks in mask_sets.items():
        rgb_frames = [np.repeat((mask > 0)[..., None], 3, axis=2).astype(np.uint8) * 255 for mask in masks]
        write_video(topology_dir / name, rgb_frames, geometry["fps"])

    geometry_dir = output_root / "relative3d_geometry"
    geometry_dir.mkdir(parents=True)
    geometry_npz = geometry_dir / "udon_relative3d_geometry.npz"
    np.savez_compressed(
        geometry_npz,
        intrinsic=camera.intrinsic.astype(np.float32),
        pinch_xyz=pinch_xyz_frames,
        strand_xyz=strand_xyz_frames,
        fixed_bowl_uv=bowl_uv.astype(np.float32),
        chopstick_depths=np.asarray(
            [depths["near_chopstick"], depths["far_chopstick"]], dtype=np.float32
        ),
    )
    review_dir = output_root / "review"
    review_dir.mkdir(parents=True)
    review_indices = geometry["review_frame_indices"]
    control_review = review_dir / "relative3d_control_contact_sheet.png"
    make_review(frames, review_indices, "relative3d").save(control_review)
    composite_masks = [
        np.stack(
            [strand_masks[index], contact_masks[index], connection_masks[index]], axis=2
        )
        for index in range(geometry["frame_count"])
    ]
    mask_review = review_dir / "topology_mask_contact_sheet.png"
    make_review(composite_masks, review_indices, "RGB masks").save(mask_review)

    metadata_specs = {
        "metadata_planar_uniform.csv": (
            copied_records["planar_vace_control"]["path"],
            False,
        ),
        "metadata_relative3d_uniform.csv": (
            relative_control_path.relative_to(output_root).as_posix(),
            False,
        ),
        "metadata_relative3d_topology_weighted.csv": (
            relative_control_path.relative_to(output_root).as_posix(),
            True,
        ),
    }
    metadata_records: dict[str, dict[str, object]] = {}
    for name, (control_path, include_topology) in metadata_specs.items():
        metadata_path = output_root / name
        write_metadata(
            metadata_path,
            video_path=copied_records["video"]["path"],
            control_path=control_path,
            reference_path=copied_records["vace_reference_image"]["path"],
            prompt=selected["prompt"],
            include_topology=include_topology,
        )
        metadata_records[name.removesuffix(".csv")] = {
            **file_record(output_root, metadata_path),
            "row_count": 2,
            "unique_sample_count": 1,
        }

    derived_files = {
        "relative_3d_vace_control_video": file_record(output_root, relative_control_path),
        "flexible_strand_mask_video": file_record(
            output_root, topology_dir / "flexible_strand.mp4"
        ),
        "pinch_contact_mask_video": file_record(
            output_root, topology_dir / "pinch_contact.mp4"
        ),
        "source_connection_mask_video": file_record(
            output_root, topology_dir / "source_connection.mp4"
        ),
        "geometry_npz": file_record(output_root, geometry_npz),
        "control_review": file_record(output_root, control_review),
        "mask_review": file_record(output_root, mask_review),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "day13_3d_guided_flexible_completion_udon_v1",
        "scientific_status": "seen_synthetic_relative_3d_flexible_completion_mechanism_pilot",
        "claim_limit": design["claim_limit"],
        "sample_id": sample["sample_id"],
        "sample_count": 2,
        "unique_sample_count": 1,
        "frame_count": geometry["frame_count"],
        "fps": geometry["fps"],
        "prompt": selected["prompt"],
        "geometry_source": camera.geometry_source,
        "camera_intrinsic": camera.intrinsic.tolist(),
        "frame_zero_equals_source_before_video_encoding": True,
        "strand_depth_crosses_both_chopstick_depths": observed_depth_crossing,
        "outside_frozen_edit_support_max_pixel_difference_before_video_encoding": int(
            np.abs(
                np.stack(frames).astype(np.int16)
                - source_image[None, ...].astype(np.int16)
            )[:, ~support, :].max(initial=0)
        ),
        "changed_union_fraction": float(changed_union.mean()),
        "frozen_source_files": copied_records,
        "derived_files": derived_files,
        "metadata": metadata_records,
        "frame_records": frame_records,
        "training_arms": {
            "planar_uniform": "metadata_planar_uniform.csv",
            "relative3d_uniform": "metadata_relative3d_uniform.csv",
            "relative3d_topology_weighted": "metadata_relative3d_topology_weighted.csv",
        },
        "topology_masks": {
            "values_after_decode": "threshold_at_127_to_binary_0_or_1",
            "source_and_approach_are_zero": True,
            "contact_has_local_strand_and_pinch": True,
            "lift_and_final_hold_have_strand_pinch_and_source_connection": True,
        },
        "provenance": {
            "design_config": design_path.relative_to(root).as_posix(),
            "design_config_sha256": sha256_file(design_path),
            "geometry_config": geometry_path.relative_to(root).as_posix(),
            "geometry_config_sha256": sha256_file(geometry_path),
            "source_dataset_manifest_sha256": sha256_file(source_manifest_path),
            "builder": Path(__file__).name,
            "builder_sha256_lf": sha256_lf(Path(__file__).resolve()),
            "copy_policy": "frozen_target_reference_planar_control_alpha_and_phase_schedule_are_byte_exact",
        },
        "status": "complete_requires_hash_frozen_training_execution_config",
    }
    manifest_path = output_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "manifest_sha256": sha256_file(manifest_path),
                "relative3d_control_sha256": sha256_file(relative_control_path),
                "observed_depth_crossing": observed_depth_crossing,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
