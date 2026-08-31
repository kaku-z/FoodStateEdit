#!/usr/bin/env python3
"""Build a two-sample phase-varying pseudo-video dataset for an overfit sanity gate.

This does not create natural motion ground truth. It deterministically moves the
bounded, audited final-state patch through approach, contact, lift, and hold
phases. The purpose is only to test whether the VACE-LoRA training path can learn
a visibly non-identity temporal mapping before more data are collected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


SCHEMA_VERSION = "foodstateedit.phase_action_overfit_dataset.v1"
DATASET_ID = "day11_phase_action_overfit_dataset_v1"
SCIENTIFIC_STATUS = "mechanism_pilot_with_synthetic_pseudotargets"
TARGET_POLICY = "disclosed_imagegen_pseudotarget_nonidentity"
EXPERIMENT_VARIANT = "phase_varying_overfit_sanity_v1"
FRAME_COUNT = 21
FPS = 8
CONTACT_SHEET_FRAMES = (0, 3, 6, 10, 15, 20)


def phase_schedule() -> list[dict[str, object]]:
    offsets = {
        0: (0, 0, 0.0, "source"),
        1: (160, -96, 0.35, "approach"),
        2: (128, -70, 0.50, "approach"),
        3: (96, -40, 0.65, "approach"),
        4: (64, -10, 0.82, "approach"),
        5: (32, 20, 1.0, "approach"),
        6: (0, 40, 1.0, "contact"),
        7: (0, 40, 1.0, "contact"),
        8: (0, 40, 1.0, "contact"),
        9: (0, 35, 1.0, "lift"),
        10: (0, 30, 1.0, "lift"),
        11: (0, 25, 1.0, "lift"),
        12: (0, 20, 1.0, "lift"),
        13: (0, 15, 1.0, "lift"),
        14: (0, 10, 1.0, "lift"),
        15: (0, 5, 1.0, "lift"),
        16: (0, 0, 1.0, "final_hold"),
        17: (0, 0, 1.0, "final_hold"),
        18: (0, 0, 1.0, "final_hold"),
        19: (0, 0, 1.0, "final_hold"),
        20: (0, 0, 1.0, "final_hold"),
    }
    return [
        {"frame": frame, "dx": dx, "dy": dy, "opacity": opacity, "phase": phase}
        for frame, (dx, dy, opacity, phase) in offsets.items()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def file_record(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def save_png_new(path: Path, array: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "L" if array.ndim == 2 else "RGB"
    Image.fromarray(array, mode=mode).save(path, optimize=True)


def translate(array: np.ndarray, dx: int, dy: int) -> np.ndarray:
    height, width = array.shape[:2]
    output = np.zeros_like(array)
    source_x0 = max(0, -dx)
    source_x1 = min(width, width - dx)
    source_y0 = max(0, -dy)
    source_y1 = min(height, height - dy)
    if source_x1 <= source_x0 or source_y1 <= source_y0:
        return output
    target_x0 = source_x0 + dx
    target_x1 = source_x1 + dx
    target_y0 = source_y0 + dy
    target_y1 = source_y1 + dy
    output[target_y0:target_y1, target_x0:target_x1] = array[
        source_y0:source_y1, source_x0:source_x1
    ]
    return output


def composite(foreground: np.ndarray, background: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    weight = alpha.astype(np.float32)[..., None] / 255.0
    output = foreground.astype(np.float32) * weight + background.astype(np.float32) * (1.0 - weight)
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


def synthesize_sequence(
    reference: np.ndarray,
    final_frame: np.ndarray,
    final_alpha: np.ndarray,
    schedule: list[dict[str, object]],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    frames: list[np.ndarray] = []
    alphas: list[np.ndarray] = []
    for item in schedule:
        frame_index = int(item["frame"])
        dx = int(item["dx"])
        dy = int(item["dy"])
        opacity = float(item["opacity"])
        if frame_index == 0:
            frame = reference.copy()
            alpha = np.zeros_like(final_alpha)
        elif dx == 0 and dy == 0 and opacity == 1.0:
            frame = final_frame.copy()
            alpha = final_alpha.copy()
        else:
            foreground = translate(final_frame, dx, dy)
            alpha = translate(final_alpha, dx, dy)
            alpha = np.rint(alpha.astype(np.float32) * opacity).astype(np.uint8)
            frame = composite(foreground, reference, alpha)
        frames.append(frame)
        alphas.append(alpha)
    return frames, alphas


def outside_union_max_difference(
    frames: list[np.ndarray], reference: np.ndarray, motion_union: np.ndarray
) -> int:
    protected = motion_union == 0
    maximum = 0
    for frame in frames:
        difference = np.abs(frame.astype(np.int16) - reference.astype(np.int16))
        maximum = max(maximum, int(difference[protected].max(initial=0)))
    return maximum


def image_difference(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    difference = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {
        "rgb_mae": float(difference.mean()),
        "changed_pixel_fraction": float(np.any(difference != 0, axis=2).mean()),
        "max_channel_difference": int(difference.max(initial=0)),
    }


def write_video_new(path: Path, frames: list[np.ndarray], ffmpeg: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="foodstateedit_phase_frames_") as temp_dir:
        temp_root = Path(temp_dir)
        for index, frame in enumerate(frames):
            Image.fromarray(frame, mode="RGB").save(temp_root / f"{index:03d}.png")
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(FPS),
            "-i",
            str(temp_root / "%03d.png"),
            "-frames:v",
            str(FRAME_COUNT),
            "-c:v",
            "mpeg4",
            "-q:v",
            "2",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(path),
        ]
        subprocess.run(command, check=True)


def make_contact_sheet(frames: list[np.ndarray]) -> np.ndarray:
    selected = [frames[index] for index in CONTACT_SHEET_FRAMES]
    height, width = selected[0].shape[:2]
    sheet = np.zeros((height * 2, width * 3, 3), dtype=np.uint8)
    for index, frame in enumerate(selected):
        row, column = divmod(index, 3)
        sheet[row * height : (row + 1) * height, column * width : (column + 1) * width] = frame
    return sheet


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=repo_root / "artifacts" / "day9_action_pseudo_dataset_v3",
    )
    parser.add_argument(
        "--target-audit",
        type=Path,
        default=repo_root / "results" / "day9_action_target_audit_v1.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "artifacts" / DATASET_ID,
    )
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    target_audit_path = args.target_audit.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse output root: {output_root}")
    source_manifest_path = source_root / "dataset_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("sample_count") != 2 or source_manifest.get("frames") != FRAME_COUNT:
        raise ValueError("Expected the frozen two-sample, 21-frame Day 9 source dataset")
    if source_manifest.get("held_out", {}).get("training_occurrences") != 0:
        raise ValueError("Fork leakage detected in source dataset")

    schedule = phase_schedule()
    if [item["frame"] for item in schedule] != list(range(FRAME_COUNT)):
        raise AssertionError("Phase schedule must cover frames 0 through 20 exactly")
    output_root.mkdir(parents=True)
    phase_path = output_root / "phase_schedule.json"
    phase_path.write_text(json.dumps(schedule, indent=2) + "\n", encoding="utf-8")

    rows: list[dict[str, str]] = []
    samples: list[dict[str, object]] = []
    for source_sample in source_manifest["samples"]:
        sample_id = source_sample["sample_id"]
        source_files = source_sample["files"]
        for record in source_files.values():
            path = source_root / record["path"]
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise ValueError(f"Source file hash mismatch: {path}")

        reference = load_rgb(source_root / source_files["vace_reference_image"]["path"])
        final_target = load_rgb(source_root / source_files["target_keyframe"]["path"])
        final_control = load_rgb(source_root / source_files["control_keyframe"]["path"])
        final_alpha = load_gray(source_root / source_files["edit_alpha"]["path"])
        expected_shape = reference.shape[:2]
        if final_target.shape[:2] != expected_shape or final_control.shape[:2] != expected_shape:
            raise ValueError(f"Image shape mismatch for {sample_id}")
        if final_alpha.shape != expected_shape:
            raise ValueError(f"Alpha shape mismatch for {sample_id}")

        target_frames, target_alphas = synthesize_sequence(
            reference, final_target, final_alpha, schedule
        )
        control_frames, control_alphas = synthesize_sequence(
            reference, final_control, final_alpha, schedule
        )
        motion_union = np.maximum.reduce(target_alphas + control_alphas)
        target_outside = outside_union_max_difference(target_frames, reference, motion_union)
        control_outside = outside_union_max_difference(control_frames, reference, motion_union)
        if target_outside != 0 or control_outside != 0:
            raise AssertionError(
                f"Protected pixels changed for {sample_id}: target={target_outside}, control={control_outside}"
            )
        if not np.array_equal(target_frames[0], reference) or not np.array_equal(
            control_frames[0], reference
        ):
            raise AssertionError("Frame 0 must be the exact no-utensil source")
        if not np.array_equal(target_frames[-1], final_target) or not np.array_equal(
            control_frames[-1], final_control
        ):
            raise AssertionError("Final hold must equal the audited Day 9 keyframes")

        reference_path = output_root / "vace_reference_image" / f"{sample_id}.png"
        target_keyframe_path = output_root / "target_keyframe" / f"{sample_id}.png"
        control_keyframe_path = output_root / "control_keyframe" / f"{sample_id}.png"
        union_path = output_root / "edit_alpha" / f"{sample_id}.png"
        target_video_path = output_root / "video" / f"{sample_id}.mp4"
        control_video_path = output_root / "vace_video" / f"{sample_id}.mp4"
        target_sheet_path = output_root / "review" / f"{sample_id}_target_contact_sheet.png"
        control_sheet_path = output_root / "review" / f"{sample_id}_control_contact_sheet.png"
        save_png_new(reference_path, reference)
        save_png_new(target_keyframe_path, final_target)
        save_png_new(control_keyframe_path, final_control)
        save_png_new(union_path, motion_union)
        save_png_new(target_sheet_path, make_contact_sheet(target_frames))
        save_png_new(control_sheet_path, make_contact_sheet(control_frames))
        write_video_new(target_video_path, target_frames, args.ffmpeg)
        write_video_new(control_video_path, control_frames, args.ffmpeg)

        records = {
            "video": file_record(output_root, target_video_path),
            "vace_video": file_record(output_root, control_video_path),
            "vace_reference_image": file_record(output_root, reference_path),
            "target_keyframe": file_record(output_root, target_keyframe_path),
            "control_keyframe": file_record(output_root, control_keyframe_path),
            "edit_alpha": file_record(output_root, union_path),
            "target_contact_sheet": file_record(output_root, target_sheet_path),
            "control_contact_sheet": file_record(output_root, control_sheet_path),
        }
        if records["video"]["sha256"] == records["vace_video"]["sha256"]:
            raise AssertionError(f"Target/control videos collapsed for {sample_id}")
        rows.append(
            {
                "video": records["video"]["path"],
                "vace_video": records["vace_video"]["path"],
                "vace_reference_image": records["vace_reference_image"]["path"],
                "prompt": source_sample["prompt"],
            }
        )
        unique_target_frames = len({hashlib.sha256(frame.tobytes()).hexdigest() for frame in target_frames})
        unique_control_frames = len({hashlib.sha256(frame.tobytes()).hexdigest() for frame in control_frames})
        if unique_target_frames < 10 or unique_control_frames < 10:
            raise AssertionError("Phase construction produced too few unique frames")
        samples.append(
            {
                "sample_id": sample_id,
                "audit_candidate_id": source_sample["audit_candidate_id"],
                "family": source_sample["family"],
                "utensil": source_sample["utensil"],
                "action": source_sample["action"],
                "split": "train_overfit_sanity_only",
                "leakage_group": sample_id,
                "target_kind": "synthetic_imagegen_pseudotarget",
                "real_photo_supervision": False,
                "frame_policy": "deterministic_patch_motion_pseudo_sequence_approach_contact_lift_hold",
                "frame_count": FRAME_COUNT,
                "fps": FPS,
                "prompt": source_sample["prompt"],
                "hard_constraints": source_sample["hard_constraints"],
                "files": records,
                "source_day9_files": source_files,
                "nonidentity_checks": {
                    "target_video_hash_differs_from_control": True,
                    "target_outside_edit_support_max_difference": target_outside,
                    "control_outside_edit_support_max_difference": control_outside,
                    "edit_support_fraction": float((motion_union > 0).mean()),
                    "frame0_equals_reference": True,
                    "final_target_equals_day9_keyframe": True,
                    "final_control_equals_day9_keyframe": True,
                    "unique_target_frames": unique_target_frames,
                    "unique_control_frames": unique_control_frames,
                    "final_target_vs_control": image_difference(final_target, final_control),
                },
            }
        )

    metadata_path = output_root / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["video", "vace_video", "vace_reference_image", "prompt"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    builder_path = Path(__file__).resolve()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "scientific_status": SCIENTIFIC_STATUS,
        "experiment_variant": EXPERIMENT_VARIANT,
        "claim_limit": (
            "Two deterministic pseudo-motion sequences test overfit capacity only. "
            "They are not real motion or real-data training and cannot establish generalization or photo realism."
        ),
        "target_policy": TARGET_POLICY,
        "motion_policy": "translated_audited_final_patch_pseudo_motion_not_physical_ground_truth",
        "held_out": source_manifest["held_out"],
        "sample_count": len(samples),
        "frames": FRAME_COUNT,
        "fps": FPS,
        "phase_schedule": file_record(output_root, phase_path),
        "metadata": {
            **file_record(output_root, metadata_path),
            "columns": ["video", "vace_video", "vace_reference_image", "prompt"],
        },
        "samples": samples,
        "provenance": {
            "builder": builder_path.name,
            "builder_sha256": sha256_text_lf(builder_path),
            "builder_hash_policy": "sha256_lf_normalized",
            "source_dataset_manifest": source_manifest_path.name,
            "source_dataset_manifest_sha256": sha256_file(source_manifest_path),
            "target_audit": target_audit_path.name,
            "target_audit_sha256": sha256_file(target_audit_path),
            "ffmpeg": args.ffmpeg,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    manifest_path = output_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
