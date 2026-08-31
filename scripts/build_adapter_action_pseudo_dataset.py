#!/usr/bin/env python3
"""Build a hash-locked two-pseudo-target VACE-LoRA mechanism dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


SCHEMA_VERSION = "foodstateedit.adapter_action_dataset.v1"
DATASET_ID = "day9_action_pseudo_dataset_v3"
SCIENTIFIC_STATUS = "mechanism_pilot_with_synthetic_pseudotargets"
TARGET_POLICY = "disclosed_imagegen_pseudotarget_nonidentity"
FRAME_COUNT = 21
FPS = 8.0


SAMPLES = (
    {
        "sample_id": "udon_chopsticks_imagegen_pseudo_v1",
        "audit_candidate_id": "udon_chopsticks_imagegen_oracle",
        "family": "strand",
        "utensil": "chopsticks",
        "action": "lift_single_bowl_connected_strand",
        "canvas": (688, 576),
        "reference": "single_patch_natural_v12/uec256_noodle_20_1897_insert/00_source.png",
        "reference_sha256": "c7f98ac69df93b1052c9e90e0907c6bf980bd8f0ea5683060a176c73636f8b75",
        "control": "single_patch_natural_v12/uec256_noodle_20_1897_insert/04_training_free_result.png",
        "control_sha256": "8985c1d4fee827abeab096058c12a6b570f28af6c60ebb0d9f7a3d70b1f9c132",
        "target": "single_patch_natural_v12/uec256_noodle_20_1897_insert/vace_naturalref_seed20260808_steps40_ctx1p25/src_ref_image_0.png",
        "target_sha256": "10ddd3fb7b36dc64bd4ede420f5295a11457f98a559f24517083f46547a46c78",
        "alpha_sources": [
            {
                "path": "single_patch_natural_v12/uec256_noodle_20_1897_insert/05_target_rods_mask.png",
                "sha256": "88981fc613f722859cdb75fd4c174c508c32d8a14d90498ba335d3fef856eec4",
            },
            {
                "path": "single_patch_natural_v12/uec256_noodle_20_1897_insert/06_target_noodle_mask.png",
                "sha256": "4e6edcef1256f0248e1ade7435c1608ce2dab7fe7445da78a087880bfdfa5efe",
            },
            {
                "path": "single_patch_natural_v12/uec256_noodle_20_1897_insert/07_source_only_mask.png",
                "sha256": "fff7317930bfee9c57165b62197e9505ee8e518939f8f324650939cc5b00ac86",
            },
        ],
        "alpha_dilate_px": 5,
        "control_style_masks": [],
        "prompt": (
            "A photorealistic close-up food photograph of the same udon bowl. "
            "Exactly two separate wooden chopsticks pinch and lift one continuous noodle strand; "
            "the strand remains visibly connected to the noodles in the bowl, with physical contact, "
            "occlusion, gloss, and a plausible local shadow. No hand and no extra utensil."
        ),
        "hard_constraints": [
            "exactly_two_separate_chopsticks",
            "one_continuous_lifted_strand",
            "strand_visibly_returns_to_bowl",
            "physical_pinch_contact_and_occlusion",
        ],
    },
    {
        "sample_id": "clear_broth_spoon_imagegen_pseudo_v1",
        "audit_candidate_id": "clear_broth_spoon_imagegen_oracle",
        "family": "liquid",
        "utensil": "spoon",
        "action": "scoop_contained_broth",
        "canvas": (736, 592),
        "reference": "outputs/foodstateedit_spoon_scoop_v0_3_staged/clear_broth_spoon_scoop/case_736x592/first_frame.png",
        "reference_sha256": "4faf3dd94bf3f7cb5978dfd89fdc27a41b0b2765e03a670293baa73c0c984486",
        "control": "outputs/foodstateedit_spoon_scoop_v0_3_staged/clear_broth_spoon_scoop/case_736x592/motion_signal.png",
        "control_sha256": "9520353481d71b9d7dab092b977ed83730833c760e6c2626d1ca063e2f71434c",
        "target": "outputs/foodstateedit_spoon_scoop_v0_3_staged/clear_broth_spoon_scoop/case_736x592/oracle_aligned.png",
        "target_sha256": "deb912f8548c8c01310f0c0b7c9ca7bb49071afb34ac6e7e9a7c42c3f44d6204",
        "alpha_sources": [
            {
                "path": "outputs/foodstateedit_spoon_scoop_v0_3_staged/clear_broth_spoon_scoop/case_736x592/edit_alpha.png",
                "sha256": "63ed4e3941da44035d5b39cd5f949edc94a43cca75f5a58541ef02868bc4596e",
            },
        ],
        "alpha_dilate_px": 0,
        "control_style_masks": [
            {
                "role": "rigid_utensil",
                "path": "outputs/foodstateedit_spoon_scoop_v0_3_staged/clear_broth_spoon_scoop/case_736x592/mask_utensil.png",
                "sha256": "e3ed8181d722563b50c86153b1baf881a973366bff5a8b861b16475b7090623a",
                "bgr": [185, 185, 185],
                "opacity": 0.82,
            },
            {
                "role": "contained_liquid",
                "path": "outputs/foodstateedit_spoon_scoop_v0_3_staged/clear_broth_spoon_scoop/case_736x592/mask_spoon_payload.png",
                "sha256": "a3e64bd9322df525f70f113c863598bb68e42354276d4bce64d8b01d2beae4e3",
                "bgr": [45, 155, 225],
                "opacity": 0.72,
            },
        ],
        "prompt": (
            "A photorealistic close-up food photograph of the same bowl of clear golden broth. "
            "Exactly one connected stainless-steel spoon scoops and contains visible broth and one "
            "scallion ring; two scallion rings remain in the bowl. Physical rim occlusion, metal "
            "reflection, and contact shadow. No hand and no other utensil."
        ),
        "hard_constraints": [
            "exactly_one_connected_spoon",
            "visible_broth_contained_by_spoon_bowl",
            "one_garnish_ring_in_spoon_two_in_bowl",
            "physical_rim_occlusion_and_local_shadow",
        ],
    },
)


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


def verify_source(experiment_root: Path, relative_path: str, expected_hash: str) -> Path:
    path = (experiment_root / relative_path).resolve()
    if experiment_root != path and experiment_root not in path.parents:
        raise ValueError(f"Source escapes experiment root: {relative_path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(f"Hash mismatch for {path}: {actual_hash} != {expected_hash}")
    return path


def load_resized(path: Path, canvas: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode {path}")
    width, height = canvas
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
    return image


def load_mask_resized(path: Path, canvas: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not decode {path}")
    width, height = canvas
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask


def composite(foreground: np.ndarray, background: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    weight = alpha.astype(np.float32)[..., None] / 255.0
    result = foreground.astype(np.float32) * weight + background.astype(np.float32) * (1.0 - weight)
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def outside_support_max_difference(image: np.ndarray, reference: np.ndarray, alpha: np.ndarray) -> int:
    protected = alpha == 0
    difference = np.abs(image.astype(np.int16) - reference.astype(np.int16))
    return int(difference[protected].max(initial=0))


def write_png_new(path: Path, image: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write {path}")


def write_static_video_new(path: Path, frame: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    try:
        for _ in range(FRAME_COUNT):
            writer.write(frame)
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(path))
    decoded = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded_fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if decoded != FRAME_COUNT or decoded_fps <= 0:
        raise RuntimeError(f"Video verification failed for {path}: frames={decoded}, fps={decoded_fps}")


def image_difference(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    difference = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {
        "rgb_mae": float(difference.mean()),
        "changed_pixel_fraction": float(np.any(difference != 0, axis=2).mean()),
        "max_channel_difference": float(difference.max(initial=0)),
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=repo_root.parent)
    parser.add_argument(
        "--audit",
        type=Path,
        default=repo_root / "results" / "day9_action_target_audit_v1.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "artifacts" / DATASET_ID,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_root = args.experiment_root.resolve()
    audit_path = args.audit.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse output root: {output_root}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audited = {
        item["candidate_id"]: item
        for item in audit["candidates"]
        if item["eligible_for_primary_pilot_training"]
    }
    expected_ids = {sample["audit_candidate_id"] for sample in SAMPLES}
    if set(audited) != expected_ids:
        raise ValueError(f"Eligible audit candidates changed: {sorted(audited)} != {sorted(expected_ids)}")
    if audit["decision"]["real_photo_target_count"] != 0:
        raise ValueError("This pilot must not imply real-photo action supervision")
    if audit["eligibility_policy"]["held_out_family"] != "fork_twirl_and_lift":
        raise ValueError("Fork must remain the held-out family")

    prepared: list[dict[str, object]] = []
    for spec in SAMPLES:
        target_audit = audited[spec["audit_candidate_id"]]
        if target_audit["classification"] != "eligible_synthetic_pseudotarget":
            raise ValueError(f"Unexpected target class for {spec['sample_id']}")
        if target_audit["sha256"] != spec["target_sha256"]:
            raise ValueError(f"Audit target hash changed for {spec['sample_id']}")
        prepared.append(
            {
                "spec": spec,
                "reference_source": verify_source(experiment_root, spec["reference"], spec["reference_sha256"]),
                "control_source": verify_source(experiment_root, spec["control"], spec["control_sha256"]),
                "target_source": verify_source(experiment_root, spec["target"], spec["target_sha256"]),
                "alpha_sources": [
                    verify_source(experiment_root, record["path"], record["sha256"])
                    for record in spec["alpha_sources"]
                ],
                "control_style_masks": [
                    {
                        "spec": record,
                        "source": verify_source(experiment_root, record["path"], record["sha256"]),
                    }
                    for record in spec["control_style_masks"]
                ],
            }
        )

    output_root.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    samples: list[dict[str, object]] = []
    for item in prepared:
        spec = item["spec"]
        sample_id = spec["sample_id"]
        reference = load_resized(item["reference_source"], spec["canvas"])
        control_raw = load_resized(item["control_source"], spec["canvas"])
        target_raw = load_resized(item["target_source"], spec["canvas"])
        alpha_binary = np.zeros(reference.shape[:2], dtype=np.uint8)
        for alpha_source in item["alpha_sources"]:
            alpha_binary = np.maximum(alpha_binary, load_mask_resized(alpha_source, spec["canvas"]))
        alpha_binary = np.where(alpha_binary > 0, 255, 0).astype(np.uint8)
        dilate_px = int(spec["alpha_dilate_px"])
        if dilate_px > 0:
            kernel_size = 2 * dilate_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            alpha_binary = cv2.dilate(alpha_binary, kernel)
        alpha = cv2.GaussianBlur(alpha_binary, (9, 9), 0)
        alpha[alpha < 2] = 0
        alpha[alpha > 253] = 255
        target = composite(target_raw, reference, alpha)
        control = composite(control_raw, reference, alpha)
        for style in item["control_style_masks"]:
            style_spec = style["spec"]
            style_mask = load_mask_resized(style["source"], spec["canvas"])
            style_mask = np.minimum(style_mask, alpha)
            opacity = float(style_spec["opacity"])
            style_alpha = np.rint(style_mask.astype(np.float32) * opacity).astype(np.uint8)
            color = np.empty_like(control)
            color[:] = np.asarray(style_spec["bgr"], dtype=np.uint8)
            control = composite(color, control, style_alpha)

        target_outside_max = outside_support_max_difference(target, reference, alpha)
        control_outside_max = outside_support_max_difference(control, reference, alpha)
        if target_outside_max != 0 or control_outside_max != 0:
            raise AssertionError(
                f"Protected pixels changed for {sample_id}: target={target_outside_max}, control={control_outside_max}"
            )

        reference_path = output_root / "vace_reference_image" / f"{sample_id}.png"
        control_keyframe_path = output_root / "control_keyframe" / f"{sample_id}.png"
        target_keyframe_path = output_root / "target_keyframe" / f"{sample_id}.png"
        alpha_path = output_root / "edit_alpha" / f"{sample_id}.png"
        control_video_path = output_root / "vace_video" / f"{sample_id}.mp4"
        target_video_path = output_root / "video" / f"{sample_id}.mp4"
        write_png_new(reference_path, reference)
        write_png_new(control_keyframe_path, control)
        write_png_new(target_keyframe_path, target)
        write_png_new(alpha_path, alpha)
        write_static_video_new(control_video_path, control)
        write_static_video_new(target_video_path, target)

        records = {
            "video": file_record(output_root, target_video_path),
            "vace_video": file_record(output_root, control_video_path),
            "vace_reference_image": file_record(output_root, reference_path),
            "target_keyframe": file_record(output_root, target_keyframe_path),
            "control_keyframe": file_record(output_root, control_keyframe_path),
            "edit_alpha": file_record(output_root, alpha_path),
        }
        if records["video"]["sha256"] == records["vace_video"]["sha256"]:
            raise AssertionError(f"Non-identity target collapsed to its control for {sample_id}")
        rows.append(
            {
                "video": records["video"]["path"],
                "vace_video": records["vace_video"]["path"],
                "vace_reference_image": records["vace_reference_image"]["path"],
                "prompt": spec["prompt"],
            }
        )
        samples.append(
            {
                "sample_id": sample_id,
                "audit_candidate_id": spec["audit_candidate_id"],
                "family": spec["family"],
                "utensil": spec["utensil"],
                "action": spec["action"],
                "split": "train_mechanism_pilot",
                "leakage_group": sample_id,
                "target_kind": "synthetic_imagegen_pseudotarget",
                "real_photo_supervision": False,
                "frame_policy": "static_keyframe_repeated_21_frames",
                "frame_count": FRAME_COUNT,
                "fps": FPS,
                "canvas": {"width": spec["canvas"][0], "height": spec["canvas"][1]},
                "prompt": spec["prompt"],
                "hard_constraints": spec["hard_constraints"],
                "source_files": {
                    "reference": {"path": spec["reference"], "sha256": spec["reference_sha256"]},
                    "control": {"path": spec["control"], "sha256": spec["control_sha256"]},
                    "target": {"path": spec["target"], "sha256": spec["target_sha256"]},
                    "alpha_sources": spec["alpha_sources"],
                    "control_style_masks": spec["control_style_masks"],
                },
                "files": records,
                "nonidentity_checks": {
                    "target_vs_control": image_difference(target, control),
                    "target_vs_reference": image_difference(target, reference),
                    "target_video_hash_differs_from_control": True,
                    "target_outside_edit_support_max_difference": target_outside_max,
                    "control_outside_edit_support_max_difference": control_outside_max,
                    "edit_support_fraction": float((alpha > 0).mean()),
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
        "claim_limit": (
            "Two disclosed ImageGen pseudo-targets can test non-identity adapter behavior only; "
            "they are not real training data and cannot establish generalization or photo realism."
        ),
        "target_policy": TARGET_POLICY,
        "held_out": {
            "family": "fork_twirl_and_lift",
            "anchor": "pasta_fork_001",
            "training_occurrences": 0,
        },
        "sample_count": len(samples),
        "frames": FRAME_COUNT,
        "metadata": {
            "path": metadata_path.relative_to(output_root).as_posix(),
            "sha256": sha256_file(metadata_path),
            "columns": ["video", "vace_video", "vace_reference_image", "prompt"],
        },
        "samples": samples,
        "provenance": {
            "builder": builder_path.name,
            "builder_sha256": sha256_text_lf(builder_path),
            "builder_hash_policy": "sha256_lf_normalized",
            "target_audit": audit_path.name,
            "target_audit_sha256": sha256_file(audit_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    manifest_path = output_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
