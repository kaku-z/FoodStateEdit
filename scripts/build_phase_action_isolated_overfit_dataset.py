#!/usr/bin/env python3
"""Build one-sample duplicated-row datasets for an isolated overfit diagnostic.

The selected Day 11 sample is copied byte for byte and written twice in the
training metadata. No target, control, prompt, phase schedule, or support mask
is changed. Step 32 therefore represents 32 dedicated exposures to that sample,
matching its approximate exposure count in the 64-step two-sample shared run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


SCHEMA_VERSION = "foodstateedit.phase_action_isolated_overfit_dataset.v1"
SOURCE_MANIFEST_SHA256 = "b7458dcdd5ec84bb44e8e67212c83680c90b97256a389d755d7c828f29e2bcde"
TARGET_AUDIT_SHA256 = "804425172ef765d38af5f943c2fde5c551564a561dd984dccb3eb8a136aa70ea"
ALLOWED_SAMPLE_IDS = {
    "udon_chopsticks_imagegen_pseudo_v1",
    "clear_broth_spoon_imagegen_pseudo_v1",
}
TRAINING_ROW_COUNT = 2
UNIQUE_SAMPLE_COUNT = 1


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


def copy_record(source_root: Path, output_root: Path, record: dict[str, object]) -> None:
    source = source_root / str(record["path"])
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.stat().st_size != record["size_bytes"] or sha256_file(source) != record["sha256"]:
        raise ValueError(f"Source record mismatch: {source}")
    target = output_root / str(record["path"])
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if target.stat().st_size != record["size_bytes"] or sha256_file(target) != record["sha256"]:
        raise ValueError(f"Copied record mismatch: {target}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=repo_root / "artifacts" / "day11_phase_action_overfit_dataset_v1",
    )
    parser.add_argument("--sample-id", choices=sorted(ALLOWED_SAMPLE_IDS), required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse output root: {output_root}")
    source_manifest_path = source_root / "dataset_manifest.json"
    if sha256_file(source_manifest_path) != SOURCE_MANIFEST_SHA256:
        raise ValueError("Frozen Day 11 source manifest hash mismatch")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("sample_count") != 2:
        raise ValueError("Expected the frozen two-sample Day 11 dataset")
    if source_manifest.get("held_out", {}).get("training_occurrences") != 0:
        raise ValueError("Held-out fork leakage detected")
    source_samples = {
        sample["sample_id"]: sample for sample in source_manifest.get("samples", [])
    }
    selected = source_samples.get(args.sample_id)
    if selected is None:
        raise ValueError(f"Sample is not in the frozen manifest: {args.sample_id}")

    output_root.mkdir(parents=True)
    phase_record = source_manifest["phase_schedule"]
    copy_record(source_root, output_root, phase_record)
    for record in selected["files"].values():
        copy_record(source_root, output_root, record)

    metadata_path = output_root / "metadata.csv"
    metadata_row = {
        "video": selected["files"]["video"]["path"],
        "vace_video": selected["files"]["vace_video"]["path"],
        "vace_reference_image": selected["files"]["vace_reference_image"]["path"],
        "prompt": selected["prompt"],
    }
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["video", "vace_video", "vace_reference_image", "prompt"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(metadata_row)
        writer.writerow(metadata_row)

    duplicated_samples = []
    for row_index in range(TRAINING_ROW_COUNT):
        record = json.loads(json.dumps(selected))
        record["training_row_index"] = row_index
        record["isolation_policy"] = "same_sample_duplicated_metadata_row"
        duplicated_samples.append(record)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": args.dataset_id,
        "scientific_status": "mechanism_pilot_with_synthetic_pseudotargets",
        "experiment_variant": "phase_varying_overfit_sanity_v1",
        "isolation_variant": "single_unique_sample_duplicated_to_two_training_rows_v1",
        "claim_limit": (
            "This duplicated-row dataset isolates one seen synthetic pseudo-motion sample. "
            "It cannot establish generalization, real-data performance, physical validity, "
            "or paper-level photo realism."
        ),
        "target_policy": source_manifest["target_policy"],
        "motion_policy": source_manifest["motion_policy"],
        "held_out": source_manifest["held_out"],
        "sample_count": TRAINING_ROW_COUNT,
        "unique_sample_count": UNIQUE_SAMPLE_COUNT,
        "selected_sample_id": args.sample_id,
        "frames": source_manifest["frames"],
        "fps": source_manifest["fps"],
        "phase_schedule": file_record(
            output_root, output_root / str(phase_record["path"])
        ),
        "metadata": {
            **file_record(output_root, metadata_path),
            "columns": ["video", "vace_video", "vace_reference_image", "prompt"],
            "row_count": TRAINING_ROW_COUNT,
            "rows_are_byte_identical": True,
        },
        "samples": duplicated_samples,
        "matched_exposure_comparison": {
            "dedicated_checkpoint_step": 32,
            "dedicated_selected_sample_exposures": 32,
            "shared_checkpoint_step": 64,
            "shared_expected_selected_sample_exposures": 32,
            "interpretation": (
                "Compare dedicated step 32 with shared step 64 for matched per-sample "
                "exposure; dedicated step 64 is an additional overfit-capacity probe."
            ),
        },
        "provenance": {
            "builder": Path(__file__).name,
            "builder_sha256": sha256_lf(Path(__file__).resolve()),
            "builder_hash_policy": "sha256_lf_normalized",
            "source_dataset_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "target_audit": "day9_action_target_audit_v1.json",
            "target_audit_sha256": TARGET_AUDIT_SHA256,
            "copy_policy": "selected_sample_files_and_phase_schedule_byte_exact",
        },
    }
    manifest_path = output_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "manifest_sha256": sha256_file(manifest_path),
                "metadata_sha256": sha256_file(metadata_path),
                "selected_sample_id": args.sample_id,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
