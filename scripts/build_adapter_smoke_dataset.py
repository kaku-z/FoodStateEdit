#!/usr/bin/env python3
"""Materialize the two-case VACE-LoRA plumbing dataset without learned models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = "foodstateedit.adapter_dataset.v1"
DATASET_ID = "day7_adapter_v0_smoke_dataset_v2"
SOURCE_METHOD = "vace_direct_dynamic_multikey"
TARGET_POLICY = "deterministic_proxy_identity_plumbing_only"
CASES = {
    "ramen_chopsticks_001": {
        "family": "strand",
        "utensil": "chopsticks",
        "action": "lift_strand",
        "constraints": [
            "exactly_two_chopsticks",
            "continuous_lifted_strand",
            "strand_returns_to_bowl",
            "rear_chopstick_strand_front_chopstick_z_order",
        ],
    },
    "pasta_fork_001": {
        "family": "strand_contact",
        "utensil": "fork",
        "action": "twirl_and_lift",
        "constraints": [
            "single_fork_with_four_tines",
            "continuous_twirl",
            "payload_contacts_tines",
            "strand_returns_to_plate",
        ],
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text_lf(path: Path) -> str:
    """Hash source text after CRLF normalization for cross-platform provenance."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def file_record(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_source_file(
    case_root: Path,
    manifest: dict[str, object],
    name: str,
    normalize_text_newlines: bool = False,
) -> Path:
    path = case_root / name
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = manifest["files"][name]["sha256"]
    if normalize_text_newlines:
        normalized = path.read_bytes().replace(b"\r\n", b"\n")
        actual = hashlib.sha256(normalized).hexdigest()
    else:
        actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Hash mismatch for {path}: {actual} != {expected}")
    return path


def copy_new(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proxy-root",
        type=Path,
        default=root / "artifacts" / "day6_dynamic_multikey_proxy_v1",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=root / "results" / "day6_dynamic_multikey_proxy_v1",
    )
    parser.add_argument(
        "--prompt-root",
        type=Path,
        default=root / "outputs" / "day6_dynamic_multikey_proxy_local_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "artifacts" / "day7_adapter_v0_smoke_dataset_v2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    proxy_root = args.proxy_root.resolve()
    manifest_root = args.manifest_root.resolve()
    prompt_root = args.prompt_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse output root: {output_root}")
    if not proxy_root.is_dir():
        raise FileNotFoundError(proxy_root)
    if not manifest_root.is_dir():
        raise FileNotFoundError(manifest_root)
    if not prompt_root.is_dir():
        raise FileNotFoundError(prompt_root)

    prepared: list[dict[str, object]] = []
    for case_id, case_spec in CASES.items():
        source_root = proxy_root / case_id
        source_manifest_path = manifest_root / case_id / "proxy_manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest["anchor_id"] != case_id:
            raise ValueError(f"Anchor mismatch in {source_manifest_path}")
        if source_manifest["method"] != SOURCE_METHOD:
            raise ValueError(f"Unexpected source method in {source_manifest_path}")
        if source_manifest["frame_count"] != 21:
            raise ValueError(f"Expected 21 frames in {source_manifest_path}")
        prepared.append(
            {
                "case_id": case_id,
                "case_spec": case_spec,
                "source_manifest_path": source_manifest_path,
                "source_manifest": source_manifest,
                "control_source": verify_source_file(source_root, source_manifest, "dynamic_control.mp4"),
                "reference_source": verify_source_file(source_root, source_manifest, "first_frame.png"),
                "prompt_source": verify_source_file(
                    prompt_root / case_id,
                    source_manifest,
                    "prompt.txt",
                    normalize_text_newlines=True,
                ),
            }
        )

    output_root.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    samples: list[dict[str, object]] = []
    for item in prepared:
        case_id = item["case_id"]
        case_spec = item["case_spec"]
        source_manifest_path = item["source_manifest_path"]
        source_manifest = item["source_manifest"]
        control_source = item["control_source"]
        reference_source = item["reference_source"]
        prompt_source = item["prompt_source"]
        prompt = prompt_source.read_text(encoding="utf-8").strip()
        if not prompt:
            raise ValueError(f"Empty prompt: {prompt_source}")

        target_path = output_root / "video" / f"{case_id}.mp4"
        control_path = output_root / "vace_video" / f"{case_id}.mp4"
        reference_path = output_root / "vace_reference_image" / f"{case_id}.png"
        copy_new(control_source, target_path)
        copy_new(control_source, control_path)
        copy_new(reference_source, reference_path)

        target_record = file_record(output_root, target_path)
        control_record = file_record(output_root, control_path)
        if target_record["sha256"] != control_record["sha256"]:
            raise AssertionError("Smoke target and control must be byte-identical")
        reference_record = file_record(output_root, reference_path)
        rows.append(
            {
                "video": target_record["path"],
                "vace_video": control_record["path"],
                "vace_reference_image": reference_record["path"],
                "prompt": prompt,
            }
        )
        samples.append(
            {
                "sample_id": f"{case_id}_proxy_identity_smoke",
                "source_case_id": case_id,
                "family": case_spec["family"],
                "utensil": case_spec["utensil"],
                "action": case_spec["action"],
                "split": "smoke",
                "leakage_group": case_id,
                "target_kind": "deterministic_proxy",
                "frame_count": source_manifest["frame_count"],
                "prompt": prompt,
                "license": "UECFOOD256_noncommercial_research_only_plus_deterministic_geometry",
                "files": {
                    "video": target_record,
                    "vace_video": control_record,
                    "vace_reference_image": reference_record,
                },
                "hard_constraints": case_spec["constraints"],
                "source_proxy_manifest_sha256": sha256_file(source_manifest_path),
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
        "scientific_status": "infrastructure_smoke_not_quality_evidence",
        "target_policy": TARGET_POLICY,
        "sample_count": len(samples),
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
            "source_method": SOURCE_METHOD,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    manifest_path = output_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
