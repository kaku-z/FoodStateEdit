#!/usr/bin/env python3
"""Audit the frozen 40-image held-out input set without opening test images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_FAMILIES = {
    "liquid": 10,
    "granular": 10,
    "strand": 10,
    "strand_contact": 10,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_heldout_inventory(
    data_manifest: Path, canonical_manifest: Path, freeze_record: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    freeze = json.loads(freeze_record.read_text(encoding="utf-8"))
    actual_manifest_hash = sha256_file(data_manifest)
    if actual_manifest_hash != freeze["frozen_manifest_sha256"]:
        raise ValueError("Frozen benchmark manifest hash mismatch")

    data_rows = [row for row in read_csv(data_manifest) if row["split"] == "test"]
    canonical_rows = {row["case_id"]: row for row in read_csv(canonical_manifest)}
    if len(data_rows) != 40:
        raise ValueError(f"Expected 40 test cases, found {len(data_rows)}")
    family_counts = Counter(row["family"] for row in data_rows)
    if family_counts != Counter(EXPECTED_FAMILIES):
        raise ValueError(f"Unexpected family balance: {family_counts}")

    inventory: list[dict[str, Any]] = []
    for row in sorted(data_rows, key=lambda item: (item["family"], item["case_id"])):
        canonical = canonical_rows.get(row["case_id"])
        if canonical is None or canonical["split"] != "test" or canonical["family"] != row["family"]:
            raise ValueError(f"Canonical manifest mismatch for {row['case_id']}")
        if row["source_kind"] != "real_dataset":
            raise ValueError(f"Held-out source is not real data: {row['case_id']}")
        if row["has_target_utensil"] != "false" or row["freeze_status"] != "frozen":
            raise ValueError(f"Frozen eligibility mismatch: {row['case_id']}")
        if not canonical["validation_status"].startswith("validated_x4_aligned"):
            raise ValueError(f"Canonical input not validated: {row['case_id']}")
        inventory.append(
            {
                "case_id": row["case_id"],
                "family": row["family"],
                "dish": row["dish"],
                "utensil": row["utensil"],
                "action": row["action"],
                "annotation_status": row["annotation_status"],
                "canonical_input_path": canonical["canonical_input_path"],
                "canonical_input_sha256": canonical["canonical_input_sha256"],
                "canonical_width": int(canonical["canonical_width"]),
                "canonical_height": int(canonical["canonical_height"]),
            }
        )
    if len({item["canonical_input_sha256"] for item in inventory}) != 40:
        raise ValueError("Held-out canonical hashes are not unique")
    return inventory, freeze


def audit_remote_hashes(host: str, inventory: list[dict[str, Any]]) -> dict[str, str]:
    paths = [item["canonical_input_path"] for item in inventory]
    completed = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=45", host, "sha256sum", "--", *paths],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Remote hash audit failed: {completed.stderr.strip()}")
    hashes: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        digest, separator, path = line.partition("  ")
        if not separator:
            raise ValueError(f"Unexpected sha256sum output: {line}")
        hashes[path] = digest
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--host", default="gp40")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite readiness audit: {output}")

    data_manifest = repo / "benchmark/data_manifest_v1.csv"
    canonical_manifest = repo / "benchmark/canonical_input_manifest_v1.csv"
    freeze_record = repo / "benchmark/freeze_record_v1.json"
    protocol = repo / "benchmark/FOODSTATEEDIT_BLIND_EVALUATION_PROTOCOL_V1.md"
    inventory, freeze = load_heldout_inventory(data_manifest, canonical_manifest, freeze_record)
    remote_hashes = audit_remote_hashes(args.host, inventory)
    mismatches = [
        item["case_id"]
        for item in inventory
        if remote_hashes.get(item["canonical_input_path"]) != item["canonical_input_sha256"]
    ]
    annotation_counts = Counter(item["annotation_status"] for item in inventory)
    input_gate = len(remote_hashes) == 40 and not mismatches
    annotations_ready = annotation_counts == Counter({"complete": 40})

    payload = {
        "schema_version": "foodstateedit.day30_heldout_readiness.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "heldout_input_integrity_audit_only_no_test_generation",
        "remote_host": args.host,
        "inputs": {
            "freeze_record": {"path": "benchmark/freeze_record_v1.json", "sha256": sha256_file(freeze_record)},
            "data_manifest": {"path": "benchmark/data_manifest_v1.csv", "sha256": sha256_file(data_manifest)},
            "canonical_manifest": {"path": "benchmark/canonical_input_manifest_v1.csv", "sha256": sha256_file(canonical_manifest)},
            "evaluation_protocol": {"path": "benchmark/FOODSTATEEDIT_BLIND_EVALUATION_PROTOCOL_V1.md", "sha256": sha256_file(protocol)},
        },
        "freeze": freeze,
        "case_count": len(inventory),
        "family_counts": dict(sorted(Counter(item["family"] for item in inventory).items())),
        "annotation_counts": dict(sorted(annotation_counts.items())),
        "remote_hash_count": len(remote_hashes),
        "remote_hash_mismatches": mismatches,
        "gates": {
            "benchmark_frozen_and_balanced": True,
            "canonical_inputs_present_and_hash_matched": input_gate,
            "annotations_complete": annotations_ready,
            "day25_development_ablation_complete_and_reviewed": False,
            "material_level_ours_rule_frozen_before_test_generation": False,
            "heldout_generation_allowed": False,
        },
        "cases": inventory,
        "next_required": [
            "Finish and review all four Day 25 development cases without inspecting held-out outputs.",
            "Freeze one material-family rule and rollback contract before any held-out generation.",
            "Complete versioned annotations/controls for all 40 frozen test cases.",
            "Freeze same-input method configs, seeds 1/2/3, frame policy and endpoint code.",
        ],
        "claim_limit": "This audit establishes only frozen held-out input availability and byte identity. It contains no held-out model output and supports no effectiveness, superiority, realism or generalization claim.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": sha256_file(output), "input_gate": input_gate, "heldout_generation_allowed": False}, indent=2))
    return 0 if input_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
