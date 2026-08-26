#!/usr/bin/env python3
"""Apply a complete visual review and create a deterministic 5/10 split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


LICENSE = "UECFOOD256_noncommercial_research_only"
REVIEWED_FIELDS = (
    "candidate_id",
    "family",
    "dish",
    "utensil",
    "action",
    "dataset",
    "class_id",
    "class_name",
    "dataset_relative_image",
    "source_path",
    "source_sha256",
    "width",
    "height",
    "format",
    "dhash64",
    "exact_duplicate_group",
    "near_duplicate_review",
    "license_status",
    "pilot_only",
    "has_target_utensil",
    "has_target_action",
    "food_visible",
    "container_visible",
    "edit_space",
    "manual_status",
    "exclusion_reason",
    "freeze_status",
    "split",
)
MANIFEST_FIELDS = (
    "case_id",
    "family",
    "dish",
    "utensil",
    "action",
    "split",
    "source_kind",
    "source_path",
    "source_sha256",
    "license",
    "has_target_utensil",
    "annotation_status",
    "freeze_status",
)


def stable_rank(seed: int, row: dict[str, str]) -> str:
    payload = f"{seed}:{row['candidate_id']}:{row['source_sha256']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--reviewed-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260826)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = read_csv(args.inventory)
    review = json.loads(args.review.read_text(encoding="utf-8"))
    reviewer = review["reviewer"]
    by_id = {row["candidate_id"]: row for row in inventory}
    if len(by_id) != len(inventory):
        raise ValueError("Candidate IDs are not unique")

    decisions: dict[str, tuple[str, str]] = {}
    for family, family_review in review["families"].items():
        for candidate_id in family_review["eligible"]:
            decisions[candidate_id] = ("eligible", "")
        for candidate_id, reason in family_review["excluded"].items():
            if candidate_id in decisions:
                raise ValueError(f"Duplicate review decision: {candidate_id}")
            decisions[candidate_id] = ("excluded", reason)
        family_inventory = {row["candidate_id"] for row in inventory if row["family"] == family}
        family_decisions = {
            candidate_id for candidate_id in decisions if by_id.get(candidate_id, {}).get("family") == family
        }
        if family_inventory != family_decisions:
            missing = sorted(family_inventory - family_decisions)
            extra = sorted(family_decisions - family_inventory)
            raise ValueError(f"Incomplete review for {family}: missing={missing}, extra={extra}")

    if set(by_id) != set(decisions):
        raise ValueError("Review does not cover the entire inventory")

    reviewed_rows: list[dict[str, str]] = []
    for original in inventory:
        row = dict(original)
        decision, reason = decisions[row["candidate_id"]]
        row["license_status"] = LICENSE
        row["exclusion_reason"] = reason
        row["freeze_status"] = "candidate" if decision == "eligible" else "excluded"
        row["split"] = "pilot" if row["pilot_only"] == "true" else "unassigned"
        if decision == "eligible":
            row.update(
                has_target_utensil="false",
                has_target_action="false",
                food_visible="true",
                container_visible="true",
                edit_space="true",
                manual_status=f"eligible:{reviewer}",
            )
        else:
            row["has_target_utensil"] = "true" if reason == "target_utensil_visible" else "unknown"
            row["has_target_action"] = "unknown"
            row["food_visible"] = "false" if reason.startswith("food_not_dominant") else "unknown"
            row["container_visible"] = "unknown"
            row["edit_space"] = "unknown"
            row["manual_status"] = f"excluded:{reviewer}"
        reviewed_rows.append(row)

    frozen_ids: set[str] = set()
    pilot_ids: set[str] = set()
    family_summary: dict[str, object] = {}
    for family in review["families"]:
        eligible = [
            row for row in reviewed_rows if row["family"] == family and row["freeze_status"] == "candidate"
        ]
        forced_pilot = [row for row in eligible if row["pilot_only"] == "true"]
        if len(eligible) < 15:
            raise ValueError(f"{family} has only {len(eligible)} eligible candidates")
        if len(forced_pilot) > 5:
            raise ValueError(f"{family} has more than five forced-pilot candidates")
        ranked = sorted(
            (row for row in eligible if row["pilot_only"] != "true"),
            key=lambda row: stable_rank(args.split_seed, row),
        )
        selected = forced_pilot + ranked[: 15 - len(forced_pilot)]
        remaining_selected = sorted(
            (row for row in selected if row["pilot_only"] != "true"),
            key=lambda row: stable_rank(args.split_seed + 1, row),
        )
        pilots = forced_pilot + remaining_selected[: 5 - len(forced_pilot)]
        tests = [row for row in selected if row["candidate_id"] not in {item["candidate_id"] for item in pilots}]
        if len(pilots) != 5 or len(tests) != 10:
            raise AssertionError(f"Bad split for {family}: pilot={len(pilots)}, test={len(tests)}")
        frozen_ids.update(row["candidate_id"] for row in selected)
        pilot_ids.update(row["candidate_id"] for row in pilots)
        family_summary[family] = {
            "eligible": len(eligible),
            "excluded": 32 - len(eligible),
            "pilot": [row["candidate_id"] for row in pilots],
            "test": [row["candidate_id"] for row in tests],
            "reserve": len(eligible) - 15,
        }

    for row in reviewed_rows:
        candidate_id = row["candidate_id"]
        if candidate_id in frozen_ids:
            row["freeze_status"] = "provisional_frozen"
            row["split"] = "pilot" if candidate_id in pilot_ids else "test"
        elif row["freeze_status"] == "candidate":
            row["freeze_status"] = "reserve"
            row["split"] = "unassigned"

    manifest_rows = [
        {
            "case_id": row["candidate_id"],
            "family": row["family"],
            "dish": row["dish"],
            "utensil": row["utensil"],
            "action": row["action"],
            "split": row["split"],
            "source_kind": "real_dataset",
            "source_path": row["source_path"],
            "source_sha256": row["source_sha256"],
            "license": LICENSE,
            "has_target_utensil": "false",
            "annotation_status": "pending",
            "freeze_status": "provisional_frozen",
        }
        for row in reviewed_rows
        if row["candidate_id"] in frozen_ids
    ]
    manifest_rows.sort(key=lambda row: (row["family"], row["split"], row["case_id"]))

    write_csv(args.reviewed_output, REVIEWED_FIELDS, reviewed_rows)
    write_csv(args.manifest_output, MANIFEST_FIELDS, manifest_rows)
    summary = {
        "schema_version": "foodstateedit.provisional_split.v1",
        "status": "provisional_requires_human_confirmation",
        "review_schema_version": review["schema_version"],
        "reviewer": reviewer,
        "split_seed": args.split_seed,
        "selection_rule": "forced development cases to pilot, then SHA-256 stable rank",
        "total_frozen": len(manifest_rows),
        "split_counts": dict(Counter(row["split"] for row in manifest_rows)),
        "family_counts": dict(Counter(row["family"] for row in manifest_rows)),
        "families": family_summary,
    }
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
