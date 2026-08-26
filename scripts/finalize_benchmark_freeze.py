#!/usr/bin/env python3
"""Finalize an explicitly confirmed provisional benchmark manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provisional", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--confirmation-source", required=True)
    parser.add_argument("--confirm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm:
        raise SystemExit("Refusing to freeze without --confirm")
    with args.provisional.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or len(rows) != 60:
        raise ValueError("Expected a 60-row provisional manifest")
    if any(row["freeze_status"] != "provisional_frozen" for row in rows):
        raise ValueError("Every input row must be provisional_frozen")
    if Counter(row["family"] for row in rows) != Counter({
        "liquid": 15,
        "granular": 15,
        "strand": 15,
        "strand_contact": 15,
    }):
        raise ValueError("Family counts are not 15/15/15/15")
    for family in {row["family"] for row in rows}:
        splits = Counter(row["split"] for row in rows if row["family"] == family)
        if splits != Counter({"pilot": 5, "test": 10}):
            raise ValueError(f"Bad split for {family}: {splits}")
    if len({row["source_sha256"] for row in rows}) != 60:
        raise ValueError("Source hashes are not unique")
    if any(row["has_target_utensil"] != "false" for row in rows):
        raise ValueError("A frozen row still contains a target utensil")

    for row in rows:
        row["freeze_status"] = "frozen"
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    record = {
        "schema_version": "foodstateedit.benchmark_freeze_record.v1",
        "freeze_date": date.today().isoformat(),
        "confirmation_source": args.confirmation_source,
        "source_provisional_manifest": str(args.provisional.as_posix()),
        "source_provisional_sha256": sha256_file(args.provisional),
        "frozen_manifest": str(args.output.as_posix()),
        "frozen_manifest_sha256": sha256_file(args.output),
        "case_count": 60,
        "pilot_count": 20,
        "test_count": 40,
        "family_count": 4,
        "replacement_policy": "no replacement based on method results; corrections require a versioned erratum",
    }
    args.record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
