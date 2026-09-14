#!/usr/bin/env python3
"""Verify the complete 4-case x 3-seed x 5-condition Day25 final-image matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from build_day25_ablation_review_grids import CONDITIONS, LEGACY_SEED1
except ModuleNotFoundError:  # package import used by the local test suite
    from scripts.build_day25_ablation_review_grids import CONDITIONS, LEGACY_SEED1


CASES = ("ramen", "soup", "rice", "cake")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    repo = args.repo.resolve()
    collection = args.collection_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    matrix = []
    for case in CASES:
        run_manifest = json.loads((collection / case / "run_manifest.json").read_text(encoding="utf-8"))
        generated = {(int(item["seed"]), item["condition"]): item for item in run_manifest["completed_conditions"]}
        for seed in (1, 2, 3):
            for condition in CONDITIONS:
                legacy = LEGACY_SEED1.get((case, condition)) if seed == 1 else None
                if legacy:
                    path = repo / legacy / "projected_final_hold.png"
                    expected = config["existing_seed_1_evidence"][case][condition]["final_sha256"]
                    source = "legacy_seed1_hash_locked_evidence"
                else:
                    item = generated[(seed, condition)]
                    record = item["files"]["projected_final_hold.png"]
                    path = collection / case / record["path"]
                    expected = record["sha256"]
                    source = "day25_collected_run"
                actual = sha256_file(path)
                if actual != expected:
                    raise ValueError(f"Final-image hash mismatch: {case} seed {seed} {condition}")
                matrix.append({
                    "case_id": case,
                    "seed": seed,
                    "condition": condition,
                    "source": source,
                    "path": path.relative_to(repo).as_posix(),
                    "sha256": actual,
                })
    if len(matrix) != 60 or len({(x["case_id"], x["seed"], x["condition"]) for x in matrix}) != 60:
        raise AssertionError("Day25 matrix is not complete and unique")
    payload = {
        "schema_version": "foodstateedit.day25_ablation_matrix.v1",
        "status": "complete_final_image_matrix_verified",
        "case_count": 4,
        "seed_count": 3,
        "condition_count": 5,
        "matrix_cell_count": 60,
        "newly_generated_cell_count": sum(x["source"] == "day25_collected_run" for x in matrix),
        "legacy_hash_locked_cell_count": sum(x["source"] != "day25_collected_run" for x in matrix),
        "unique_final_image_hash_count": len({x["sha256"] for x in matrix}),
        "cells": matrix,
        "claim_limit": "A complete hash-verified final-image matrix is not an action-success, photo-realism, superiority or held-out result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "cells"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
