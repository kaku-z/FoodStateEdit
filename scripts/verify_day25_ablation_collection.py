#!/usr/bin/env python3
"""Verify locally collected Day 25 case roots against their run manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


COMPLETE_STATUS = "complete_requires_ablation_mapping_and_review"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_case(case_root: Path) -> dict[str, Any]:
    manifest_path = case_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != COMPLETE_STATUS or not (case_root / "COMPLETE").is_file():
        raise ValueError(f"Case is not complete: {case_root}")
    if manifest.get("pipeline_load_count") != 1:
        raise ValueError(f"Pipeline load count is not one: {case_root}")
    expected = {(int(item["seed"]), item["condition"]) for item in manifest["expected_jobs"]}
    completed = {
        (int(item["seed"]), item["condition"])
        for item in manifest["completed_conditions"]
    }
    if completed != expected or len(completed) != len(manifest["completed_conditions"]):
        raise ValueError(f"Expected/completed jobs differ: {case_root}")

    file_count = 0
    total_bytes = 0
    output_hashes: list[str] = []
    for item in manifest["completed_conditions"]:
        if item["frames"] != 21 or item["steps"] != 20:
            raise ValueError(f"Unexpected inference budget: {case_root}")
        if item["outside_support_max_pixel_difference"] != 0:
            raise ValueError(f"Protected pixels changed: {case_root}")
        for record in item["files"].values():
            path = case_root / record["path"]
            if not path.is_file() or path.stat().st_size != record["size_bytes"]:
                raise ValueError(f"Missing or wrong-sized file: {path}")
            digest = sha256_file(path)
            if digest != record["sha256"]:
                raise ValueError(f"Hash mismatch: {path}")
            file_count += 1
            total_bytes += path.stat().st_size
            output_hashes.append(digest)
    return {
        "case_id": manifest["case_id"],
        "dataset_case_id": manifest["dataset_case_id"],
        "config_sha256": manifest["config_sha256"],
        "run_manifest_sha256": sha256_file(manifest_path),
        "expected_job_count": len(expected),
        "completed_job_count": len(completed),
        "verified_file_count": file_count,
        "verified_bytes": total_bytes,
        "unique_output_hash_count": len(set(output_hashes)),
        "wall_time_seconds": manifest["wall_time_seconds"],
        "all_outside_support_exact": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = [verify_case(args.collection_root / case) for case in args.case]
    payload = {
        "schema_version": "foodstateedit.day25_ablation_collection.v1",
        "status": "complete_collection_verified" if len(cases) == 4 else "partial_collection_verified",
        "case_count": len(cases),
        "expected_job_count": sum(item["expected_job_count"] for item in cases),
        "completed_job_count": sum(item["completed_job_count"] for item in cases),
        "verified_file_count": sum(item["verified_file_count"] for item in cases),
        "all_outside_support_exact": all(item["all_outside_support_exact"] for item in cases),
        "cases": cases,
        "claim_limit": "Byte verification and technical completion do not establish semantic action success, photo realism or held-out effectiveness.",
    }
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
