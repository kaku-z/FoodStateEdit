#!/usr/bin/env python3
"""Verify collected benchmark VACE shards without assigning visual success."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


COMPLETE_STATUS = "complete_requires_blind_review"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def job_key(item: dict) -> tuple[str, int, str]:
    return item["case_id"], int(item["seed"]), item["condition"]


def verify_shard(shard_root: Path) -> dict[str, Any]:
    manifest_path = shard_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != COMPLETE_STATUS or not (shard_root / "COMPLETE").is_file():
        raise ValueError(f"Shard is not complete: {shard_root}")
    if manifest.get("pipeline_load_count") != 1:
        raise ValueError(f"Pipeline load count is not one: {shard_root}")
    expected = {job_key(item) for item in manifest["expected_jobs"]}
    completed = {job_key(item) for item in manifest["completed_jobs"]}
    if expected != completed or len(completed) != len(manifest["completed_jobs"]):
        raise ValueError(f"Expected/completed jobs differ: {shard_root}")
    file_count = 0
    total_bytes = 0
    output_hashes = []
    for item in manifest["completed_jobs"]:
        if item["frames"] != 21 or item["steps"] != 20:
            raise ValueError(f"Unexpected inference budget: {shard_root} {job_key(item)}")
        if item["outside_support_max_pixel_difference"] != 0:
            raise ValueError(f"Protected pixels changed: {shard_root} {job_key(item)}")
        if item["cuda_max_memory_allocated_mib"] is None or item["cuda_max_memory_reserved_mib"] is None:
            raise ValueError(f"Missing CUDA memory evidence: {shard_root} {job_key(item)}")
        for record in item["files"].values():
            path = shard_root / record["path"]
            if not path.is_file() or path.stat().st_size != record["size_bytes"]:
                raise ValueError(f"Missing or wrong-sized file: {path}")
            digest = sha256_file(path)
            if digest != record["sha256"]:
                raise ValueError(f"Hash mismatch: {path}")
            file_count += 1
            total_bytes += path.stat().st_size
            output_hashes.append(digest)
    return {
        "shard_id": manifest["shard_id"],
        "host": manifest["host"],
        "physical_gpu": manifest["physical_gpu"],
        "config_sha256": manifest["config_sha256"],
        "run_manifest_sha256": sha256_file(manifest_path),
        "case_ids": manifest["case_ids"],
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
    parser.add_argument("--shard", action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    shards = [verify_shard(args.collection_root / shard) for shard in args.shard]
    case_ids = [case_id for shard in shards for case_id in shard["case_ids"]]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("A case appears in more than one collected shard")
    config_hashes = {shard["config_sha256"] for shard in shards}
    if len(config_hashes) != 1:
        raise ValueError("Collected shards do not share one config hash")
    payload = {
        "schema_version": "foodstateedit.benchmark_vace_collection.v1",
        "status": "collection_verified_requires_blind_review",
        "shard_count": len(shards),
        "case_count": len(case_ids),
        "expected_job_count": sum(shard["expected_job_count"] for shard in shards),
        "completed_job_count": sum(shard["completed_job_count"] for shard in shards),
        "verified_file_count": sum(shard["verified_file_count"] for shard in shards),
        "verified_bytes": sum(shard["verified_bytes"] for shard in shards),
        "all_outside_support_exact": all(shard["all_outside_support_exact"] for shard in shards),
        "shards": shards,
        "claim_limit": "Technical and byte-level verification does not assign action, photo-realism or held-out effectiveness success.",
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
