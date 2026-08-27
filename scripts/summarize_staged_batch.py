#!/usr/bin/env python3
"""Verify and summarize a two-worker FoodStateEdit staged pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ANCHORS = (
    "soup_spoon_001",
    "fried_rice_spatula_001",
    "ramen_chopsticks_001",
    "pasta_fork_001",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", default="foodstateedit_staged")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite summary: {args.output}")

    worker_dirs = sorted(path for path in args.results_root.glob("worker_gpu*") if path.is_dir())
    if len(worker_dirs) != 2:
        raise ValueError(f"Expected two worker directories, found {len(worker_dirs)}")

    workers = []
    manifests: dict[str, tuple[Path, dict[str, object]]] = {}
    for worker_dir in worker_dirs:
        worker = load_json(worker_dir / "worker_summary.json")
        if worker["method"] != args.method or worker["seed"] != 1:
            raise ValueError(f"Worker identity mismatch: {worker_dir}")
        workers.append({
            "worker": worker_dir.name,
            "gpu": worker["worker_gpu"],
            "case_count": worker["case_count"],
            "complete_count": worker["complete_count"],
            "technical_failure_count": worker["technical_failure_count"],
            "pipeline_load_count": worker["pipeline_load_count"],
            "resident_contract_passed": worker["resident_contract_passed"],
        })
        for path in (worker_dir / "runs").glob("*/run_manifest.json"):
            manifest = load_json(path)
            case_id = str(manifest["case_id"])
            if case_id in manifests:
                raise ValueError(f"Duplicate run manifest for {case_id}")
            manifests[case_id] = (path, manifest)

    if set(manifests) != set(ANCHORS):
        raise ValueError(f"Anchor mismatch: {sorted(manifests)}")

    runs = []
    for anchor in ANCHORS:
        path, manifest = manifests[anchor]
        if manifest["method"] != args.method or manifest["seed"] != 1:
            raise ValueError(f"Run identity mismatch: {path}")
        edited_record = next(
            item for item in manifest["outputs"]
            if Path(item["path"]).name == "edited_2d.png"
        )
        artifact = args.artifact_root / f"{anchor}_edited_2d.png"
        actual_hash = sha256_file(artifact)
        if actual_hash != edited_record["sha256"]:
            raise ValueError(f"Edited image hash mismatch: {anchor}")
        inference = manifest["inference"]
        runs.append({
            "anchor_id": anchor,
            "run_manifest": path.as_posix(),
            "status": manifest["status"],
            "edited_2d_artifact": artifact.as_posix(),
            "edited_2d_sha256": actual_hash,
            "outside_edit_alpha_max_pixel_difference": inference[
                "outside_edit_alpha_max_pixel_difference"
            ],
            "outside_edit_alpha_rgb_mae": inference["outside_edit_alpha_rgb_mae"],
            "feather_band_rgb_mae": inference["feather_band_rgb_mae"],
            "pipeline_load_count_at_completion": inference[
                "pipeline_load_count_at_completion"
            ],
            "wall_time_seconds": inference["wall_time_seconds"],
        })

    summary = {
        "schema_version": "foodstateedit.staged_batch_summary.v1",
        "method": args.method,
        "seed": 1,
        "case_count": len(runs),
        "complete_count": sum(item["status"] == "complete" for item in runs),
        "technical_failure_count": sum(
            item["status"] == "technical_failure" for item in runs
        ),
        "all_protected_pixels_exact": all(
            item["status"] == "complete"
            and item["outside_edit_alpha_max_pixel_difference"] == 0
            for item in runs
        ),
        "all_edited_images_hash_verified": True,
        "resident_worker_count": len(workers),
        "resident_contract_passed": all(
            worker["resident_contract_passed"]
            and worker["pipeline_load_count"] == 1
            for worker in workers
        ),
        "workers": workers,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"{summary['complete_count']}/{summary['case_count']} complete; "
        f"resident={summary['resident_contract_passed']}; "
        f"exact={summary['all_protected_pixels_exact']}"
    )


if __name__ == "__main__":
    main()
