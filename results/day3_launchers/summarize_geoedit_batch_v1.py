#!/usr/bin/env python3
"""Summarize a four-anchor GeoEdit batch without copying restricted images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ANCHORS = (
    "soup_spoon_001",
    "fried_rice_spatula_001",
    "ramen_chopsticks_001",
    "pasta_fork_001",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.summary.exists():
        raise FileExistsError(f"Refusing to overwrite summary: {args.summary}")

    runs = []
    for anchor in ANCHORS:
        path = args.output_root / anchor / f"seed_{args.seed}" / "run_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing run manifest: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["case_id"] != anchor or manifest["method"] != args.method:
            raise ValueError(f"Run identity mismatch: {path}")
        if manifest["seed"] != args.seed:
            raise ValueError(f"Seed mismatch: {path}")
        runs.append({
            "anchor_id": anchor,
            "run_manifest": str(path.resolve()),
            "status": manifest["status"],
            "outside_edit_alpha_max_pixel_difference": manifest["inference"].get(
                "outside_edit_alpha_max_pixel_difference"
            ),
            "outputs": manifest["outputs"],
        })

    summary = {
        "schema_version": 1,
        "method": args.method,
        "seed": args.seed,
        "case_count": len(runs),
        "complete_count": sum(run["status"] == "complete" for run in runs),
        "technical_failure_count": sum(
            run["status"] == "technical_failure" for run in runs
        ),
        "all_protected_pixels_exact": all(
            run["status"] == "complete"
            and run["outside_edit_alpha_max_pixel_difference"] == 0
            for run in runs
        ),
        "runs": runs,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"{args.method}: {summary['complete_count']}/{summary['case_count']} complete; "
        f"{summary['technical_failure_count']} technical failures"
    )


if __name__ == "__main__":
    main()
