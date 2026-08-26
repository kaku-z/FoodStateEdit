#!/usr/bin/env python3
"""Run the deterministic input/no-edit control without copying images locally."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import socket
import sys
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_identity_baseline(
    canonical_manifest: Path,
    anchor_manifest: Path,
    output_root: Path,
    summary_path: Path,
    code_commit: str,
    builder_path: Path,
    created_at: str,
) -> dict[str, object]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output root: {output_root}")
    canonical = {row["case_id"]: row for row in read_csv(canonical_manifest)}
    anchors = read_csv(anchor_manifest)
    if len(anchors) != 4:
        raise ValueError(f"Expected four anchors, found {len(anchors)}")

    builder_path = builder_path.resolve()
    builder_hash = sha256_file(builder_path)
    builder_size = builder_path.stat().st_size
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for anchor in anchors:
        row = canonical[anchor["case_id"]]
        if row["split"] != "pilot":
            raise ValueError(f"Anchor is not in pilot split: {anchor['anchor_id']}")
        source = Path(row["canonical_input_path"])
        if not source.is_file():
            raise FileNotFoundError(f"Missing canonical input: {source}")
        source_hash = sha256_file(source)
        if source_hash != row["canonical_input_sha256"] or source_hash != anchor["canonical_input_sha256"]:
            raise ValueError(f"Canonical hash mismatch for {anchor['anchor_id']}")

        run_dir = output_root / anchor["anchor_id"] / "deterministic"
        run_dir.mkdir(parents=True, exist_ok=False)
        output = run_dir / f"edited_2d{source.suffix.lower()}"
        shutil.copy2(source, output)
        output_hash = sha256_file(output)
        if output_hash != source_hash:
            raise ValueError(f"Identity copy is not bit-exact for {anchor['anchor_id']}")

        run_manifest = {
            "schema_version": "foodstateedit.run.v1",
            "run_id": f"{anchor['anchor_id']}__input_no_edit__deterministic",
            "case_id": anchor["anchor_id"],
            "method": "input_no_edit",
            "seed": 0,
            "model": {
                "name": "identity_copy_v1",
                "root": str(builder_path.parent),
                "files": [
                    {"path": str(builder_path), "size_bytes": builder_size, "sha256": builder_hash}
                ],
            },
            "inference": {
                "width": int(row["canonical_width"]),
                "height": int(row["canonical_height"]),
                "frames": 1,
                "steps": 1,
                "warm_start": False,
                "schedule": {"rigid": 0, "contact": 0, "material": 0, "hole": 0},
            },
            "environment": {
                "host": socket.gethostname(),
                "gpu": "none_identity_copy",
                "python": sys.executable,
                "code_commit": code_commit,
                "created_at": created_at,
            },
            "status": "complete",
            "outputs": [{"path": str(output), "sha256": output_hash}],
        }
        manifest_path = run_dir / "run_manifest.json"
        manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
        results.append(
            {
                "anchor_id": anchor["anchor_id"],
                "source_case_id": anchor["case_id"],
                "run_manifest": str(manifest_path),
                "output": str(output),
                "source_sha256": source_hash,
                "output_sha256": output_hash,
                "bit_exact": True,
                "status": "complete",
            }
        )

    summary = {
        "schema_version": 1,
        "method": "input_no_edit",
        "deterministic": True,
        "case_count": len(results),
        "complete_count": sum(item["status"] == "complete" for item in results),
        "all_outputs_bit_exact": all(item["bit_exact"] for item in results),
        "code_commit": code_commit,
        "created_at": created_at,
        "runs": results,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--builder-path", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_identity_baseline(
        args.canonical_manifest,
        args.anchor_manifest,
        args.output_root,
        args.summary,
        args.code_commit,
        args.builder_path,
        args.created_at,
    )
    print(f"Completed {summary['complete_count']}/{summary['case_count']} no-edit anchor runs")


if __name__ == "__main__":
    main()
