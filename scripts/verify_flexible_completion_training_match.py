#!/usr/bin/env python3
"""Verify initialization and stochastic traces across all Day 13 arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from preflight_vace_fork_3d_compare import sha256_file


ARM_IDS = (
    "planar_uniform",
    "relative3d_uniform",
    "relative3d_topology_weighted",
)
MATCHED_KEYS = (
    "optimizer_step",
    "training_row_id",
    "training_sample_id",
    "timestep_seed",
    "noise_seed",
    "timestep_id",
    "timestep",
    "noise_sha256",
    "input_latent_shape",
    "prediction_shape_after_first_frame_policy",
    "conditioned_first_frame_excluded",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = args.report.resolve()
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite report: {report_path}")
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    initializations: dict[str, dict[str, object]] = {}
    traces: dict[str, list[dict[str, object]]] = {}
    run_manifests: dict[str, dict[str, object]] = {}
    file_records: dict[str, dict[str, object]] = {}
    for arm_id in ARM_IDS:
        root = Path(config["arms"][arm_id]["output_root"])
        initialization_path = root / "initialization_manifest.json"
        trace_path = root / "randomness_trace.jsonl"
        run_manifest_path = root / "run_manifest.json"
        initializations[arm_id] = json.loads(initialization_path.read_text(encoding="utf-8"))
        traces[arm_id] = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        run_manifests[arm_id] = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        file_records[arm_id] = {
            "initialization_sha256": sha256_file(initialization_path),
            "trace_sha256": sha256_file(trace_path),
            "run_manifest_sha256": sha256_file(run_manifest_path),
        }
    initialization_hashes = {
        arm_id: value["aggregate_sha256"] for arm_id, value in initializations.items()
    }
    reference = [
        {key: record[key] for key in MATCHED_KEYS} for record in traces[ARM_IDS[0]]
    ]
    trace_matches = {
        arm_id: [
            {key: record[key] for key in MATCHED_KEYS} for record in traces[arm_id]
        ]
        == reference
        for arm_id in ARM_IDS
    }
    expected_steps = config["training"]["expected_optimizer_steps"]
    passed = (
        len(set(initialization_hashes.values())) == 1
        and all(trace_matches.values())
        and all(len(trace) == expected_steps for trace in traces.values())
        and all(
            manifest.get("status")
            == "complete_requires_cross_arm_match_and_checkpoint_validation"
            for manifest in run_manifests.values()
        )
    )
    report = {
        "schema_version": "foodstateedit.flexible_completion_cross_arm_match.v1",
        "status": "complete" if passed else "failed_preserved",
        "initial_trainable_state_aggregate_sha256_by_arm": initialization_hashes,
        "initial_trainable_state_identical": len(set(initialization_hashes.values())) == 1,
        "matched_trace_by_arm": trace_matches,
        "trace_count_by_arm": {arm_id: len(value) for arm_id, value in traces.items()},
        "matched_fields": list(MATCHED_KEYS),
        "files": file_records,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
