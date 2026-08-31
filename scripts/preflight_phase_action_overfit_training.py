#!/usr/bin/env python3
"""Variant guard for the frozen phase-varying overfit training preflight."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

VARIANT = "phase_varying_overfit_sanity_v1"
DEPENDENCY_SHA256 = "655a74a41acfec32d959c1697836f6538b89f2726d9afc5bdf35358e09a32d35"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_path_from_argv(argv: list[str]) -> Path:
    try:
        index = argv.index("--config")
        return Path(argv[index + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise ValueError("--config is required") from error


def main() -> int:
    dependency_path = Path(__file__).resolve().with_name("preflight_adapter_action_training.py")
    dependency_hash = sha256_file(dependency_path) if dependency_path.is_file() else None
    if dependency_hash != DEPENDENCY_SHA256:
        raise ValueError(f"Frozen preflight dependency hash mismatch: {dependency_hash}")
    config_path = config_path_from_argv(sys.argv[1:])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_variant") != VARIANT:
        raise ValueError(f"Unexpected experiment variant: {config.get('experiment_variant')}")
    if config.get("dataset", {}).get("frame_policy") != (
        "deterministic_patch_motion_pseudo_sequence_approach_contact_lift_hold"
    ):
        raise ValueError("Phase-varying frame policy is not frozen")
    if config.get("training", {}).get("expected_optimizer_steps") != 64:
        raise ValueError("Overfit sanity must contain exactly 64 optimizer steps")
    import preflight_adapter_action_training

    return preflight_adapter_action_training.main()


if __name__ == "__main__":
    raise SystemExit(main())
