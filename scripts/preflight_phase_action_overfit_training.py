#!/usr/bin/env python3
"""Variant guard for the frozen phase-varying overfit training preflight."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import preflight_adapter_action_training


VARIANT = "phase_varying_overfit_sanity_v1"


def config_path_from_argv(argv: list[str]) -> Path:
    try:
        index = argv.index("--config")
        return Path(argv[index + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise ValueError("--config is required") from error


def main() -> int:
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
    return preflight_adapter_action_training.main()


if __name__ == "__main__":
    raise SystemExit(main())
