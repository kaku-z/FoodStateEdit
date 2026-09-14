#!/usr/bin/env python3
"""Validate E2E v1 cases and write a non-overwriting frozen run plan."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foodstateedit.e2e_pipeline import build_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    payload = json.loads(args.cases.read_text(encoding="utf-8"))
    plan = build_plan(config, payload["cases"], args.workspace_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "jobs": len(plan["jobs"]), "generation_jobs": plan["generation_job_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
