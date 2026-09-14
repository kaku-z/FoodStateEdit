#!/usr/bin/env python3
"""Apply the frozen ordinary or event-aware E2E v1 frame selector."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foodstateedit.e2e_pipeline import FrameMetric, select_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--event-aware", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    result = select_frame([FrameMetric.from_dict(row) for row in payload["frames"]], payload["weights"], require_events=args.event_aware)
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / "selection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if result["selected_index"] is not None:
        source = args.frames / f"frame_{result['selected_index']:03d}.png"
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, args.output_root / "06_selected_frame.png")
        shutil.copy2(source, args.output_root / "07_final.png")
    print(json.dumps({"status": result["status"], "selected_index": result["selected_index"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
