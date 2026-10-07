"""Run exact-geometry synthetic oracle; deliberately not image-edit training."""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodstateedit.material_transfer.core import make_cake, transport, render, MaterialState


def diagnostics(source, grid, delta):
    ids = grid[grid >= 0]
    expected = source.grid[source.grid >= 0]
    positions = np.argwhere(np.isin(source.grid, source.selected_ids))
    destination = positions + delta
    return {
        "source_volume": int(len(expected)), "edited_volume": int(len(ids)),
        "volume_relative_error": float(abs(len(ids) - len(expected)) / len(expected)),
        "duplicate_ids": int(len(ids) - len(np.unique(ids))),
        "missing_ids": int(len(np.setdiff1d(expected, ids))),
        "source_residual_cells": int(np.sum(grid[tuple(positions.T)] >= 0)),
        "destination_correspondence_errors": int(np.sum(
            grid[tuple(destination.T)] != source.grid[tuple(positions.T)])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=160)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("cases must be positive")
    # A fresh directory prevents accidental replacement of previous experiments.
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    rows = []
    delta = np.array([15, 0, 0])
    for seed in range(args.cases):
        source = make_cake(seed)
        edited = transport(source, delta)
        exact = diagnostics(source, edited.grid, delta)
        assert all(exact[k] == 0 for k in (
            "volume_relative_error", "duplicate_ids", "missing_ids",
            "source_residual_cells", "destination_correspondence_errors"))
        moving = np.argwhere(np.isin(source.grid, source.selected_ids))
        copied = edited.grid.copy()
        copied[tuple(moving.T)] = source.grid[tuple(moving.T)]
        copied_metrics = diagnostics(source, copied, delta)
        assert copied_metrics["duplicate_ids"] > 0 and copied_metrics["source_residual_cells"] > 0
        shuffled = edited.grid.copy()
        target = moving + delta
        ids = shuffled[tuple(target.T)].copy()
        shuffled[tuple(target.T)] = np.roll(ids, 1)
        shuffled_metrics = diagnostics(source, shuffled, delta)
        assert shuffled_metrics["destination_correspondence_errors"] == len(ids)
        assert shuffled_metrics["volume_relative_error"] == 0
        # A real rendering ablation: ID permutation keeps geometry but changes
        # which canonical material is visible. It is not an external baseline.
        wrong = MaterialState(shuffled, source.canonical, source.selected_ids)
        case_dir = args.output / f"case_{seed:03d}"
        case_dir.mkdir()
        images = []
        renders = {}
        for name, state in (("source", source), ("transport", edited), ("shuffled", wrong)):
            result = render(state, args.resolution)
            renders[name] = result
            image = Image.fromarray(np.round(result["rgb"] * 255).astype(np.uint8))
            image.save(case_dir / f"{name}.png")
            images.append(image)
            np.savez_compressed(case_dir / f"{name}.npz", grid=state.grid,
                                canonical=state.canonical, selected_ids=state.selected_ids,
                                **{k: v for k, v in result.items() if k != "rgb"})
        sheet = Image.new("RGB", (3 * args.resolution, args.resolution))
        for i, image in enumerate(images):
            sheet.paste(image, (i * args.resolution, 0))
        sheet.save(case_dir / "comparison_source_transport_shuffled.png")
        mask = np.isin(renders["transport"]["material_ids"], source.selected_ids)
        assert mask.any(), "Selected material must be visible in this fixture"
        appearance_error = float(np.abs(renders["transport"]["rgb"][mask] -
                                       renders["shuffled"]["rgb"][mask]).mean())
        rows.append({"seed": seed, "exact": exact, "copy_negative_control": copied_metrics,
                     "shuffle_negative_control": shuffled_metrics,
                     "shuffle_visible_material_rgb_mae": appearance_error})
    code_root = Path(__file__).resolve().parents[1]
    code_paths = [Path(__file__), code_root / "foodstateedit/material_transfer/core.py"]
    report = {"stage": "synthetic_oracle_integer_translation", "host": platform.node(),
              "python": sys.version, "numpy": np.__version__, "seconds": time.time() - started,
              "cases": rows, "all_exact_invariants_passed": True,
              "code_sha256": {str(p.relative_to(code_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in code_paths},
              "limitations": ["Oracle source geometry and selection; no learned single-image model",
                              "Integer translations, discrete path collisions, fixed orthographic camera",
                              "No tool/contact model, lighting transport, continuous dynamics or real-image claims",
                              "Copy/shuffle are implementation negative controls, not competitive baselines"]}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "cases": len(rows),
                      "all_exact_invariants_passed": True, "seconds": report["seconds"]}))


if __name__ == "__main__":
    main()
