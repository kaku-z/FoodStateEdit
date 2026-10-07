"""State perturbation sweep: independent GT, no learned model or real-image claims."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodstateedit.material_transfer.core import transport, render
from foodstateedit.material_transfer.sensitivity import make_fixture, edit_metrics, planar_comparator


def save_sheet(path, panels):
    n = panels[0][1].shape[0]
    sheet = Image.new("RGB", (n * len(panels), n + 24), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, rgb) in enumerate(panels):
        sheet.paste(Image.fromarray(np.round(rgb * 255).astype(np.uint8)), (index * n, 24))
        draw.text((index * n + 3, 5), label, fill="black")
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=128)
    args = parser.parse_args()
    if args.cases < 1 or args.resolution < 32:
        parser.error("Positive cases and resolution >=32 required")
    args.output.mkdir(parents=True, exist_ok=False)
    conditions = [("exact", 0, 0)] + [(f"selection_{v:+}", v, 0) for v in (-2, -1, 1, 2)] + \
                 [(f"thickness_{v:+}", 0, v) for v in (-2, -1, 1, 2)]
    # Freeze this before producing results. No success threshold tuned to outputs.
    protocol = {"cases": args.cases, "resolution": args.resolution, "conditions": conditions,
                "perturbation_units": "world-grid cells; thickness is z-axis, not camera-ray depth",
                "primary_metrics": ["selection_iou", "true_destination_error_fraction", "edit_rgb_mae"],
                "image_roi": "fixed ground-truth changed pixels per scene; never predicted ROI",
                "planar_selection": "visible source partition with boundary error; independent of thickness error",
                "comparison": "source-only 2D diagnostic; privileged 3D hidden material prevents fair superiority claim",
                "inference": "descriptive controlled synthetic sensitivity; no real generalization probability"}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    rows, hashes = [], []
    started = time.time()
    for seed in range(args.cases):
        source, delta, meta = make_fixture(seed)
        reference = transport(source, delta)
        hashes.append(hashlib.sha256(source.grid.tobytes()).hexdigest())
        source_render, target = render(source, args.resolution), render(reference, args.resolution)
        roi = np.any(np.abs(source_render["rgb"] - target["rgb"]) > 1e-5, axis=-1)
        preserved_food = (source_render["material_ids"] >= 0) & ~roi
        assert roi.any()
        panels = [("source", source_render["rgb"]), ("GT final", target["rgb"])]
        for name, boundary, thickness in conditions:
            predicted, _, _ = make_fixture(seed, boundary, thickness)
            row = {"seed": seed, "condition": name, "scene": meta}
            try:
                final = transport(predicted, delta)
            except ValueError as exc:
                row.update(status="rejected", error=str(exc))
                rows.append(row)
                continue
            result = render(final, args.resolution)
            # A 2D source mask is observable independently of the estimated
            # thickness. Do not erase its visible top pixels when predicted
            # geometry is too thin: that would confound the depth comparison.
            mask_state, _, _ = make_fixture(seed, boundary, 0)
            selected_mask = np.isin(source_render["material_ids"], mask_state.selected_ids)
            planar = planar_comparator(source_render["rgb"], selected_mask, delta)
            row.update(status="generated", **edit_metrics(source, reference, predicted, final))
            error = np.abs(result["rgb"] - target["rgb"])
            planar_error = np.abs(planar - target["rgb"])
            row.update(edit_rgb_mae=float(error[roi].mean()),
                       outside_edit_rgb_mae=float(error[~roi].mean()),
                       preserved_food_rgb_mae=float(error[preserved_food].mean()) if preserved_food.any() else 0.,
                       planar_edit_rgb_mae=float(planar_error[roi].mean()),
                       planar_outside_edit_rgb_mae=float(planar_error[~roi].mean()),
                       visible_id_error=float(np.mean(result["material_ids"][roi] != target["material_ids"][roi])))
            if name == "exact":
                assert row["true_destination_error_fraction"] == 0 and row["edit_rgb_mae"] == 0
            if name in ("exact", "selection_+1", "thickness_+1"):
                if name == "exact":
                    panels.append(("2D exact selection", planar))
                else:
                    panels.append((name, result["rgb"]))
            rows.append(row)
        save_sheet(args.output / f"case_{seed:03d}.png", panels)
        (args.output / "rows.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps({"completed_cases": seed + 1, "total": args.cases}), flush=True)
    if len(set(hashes)) != args.cases:
        raise RuntimeError("Duplicate fixtures: do not count repeated geometry as independent cases")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    aggregate = {}
    for name, group in grouped.items():
        generated = [r for r in group if r["status"] == "generated"]
        aggregate[name] = {"attempts": len(group), "generated": len(generated),
                           "rejected": len(group) - len(generated)}
        if generated:
            for key, value in generated[0].items():
                if isinstance(value, float):
                    vals = [r[key] for r in generated]
                    aggregate[name][key] = {"mean": float(np.mean(vals)), "min": min(vals), "max": max(vals)}
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / "foodstateedit/material_transfer/core.py",
             root / "foodstateedit/material_transfer/sensitivity.py"]
    report = {"host": platform.node(), "python": sys.version, "numpy": np.__version__,
              "seconds": time.time() - started, "unique_source_grids": len(set(hashes)),
              "source_grid_hashes": hashes, "attempts": len(rows), "aggregate": aggregate,
              "code_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "limits": ["synthetic oracle material and geometry; no learned model",
                         "world-z thickness errors; no perspective depth uncertainty",
                         "no spoon, support or continuous collision validation",
                         "2D comparator is weak and lacks 3D hidden material privileges"]}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"completed": True, "attempts": len(rows), "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
