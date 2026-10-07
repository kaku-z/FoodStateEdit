"""Source ambiguity, observed-appearance ablation, image-resolution and blade tests."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodstateedit.material_transfer.core import render, transport
from foodstateedit.material_transfer.sensitivity import make_fixture, edit_metrics
from foodstateedit.material_transfer.identifiability import CutSurfaceAlternative, SourceAppearance, tool_probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=128)
    args = parser.parse_args()
    if args.cases < 1 or args.resolution < 32:
        parser.error("Positive cases and resolution >=32 required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    resolutions = [args.resolution // 2, args.resolution, args.resolution * 2]
    protocol = {"cases": args.cases, "image_resolutions": resolutions,
                "voxel_grid": "56 cubed; fixed across image-resolution experiment",
                "appearance_hypotheses": "A analytic layers; B modifies only initially internal partition interfaces",
                "source_only_predictor": "nearest source-observed RGB in oracle canonical coordinates, same normals preferred",
                "appearance_scope": "food-only RGB ablation; geometry, correspondence and background remain oracle",
                "perturbation": "source selection boundary +1 cell, evaluated against frozen correct source/target",
                "tool": "rectangular flat blade, straight +x-side insertion, prescribed translation; no curvature, handle or forces",
                "tool_gap": [0, .5, 1, 2], "tool_thickness": [.5, 1, 2], "approach_blocked": [False, True],
                "inference": "descriptive paired synthetic examples; no real-world accuracy or posterior probability claims"}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    appearance_rows, tool_rows, hashes = [], [], []
    for seed in range(args.cases):
        source, delta, meta = make_fixture(seed)
        target = transport(source, delta)
        alternate = CutSurfaceAlternative(source)
        incorrect_source, _, _ = make_fixture(seed, boundary_error=1)
        incorrect_target = transport(incorrect_source, delta)
        independent = edit_metrics(source, target, incorrect_source, incorrect_target)
        hashes.append(hashlib.sha256(source.grid.tobytes()).hexdigest())
        for resolution in resolutions:
            source_a = render(source, resolution)
            source_b = render(source, resolution, alternate)
            assert np.array_equal(source_a["rgb"], source_b["rgb"]), "Ambiguous sources differ"
            target_a = render(target, resolution)
            target_b = render(target, resolution, alternate)
            observed = render(target, resolution, SourceAppearance(source_a))
            wrong = render(incorrect_target, resolution)
            food = target_a["material_ids"] >= 0
            cut = np.zeros(food.shape, dtype=bool)
            cut[food] = alternate.hidden_faces(target_a["canonical_hits"][food], target_a["surface_normals"][food])
            if not cut.any():
                raise RuntimeError("Fixture exposes no initially internal faces; retain and inspect failure")
            roi = np.any(np.abs(source_a["rgb"] - target_a["rgb"]) > 1e-5, axis=-1)
            error_a = np.abs(observed["rgb"] - target_a["rgb"])
            error_b = np.abs(observed["rgb"] - target_b["rgb"])
            gap = float(np.abs(target_a["rgb"][cut] - target_b["rgb"][cut]).mean())
            mean_error = float((error_a[cut].mean() + error_b[cut].mean()) / 2)
            assert mean_error + 1e-9 >= gap / 2, "L1 triangle bound violated"
            appearance_rows.append({"seed": seed, "resolution": resolution, "scene": meta,
                "source_pair_max_abs_difference": float(np.abs(source_a["rgb"] - source_b["rgb"]).max()),
                "visible_cut_pixels": int(cut.sum()), "cut_fraction_of_visible_food": float(cut.sum() / food.sum()),
                "target_pair_cut_rgb_mae": gap, "equal_pair_cut_mae_lower_bound": gap / 2,
                "source_only_cut_mae_a": float(error_a[cut].mean()),
                "source_only_cut_mae_b": float(error_b[cut].mean()),
                "source_only_equal_pair_cut_mae": mean_error,
                "source_only_other_food_mae_a": float(error_a[food & ~cut].mean()),
                "source_only_all_food_mae_a": float(error_a[food].mean()),
                "selection_plus1_edit_rgb_mae": float(np.abs(wrong["rgb"][roi] - target_a["rgb"][roi]).mean()),
                "selection_plus1_true_residual": independent["true_source_residual_fraction"]})
            if resolution == args.resolution:
                panels = [("same source A/B", source_a["rgb"]), ("target A", target_a["rgb"]),
                          ("target B", target_b["rgb"]), ("source RGB only", observed["rgb"])]
                sheet = Image.new("RGB", (4 * resolution, resolution + 24), "white")
                draw = ImageDraw.Draw(sheet)
                for i, (label, rgb) in enumerate(panels):
                    sheet.paste(Image.fromarray(np.round(rgb * 255).astype(np.uint8)), (i * resolution, 24))
                    draw.text((i * resolution + 3, 5), label, fill="black")
                sheet.save(args.output / f"ambiguity_{seed:03d}.png")
                Image.fromarray((cut * 255).astype(np.uint8)).save(args.output / f"cut_mask_{seed:03d}.png")
        for gap in protocol["tool_gap"]:
            for thickness in protocol["tool_thickness"]:
                for blocked in protocol["approach_blocked"]:
                    row = {"seed": seed, **tool_probe(source, delta, gap, thickness, blocked)}
                    assert row["feasible"] == row["expected_for_fixture"], row
                    tool_rows.append(row)
        (args.output / "appearance_rows.json").write_text(json.dumps(appearance_rows, indent=2))
        (args.output / "tool_rows.json").write_text(json.dumps(tool_rows, indent=2))
        print(json.dumps({"completed_cases": seed + 1, "total": args.cases}), flush=True)
    assert len(set(hashes)) == args.cases, "Duplicate source geometry"
    aggregate = {}
    for resolution in resolutions:
        group = [r for r in appearance_rows if r["resolution"] == resolution]
        aggregate[str(resolution)] = {k: {"mean": float(np.mean([r[k] for r in group])),
                                         "min": min(r[k] for r in group), "max": max(r[k] for r in group)}
                                      for k, v in group[0].items() if isinstance(v, float)}
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / "foodstateedit/material_transfer/core.py",
             root / "foodstateedit/material_transfer/sensitivity.py", root / "foodstateedit/material_transfer/identifiability.py"]
    report = {"host": platform.node(), "python": sys.version, "numpy": np.__version__,
              "seconds": time.time() - started, "unique_geometries": len(set(hashes)),
              "appearance_trials": len(appearance_rows), "tool_trials": len(tool_rows),
              "tool_feasible": sum(r["feasible"] for r in tool_rows),
              "tool_rejected": sum(not r["feasible"] for r in tool_rows),
              "tool_expected_mismatches": sum(r["feasible"] != r["expected_for_fixture"] for r in tool_rows),
              "aggregate_by_resolution": aggregate,
              "code_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "limitations": protocol}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"completed": True, "appearance_trials": len(appearance_rows),
                      "tool_trials": len(tool_rows), "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
