"""Direct source-pixel material baseline, with no geometry or learned prediction."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F


def sample_linear(image, uv):
    observed = F.grid_sample(image, uv[:, :, None], mode="bilinear",
                             padding_mode="border", align_corners=False)
    srgb = observed.squeeze(-1).transpose(1, 2)
    return torch.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)


def summarize(rows):
    result = {"scene_count": len(rows)}
    for key in ("occupied", "visible"):
        count = sum(row[f"{key}_queries"] for row in rows)
        result[f"{key}_query_count"] = count
        result[f"{key}_empty_scene_count"] = sum(row[f"{key}_queries"] == 0 for row in rows)
        result[f"{key}_linear_rgb_mse_micro"] = sum(row[f"{key}_color_squared_error"] for row in rows) / max(3 * count, 1)
        values = [row[f"{key}_linear_rgb_mse"] for row in rows if row[f"{key}_queries"] > 0]
        result[f"{key}_linear_rgb_mse_macro"] = float(np.mean(values))
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    started = time.time()
    names = ("source_rgb", "source_uv", "joint", "occupancy", "surface_uv", "surface_rgb", "surface_valid", "splits")
    arrays = {key: np.load(args.data / f"{key}.npy", mmap_mode="r") for key in names}
    records = [json.loads(line) for line in (args.data / "scenes.jsonl").read_text().splitlines()]
    scenes = np.flatnonzero(arrays["splits"] == 2)
    rows = []
    for start in range(0, len(scenes), 32):
        selected = scenes[start:start + 32]
        image = torch.from_numpy(np.asarray(arrays["source_rgb"][selected], dtype=np.float32).transpose(0, 3, 1, 2).copy() / 255).to(device)
        uv = torch.from_numpy(np.broadcast_to(arrays["source_uv"], (len(selected), *arrays["source_uv"].shape)).copy()).to(device)
        predicted = sample_linear(image, uv)
        target = torch.from_numpy((np.asarray(arrays["joint"][selected, :, 1:], dtype=np.float32) + 1) / 2).to(device)
        occupied = torch.from_numpy(np.asarray(arrays["occupancy"][selected]).copy()).to(device)
        occupied_error = ((predicted - target).square() * occupied[..., None]).sum((1, 2))
        occupied_count = occupied.sum(1)
        surface_uv = torch.from_numpy(np.asarray(arrays["surface_uv"][selected], dtype=np.float32).copy()).to(device)
        surface_predicted = sample_linear(image, surface_uv)
        surface_target = torch.from_numpy(np.asarray(arrays["surface_rgb"][selected], dtype=np.float32).copy()).to(device)
        valid = torch.from_numpy(np.asarray(arrays["surface_valid"][selected]).copy()).to(device)
        visible_error = ((surface_predicted - surface_target).square() * valid[..., None]).sum((1, 2))
        visible_count = valid.sum(1)
        for j, scene in enumerate(selected):
            row = {"scene_index": int(scene), "scene_id": records[int(scene)]["scene_id"],
                   "shape_family": records[int(scene)]["shape_family"], "development_diagnostic": int(scene) < 7176}
            for key, error, count in (("occupied", occupied_error, occupied_count), ("visible", visible_error, visible_count)):
                row[f"{key}_queries"] = int(count[j])
                row[f"{key}_color_squared_error"] = float(error[j])
                row[f"{key}_linear_rgb_mse"] = float(error[j] / (3 * count[j]).clamp_min(1))
            rows.append(row)
    aggregate = summarize(rows)
    families = sorted({row["shape_family"] for row in rows})
    result = {"method": "source projection color only", "model": "none", "dataset": str(args.data),
              "dataset_manifest_sha256": hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest(),
              "prediction": "Bilinear F.grid_sample of source_rgb/255 at known camera UV, align_corners=False, border padding; sampled sRGB then converted to linear RGB. No learned calibration.",
              "inputs": ["source_rgb", "known camera query UV"],
              "GT_scope": "GT occupied mask and valid first-visible query indices only choose scoring points; GT material supplies targets. No GT geometry, mask, depth or RGB is an input to color prediction.",
              "target_scope": "Canonical persistent unlit linear material RGB. Source pixels include shading and sRGB quantization; copying source retains observed lighting.",
              "aggregation": "Micro = all squared channel errors / (3 * selected query count). Macro = mean per-scene channel MSE, excluding empty scoring scenes.",
              "all_1024_test": aggregate,
              "per_family": {family: summarize([row for row in rows if row["shape_family"] == family]) for family in families},
              "blind_1016_excluding_eight_development_diagnostics": summarize([row for row in rows if not row["development_diagnostic"]]),
              "source_anchor_interpretation": "A color anchor is provided by observed pixels. Network improvements may reflect supervised unshading and inferred interior material; they do not make source copying a novel learned contribution."}
    comparisons = []
    for seed in (41, 163, 907):
        path = args.experiment / "runs_continuous" / f"shared_s{seed}" / "evaluation.json"
        learned = json.loads(path.read_text())
        values = {"seed": seed, "existing_evaluation": str(path),
                  "occupied_linear_rgb_mse_micro": learned["source"]["occupied_linear_rgb_mse"],
                  "occupied_linear_rgb_mse_macro": learned["macro_per_scene"]["occupied_linear_rgb_mse"],
                  "visible_linear_rgb_mse_micro": learned["visible_linear_rgb_mse"],
                  "per_family_occupied_linear_rgb_mse_macro": {family: learned["per_family"][family]["occupied_linear_rgb_mse"] for family in families}}
        values["source_baseline_to_network_mse_ratio"] = {key: aggregate[key] / values[key] for key in
                                                       ("occupied_linear_rgb_mse_micro", "occupied_linear_rgb_mse_macro", "visible_linear_rgb_mse_micro")}
        comparisons.append(values)
    result["completed_65k_network_comparisons"] = comparisons
    result["network_comparison_scope"] = "Previously completed same-dataset 1024-scene posterior RGB evaluation, not D samples. Visible macro/per-family metrics were not saved by that evaluation and are not inferred here."
    result["elapsed_seconds"] = time.time() - started
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out / "per_scene.jsonl").write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"all_1024_test": aggregate, "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
