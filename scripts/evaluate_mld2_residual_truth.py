"""Finite paired GT comparison of a frozen posterior and three fixed DDIM draws."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodstateedit.material_lineage.mld2 import MLD2Config, MLD2Model, spatial_noise, sample_residual
from train_mld2 import make_actions


def metric_row(prediction, target, occupied, cut, cell_volume):
    sdf = prediction[:, 0].astype(np.float64)
    soft = 1 / (1 + np.exp(np.clip(sdf / .025, -60, 60)))
    hard = sdf <= 0
    true_count = int(occupied.sum())
    error = ((prediction[:, 1:].astype(np.float64) - target[:, 1:]) / 2) ** 2
    true_volume = true_count * cell_volume
    return {"intersection": int((hard & occupied).sum()), "union": int((hard | occupied).sum()),
            "occupied_queries": true_count, "occupied_rgb_squared_error": float(error[occupied].sum()),
            "source_iou": float((hard & occupied).sum() / max((hard | occupied).sum(), 1)),
            "sdf_mae_normalized": float(np.abs(sdf - target[:, 0]).mean()),
            "source_center_soft_volume": float(soft.sum() * cell_volume), "true_center_volume": true_volume,
            "source_center_soft_volume_l1": float(abs(soft.sum() - true_count) * cell_volume),
            "source_center_soft_volume_l1_normalized": float(abs(soft.sum() - true_count) / max(true_count, 1)),
            "source_center_soft_occupancy_l1_normalized": float(np.abs(soft - occupied).sum() / max(true_count, 1)),
            "occupied_linear_rgb_mse": float(error[occupied].sum() / max(3 * true_count, 1)),
            "id_carried_center_soft_volume_l1_normalized": float(abs(soft[cut].sum() - occupied[cut].sum()) / max(true_count, 1))}


def aggregate(rows):
    means = ("source_iou", "sdf_mae_normalized", "source_center_soft_volume_l1",
             "source_center_soft_volume_l1_normalized", "source_center_soft_occupancy_l1_normalized",
             "occupied_linear_rgb_mse", "id_carried_center_soft_volume_l1_normalized")
    return {"scene_count": len(rows), **{key + "_macro": float(np.mean([row[key] for row in rows])) for key in means},
            "source_iou_micro": sum(row["intersection"] for row in rows) / max(sum(row["union"] for row in rows), 1),
            "occupied_linear_rgb_mse_micro": sum(row["occupied_rgb_squared_error"] for row in rows) / max(3 * sum(row["occupied_queries"] for row in rows), 1)}


@torch.inference_mode()
def run(args):
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    payload = args.checkpoint.read_bytes()
    saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    model = MLD2Model(MLD2Config(**saved["model_config"]))
    model.load_state_dict(saved["state_dict"])
    residual_steps = [int(saved["optimizer_state_dict"]["state"][i]["step"]) for i, (name, _) in enumerate(model.named_parameters()) if name.startswith("residual_")]
    device = torch.device("cuda")
    model = model.to(device).eval()
    arrays = {key: np.load(args.data / f"{key}.npy", mmap_mode="r") for key in
              ("source_rgb", "canonical_xyz", "joint", "occupancy", "splits")}
    records = [json.loads(line) for line in (args.data / "scenes.jsonl").read_text().splitlines()]
    indices = np.load(args.replay / "predictions.npz")["scene_indices"]
    if args.scenes > len(indices):
        indices = np.flatnonzero(arrays["splits"] == 2)[:args.scenes]
    else:
        indices = indices[:args.scenes]
    xyz_np = np.asarray(arrays["canonical_xyz"], dtype=np.float32)
    xyz = torch.from_numpy(xyz_np.copy()).to(device)
    seeds = [20261004, 20261005, 20261006]
    cell_volume = (2 / model.config.grid_size) ** 3
    posterior = np.empty((len(indices), len(xyz_np), 4), np.float32)
    draws = np.empty((len(seeds), *posterior.shape), np.float32)
    ids = [records[int(i)]["scene_id"] for i in indices]
    actions = make_actions(ids, "id")
    print(json.dumps({"status": "running", "optimizer_steps": saved["optimizer_steps"], "residual_optimizer_steps": residual_steps,
                      "scene_count": len(indices), "fixed_sampling_seeds": seeds}), flush=True)
    for start in range(0, len(indices), 8):
        selected = indices[start:start + 8]
        image = np.asarray(arrays["source_rgb"][selected], dtype=np.float32).transpose(0, 3, 1, 2).copy() / 255
        source = model.query(model.encode_image(torch.from_numpy(image).to(device)), xyz[None].expand(len(selected), -1, -1))
        posterior[start:start + len(selected)] = torch.cat((source["sdf"], source["rgb"]), -1).cpu().numpy()
        for sample, seed in enumerate(seeds):
            noise = torch.from_numpy(spatial_noise(xyz_np, ids[start:start + len(selected)], seed)).to(device)
            draws[sample, start:start + len(selected)] = sample_residual(model, source, noise, 50).cpu().numpy()
        print(json.dumps({"completed_scenes": start + len(selected), "seconds": time.time() - started}), flush=True)
    rows = []
    for method, seed, predicted in [("posterior", None, posterior)] + [("ddim", seed, draws[j]) for j, seed in enumerate(seeds)]:
        for j, index in enumerate(indices):
            target = np.asarray(arrays["joint"][index], dtype=np.float32)
            occupied = np.asarray(arrays["occupancy"][index])
            cut = xyz_np @ actions[j, :3] > actions[j, 3]
            row = {"method": method, "sampling_seed": seed, "scene_index": int(index), "scene_id": ids[j],
                   "shape_family": records[int(index)]["shape_family"], "development_diagnostic": int(index) < 7176}
            row.update(metric_row(predicted[j], target, occupied, cut, cell_volume))
            rows.append(row)
    families = sorted({row["shape_family"] for row in rows})
    summaries = []
    scope_key = f"all_{len(indices)}"
    for method, seed in [("posterior", None)] + [("ddim", seed) for seed in seeds]:
        chosen = [row for row in rows if row["method"] == method and row["sampling_seed"] == seed]
        summaries.append({"method": method, "sampling_seed": seed, scope_key: aggregate(chosen),
                          "per_family": {family: aggregate([row for row in chosen if row["shape_family"] == family]) for family in families},
                          "excluding_eight_development_diagnostics": aggregate([row for row in chosen if not row["development_diagnostic"]])})
    numeric = [key for key in summaries[0][scope_key] if key != "scene_count"]
    draw_mean = {key: float(np.mean([summary[scope_key][key] for summary in summaries[1:]])) for key in numeric}
    mean_family = {family: {key: float(np.mean([summary["per_family"][family][key] for summary in summaries[1:]])) for key in numeric} for family in families}
    delta = {key: draw_mean[key] - summaries[0][scope_key][key] for key in numeric}
    bootstrap_metrics = ("source_iou", "source_center_soft_volume_l1_normalized",
                         "source_center_soft_occupancy_l1_normalized", "occupied_linear_rgb_mse",
                         "id_carried_center_soft_volume_l1_normalized")
    posterior_rows = rows[:len(indices)]
    draw_rows = [rows[(j + 1) * len(indices):(j + 2) * len(indices)] for j in range(len(seeds))]
    paired_deltas = np.asarray([[np.mean([draw[j][key] for draw in draw_rows]) - posterior_rows[j][key]
                                for key in bootstrap_metrics] for j in range(len(indices))])
    bootstrap_indices = np.random.default_rng(20261007).integers(0, len(indices), (2000, len(indices)))
    bootstrap = {}
    for column, key in enumerate(bootstrap_metrics):
        boot_means = paired_deltas[bootstrap_indices, column].mean(1)
        bootstrap[key + "_macro_draw_mean_minus_posterior"] = {
            "paired_scene_mean_delta": float(paired_deltas[:, column].mean()),
            "percentile_95_CI": np.percentile(boot_means, [2.5, 97.5]).tolist()}
    np.savez_compressed(args.out / "predictions.npz", scene_indices=indices, canonical_xyz=xyz_np,
                        posterior_joint=posterior, ddim_joint=draws, sampling_seeds=np.asarray(seeds), id_actions=actions)
    result = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
              "optimizer_steps": int(saved["optimizer_steps"]), "actual_residual_optimizer_steps_min": min(residual_steps),
              "actual_residual_optimizer_steps_max": max(residual_steps), "source_replay": str(args.replay),
              "dataset": str(args.data), "dataset_manifest_sha256": hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest(),
              "scene_count": len(indices), "points_per_scene": len(xyz_np), "DDIM_steps": 50, "sampling_seeds": seeds,
              "methods": summaries, "ddim_draw_metric_mean": draw_mean, "ddim_draw_metric_mean_per_family": mean_family,
              "ddim_mean_minus_posterior": delta,
              "comparison_policy": f"Every method uses the exact same {len(indices)} scenes and 4096 center queries. Three fixed draws are all reported; their metric mean is not a best draw or a metric of averaged fields. No oracle or GT projection.",
              "paired_scene_bootstrap_2000": {"replicates": 2000, "seed": 20261007,
                                               "unit": "Scene. Each scene's three draw metrics are averaged before pairing with its posterior. Draws are not independent bootstrap units.",
                                               "metrics": bootstrap},
              "GT_center_empty_scene_counts": {"all_test": sum(row["occupied_queries"] == 0 for row in posterior_rows),
                                               "per_family": {family: sum(row["occupied_queries"] == 0 for row in posterior_rows if row["shape_family"] == family) for family in families}},
              "volume_scope": "Center-lattice unit-density proxy only, cell volume=(2/16)^3, soft occupancy=sigmoid(-normalized_sdf/.025), normalized denominator=max(GT occupied center count,1). Empty-center scenes would not measure true shape volume; their counts are reported. Global volume error and per-point occupancy error are separate; no eight-subsite or physical mass claim.",
              "GT_scope": "Stored GT SDF, occupancy and material are used only after sampling for scoring. Cache contains source RGB and known canonical coordinates, with no GT geometry/depth/mask.",
              "RGB_scope": "MSE of linear canonical material on GT occupied queries; normalized [-1,1] output difference divided by 2.",
              "DDIM_scope": "Existing learned pointwise residual sampler perturbs the source posterior joint SDF/RGB. No assumption that persistence implies truth improvement.",
              "elapsed_seconds": time.time() - started}
    (args.out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out / "per_scene.jsonl").write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"status": "complete", "posterior": summaries[0][scope_key], "ddim_metric_mean": draw_mean,
                      "mean_minus_posterior": delta, "seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for flag in ("checkpoint", "data", "replay", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=64)
    run(parser.parse_args())
