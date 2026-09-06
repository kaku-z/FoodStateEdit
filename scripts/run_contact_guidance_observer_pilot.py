#!/usr/bin/env python3
"""Run CPU objective/observer falsification before any VACE guidance launch.

Mask optimization is an explicitly synthetic algorithm test, never a generated
video. RGB counterfactuals are measurement fixtures, never new image results.
All thresholds are read from a config hashed before any observation is scored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from foodstateedit.contact_guidance import (geometry_frame, image_tensor,
    rgb_observer, structure_energy, sample_curve)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def line_mask(size, points, width):
    canvas = Image.new("L", size)
    ImageDraw.Draw(canvas).line([tuple(x) for x in points], fill=255, width=width, joint="curve")
    return np.asarray(canvas).copy() > 0


def font(size=16):
    for name in ["C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def scalar_record(energy):
    return {key: bool(value) if key == "active" else float(value.detach())
            for key, value in energy.items()}


def synthetic_trial(settings, output_root):
    torch.manual_seed(settings["seed"])
    height, width = 96, 160
    xs = torch.linspace(15, 140, 100)
    ys = 62 - (xs-15) * .24 + torch.sin((xs-15)/125 * torch.pi) * 10
    uv = torch.stack([xs, ys], -1)
    tips = torch.tensor([[140., 28.], [140., 36.]])
    visible = torch.ones(100, dtype=torch.bool)
    visible[60:70] = False
    strand = line_mask((width, height), uv.numpy(), 5)
    hidden = line_mask((width, height), uv[60:70].numpy(), 7)
    # The finite-width known occluder covers more than its centerline indices.
    # Account for bilinear footprints from the known occluder raster, rather
    # than inferring visibility from the observed/missing strand probabilities.
    occluder = torch.as_tensor(hidden.copy(), dtype=torch.float32)[None, None]
    visible = sample_curve(occluder, uv)[0] < 1e-6
    broken = line_mask((width, height), uv[30:40].numpy(), 7)
    utensil = np.zeros((height, width), dtype=bool)
    for tip in tips.numpy():
        utensil |= line_mask((width, height), [tip, tip + [17, -10]], 3)
    complete = torch.as_tensor(np.where(strand & ~hidden, .98, .02), dtype=torch.float32)[None, None]
    tools = torch.as_tensor(np.where(utensil, .98, .02), dtype=torch.float32)[None, None]
    damaged = complete.clone()
    damaged[0, 0, broken] = .02
    missing_tool = torch.full_like(tools, .02)
    reference_e = structure_energy(complete, tools, uv, visible, tips, phase="lift")
    break_e = structure_energy(damaged, tools, uv, visible, tips, phase="lift")
    no_tool_e = structure_energy(complete, missing_tool, uv, visible, tips, phase="lift")
    no_occlusion_mask = complete.clone()
    no_occlusion_mask[0, 0, strand & hidden] = .98
    unhidden_e = structure_energy(no_occlusion_mask, tools, uv, visible, tips, phase="lift")
    approach_e = structure_energy(damaged, missing_tool, uv, visible, tips, phase="approach")
    metrics = {"reference": scalar_record(reference_e), "broken": scalar_record(break_e),
               "missing_utensil": scalar_record(no_tool_e), "unoccluded": scalar_record(unhidden_e),
               "approach": scalar_record(approach_e), "optimization": {}}
    boards = []
    for mode in ["none", "fixed", "phase_visibility"]:
        start = torch.cat([damaged, missing_tool], 1)
        start_logits = torch.logit(start.clamp(.001, .999))
        logits = start_logits.clone().requires_grad_()
        # Explicit Adam avoids torch.optim's optional Dynamo/SymPy dependency
        # on this local CPU runtime. Standard bias correction and epsilon.
        moment = torch.zeros_like(logits)
        variance = torch.zeros_like(logits)
        trace = []
        if mode != "none":
            for step in range(settings["steps"]):
                fields = logits.sigmoid()
                e = structure_energy(fields[:, :1], fields[:, 1:2], uv, visible, tips, phase="lift", mode=mode)
                objective = e["total"] + .005 * (logits-start_logits).square().mean()
                gradient, = torch.autograd.grad(objective, logits)
                with torch.no_grad():
                    moment.mul_(.9).add_(gradient, alpha=.1)
                    variance.mul_(.999).addcmul_(gradient, gradient, value=.001)
                    corrected_m = moment / (1 - .9**(step+1))
                    corrected_v = variance / (1 - .999**(step+1))
                    logits.addcdiv_(corrected_m, corrected_v.sqrt().add_(1e-8), value=-settings["learning_rate"])
                trace.append(float(objective.detach()))
        fields = logits.detach().sigmoid()
        result = scalar_record(structure_energy(fields[:, :1], fields[:, 1:2], uv, visible, tips, phase="lift"))
        result["hidden_region_predicted_noodle_mean"] = float(fields[0, 0][strand & hidden].mean())
        result["trace"] = trace
        metrics["optimization"][mode] = result
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[..., 1] = (fields[0, 0].numpy()*255).astype(np.uint8)
        panel[..., 0] = (fields[0, 1].numpy()*255).astype(np.uint8)
        boards.append((mode, Image.fromarray(panel).resize((480, 288))))
    board = Image.new("RGB", (1440, 350), "white")
    draw = ImageDraw.Draw(board)
    draw.text((10, 5), "SYNTHETIC MASK OPTIMIZATION ONLY - green: strand, red: utensil; not VACE output", font=font(18), fill="black")
    for index, (name, panel) in enumerate(boards):
        board.paste(panel, (index*480, 60))
        draw.text((index*480+8, 33), name, font=font(17), fill="black")
    board.save(output_root / "synthetic_mask_optimization.png")
    return metrics


def natural_trial(config, dataset, geometry, output_root):
    sample = "udon_chopsticks_imagegen_pseudo_v1.png"
    source = np.asarray(Image.open(dataset / "vace_reference_image" / sample).convert("RGB"))
    target = np.asarray(Image.open(dataset / "target_keyframe" / sample).convert("RGB"))
    height, width = source.shape[:2]
    with np.load(dataset / "relative3d_geometry" / "udon_relative3d_geometry.npz", allow_pickle=False) as arrays:
        uv, visible, tips, handles = geometry_frame(arrays, geometry, config["frame"], width, height)
    line = line_mask((width, height), uv.numpy(), geometry["strand"]["shadow_width_px"]+4)
    source_change = torch.as_tensor(np.abs(target.astype(np.float32)-source).mean(-1))[None, None]
    sampled_change = sample_curve(source_change, uv)[0]
    chopsticks = np.zeros((height, width), dtype=bool)
    for tip, handle in zip(tips.numpy(), handles.numpy()):
        chopsticks |= line_mask((width, height), [tip, handle], geometry["chopsticks"]["shadow_width_px"]+4)
    erase_strand = target.copy()
    erase_strand[line] = source[line]
    erase_tool = target.copy()
    erase_tool[chopsticks] = source[chopsticks]
    blob = target.copy()
    all_points = np.concatenate([uv.numpy(), tips.numpy()])
    x0, y0 = np.maximum(np.floor(all_points.min(0)-10).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(all_points.max(0)+10).astype(int), [width, height])
    blob[y0:y1, x0:x1] = [245, 224, 164]
    cases = {"no_edit": source, "synthetic_target": target,
             "strand_restored_to_source": erase_strand,
             "utensil_restored_to_source": erase_tool,
             "solid_color_blob": blob}
    metrics = {}
    for name, rgb in cases.items():
        tensor = image_tensor(rgb).requires_grad_()
        noodle, utensil = rgb_observer(tensor)
        energy = structure_energy(noodle, utensil, uv, visible, tips, phase="final_hold")
        energy["total"].backward()
        record = scalar_record(energy)
        record["gradient_finite"] = bool(torch.isfinite(tensor.grad).all())
        record["gradient_absolute_sum"] = float(tensor.grad.abs().sum())
        record["rgb_sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest()
        record["changed_pixels_vs_target"] = int(np.any(rgb != target, -1).sum())
        metrics[name] = record
    crop = (max(0, x0-75), max(0, y0-70), min(width, x1+65), min(height, y1+65))
    panel_w, panel_h = 440, 400
    board = Image.new("RGB", (panel_w*3, panel_h*2+40), "#eeeeee")
    draw = ImageDraw.Draw(board)
    draw.text((10, 6), "RGB OBSERVER FALSIFICATION - lower energy is preferred; counterfactuals are NOT generated results", font=font(17), fill="black")
    for index, (name, rgb) in enumerate(cases.items()):
        px, py = (index%3)*panel_w, (index//3)*panel_h+40
        draw.text((px+8, py+8), name, font=font(16), fill="black")
        score = metrics[name]
        draw.text((px+8, py+30), f"E={score['total']:.4f} connection={score['connection']:.4f} contact={score['contact']:.4f}", font=font(14), fill="black")
        panel = Image.fromarray(rgb).crop(crop)
        factor = min(420/panel.width, 340/panel.height)
        panel = panel.resize((round(panel.width*factor), round(panel.height*factor)), Image.Resampling.LANCZOS)
        board.paste(panel, (px+8, py+55))
    px, py = panel_w*2, panel_h+40
    overlay = Image.fromarray(target).copy()
    overdraw = ImageDraw.Draw(overlay)
    for a in range(len(uv)-1):
        overdraw.line([tuple(uv[a].numpy()), tuple(uv[a+1].numpy())], fill="lime" if visible[a] else "red", width=3)
    for tip in tips.numpy():
        x, y = tip
        overdraw.ellipse((x-4,y-4,x+4,y+4), fill="cyan")
    panel = overlay.crop(crop)
    factor = min(420/panel.width, 340/panel.height)
    panel = panel.resize((round(panel.width*factor), round(panel.height*factor)), Image.Resampling.LANCZOS)
    board.paste(panel, (px+8, py+55))
    draw.text((px+8, py+8), "Measured geometry over target", font=font(16), fill="black")
    draw.text((px+8, py+30), "green: visible / red: hidden / cyan: tips", font=font(14), fill="black")
    board.save(output_root / "rgb_observer_counterfactuals.png")
    return {"frame": config["frame"], "visible_samples": int(visible.sum()),
            "hidden_samples": int((~visible).sum()), "cases": metrics,
            "geometry_alignment_diagnostic": {
                "sampled_target_source_rgb_mae_mean": float(sampled_change[visible].mean()),
                "fraction_visible_path_with_target_source_mae_over_10": float((sampled_change[visible] > 10).float().mean()),
                "interpretation": "Change on a planned path is not proof of correct geometry; the overlay must be reviewed independently."},
            "observer_limit": "Analytic RGB likelihood; no validated semantic instance segmentation. Geometry positions are expected positions, not recovered generated endpoints."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/contact_guidance_observer_pilot_v1.json")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    dataset = ROOT / config["dataset_root"]
    geometry_path = ROOT / config["geometry_config"]
    if sha(dataset / "dataset_manifest.json") != config["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest hash mismatch")
    if sha(geometry_path) != config["geometry_sha256"]:
        raise ValueError("Geometry config hash mismatch")
    manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    for section in ["frozen_source_files", "derived_files"]:
        for record in manifest[section].values():
            if sha(dataset / record["path"]) != record["sha256"]:
                raise ValueError(f"Dataset byte mismatch: {record['path']}")
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    write_json(args.output_root / "pre_run_freeze.json", {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha(args.config),
        "runner_sha256": sha(Path(__file__)),
        "objective_sha256": sha(ROOT / "foodstateedit/contact_guidance.py"),
        "config": config,
    })
    synthetic = synthetic_trial(config["synthetic_mask_optimization"], args.output_root)
    natural = natural_trial(config, dataset, geometry, args.output_root)
    gap = config["minimum_gap"]
    reference = synthetic["reference"]
    cases = natural["cases"]
    target = cases["synthetic_target"]
    gates = {
        "broken_strand_has_more_energy_than_connected": synthetic["broken"]["connection"] > reference["connection"] + gap,
        "missing_utensil_has_more_energy_than_connected": synthetic["missing_utensil"]["contact"] > reference["contact"] + gap,
        "phase_guidance_inactive_during_approach": synthetic["approach"]["total"] == 0 and not synthetic["approach"]["active"],
        "known_occlusion_is_not_penalized_as_break": abs(synthetic["unoccluded"]["connection"] - reference["connection"]) < .01,
        "rgb_target_beats_no_edit": cases["no_edit"]["total"] > target["total"] + gap,
        "rgb_target_beats_same_color_solid_blob": cases["solid_color_blob"]["total"] > target["total"] + gap,
        "rgb_erasing_strand_increases_energy": cases["strand_restored_to_source"]["total"] > target["total"] + gap,
        "rgb_erasing_utensil_increases_contact_energy": cases["utensil_restored_to_source"]["contact"] > target["contact"] + gap,
        "rgb_objective_has_finite_nonzero_gradient": target["gradient_finite"] and target["gradient_absolute_sum"] > 0,
    }
    if set(gates) != set(config["required_gates"]):
        raise ValueError("Gate list differs from freeze")
    passed = all(gates.values())
    result = {
        "schema_version": "foodstateedit.contact_guidance_observer_result.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "observer_pass_requires_vace_integration" if passed else "observer_gate_failed_do_not_guide_vace",
        "gates": gates, "all_observer_gates_passed": passed,
        "synthetic_mask_experiment": synthetic, "rgb_observer_experiment": natural,
        "new_vace_inference_count": 0,
        "claim_limit": "CPU objective and measurement diagnostic only. No VACE-generated effectiveness, photo-realism, or generalization claim.",
    }
    write_json(args.output_root / "result.json", result)
    write_json(args.output_root / "file_hashes.json", {
        path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
        for path in sorted(args.output_root.iterdir()) if path.is_file()
    })
    print(json.dumps({"output_root": str(args.output_root), "status": result["status"],
                      "gates": gates, "synthetic": {key: value["total"] for key,value in synthetic["optimization"].items()},
                      "rgb": {key: value["total"] for key,value in cases.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
