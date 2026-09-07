#!/usr/bin/env python3
"""Run the frozen Day 16 geometry-prompted SAM3 observer pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np
from PIL import Image, ImageDraw


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_x(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def run_text(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_file(path: Path, record: dict) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if "bytes" in record and path.stat().st_size != record["bytes"]:
        raise ValueError(f"Size mismatch: {path}")
    if sha256(path) != record["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")


def gpu_preflight(config: dict, output_root: Path, report_path: Path) -> dict:
    runtime = config["runtime"]
    gpu_index = str(runtime["physical_gpu"])
    query = run_text([
        "nvidia-smi", "-i", gpu_index,
        "--query-gpu=uuid,name,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    fields = [item.strip() for item in query.split(",")]
    if len(fields) != 5:
        raise RuntimeError(f"Unexpected nvidia-smi output: {query}")
    uuid, name, total, free, utilization = fields
    apps_text = run_text([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    apps = []
    for line in apps_text.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) >= 4 and parts[0] == uuid:
            owner = subprocess.run(
                ["ps", "-o", "user=,pid=,cmd=", "-p", parts[1]],
                check=False, text=True, capture_output=True,
            ).stdout.strip()
            apps.append({"pid": parts[1], "process": parts[2],
                         "used_memory_mib": parts[3], "owner_record": owner})
    available_kib = 0
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            available_kib = int(line.split()[1])
            break
    checks = {
        "output_root_absent": not output_root.exists(),
        "preflight_report_absent": not report_path.exists(),
        "gpu_name": name == runtime["required_gpu_name"],
        "gpu_free_memory": int(free) >= runtime["minimum_free_memory_mib"],
        "gpu_utilization": int(utilization) <= runtime["maximum_utilization_percent"],
        "zero_compute_processes": len(apps) == 0,
        "host_available_memory": available_kib // 1024 >= runtime["minimum_available_system_memory_mib"],
    }
    report = {
        "schema_version": "foodstateedit.geometry_prompted_sam3_preflight.v1",
        "timestamp_utc": utc_now(),
        "gpu": {"physical_index": int(gpu_index), "uuid": uuid, "name": name,
                "total_memory_mib": int(total), "free_memory_mib": int(free),
                "utilization_percent": int(utilization), "compute_processes": apps},
        "host_available_memory_mib": available_kib // 1024,
        "checks": checks,
        "passed": all(checks.values()),
    }
    write_json_x(report_path, report)
    if not report["passed"]:
        raise RuntimeError(f"Resource preflight failed: {checks}")
    return report


def read_video_frame(path: Path, index: int, rgb: bool = True) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"Could not decode frame {index}: {path}")
    if rgb:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def pixel_box(normalized: list[float], width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = normalized
    return [int(round(x0 * width)), int(round(y0 * height)),
            int(round(x1 * width)), int(round(y1 * height))]


def pixel_points(normalized: list[list[float]], width: int, height: int) -> list[list[float]]:
    return [[min(width - 1, max(0, x * width)), min(height - 1, max(0, y * height))]
            for x, y in normalized]


def make_counterfactuals(config: dict, images: dict[str, Image.Image], root: Path) -> dict[str, Image.Image]:
    fixtures = {}
    for name, spec in config["counterfactuals"].items():
        base = np.asarray(images[spec["base"]].convert("RGB")).copy()
        height, width = base.shape[:2]
        x0, y0, x1, y1 = pixel_box(spec["box_xyxy_normalized"], width, height)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"Invalid counterfactual box: {name}")
        if "replacement" in spec:
            replacement = np.asarray(images[spec["replacement"]].convert("RGB"))
            if replacement.shape != base.shape:
                raise ValueError(f"Counterfactual shape mismatch: {name}")
            base[y0:y1, x0:x1] = replacement[y0:y1, x0:x1]
        else:
            base[y0:y1, x0:x1] = np.asarray(spec["rgb"], dtype=np.uint8)
        image = Image.fromarray(base)
        image.save(root / f"{name}.png")
        fixtures[name] = image
    return fixtures


def stick_masks(geometry: dict, width: int, height: int) -> dict[str, np.ndarray]:
    anchors = geometry["anchors_normalized"]
    sticks = geometry["chopsticks"]
    pinch = np.asarray(anchors["final_pinch_uv"], dtype=np.float64) * [width, height]
    offset = np.asarray(sticks["handle_offset_normalized"], dtype=np.float64) * [width, height]
    tip_separation = sticks["tip_separation_normalized"] * height
    handle_separation = sticks["handle_separation_normalized"] * height
    masks = {}
    for name, side in (("near_chopstick", -1.0), ("far_chopstick", 1.0)):
        tip = pinch + [0.0, side * tip_separation / 2.0]
        handle = pinch + offset + [0.0, side * handle_separation / 2.0]
        canvas = Image.new("L", (width, height), 0)
        ImageDraw.Draw(canvas).line([tuple(tip), tuple(handle)], fill=255,
                                    width=int(sticks["line_width_px"]))
        masks[name] = np.asarray(canvas) > 0
    return masks


def overlap_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    intersection = int(np.logical_and(predicted, target).sum())
    union = int(np.logical_or(predicted, target).sum())
    predicted_pixels = int(predicted.sum())
    target_pixels = int(target.sum())
    return {
        "iou": intersection / union if union else 0.0,
        "precision": intersection / predicted_pixels if predicted_pixels else 0.0,
        "recall": intersection / target_pixels if target_pixels else 0.0,
        "intersection_pixels": intersection,
        "predicted_pixels": predicted_pixels,
        "target_pixels": target_pixels,
    }


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 0.0


def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return None
    return float(cols.mean()), float(rows.mean())


def pair_metrics(first: np.ndarray, second: np.ndarray) -> dict[str, float | None]:
    first_center, second_center = centroid(first), centroid(second)
    distance = 0.0
    if first_center is not None and second_center is not None:
        distance = float(np.linalg.norm(np.asarray(first_center) - np.asarray(second_center)))
    return {"mask_iou": mask_iou(first, second), "centroid_distance_px": distance,
            "first_centroid_xy": first_center, "second_centroid_xy": second_center}


def render_overlay(image: Image.Image, predicted: np.ndarray, target: np.ndarray,
                   box: list[int], positives: list[list[float]], negatives: list[list[float]]) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    base[target] = base[target] * 0.65 + np.asarray([45, 130, 245]) * 0.35
    base[predicted] = base[predicted] * 0.55 + np.asarray([35, 220, 90]) * 0.45
    rendered = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(rendered)
    draw.rectangle(tuple(box), outline=(255, 40, 40), width=2)
    for x, y in positives:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(30, 230, 70), outline="white")
    for x, y in negatives:
        draw.line((x - 4, y - 4, x + 4, y + 4), fill=(255, 40, 40), width=2)
        draw.line((x - 4, y + 4, x + 4, y - 4), fill=(255, 40, 40), width=2)
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    preflight = gpu_preflight(config, args.output_root, args.preflight_report)
    root = Path(config["remote_input_root"])
    input_paths = {}
    for record in config["inputs"]:
        path = root / record["path"]
        validate_file(path, record)
        input_paths[record["id"]] = path
    for name in ("geometry_control_video", "strand_mask_video"):
        record = config["derived_inputs"][name]
        validate_file(root / record["path"], record)
    validate_file(args.geometry_config, config["derived_inputs"]["geometry_config"])
    geometry = json.loads(args.geometry_config.read_text(encoding="utf-8"))

    sam = config["sam3"]
    code_root = Path(sam["code_root"])
    checkpoint = code_root / sam["checkpoint"]["path"]
    bpe = code_root / sam["bpe"]["path"]
    validate_file(checkpoint, sam["checkpoint"])
    validate_file(bpe, sam["bpe"])
    if sha256(code_root / "sam3/model_builder.py") != sam["model_builder_sha256"]:
        raise ValueError("SAM3 model builder hash mismatch")
    if sha256(code_root / "sam3/model/sam3_image_processor.py") != sam["image_processor_sha256"]:
        raise ValueError("SAM3 processor hash mismatch")
    if run_text(["git", "-C", str(code_root), "rev-parse", "HEAD"]) != sam["git_commit"]:
        raise ValueError("SAM3 source commit mismatch")

    args.output_root.mkdir(parents=True, exist_ok=False)
    for name in ("fixtures", "ground_truth", "masks", "overlays"):
        (args.output_root / name).mkdir()
    write_json_x(args.output_root / "pre_run_freeze.json", {
        "timestamp_utc": utc_now(), "config_sha256": sha256(args.config),
        "runner_sha256": sha256(Path(__file__)), "geometry_config_sha256": sha256(args.geometry_config),
        "config": config, "resource_preflight": preflight,
    })

    images = {case_id: Image.open(path).convert("RGB") for case_id, path in input_paths.items()}
    control_record = config["derived_inputs"]["geometry_control_video"]
    control = read_video_frame(root / control_record["path"], control_record["frame"])
    images["geometry_control"] = Image.fromarray(control)
    if len({image.size for image in images.values()}) != 1:
        raise ValueError("All images must have identical aligned dimensions")
    images.update(make_counterfactuals(config, images, args.output_root / "fixtures"))
    width, height = images["source"].size

    strand_record = config["derived_inputs"]["strand_mask_video"]
    strand_frame = read_video_frame(root / strand_record["path"], strand_record["frame"], rgb=False)
    strand_gray = cv2.cvtColor(strand_frame, cv2.COLOR_BGR2GRAY) if strand_frame.ndim == 3 else strand_frame
    ground_truth = {"strand": strand_gray > 127}
    ground_truth.update(stick_masks(geometry, width, height))
    for name, mask in ground_truth.items():
        Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(
            args.output_root / "ground_truth" / f"{name}.png")

    sys.path.insert(0, str(code_root))
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if torch.__version__ != config["runtime"]["torch_expected"]:
        raise RuntimeError(f"Torch version mismatch: {torch.__version__}")
    model = build_sam3_image_model(
        bpe_path=str(bpe), device="cuda", eval_mode=True, checkpoint_path=str(checkpoint),
        load_from_HF=False, enable_segmentation=True, enable_inst_interactivity=False,
        compile=False,
    )
    processor = Sam3Processor(model, device="cuda",
                              confidence_threshold=config["observer"]["confidence_threshold"])
    metrics = {}
    selected_masks = {}
    try:
        for case_id, image in images.items():
            metrics[case_id] = {}
            selected_masks[case_id] = {}
            state = processor.set_image(image)
            for prompt_id, spec in config["prompts_normalized"].items():
                box = pixel_box(spec["box_xyxy"], width, height)
                positives = pixel_points(spec["positive_points"], width, height)
                negatives = pixel_points(spec["negative_points"], width, height)
                processor.reset_all_prompts(state)
                output = processor.add_box_and_point_prompt(
                    box=box, points=positives + negatives,
                    point_labels=[1] * len(positives) + [0] * len(negatives), state=state,
                )
                scores = output["scores"].detach().cpu().tolist()
                raw_masks = output["masks"].detach().cpu().numpy()
                candidates = []
                for score, raw_mask in zip(scores, raw_masks):
                    mask = np.asarray(raw_mask).squeeze().astype(bool)
                    if int(mask.sum()) >= config["observer"]["minimum_mask_pixels"]:
                        candidates.append((float(score), mask))
                candidates.sort(key=lambda item: item[0], reverse=True)
                best_score, best_mask = candidates[0] if candidates else (
                    0.0, np.zeros((height, width), dtype=bool))
                selected_masks[case_id][prompt_id] = best_mask
                target = ground_truth[prompt_id]
                record = overlap_metrics(best_mask, target)
                record.update({"score": best_score, "candidate_count": len(candidates),
                               "box_xyxy": box, "positive_points_xy": positives,
                               "negative_points_xy": negatives})
                metrics[case_id][prompt_id] = record
                mask_path = args.output_root / "masks" / f"{case_id}__{prompt_id}.png"
                Image.fromarray((best_mask * 255).astype(np.uint8), mode="L").save(mask_path)
                render_overlay(image, best_mask, target, box, positives, negatives).save(
                    args.output_root / "overlays" / f"{case_id}__{prompt_id}.png")
            metrics[case_id]["stick_pair"] = pair_metrics(
                selected_masks[case_id]["near_chopstick"],
                selected_masks[case_id]["far_chopstick"],
            )
    finally:
        del processor
        del model
        torch.cuda.empty_cache()

    obs = config["observer"]
    def iou(case: str, prompt: str) -> float:
        return float(metrics[case][prompt]["iou"])
    def distinct(case: str) -> bool:
        pair = metrics[case]["stick_pair"]
        return (pair["mask_iou"] <= obs["maximum_pair_mask_iou"] and
                pair["centroid_distance_px"] >= obs["minimum_pair_centroid_distance_px"])
    control_stick = (iou("geometry_control", "near_chopstick") +
                     iou("geometry_control", "far_chopstick")) / 2.0
    erased_stick = (iou("sticks_erased_to_source", "near_chopstick") +
                    iou("sticks_erased_to_source", "far_chopstick")) / 2.0
    margin = obs["contrast_margin_iou"]
    gates = {
        "control_strand_aligns": iou("geometry_control", "strand") >= obs["control_strand_iou_threshold"],
        "control_near_stick_aligns": iou("geometry_control", "near_chopstick") >= obs["control_stick_iou_threshold"],
        "control_far_stick_aligns": iou("geometry_control", "far_chopstick") >= obs["control_stick_iou_threshold"],
        "control_sticks_are_distinct": distinct("geometry_control"),
        "control_strand_beats_source": iou("geometry_control", "strand") >= iou("source", "strand") + margin,
        "strand_erasure_lowers_alignment": iou("geometry_control", "strand") >= iou("strand_erased_to_source", "strand") + margin,
        "stick_erasure_lowers_alignment": control_stick >= erased_stick + margin,
        "solid_block_not_preferred": iou("geometry_control", "strand") >= iou("solid_strand_block", "strand") + margin,
        "weighted_output_has_all_three_structures": (
            iou("relative3d_topology_weighted_step32", "strand") >= obs["weighted_strand_iou_threshold"] and
            iou("relative3d_topology_weighted_step32", "near_chopstick") >= obs["weighted_stick_iou_threshold"] and
            iou("relative3d_topology_weighted_step32", "far_chopstick") >= obs["weighted_stick_iou_threshold"] and
            distinct("relative3d_topology_weighted_step32")
        ),
    }
    if set(gates) != set(config["required_gates"]):
        raise RuntimeError("Gate contract mismatch")
    passed = all(gates.values())
    result = {
        "schema_version": "foodstateedit.geometry_prompted_sam3_observer_result.v1",
        "timestamp_utc": utc_now(),
        "status": "automated_gate_pass_requires_independent_labels" if passed else "observer_gate_failed_do_not_guide_vace",
        "automated_gates": gates,
        "all_automated_gates_passed": passed,
        "metrics": metrics,
        "ground_truth_kind": "relative_3d_scaffold_derived_not_independent_annotation",
        "new_vace_inference_count": 0,
        "runtime": {"python": sys.executable, "torch": torch.__version__,
                    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
        "claim_limit": config["claim_limit"],
    }
    write_json_x(args.output_root / "result.json", result)
    write_json_x(args.output_root / "file_hashes.json", {
        str(path.relative_to(args.output_root)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(args.output_root.rglob("*"))
        if path.is_file() and path.name != "file_hashes.json"
    })
    print(json.dumps({"output_root": str(args.output_root), "status": result["status"],
                      "gates": gates,
                      "selected_iou": {case: {name: metrics[case][name]["iou"]
                                               for name in config["prompts_normalized"]}
                                       for case in metrics}}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
