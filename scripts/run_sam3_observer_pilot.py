#!/usr/bin/env python3
"""Run a frozen SAM3 observer falsification pilot without VACE inference."""
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

import numpy as np
from PIL import Image, ImageDraw, ImageFont


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_text(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip()


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
        raise RuntimeError(f"Unexpected nvidia-smi GPU query: {query}")
    uuid, name, total, free, utilization = fields
    apps_text = run_text([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    apps = []
    for line in apps_text.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) >= 4 and parts[0] == uuid:
            pid = parts[1]
            owner = subprocess.run(
                ["ps", "-o", "user=,pid=,cmd=", "-p", pid],
                check=False, text=True, capture_output=True,
            ).stdout.strip()
            apps.append({"gpu_uuid": parts[0], "pid": pid, "process": parts[2],
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
        "schema_version": "foodstateedit.sam3_observer_preflight.v1",
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


def validate_file(path: Path, record: dict) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if "bytes" in record and path.stat().st_size != record["bytes"]:
        raise ValueError(f"Size mismatch: {path}")
    if sha256(path) != record["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")


def font(size: int = 16):
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def make_counterfactuals(config: dict, images: dict[str, Image.Image], root: Path) -> dict[str, Image.Image]:
    fixtures = {}
    for name, spec in config["counterfactuals"].items():
        base = np.asarray(images[spec["base"]].convert("RGB")).copy()
        x0, y0, x1, y1 = spec["xyxy"]
        if not (0 <= x0 < x1 <= base.shape[1] and 0 <= y0 < y1 <= base.shape[0]):
            raise ValueError(f"Counterfactual bounds invalid: {name}")
        if "replacement" in spec:
            replacement = np.asarray(images[spec["replacement"]].convert("RGB"))
            if replacement.shape != base.shape:
                raise ValueError(f"Counterfactual shape mismatch: {name}")
            base[y0:y1, x0:x1] = replacement[y0:y1, x0:x1]
        else:
            base[y0:y1, x0:x1] = np.asarray(spec["rgb"], dtype=np.uint8)
        image = Image.fromarray(base)
        path = root / f"{name}.png"
        image.save(path)
        fixtures[name] = image
    return fixtures


def overlay(image: Image.Image, masks: list[np.ndarray], boxes: list[list[float]], scores: list[float]) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    colors = np.asarray([[30, 210, 95], [45, 135, 245], [245, 160, 35], [210, 45, 160]], dtype=np.float32)
    for index, mask in enumerate(masks):
        color = colors[index % len(colors)]
        base[mask] = base[mask] * 0.52 + color * 0.48
    result = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)
    for index, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        draw.rectangle((x0, y0, x1, y1), outline="red", width=3)
        score = scores[index] if index < len(scores) else 0.0
        draw.text((x0 + 3, max(0, y0 - 18)), f"{score:.3f}", fill="red", font=font(15))
    return result


def make_sheet(case_id: str, original: Image.Image, panels: list[tuple[str, Image.Image]], path: Path) -> None:
    thumb_w, thumb_h = 344, 288
    top = 34
    sheet = Image.new("RGB", (thumb_w * (len(panels) + 1), thumb_h + top), "white")
    draw = ImageDraw.Draw(sheet)
    all_panels = [("original", original)] + panels
    for index, (label, image) in enumerate(all_panels):
        fitted = image.copy()
        fitted.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = index * thumb_w + (thumb_w - fitted.width) // 2
        y = top + (thumb_h - fitted.height) // 2
        sheet.paste(fitted, (x, y))
        draw.text((index * thumb_w + 6, 7), label, fill="black", font=font(15))
    draw.text((6, top + 2), case_id, fill="white", stroke_width=2, stroke_fill="black", font=font(16))
    sheet.save(path)


def group_score(records: dict, prompt_ids: list[str]) -> float:
    return max([records[prompt_id]["max_score"] for prompt_id in prompt_ids] + [0.0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    preflight = gpu_preflight(config, args.output_root, args.preflight_report)
    input_root = Path(config["remote_input_root"])
    input_paths = {}
    for record in config["inputs"]:
        path = input_root / record["path"]
        validate_file(path, record)
        input_paths[record["id"]] = path

    sam = config["sam3"]
    code_root = Path(sam["code_root"])
    checkpoint = code_root / sam["checkpoint"]["path"]
    bpe = code_root / sam["bpe"]["path"]
    validate_file(checkpoint, sam["checkpoint"])
    validate_file(bpe, sam["bpe"])
    if sha256(code_root / "sam3/model_builder.py") != sam["model_builder_sha256"]:
        raise ValueError("SAM3 model_builder.py hash mismatch")
    if sha256(code_root / "sam3/model/sam3_image_processor.py") != sam["image_processor_sha256"]:
        raise ValueError("SAM3 image processor hash mismatch")
    commit = run_text(["git", "-C", str(code_root), "rev-parse", "HEAD"])
    if commit != sam["git_commit"]:
        raise ValueError("SAM3 git commit mismatch")

    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / "fixtures").mkdir()
    (args.output_root / "masks").mkdir()
    (args.output_root / "overlays").mkdir()
    (args.output_root / "sheets").mkdir()
    write_json_x(args.output_root / "pre_run_freeze.json", {
        "timestamp_utc": utc_now(),
        "config_sha256": sha256(args.config),
        "runner_sha256": sha256(Path(__file__)),
        "config": config,
        "resource_preflight": preflight,
    })

    images = {case_id: Image.open(path).convert("RGB") for case_id, path in input_paths.items()}
    images.update(make_counterfactuals(config, images, args.output_root / "fixtures"))

    sys.path.insert(0, str(code_root))
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if torch.__version__ != config["runtime"]["torch_expected"]:
        raise RuntimeError(f"Torch version mismatch: {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    model = build_sam3_image_model(
        bpe_path=str(bpe), device="cuda", eval_mode=True,
        checkpoint_path=str(checkpoint), load_from_HF=False,
        enable_segmentation=True, enable_inst_interactivity=False, compile=False,
    )
    processor = Sam3Processor(
        model, device="cuda",
        confidence_threshold=config["observer"]["confidence_threshold"],
    )

    prompts = []
    prompt_groups = {}
    for group, records in config["prompt_groups"].items():
        prompt_groups[group] = [record["id"] for record in records]
        prompts.extend(records)
    metrics = {}
    try:
        for case_id, image in images.items():
            state = processor.set_image(image)
            case_metrics = {}
            panels = []
            for prompt in prompts:
                processor.reset_all_prompts(state)
                output = processor.set_text_prompt(prompt=prompt["text"], state=state)
                raw_scores = [float(value) for value in output["scores"].detach().cpu().tolist()]
                raw_boxes = output["boxes"].detach().cpu().tolist()
                raw_masks = output["masks"].detach().cpu().numpy()
                masks = []
                scores = []
                boxes = []
                for index, raw in enumerate(raw_masks):
                    mask = np.asarray(raw).squeeze().astype(bool)
                    if int(mask.sum()) >= config["observer"]["minimum_mask_pixels"]:
                        masks.append(mask)
                        scores.append(raw_scores[index])
                        boxes.append(raw_boxes[index])
                if masks:
                    union = np.logical_or.reduce(masks)
                else:
                    union = np.zeros((image.height, image.width), dtype=bool)
                mask_path = args.output_root / "masks" / f"{case_id}__{prompt['id']}.png"
                Image.fromarray((union * 255).astype(np.uint8), mode="L").save(mask_path)
                rendered = overlay(image, masks, boxes, scores)
                overlay_path = args.output_root / "overlays" / f"{case_id}__{prompt['id']}.png"
                rendered.save(overlay_path)
                panels.append((f"{prompt['id']} max={max(scores, default=0.0):.3f}", rendered))
                case_metrics[prompt["id"]] = {
                    "text": prompt["text"],
                    "instance_count": len(masks),
                    "scores": scores,
                    "max_score": max(scores, default=0.0),
                    "union_pixels": int(union.sum()),
                    "union_fraction": float(union.mean()),
                    "boxes_xyxy": boxes,
                    "mask_sha256": sha256(mask_path),
                }
            case_metrics["group_scores"] = {
                group: group_score(case_metrics, ids) for group, ids in prompt_groups.items()
            }
            metrics[case_id] = case_metrics
            make_sheet(case_id, image, panels, args.output_root / "sheets" / f"{case_id}.png")
    finally:
        del processor
        del model
        torch.cuda.empty_cache()

    threshold = config["observer"]["positive_score_threshold"]
    margin = config["observer"]["contrast_margin"]
    def score(case: str, group: str) -> float:
        return metrics[case]["group_scores"][group]
    gates = {
        "target_has_strand_candidate": score("synthetic_target", "strand") >= threshold,
        "target_has_utensil_candidate": score("synthetic_target", "utensil") >= threshold,
        "target_strand_beats_source": score("synthetic_target", "strand") >= score("source", "strand") + margin,
        "target_utensil_beats_source": score("synthetic_target", "utensil") >= score("source", "utensil") + margin,
        "strand_erasure_lowers_strand_score": score("synthetic_target", "strand") >= score("strand_restored_to_source", "strand") + margin,
        "utensil_erasure_lowers_utensil_score": score("synthetic_target", "utensil") >= score("utensil_restored_to_source", "utensil") + margin,
        "solid_block_not_preferred_as_strand": score("synthetic_target", "strand") >= score("solid_noodle_color_block", "strand") + margin,
        "weighted_output_has_strand_candidate": score("relative3d_topology_weighted_step32", "strand") >= threshold,
        "weighted_output_has_utensil_candidate": score("relative3d_topology_weighted_step32", "utensil") >= threshold,
    }
    if set(gates) != set(config["required_gates"]):
        raise RuntimeError("Computed gates do not match frozen gate list")
    passed = all(gates.values())
    result = {
        "schema_version": "foodstateedit.sam3_observer_result.v1",
        "timestamp_utc": utc_now(),
        "status": "automated_gate_pass_pending_independent_manual_review" if passed else "observer_gate_failed_do_not_guide_vace",
        "automated_gates": gates,
        "all_automated_gates_passed": passed,
        "manual_review_complete": False,
        "metrics": metrics,
        "runtime": {"python": sys.executable, "torch": torch.__version__,
                    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
        "new_vace_inference_count": 0,
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
                      "group_scores": {case: record["group_scores"] for case, record in metrics.items()}},
                     indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
