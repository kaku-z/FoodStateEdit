"""Freeze E1 controls that factor coarse action structure from source appearance."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_multimaterial_pilot as base


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def file_record(path: Path, root: Path) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def render_case(source: np.ndarray, case_id: str):
    if case_id == "noodle":
        return base.render_noodle(source, "relative3d")
    return base.render_solid_case(source, case_id, "relative3d")


def structure_scribble_frames(source: np.ndarray, appearance: np.ndarray, alpha: np.ndarray, cfg: dict) -> np.ndarray:
    """Create a black/white structural diagnostic while keeping inactive pixels identical."""
    result = []
    active = alpha > 0
    blur_k = int(cfg["structure_blur_kernel"])
    edge_k = int(cfg["structure_edge_dilation"])
    kernel = np.ones((edge_k, edge_k), np.uint8)
    for index, frame in enumerate(appearance):
        if index == 0:
            result.append(source.copy())
            continue
        changed = np.any(frame != source, axis=2).astype(np.uint8) * 255
        changed = cv2.dilate(changed, np.ones((5, 5), np.uint8), iterations=1)
        smooth = cv2.GaussianBlur(frame, (blur_k, blur_k), 0)
        edges = cv2.Canny(smooth, int(cfg["structure_canny_low"]), int(cfg["structure_canny_high"]))
        edges = cv2.bitwise_and(edges, changed)
        edges = cv2.dilate(edges, kernel, iterations=1) > 0
        outline = cv2.morphologyEx(changed, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
        edges = (edges | outline) & active
        control = source.copy()
        control[active] = 0
        control[edges] = 255
        result.append(control)
    return np.stack(result)


def payload_reference(source: np.ndarray, polygon_normalized, background_rgb) -> np.ndarray:
    h, w = source.shape[:2]
    polygon = np.asarray(polygon_normalized, dtype=np.float32) * np.asarray([w, h], dtype=np.float32)
    x, y, bw, bh = cv2.boundingRect(np.rint(polygon).astype(np.int32))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + bw), min(h, y + bh)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Empty payload reference crop")
    crop = source[y0:y1, x0:x1]
    scale = min((0.62 * w) / crop.shape[1], (0.62 * h) / crop.shape[0])
    nw, nh = max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.empty_like(source)
    canvas[:] = np.asarray(background_rgb, dtype=np.uint8)
    left, top = (w - nw) // 2, (h - nh) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def make_review_sheet(frames: np.ndarray, labels, indices) -> Image.Image:
    h, w = frames.shape[1:3]
    sheet = Image.new("RGB", (w * 3, (h + 28) * 2), "white")
    draw = ImageDraw.Draw(sheet)
    for k, index in enumerate(indices):
        x, y = (k % 3) * w, (k // 3) * (h + 28)
        sheet.paste(Image.fromarray(frames[index]), (x, y + 28))
        draw.text((x + 5, y + 5), f"{labels} / f{index}", fill="black")
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "foodstateedit.e1_control_factorization.v1":
        raise ValueError("Unexpected E1 config")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    dataset = ROOT / "artifacts/day19_multimaterial_dataset_v3"
    source_manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    arms = [arm["id"] for arm in config["arms"]]
    if arms != ["appearance_laden_rgb", "structure_scribble", "structure_scribble_payload_reference"]:
        raise ValueError("Frozen E1 arm order changed")
    manifest = {
        "schema_version": "foodstateedit.e1_control_factorization_controls.v1",
        "status": "controls_only_require_review_before_inference",
        "config_sha256": sha256(args.config),
        "builder_sha256": sha256(Path(__file__)),
        "config": config,
        "cases": [],
    }
    build_cfg = config["control_builder"]
    for case_spec in config["cases"]:
        case_id = case_spec["id"]
        case_root = output / case_id
        case_root.mkdir()
        source_path = dataset / case_id / "reference.png"
        source_record = next(item for item in source_manifest["cases"] if item["case_id"] == case_id)
        expected_source = source_record["files"]["reference.png"]["sha256"]
        if sha256(source_path) != expected_source:
            raise ValueError(f"Source hash mismatch: {case_id}")
        source = np.asarray(Image.open(source_path).convert("RGB"))
        appearance_frames, geometry = render_case(source, case_id)
        appearance = np.stack(appearance_frames)
        if appearance.shape != (21, *source.shape):
            raise ValueError(f"Unexpected control shape: {case_id}: {appearance.shape}")
        if not np.array_equal(appearance[0], source):
            raise ValueError(f"Frame zero is not the source image: {case_id}")
        union = np.any(appearance != source, axis=3).any(axis=0)
        window = int(build_cfg["support_dilation_window"])
        if window % 2 != 1:
            raise ValueError("Support dilation window must be odd")
        alpha = np.asarray(Image.fromarray(union.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(window)))
        structure = structure_scribble_frames(source, appearance, alpha, build_cfg)
        payload = payload_reference(source, case_spec["payload_polygon_normalized"], build_cfg["payload_reference_background_rgb"])
        controls = {
            "appearance_laden_rgb": appearance,
            "structure_scribble": structure,
            "structure_scribble_payload_reference": structure.copy(),
        }
        Image.fromarray(source).save(case_root / "reference.png")
        Image.fromarray(payload).save(case_root / "payload_reference.png")
        Image.fromarray(alpha).save(case_root / "shared_alpha.png")
        np.savez_compressed(case_root / "controls.npz", **controls)
        write_json(case_root / "geometry.json", geometry)
        for arm, frames in controls.items():
            if np.any(frames[:, alpha == 0] != source[alpha == 0]):
                raise ValueError(f"Control changes outside shared support: {case_id}/{arm}")
            Image.fromarray(frames[-1]).save(case_root / f"{arm}_final.png")
            make_review_sheet(frames, f"{case_id} / {arm}", config["inference"]["review_frame_indices"]).save(case_root / f"{arm}_review.png")
        manifest["cases"].append({
            "case_id": case_id,
            "width": source.shape[1],
            "height": source.shape[0],
            "prompt": source_record["prompt"],
            "negative_prompt": source_record["negative_prompt"],
            "source_sha256": expected_source,
            "support_fraction": float((alpha > 0).mean()),
            "reference_policy": {
                "appearance_laden_rgb": ["reference.png"],
                "structure_scribble": ["reference.png"],
                "structure_scribble_payload_reference": ["reference.png", "payload_reference.png"],
            },
            "files": {path.name: file_record(path, output) for path in sorted(case_root.iterdir()) if path.is_file()},
        })
    write_json(output / "manifest.json", manifest)
    print(json.dumps({
        "output_root": str(output),
        "manifest_sha256": sha256(output / "manifest.json"),
        "cases": [{"case_id": c["case_id"], "support_fraction": c["support_fraction"]} for c in manifest["cases"]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
