"""Freeze the paired E2 RGB-proxy versus full-frame-Canny controls."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ARMS = ["appearance_laden_rgb", "fullframe_canny"]


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


def fullframe_canny(frames: np.ndarray, blur_kernel: int, low: int, high: int) -> np.ndarray:
    result = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        edges = cv2.Canny(blurred, low, high)
        result.append(np.repeat(edges[:, :, None], 3, axis=2))
    return np.stack(result).astype(np.uint8)


def make_review_sheet(frames: np.ndarray, label: str, indices: list[int]) -> Image.Image:
    height, width = frames.shape[1:3]
    sheet = Image.new("RGB", (width * 3, (height + 28) * 2), "white")
    draw = ImageDraw.Draw(sheet)
    for position, frame_index in enumerate(indices):
        left = (position % 3) * width
        top = (position // 3) * (height + 28)
        sheet.paste(Image.fromarray(frames[frame_index]), (left, top + 28))
        draw.text((left + 5, top + 5), f"{label} / f{frame_index}", fill="black")
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "foodstateedit.e2_control_representation_paired.v1":
        raise ValueError("Unexpected E2 config")
    if [arm["id"] for arm in config["arms"]] != EXPECTED_ARMS:
        raise ValueError("Frozen E2 arm order changed")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    source_root = ROOT / config["source_controls"]["path"]
    source_manifest_path = source_root / "manifest.json"
    if sha256(source_manifest_path) != config["source_controls"]["manifest_sha256"]:
        raise ValueError("Frozen E1 control manifest hash mismatch")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))

    output.mkdir(parents=True)
    manifest = {
        "schema_version": "foodstateedit.e2_control_representation_controls.v1",
        "status": "controls_only_require_review_before_inference",
        "config_sha256": sha256(args.config),
        "builder_sha256": sha256(Path(__file__)),
        "source_manifest_sha256": sha256(source_manifest_path),
        "config": config,
        "cases": [],
    }
    builder_cfg = config["control_builder"]
    blur_kernel = int(builder_cfg["gaussian_blur_kernel"])
    if blur_kernel <= 0 or blur_kernel % 2 != 1:
        raise ValueError("gaussian_blur_kernel must be positive and odd")

    for case_id in config["cases"]:
        source_case = next(item for item in source_manifest["cases"] if item["case_id"] == case_id)
        source_dir = source_root / case_id
        for record in source_case["files"].values():
            path = source_root / record["path"]
            if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
                raise ValueError(f"Frozen E1 file mismatch: {record['path']}")
        arrays = np.load(source_dir / "controls.npz", allow_pickle=False)
        appearance = arrays["appearance_laden_rgb"]
        canny = fullframe_canny(
            appearance,
            blur_kernel,
            int(builder_cfg["canny_low"]),
            int(builder_cfg["canny_high"]),
        )
        if appearance.shape != canny.shape or appearance.dtype != np.uint8:
            raise ValueError(f"Invalid controls: {case_id}")
        edge_fraction = [float(np.any(frame > 0, axis=2).mean()) for frame in canny]
        if min(edge_fraction) <= 0 or max(edge_fraction) >= 0.35:
            raise ValueError(f"Unexpected Canny density: {case_id}: {edge_fraction}")

        case_root = output / case_id
        case_root.mkdir()
        shutil.copyfile(source_dir / "reference.png", case_root / "reference.png")
        shutil.copyfile(source_dir / "shared_alpha.png", case_root / "shared_alpha.png")
        shutil.copyfile(source_dir / "geometry.json", case_root / "geometry.json")
        controls = {
            "appearance_laden_rgb": appearance,
            "fullframe_canny": canny,
        }
        np.savez_compressed(case_root / "controls.npz", **controls)
        for arm, frames in controls.items():
            Image.fromarray(frames[-1]).save(case_root / f"{arm}_final.png")
            make_review_sheet(
                frames,
                f"{case_id} / {arm}",
                config["inference"]["review_frame_indices"],
            ).save(case_root / f"{arm}_review.png")
        manifest["cases"].append({
            "case_id": case_id,
            "width": int(appearance.shape[2]),
            "height": int(appearance.shape[1]),
            "prompt": source_case["prompt"],
            "negative_prompt": source_case["negative_prompt"],
            "source_sha256": source_case["source_sha256"],
            "support_fraction": source_case["support_fraction"],
            "reference_policy": {arm: ["reference.png"] for arm in EXPECTED_ARMS},
            "canny_edge_fraction_by_frame": edge_fraction,
            "files": {
                path.name: file_record(path, output)
                for path in sorted(case_root.iterdir()) if path.is_file()
            },
        })
    write_json(output / "manifest.json", manifest)
    print(json.dumps({
        "output_root": str(output),
        "manifest_sha256": sha256(output / "manifest.json"),
        "cases": [item["case_id"] for item in manifest["cases"]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
