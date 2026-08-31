#!/usr/bin/env python3
"""Build the tracked Day 8 planar-versus-3-D fork VACE review package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


FRAME_INDICES = [0, 5, 8, 12, 16, 18, 20]
DAY6_RAW_SHA256 = "be99b543bed78111f5c28994f65e250aee2a5f90d29517b0c670549366628afe"
DAY6_EDITED_SHA256 = "8d9fd19d034e51e6afba8a66f7a5bc7d2a73752c7810111504228c1cb1950f63"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def labelled_panel(path: Path, label: str, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        resized = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (size[0], size[1] + 30), "white")
    panel.paste(resized, (0, 30))
    ImageDraw.Draw(panel).text((8, 8), label, fill="black")
    return panel


def main() -> None:
    args = parse_args()
    artifact_root = args.artifact_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse review output: {output_dir}")

    run_manifest_path = artifact_root / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("status") != "complete_requires_separate_action_and_photo_review":
        raise ValueError("The Day 8 VACE run is not technically complete")
    if run_manifest["inference"].get("pipeline_load_count") != 1:
        raise ValueError("Expected exactly one pipeline load")
    if run_manifest["inference"].get("decoded_frames") != 21:
        raise ValueError("Expected exactly 21 decoded frames")
    if run_manifest["inference"].get("outside_motion_support_max_pixel_difference") != 0:
        raise ValueError("Exact scene protection did not pass")
    for record in run_manifest["outputs"]:
        local = artifact_root / Path(record["path"]).name
        if not local.is_file() or sha256_file(local) != record["sha256"]:
            raise ValueError(f"Pulled output mismatch: {local}")
    if sha256_file(artifact_root / "preflight.json") != run_manifest["preflight_report_sha256"]:
        raise ValueError("Passing preflight report hash mismatch")
    if sha256_file(artifact_root / "day6_planar_raw_selected_frame.png") != DAY6_RAW_SHA256:
        raise ValueError("Day 6 raw selected frame hash mismatch")
    if sha256_file(artifact_root / "day6_planar_edited_2d.png") != DAY6_EDITED_SHA256:
        raise ValueError("Day 6 exact-projection frame hash mismatch")

    capture = cv2.VideoCapture(str(artifact_root / "result.mp4"))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) != 21:
        raise ValueError(f"Review decoder found {len(frames)} frames instead of 21")

    output_dir.mkdir(parents=True, exist_ok=False)
    copied_names = [
        "run_manifest.json",
        "preflight.json",
        "command.json",
        "stdout.log",
        "stderr.log",
        "raw_selected_frame.png",
        "edited_2d.png",
        "day6_planar_raw_selected_frame.png",
        "day6_planar_edited_2d.png",
    ]
    for name in copied_names:
        shutil.copy2(artifact_root / name, output_dir / name)

    width = 368
    height = 272
    board = Image.new("RGB", (width * 4, 30 + height + 30 + height), "white")
    top_panels = [
        labelled_panel(artifact_root / "day6_planar_raw_selected_frame.png", "Day 6 planar: raw frame 18", (width, height)),
        labelled_panel(artifact_root / "raw_selected_frame.png", "Day 8 3-D: raw frame 18", (width, height)),
        labelled_panel(artifact_root / "day6_planar_edited_2d.png", "Day 6 planar: exact projection", (width, height)),
        labelled_panel(artifact_root / "edited_2d.png", "Day 8 3-D: exact projection", (width, height)),
    ]
    for column, panel in enumerate(top_panels):
        board.paste(panel, (column * width, 0))

    temporal_width = 210
    temporal_height = 155
    temporal_y = height + 30
    draw = ImageDraw.Draw(board)
    for column, index in enumerate(FRAME_INDICES):
        frame = Image.fromarray(frames[index]).resize((temporal_width, temporal_height), Image.Resampling.LANCZOS)
        x = column * temporal_width
        draw.text((x + 6, temporal_y + 7), f"Day 8 frame {index}", fill="black")
        board.paste(frame, (x, temporal_y + 30))
    review_path = output_dir / "comparison_review.png"
    board.save(review_path)

    files = {}
    for path in sorted(output_dir.iterdir()):
        if path.name == "evidence_manifest.json":
            continue
        files[path.name] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    evidence = {
        "schema_version": "foodstateedit.day8_fork_3d_vace_evidence.v0",
        "method": run_manifest["method"],
        "scientific_status": run_manifest["scientific_status"],
        "technical_status": "complete_hash_verified_requires_human_visual_review",
        "source_video": {
            "size_bytes": (artifact_root / "result.mp4").stat().st_size,
            "sha256": sha256_file(artifact_root / "result.mp4"),
            "tracked_in_git": False,
        },
        "review_frame_indices": FRAME_INDICES,
        "pipeline_load_count": run_manifest["inference"]["pipeline_load_count"],
        "decoded_frames": run_manifest["inference"]["decoded_frames"],
        "outside_motion_support_max_pixel_difference": run_manifest["inference"]["outside_motion_support_max_pixel_difference"],
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "files": files,
    }
    (output_dir / "evidence_manifest.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
