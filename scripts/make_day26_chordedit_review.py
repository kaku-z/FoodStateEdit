#!/usr/bin/env python3
"""Build a hash-verified review grid for the Day 26 ChordEdit baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CASES = ("ramen", "soup", "rice", "cake")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def fit_panel(path: Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (width, height), (244, 244, 244))
    panel.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=ROOT / "artifacts/day26_chordedit_direct_baseline_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/day26_chordedit_direct_baseline_v1/review_grid.png",
    )
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite review figure: {output}")

    panel_width, panel_height = 326, 248
    row_label_width, title_height, header_height, gap = 150, 78, 48, 8
    columns = (("input", "Input"), ("seed_1", "Seed 1"), ("seed_2", "Seed 2"), ("seed_3", "Seed 3"))
    width = row_label_width + gap + len(columns) * (panel_width + gap)
    height = title_height + header_height + len(CASES) * (panel_height + gap) + 42
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), "Day 26: ChordEdit direct-input baseline", font=font(28, True), fill=(22, 28, 40))
    draw.text((14, 47), "Same input and edit support | three fixed seeds | internal non-blind review", font=font(17), fill=(76, 82, 94))

    for index, (_, title) in enumerate(columns):
        x = row_label_width + gap + index * (panel_width + gap)
        bbox = draw.textbbox((0, 0), title, font=font(18, True))
        draw.text((x + (panel_width - bbox[2] + bbox[0]) / 2, title_height + 12), title, font=font(18, True), fill=(22, 28, 40))

    records: list[dict[str, object]] = []
    for row, case_id in enumerate(CASES):
        case_root = artifact_root / case_id
        manifest_path = case_root / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete_requires_blind_review" or manifest.get("pipeline_load_count") != 1:
            raise ValueError(f"Unexpected run state: {manifest_path}")
        expected = {
            (int(seed["seed"]), filename): data["sha256"]
            for seed in manifest["completed_seeds"]
            for filename, data in seed["files"].items()
        }
        paths = [case_root / "seed_1/input.png"] + [case_root / f"seed_{seed}/final.png" for seed in (1, 2, 3)]
        for index, path in enumerate(paths):
            seed = 1 if index == 0 else index
            filename = "input.png" if index == 0 else "final.png"
            actual = sha256_file(path)
            if actual != expected[(seed, filename)]:
                raise ValueError(f"Hash mismatch: {path}")
            x = row_label_width + gap + index * (panel_width + gap)
            y = title_height + header_height + row * (panel_height + gap)
            canvas.paste(fit_panel(path, panel_width, panel_height), (x, y))
            records.append({"case_id": case_id, "column": columns[index][0], "path": str(path.relative_to(ROOT)), "sha256": actual})
        y = title_height + header_height + row * (panel_height + gap)
        draw.text((14, y + 105), case_id.upper(), font=font(20, True), fill=(22, 28, 40))

    draw.text((14, height - 30), "Development evidence only; cake is synthetic; outputs require task/realism scoring.", font=font(16), fill=(122, 40, 40))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    review_manifest = {
        "schema_version": "foodstateedit.day26_chordedit_review.v1",
        "status": "complete_hash_verified_requires_human_review",
        "figure": {"path": str(output.relative_to(ROOT)), "sha256": sha256_file(output), "width": width, "height": height},
        "records": records,
    }
    manifest_output = output.with_name("review_grid_manifest.json")
    manifest_output.write_text(json.dumps(review_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(review_manifest["figure"], indent=2))


if __name__ == "__main__":
    main()
