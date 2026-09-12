#!/usr/bin/env python3
"""Build a hash-verified same-input cross-method pilot comparison grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
ANCHORS = (
    ("soup_spoon_001", "Soup + spoon", "soup_002.jpg", "6f633e7416d069c149fecfb37f5e67d7f4b568487788ee73cba75b4702d188e3", "92c064768c4b8f0dee95d92265cbdd48b95e97a39fa852594383532043b102f4"),
    ("fried_rice_spatula_001", "Fried rice + spatula", "rice_007.jpg", "60c6e0870bf35caf686db9bcdf636f2cdd302762d5902a4b5937214f5c5b88cc", "aa49b1b0e478e87c4951ec4e7abbc0b4fda7d4855681cfb420a6945eb38fc64f"),
    ("ramen_chopsticks_001", "Udon + chopsticks", "noodle_001.jpg", "44b4bc5461d6ac0d7a3a811859db4914e5cd0e29db14a66c12e2ec2a64ee89bf", "5009276c8f68a133a17a9694cfca304395d1afd9ac17b39e2a662016938a9011"),
    ("pasta_fork_001", "Pasta + fork", "pasta_006.jpg", "9338301493ad89273ce5e881e5e45dac48eb5eb541b3809de9ebddde47b9a8a9", "a778938b0c77594761a936786f1893ff843a6b15e72a63ff824d1231cf5b610a"),
)
COLUMNS = (
    ("input", "Input"),
    ("vanilla_geoedit", "Vanilla GeoEdit"),
    ("geoedit_unified_action_mask", "GeoEdit + union mask"),
    ("vace_direct_static_proxy", "VACE static"),
    ("vace_direct_dynamic_multikey", "Ours (dynamic 2-D)"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(path: Path, expected: str) -> Path:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Hash mismatch for {path}: {actual} != {expected}")
    return path


def working_size(width: int, height: int, long_side: int = 736) -> tuple[int, int]:
    scale = long_side / max(width, height)
    return (
        max(16, round(width * scale / 16) * 16),
        max(16, round(height * scale / 16) * 16),
    )


def prepare_input(source: Path, expected_source_sha256: str) -> Image.Image:
    verify(source, expected_source_sha256)
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(working_size(*image.size), Image.Resampling.LANCZOS)
    return image


def output_digest(manifest: dict[str, object], filename: str = "edited_2d.png") -> str:
    matches = [row for row in manifest["outputs"] if Path(row["path"]).name == filename]
    if len(matches) != 1:
        raise ValueError(f"Expected one {filename} output")
    return matches[0]["sha256"]


def find_manifest(pattern: str, anchor: str, method: str) -> Path:
    matches = []
    for path in ROOT.glob(pattern):
        data = load_json(path)
        if data.get("case_id") == anchor and data.get("method") == method:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"Expected one manifest for {anchor}/{method}, found {len(matches)}")
    return matches[0]


def method_image(anchor: str, method: str) -> tuple[Path, Path, str]:
    if method == "vanilla_geoedit":
        image = ROOT / "artifacts/day3_vanilla_geoedit_smoke_v1" / f"{anchor}_edited_2d.png"
        manifest = ROOT / "artifacts/day3_vanilla_geoedit_smoke_v1" / f"{anchor}_run_manifest.json"
    elif method == "geoedit_unified_action_mask":
        image = ROOT / "artifacts/day3_unified_geoedit_smoke_v1" / f"{anchor}_edited_2d.png"
        manifest = ROOT / "results/day3_unified_geoedit_smoke_v1/runs" / f"{anchor}.json"
    elif method == "vace_direct_static_proxy":
        manifest = find_manifest(
            "results/day5_vace_direct_v1/gpu*/**/run_manifest.json", anchor, method
        )
        image = ROOT / "artifacts" / manifest.relative_to(ROOT / "results")
        image = image.parent / "edited_2d.png"
    elif method == "vace_direct_dynamic_multikey":
        manifest = find_manifest(
            "results/day6_vace_dynamic_multikey_v1/gpu*/**/run_manifest.json",
            anchor,
            method,
        )
        image = ROOT / "artifacts" / manifest.relative_to(ROOT / "results")
        image = image.parent / "edited_2d.png"
    else:
        raise ValueError(f"Unsupported method: {method}")
    data = load_json(manifest)
    if data.get("status") != "complete" or data.get("seed") != 1:
        raise ValueError(f"Incomplete or non-matched run: {manifest}")
    if data["inference"]["decoded_frames"] != 21 or data["inference"]["steps"] != 20:
        raise ValueError(f"Inference budget mismatch: {manifest}")
    expected = output_digest(data)
    return verify(image, expected), manifest, expected


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        path = Path(name)
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def fit_panel(image: Image.Image, width: int, height: int) -> Image.Image:
    panel = image.convert("RGB")
    panel.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (246, 246, 246))
    canvas.paste(panel, ((width - panel.width) // 2, (height - panel.height) // 2))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/day21_cross_method_pilot_v1",
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse output root: {output_root}")

    panel_width, panel_height = 326, 248
    row_label_width, title_height, header_height, gutter = 205, 80, 52, 8
    width = row_label_width + len(COLUMNS) * (panel_width + gutter) + gutter
    height = title_height + header_height + len(ANCHORS) * (panel_height + gutter) + 44
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), "FoodStateEdit cross-method development pilot", font=font(28, True), fill=(22, 28, 40))
    draw.text((14, 48), "Same four inputs | seed 1 | 21 frames | 20 steps | internal non-blind review", font=font(18), fill=(76, 82, 94))
    records: list[dict[str, object]] = []

    for column_index, (_, label) in enumerate(COLUMNS):
        x = row_label_width + gutter + column_index * (panel_width + gutter)
        box = draw.textbbox((0, 0), label, font=font(18, True))
        draw.text((x + (panel_width - (box[2] - box[0])) / 2, title_height + 14), label, font=font(18, True), fill=(22, 28, 40))

    source_root = ROOT / "artifacts/day2_anchor_sources_x4_v1"
    for row_index, (anchor, label, source_name, source_hash, historical_input_png_hash) in enumerate(ANCHORS):
        y = title_height + header_height + row_index * (panel_height + gutter)
        input_image = prepare_input(source_root / source_name, source_hash)
        images: dict[str, Image.Image] = {"input": input_image}
        records.append({
            "anchor_id": anchor,
            "column": "input",
            "source_path": str((source_root / source_name).relative_to(ROOT)),
            "source_sha256": source_hash,
            "historical_first_frame_png_sha256": historical_input_png_hash,
            "locally_resized_rgb_sha256": hashlib.sha256(input_image.tobytes()).hexdigest(),
        })
        for method, _ in COLUMNS[1:]:
            path, manifest, digest = method_image(anchor, method)
            with Image.open(path) as loaded:
                images[method] = loaded.convert("RGB")
            records.append({"anchor_id": anchor, "column": method, "path": str(path.relative_to(ROOT)), "manifest": str(manifest.relative_to(ROOT)), "sha256": digest})
        draw.multiline_text((12, y + 78), label.replace(" + ", "\n+ "), font=font(19, True), fill=(22, 28, 40), spacing=6)
        draw.text((12, y + 139), anchor, font=font(13), fill=(92, 98, 108))
        for column_index, (key, _) in enumerate(COLUMNS):
            x = row_label_width + gutter + column_index * (panel_width + gutter)
            canvas.paste(fit_panel(images[key], panel_width, panel_height), (x, y))

    draw.text((14, height - 32), "Development evidence only: 4 selected anchors, one seed, no independent blinded ratings; not a foundation-model leaderboard.", font=font(16), fill=(122, 40, 40))
    output_root.mkdir(parents=True, exist_ok=False)
    figure = output_root / "cross_method_pilot_grid.png"
    canvas.save(figure, optimize=True)
    manifest = {
        "schema_version": "foodstateedit.cross_method_pilot_figure.v1",
        "status": "complete_hash_verified_development_figure",
        "rows": len(ANCHORS),
        "columns": [key for key, _ in COLUMNS],
        "matched_factors": {"inputs": 4, "seed": 1, "frames": 21, "steps": 20},
        "review_scope": "internal_non_blind_development_pilot_not_foundation_model_leaderboard",
        "records": records,
        "output": {
            "path": str(figure.relative_to(ROOT)),
            "sha256": sha256_file(figure),
            "width": canvas.width,
            "height": canvas.height,
        },
    }
    (output_root / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["output"], indent=2))


if __name__ == "__main__":
    main()
