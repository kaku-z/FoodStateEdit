#!/usr/bin/env python3
"""Build a hash-traceable four-anchor method comparison grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ANCHORS = (
    ("soup_spoon_001", "Soup + spoon"),
    ("fried_rice_spatula_001", "Fried rice + spatula"),
    ("ramen_chopsticks_001", "Udon + chopsticks"),
    ("pasta_fork_001", "Pasta + fork"),
)
COLUMNS = (
    ("source", "Source"),
    ("proxy", "Geometry proxy"),
    ("unified", "Unified GeoEdit"),
    ("exclusive", "Staged exclusive"),
    ("geometry", "Geometry lock"),
    ("material", "Material render"),
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
    if sha256_file(path) != expected:
        raise ValueError(f"Hash mismatch: {path}")
    return path


def proxy_images(root: Path, anchor: str) -> dict[str, Path]:
    case_root = root / anchor
    manifest = load_json(case_root / "proxy_manifest.json")
    if manifest["anchor_id"] != anchor:
        raise ValueError(f"Proxy identity mismatch: {case_root}")
    return {
        "source": verify(
            case_root / "first_frame.png", manifest["files"]["first_frame.png"]["sha256"]
        ),
        "proxy": verify(
            case_root / "motion_signal.png",
            manifest["files"]["motion_signal.png"]["sha256"],
        ),
    }


def stochastic_image(roots: list[Path], anchor: str, method: str) -> Path:
    matches = []
    for root in roots:
        manifest_path = root / anchor / "seed_1" / "run_manifest.json"
        if manifest_path.is_file():
            manifest = load_json(manifest_path)
            if manifest.get("method") == method and manifest.get("status") == "complete":
                output = next(
                    item for item in manifest["outputs"]
                    if Path(item["path"]).name == "edited_2d.png"
                )
                matches.append(verify(Path(output["path"]), output["sha256"]))
    if len(matches) != 1:
        raise ValueError(f"Expected one {method} image for {anchor}, found {len(matches)}")
    return matches[0]


def deterministic_image(root: Path, anchor: str, method: str) -> Path:
    case_root = root / anchor
    manifest = load_json(case_root / "run_manifest.json")
    if manifest.get("method") != method or manifest.get("status") != "complete":
        raise ValueError(f"Deterministic result mismatch: {case_root}")
    path = case_root / "edited_2d.png"
    return verify(path, manifest["outputs"]["edited_2d.png"])


def font(size: int, bold: bool = False):
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def fit_panel(path: Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as image:
        panel = image.convert("RGB")
    panel.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    canvas.paste(panel, ((width - panel.width) // 2, (height - panel.height) // 2))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--unified-root", type=Path, required=True)
    parser.add_argument("--exclusive-root", action="append", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--material-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to reuse output root: {args.output_root}")

    panel_width, panel_height = 368, 280
    row_label_width, header_height = 220, 58
    gutter = 8
    canvas_width = row_label_width + len(COLUMNS) * (panel_width + gutter) + gutter
    canvas_height = header_height + len(ANCHORS) * (panel_height + gutter) + gutter
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = font(22, bold=True)
    row_font = font(20, bold=True)
    small_font = font(15)
    records = []

    for column_index, (_, label) in enumerate(COLUMNS):
        x = row_label_width + gutter + column_index * (panel_width + gutter)
        box = draw.textbbox((0, 0), label, font=header_font)
        draw.text(
            (x + (panel_width - (box[2] - box[0])) / 2, 17),
            label,
            fill=(26, 32, 44),
            font=header_font,
        )

    for row_index, (anchor, row_label) in enumerate(ANCHORS):
        y = header_height + gutter + row_index * (panel_height + gutter)
        proxy = proxy_images(args.proxy_root.resolve(), anchor)
        paths = {
            **proxy,
            "unified": stochastic_image(
                [args.unified_root.resolve()], anchor, "geoedit_unified_action_mask"
            ),
            "exclusive": stochastic_image(
                [root.resolve() for root in args.exclusive_root],
                anchor,
                "foodstateedit_staged_exclusive",
            ),
            "geometry": deterministic_image(
                args.geometry_root.resolve(), anchor, "foodstateedit_geometry_lock"
            ),
            "material": deterministic_image(
                args.material_root.resolve(), anchor, "foodstateedit_material_render"
            ),
        }
        draw.multiline_text(
            (14, y + 88),
            row_label.replace(" + ", "\n+ "),
            fill=(26, 32, 44),
            font=row_font,
            spacing=7,
        )
        draw.text((14, y + 154), anchor, fill=(90, 96, 108), font=small_font)
        for column_index, (key, _) in enumerate(COLUMNS):
            x = row_label_width + gutter + column_index * (panel_width + gutter)
            canvas.paste(fit_panel(paths[key], panel_width, panel_height), (x, y))
            records.append({
                "anchor_id": anchor,
                "column": key,
                "path": str(paths[key].resolve()),
                "sha256": sha256_file(paths[key]),
            })

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    output_path = output_root / "day5_method_grid.png"
    canvas.save(output_path, optimize=True)
    manifest = {
        "schema_version": "foodstateedit.paper_figure.v1",
        "figure": "day5_method_grid",
        "code_commit": args.code_commit,
        "rows": len(ANCHORS),
        "columns": len(COLUMNS),
        "sources": records,
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "width": canvas.width,
            "height": canvas.height,
        },
        "status": "complete",
    }
    (output_root / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["output"], indent=2))


if __name__ == "__main__":
    main()
