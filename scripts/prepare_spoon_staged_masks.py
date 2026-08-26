"""Build the material mask used by the staged spoon-scooping ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True, type=Path)
    return parser.parse_args()


def load_binary(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 127


def main() -> None:
    args = parse_args()
    case_dir = args.case_dir.resolve()
    output = case_dir / "mask_material.png"
    stats_path = case_dir / "staged_mask_stats.json"
    if output.exists() or stats_path.exists():
        raise FileExistsError("refusing to overwrite staged-mask outputs")

    action = load_binary(case_dir / "mask.png")
    rigid = load_binary(case_dir / "mask_utensil.png")
    payload = load_binary(case_dir / "mask_spoon_payload.png")
    target_garnish = load_binary(case_dir / "mask_garnish_target.png")
    source_hole = load_binary(case_dir / "mask_moved_garnish_removal.png")
    material = payload | target_garnish | source_hole
    staged_union = rigid | material

    Image.fromarray(material.astype(np.uint8) * 255, mode="L").save(output)
    stats = {
        "schema_version": "foodstateedit.staged_mask_stats.v0.1",
        "rigid_mask": "mask_utensil.png",
        "material_mask": "mask_material.png",
        "material_components": [
            "mask_spoon_payload.png",
            "mask_garnish_target.png",
            "mask_moved_garnish_removal.png",
        ],
        "action_area_px": int(action.sum()),
        "rigid_area_px": int(rigid.sum()),
        "material_area_px": int(material.sum()),
        "staged_union_area_px": int(staged_union.sum()),
        "action_minus_staged_union_px": int((action & ~staged_union).sum()),
        "staged_union_minus_action_px": int((staged_union & ~action).sum()),
        "image_area_px": int(action.size),
    }
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
