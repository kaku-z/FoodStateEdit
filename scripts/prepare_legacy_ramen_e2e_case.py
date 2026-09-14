#!/usr/bin/env python3
"""Create a same-resolution legacy ramen staging case without claiming SAM output."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/day19_multimaterial_dataset_v3/noodle/reference.png"
OLD_MATERIAL = ROOT / "benchmark/anchors_v1/ramen_chopsticks_001/mask_material.png"
OLD_BITE = ROOT / "benchmark/anchors_v1/ramen_chopsticks_001/mask_hole.png"
EDIT = ROOT / "artifacts/day18_high_lift_swept_support_v1/edit_alpha/udon_chopsticks_imagegen_pseudo_v1.png"
OUTPUT = ROOT / "artifacts/e2e_interaction_v1_case_staging/ramen_chopsticks_001"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_mask(source: Path, target: Path, size: tuple[int, int]) -> None:
    with Image.open(source) as image:
        image.convert("L").resize(size, Image.Resampling.NEAREST).save(target)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with Image.open(SOURCE) as image:
        size = image.size
    shutil.copy2(SOURCE, OUTPUT / "01_input.png")
    save_mask(OLD_MATERIAL, OUTPUT / "food_mask.png", size)
    save_mask(OLD_BITE, OUTPUT / "bite_mask.png", size)
    shutil.copy2(EDIT, OUTPUT / "allowed_edit_region.png")

    # A coarse manually parameterized bowl interior mask for interface staging.
    # It is explicitly not presented as SAM output or an evaluation annotation.
    container = Image.new("L", size, 0)
    draw = ImageDraw.Draw(container)
    width, height = size
    draw.ellipse((0.135 * width, 0.105 * height, 0.865 * width, 0.94 * height), fill=255)
    container.save(OUTPUT / "container_mask.png")

    names = ["01_input.png", "food_mask.png", "bite_mask.png", "container_mask.png", "allowed_edit_region.png"]
    manifest = {
        "schema_version": "foodstateedit.e2e_staging_case.v1",
        "case_id": "ramen_chopsticks_001",
        "size_wh": list(size),
        "annotation_method": "legacy_masks_resized_plus_manual_container_ellipse",
        "eligible_for_formal_sam_comparison": False,
        "purpose": "interface smoke only; replace with shared SAM/manual annotation before development evaluation",
        "files": {name: {"sha256": sha256(OUTPUT / name), "bytes": (OUTPUT / name).stat().st_size} for name in names},
    }
    (OUTPUT / "task.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "size_wh": size, "formal_evaluation": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
