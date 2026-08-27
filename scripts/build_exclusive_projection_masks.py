#!/usr/bin/env python3
"""Build disjoint TTM projection masks from a frozen common proxy package."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path


LAYERS = ("rigid", "contact", "material", "hole")
OWNERSHIP_PRIORITY = ("hole", "contact", "material", "rigid")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def owner_for_membership(membership: set[str]) -> str | None:
    """Return the unique semantic owner for one pixel's layer membership."""
    unknown = membership.difference(LAYERS)
    if unknown:
        raise ValueError(f"Unknown semantic layers: {sorted(unknown)}")
    return next(
        (layer for layer in OWNERSHIP_PRIORITY if layer in membership),
        None,
    )


def load_mask(path: Path) -> np.ndarray:
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 127


def make_exclusive(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Assign every union pixel to its highest-priority semantic owner."""
    import numpy as np

    if set(masks) != set(LAYERS):
        raise ValueError(f"Expected masks {LAYERS}, got {sorted(masks)}")
    shapes = {mask.shape for mask in masks.values()}
    if len(shapes) != 1:
        raise ValueError(f"Mask shape mismatch: {sorted(shapes)}")

    claimed = np.zeros_like(next(iter(masks.values())), dtype=bool)
    exclusive: dict[str, np.ndarray] = {}
    for layer in OWNERSHIP_PRIORITY:
        owned = np.logical_and(masks[layer], np.logical_not(claimed))
        exclusive[layer] = owned
        claimed = np.logical_or(claimed, masks[layer])
    return {layer: exclusive[layer] for layer in LAYERS}


def save_mask(mask: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path, optimize=True)


def build_case(source_dir: Path, output_dir: Path) -> dict[str, object]:
    import numpy as np

    source_manifest_path = source_dir / "proxy_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest["anchor_id"] != source_dir.name:
        raise ValueError(f"Source proxy identity mismatch: {source_dir}")
    masks = {
        layer: load_mask(source_dir / f"mask_{layer}.png")
        for layer in LAYERS
    }
    exclusive = make_exclusive(masks)
    source_union = np.logical_or.reduce(list(masks.values()))
    exclusive_union = np.logical_or.reduce(list(exclusive.values()))
    if not np.array_equal(source_union, exclusive_union):
        raise ValueError(f"Exclusive masks changed union coverage: {source_dir.name}")
    for left, right in combinations(LAYERS, 2):
        if np.logical_and(exclusive[left], exclusive[right]).any():
            raise ValueError(f"Exclusive masks overlap: {source_dir.name} {left}/{right}")
    if any(not mask.any() for mask in exclusive.values()):
        raise ValueError(f"Exclusive mask became empty: {source_dir.name}")

    output_dir.mkdir(parents=True, exist_ok=False)
    for layer in LAYERS:
        save_mask(exclusive[layer], output_dir / f"mask_{layer}.png")

    source_counts = {layer: int(masks[layer].sum()) for layer in LAYERS}
    exclusive_counts = {layer: int(exclusive[layer].sum()) for layer in LAYERS}
    manifest = {
        "schema_version": "foodstateedit.exclusive_projection_masks.v1",
        "anchor_id": source_dir.name,
        "source_proxy_manifest": str(source_manifest_path.resolve()),
        "source_proxy_manifest_sha256": sha256_file(source_manifest_path),
        "ownership_priority_high_to_low": list(OWNERSHIP_PRIORITY),
        "rule": "Each union pixel is assigned once to the first matching layer in ownership_priority_high_to_low.",
        "source_union_pixel_count": int(source_union.sum()),
        "exclusive_union_pixel_count": int(exclusive_union.sum()),
        "union_coverage_preserved": True,
        "pairwise_overlap_pixel_count": 0,
        "source_layer_pixel_counts": source_counts,
        "exclusive_layer_pixel_counts": exclusive_counts,
        "reassigned_from_layer_pixel_counts": {
            layer: source_counts[layer] - exclusive_counts[layer]
            for layer in LAYERS
        },
        "source_mask_files": {
            layer: {
                "path": str((source_dir / f"mask_{layer}.png").resolve()),
                "sha256": sha256_file(source_dir / f"mask_{layer}.png"),
            }
            for layer in LAYERS
        },
        "projection_mask_files": {
            layer: {
                "path": str((output_dir / f"mask_{layer}.png").resolve()),
                "sha256": sha256_file(output_dir / f"mask_{layer}.png"),
            }
            for layer in LAYERS
        },
        "status": "complete",
    }
    (output_dir / "projection_mask_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    proxy_root = args.proxy_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse output root: {output_root}")

    source_dirs = sorted(
        path for path in proxy_root.iterdir()
        if path.is_dir() and (path / "proxy_manifest.json").is_file()
    )
    if len(source_dirs) != 4:
        raise ValueError(f"Expected four proxy cases, found {len(source_dirs)}")
    output_root.mkdir(parents=True, exist_ok=False)
    cases = [build_case(path, output_root / path.name) for path in source_dirs]
    summary = {
        "schema_version": "foodstateedit.exclusive_projection_mask_summary.v1",
        "case_count": len(cases),
        "ownership_priority_high_to_low": list(OWNERSHIP_PRIORITY),
        "all_union_coverage_preserved": all(
            case["union_coverage_preserved"] for case in cases
        ),
        "all_pairwise_disjoint": all(
            case["pairwise_overlap_pixel_count"] == 0 for case in cases
        ),
        "cases": cases,
    }
    (output_root / "exclusive_projection_mask_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Built {len(cases)} exclusive projection-mask packages -> {output_root}"
    )


if __name__ == "__main__":
    main()
