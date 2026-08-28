#!/usr/bin/env python3
"""Compose geometry-locked FoodStateEdit images from frozen proxy/run inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ANCHORS = (
    "soup_spoon_001",
    "fried_rice_spatula_001",
    "ramen_chopsticks_001",
    "pasta_fork_001",
)
METHOD = "foodstateedit_geometry_lock"
DONOR_METHOD = "foodstateedit_staged_exclusive"
MASK_FILES = {
    "rigid": "mask_rigid.png",
    "contact": "mask_contact.png",
    "material": "mask_material.png",
    "hole": "mask_hole.png",
    "edit_alpha": "edit_alpha.png",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_proxy(proxy_dir: Path) -> dict[str, object]:
    manifest_path = proxy_dir / "proxy_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != "foodstateedit.common_proxy.v1":
        raise ValueError(f"Unexpected proxy schema: {manifest_path}")
    if manifest.get("anchor_id") != proxy_dir.name:
        raise ValueError(f"Proxy identity mismatch: {manifest_path}")
    for name, item in manifest["files"].items():
        path = proxy_dir / name
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Proxy hash mismatch: {path}")
    return manifest


def find_donors(roots: list[Path]) -> dict[str, dict[str, object]]:
    donors = {}
    for root in roots:
        for manifest_path in root.glob("*/seed_1/run_manifest.json"):
            manifest = load_json(manifest_path)
            anchor = manifest["case_id"]
            if anchor in donors:
                raise ValueError(f"Duplicate donor run for {anchor}")
            if manifest["method"] != DONOR_METHOD or manifest["seed"] != 1:
                raise ValueError(f"Donor identity mismatch: {manifest_path}")
            if manifest["status"] != "complete":
                raise ValueError(f"Donor run is incomplete: {manifest_path}")
            output = next(
                item for item in manifest["outputs"]
                if Path(item["path"]).name == "edited_2d.png"
            )
            output_path = Path(output["path"])
            if sha256_file(output_path) != output["sha256"]:
                raise ValueError(f"Donor image hash mismatch: {output_path}")
            donors[anchor] = {
                "manifest_path": manifest_path,
                "manifest": manifest,
                "output_path": output_path,
                "output_sha256": output["sha256"],
            }
    if set(donors) != set(ANCHORS):
        raise ValueError(f"Donor anchor mismatch: {sorted(donors)}")
    return donors


def read_image(path: Path, flags: int):
    import cv2

    image = cv2.imread(str(path), flags)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def compose_case(
    proxy_dir: Path,
    donor: dict[str, object],
    config: dict[str, object],
    output_dir: Path,
    code_commit: str,
) -> dict[str, object]:
    import cv2
    import numpy as np

    proxy_manifest = verify_proxy(proxy_dir)
    source_path = proxy_dir / "first_frame.png"
    proxy_path = proxy_dir / "motion_signal.png"
    source = read_image(source_path, cv2.IMREAD_COLOR)
    proxy = read_image(proxy_path, cv2.IMREAD_COLOR)
    generated = read_image(donor["output_path"], cv2.IMREAD_COLOR)
    if source.shape != proxy.shape or source.shape != generated.shape:
        raise ValueError(f"Image shape mismatch: {proxy_dir.name}")

    threshold = int(config["binary_mask_threshold"])
    masks = {
        name: read_image(proxy_dir / filename, cv2.IMREAD_GRAYSCALE)
        for name, filename in MASK_FILES.items()
    }
    foreground = np.logical_or.reduce([
        masks[layer] > threshold for layer in config["foreground_layers"]
    ])
    hole = masks["hole"] > threshold
    semantic_core = np.logical_or(foreground, hole)
    support = masks["edit_alpha"] > 0
    support_alpha = masks["edit_alpha"].astype(np.float32) / 255.0

    source_f = source.astype(np.float32)
    proxy_f = proxy.astype(np.float32)
    generated_f = generated.astype(np.float32)

    hole_binary = hole.astype(np.float32)
    hole_blur = cv2.GaussianBlur(
        hole_binary,
        (0, 0),
        sigmaX=float(config["hole_outer_feather_sigma"]),
    )
    hole_alpha = np.maximum(hole_binary, hole_blur) * support_alpha
    result = source_f * (1.0 - hole_alpha[..., None]) + proxy_f * hole_alpha[..., None]

    contact_binary = (masks["contact"] > threshold).astype(np.float32)
    contact_halo = cv2.GaussianBlur(
        contact_binary,
        (0, 0),
        sigmaX=float(config["contact_halo_sigma"]),
    )
    contact_halo[semantic_core] = 0.0
    contact_halo *= support_alpha * float(config["contact_halo_strength"])
    generated_weight = float(config["generated_contact_donor_weight"])
    contact_donor = proxy_f * (1.0 - generated_weight) + generated_f * generated_weight
    result = result * (1.0 - contact_halo[..., None]) + contact_donor * contact_halo[..., None]

    foreground_binary = foreground.astype(np.float32)
    foreground_blur = cv2.GaussianBlur(
        foreground_binary,
        (0, 0),
        sigmaX=float(config["foreground_outer_feather_sigma"]),
    )
    foreground_alpha = np.maximum(foreground_binary, foreground_blur) * support_alpha
    # The generated image is allowed to affect only the bounded contact halo.
    # Foreground edge feathering remains a deterministic proxy/source blend.
    edge_donor = proxy_f
    result = result * (1.0 - foreground_alpha[..., None]) + edge_donor * foreground_alpha[..., None]

    result = np.clip(np.rint(result), 0, 255).astype(np.uint8)
    result[semantic_core] = proxy[semantic_core]
    result[np.logical_not(support)] = source[np.logical_not(support)]

    core_difference = np.abs(
        result.astype(np.int16) - proxy.astype(np.int16)
    )[semantic_core]
    outside_difference = np.abs(
        result.astype(np.int16) - source.astype(np.int16)
    )[np.logical_not(support)]
    if int(core_difference.max(initial=0)) != 0:
        raise ValueError(f"Semantic core changed: {proxy_dir.name}")
    if int(outside_difference.max(initial=0)) != 0:
        raise ValueError(f"Protected pixels changed: {proxy_dir.name}")

    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "edited_2d.png"
    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError(f"Cannot write output: {output_path}")
    review = np.concatenate([
        np.concatenate([source, proxy], axis=1),
        np.concatenate([generated, result], axis=1),
    ], axis=0)
    review_path = output_dir / "review_board.png"
    if not cv2.imwrite(str(review_path), review):
        raise RuntimeError(f"Cannot write review board: {review_path}")

    generated_influence = np.logical_and(contact_halo > 0, np.logical_not(semantic_core))
    changed = np.any(result != source, axis=2)
    manifest = {
        "schema_version": "foodstateedit.geometry_lock_run.v1",
        "anchor_id": proxy_dir.name,
        "method": METHOD,
        "deterministic": True,
        "code_commit": code_commit,
        "config_sha256": sha256_file(Path(config["_config_path"])),
        "proxy_manifest_sha256": sha256_file(proxy_dir / "proxy_manifest.json"),
        "proxy_status": proxy_manifest["status"],
        "donor_method": DONOR_METHOD,
        "donor_seed": 1,
        "donor_manifest": str(donor["manifest_path"].resolve()),
        "donor_manifest_sha256": sha256_file(donor["manifest_path"]),
        "donor_image_sha256": donor["output_sha256"],
        "invariants": {
            "semantic_core_pixel_count": int(semantic_core.sum()),
            "semantic_core_vs_proxy_max_pixel_difference": 0,
            "protected_pixel_count": int(np.logical_not(support).sum()),
            "outside_edit_alpha_max_pixel_difference": 0,
            "generated_influence_pixel_count": int(generated_influence.sum()),
            "generated_influence_outside_semantic_core": True,
        },
        "metrics": {
            "changed_pixel_fraction": round(float(changed.mean()), 6),
            "edit_support_pixel_count": int(support.sum()),
        },
        "outputs": {
            "edited_2d.png": sha256_file(output_path),
            "review_board.png": sha256_file(review_path),
        },
        "status": "complete",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--donor-root", action="append", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to reuse output root: {args.output_root}")

    config_path = args.config.resolve()
    config = load_json(config_path)
    if config.get("schema_version") != "foodstateedit.geometry_lock.v1":
        raise ValueError(f"Unexpected config schema: {config_path}")
    if config.get("method") != METHOD:
        raise ValueError(f"Config method mismatch: {config_path}")
    config["_config_path"] = str(config_path)
    donors = find_donors([path.resolve() for path in args.donor_root])

    proxy_root = args.proxy_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    cases = [
        compose_case(
            proxy_root / anchor,
            donors[anchor],
            config,
            output_root / anchor,
            args.code_commit,
        )
        for anchor in ANCHORS
    ]
    summary = {
        "schema_version": "foodstateedit.geometry_lock_summary.v1",
        "method": METHOD,
        "case_count": len(cases),
        "complete_count": sum(case["status"] == "complete" for case in cases),
        "all_semantic_cores_exact": all(
            case["invariants"]["semantic_core_vs_proxy_max_pixel_difference"] == 0
            for case in cases
        ),
        "all_protected_pixels_exact": all(
            case["invariants"]["outside_edit_alpha_max_pixel_difference"] == 0
            for case in cases
        ),
        "all_generated_influence_outside_core": all(
            case["invariants"]["generated_influence_outside_semantic_core"]
            for case in cases
        ),
        "cases": cases,
    }
    (output_root / "geometry_lock_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
