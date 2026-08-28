#!/usr/bin/env python3
"""Render topology-preserving utensil/food appearance from frozen proxy masks."""

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
METHOD = "foodstateedit_material_render"
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


def read_image(path: Path, flags: int):
    import cv2

    image = cv2.imread(str(path), flags)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def verify_proxy(proxy_dir: Path) -> dict[str, object]:
    manifest_path = proxy_dir / "proxy_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != "foodstateedit.common_proxy.v1":
        raise ValueError(f"Unexpected proxy schema: {manifest_path}")
    if manifest.get("anchor_id") != proxy_dir.name:
        raise ValueError(f"Proxy identity mismatch: {manifest_path}")
    for name, item in manifest["files"].items():
        if sha256_file(proxy_dir / name) != item["sha256"]:
            raise ValueError(f"Proxy hash mismatch: {proxy_dir / name}")
    return manifest


def soft_outer_alpha(binary, sigma: float, support_alpha):
    import cv2
    import numpy as np

    core = binary.astype(np.float32)
    blurred = cv2.GaussianBlur(core, (0, 0), sigmaX=sigma)
    return np.maximum(core, blurred) * support_alpha


def composite(base, overlay, alpha):
    return base * (1.0 - alpha[..., None]) + overlay * alpha[..., None]


def surface_geometry(mask, radius: float):
    import cv2
    import numpy as np

    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    profile = np.clip(distance / radius, 0.0, 1.0)
    gradient_x = cv2.Sobel(distance, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(distance, cv2.CV_32F, 0, 1, ksize=3)
    normal_z = np.full_like(distance, 3.0)
    normal_length = np.sqrt(
        gradient_x * gradient_x + gradient_y * gradient_y + normal_z * normal_z
    )
    normal_x = gradient_x / normal_length
    normal_y = gradient_y / normal_length
    normal_z = normal_z / normal_length
    light = np.asarray([-0.38, -0.55, 0.74], dtype=np.float32)
    light /= np.linalg.norm(light)
    diffuse = np.clip(
        normal_x * light[0] + normal_y * light[1] + normal_z * light[2],
        0.0,
        1.0,
    )
    return profile, diffuse


def rigid_texture(source, mask, utensil: str, config: dict[str, object]):
    import cv2
    import numpy as np

    profile, diffuse = surface_geometry(mask, float(config["surface_edge_radius"]))
    environment = cv2.GaussianBlur(source, (0, 0), sigmaX=9.0).astype(np.float32)
    if utensil in config["utensil_material_rule"]["wood"]:
        base = np.asarray(config["wood_base_rgb"][::-1], dtype=np.float32)
        ys, xs = np.where(mask)
        coords = np.column_stack([xs, ys]).astype(np.float32)
        centered = coords - coords.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / max(1, len(coords) - 1)
        _, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, -1]
        yy, xx = np.indices(mask.shape, dtype=np.float32)
        along = (xx - coords[:, 0].mean()) * axis[0] + (
            yy - coords[:, 1].mean()
        ) * axis[1]
        grain = 0.65 * np.sin(along / 8.0) + 0.35 * np.sin(along / 3.2)
        shade = 0.62 + 0.30 * profile + 0.12 * diffuse + 0.07 * grain
        mix = float(config["wood_environment_mix"])
        texture = base[None, None, :] * shade[..., None]
    else:
        base = np.asarray(config["steel_base_rgb"][::-1], dtype=np.float32)
        specular = np.power(diffuse, 12.0) * float(
            config["steel_specular_strength"]
        )
        shade = 0.52 + 0.30 * profile + 0.23 * diffuse
        mix = float(config["steel_environment_mix"])
        texture = base[None, None, :] * shade[..., None]
        texture += 255.0 * specular[..., None]
    return np.clip(texture * (1.0 - mix) + environment * mix, 0, 255)


def enhance_food(proxy, config: dict[str, object]):
    import cv2
    import numpy as np

    proxy_f = proxy.astype(np.float32)
    blur = cv2.GaussianBlur(proxy_f, (0, 0), sigmaX=1.0)
    detail = proxy_f - blur
    enhanced = proxy_f + float(config["food_detail_gain"]) * detail
    enhanced_u8 = np.clip(np.rint(enhanced), 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(enhanced_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= float(config["food_saturation_scale"])
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)


def split_chopsticks(rigid, primitives, width: int, height: int):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    def to_pixel(point):
        return round(point[0] * (width - 1)), round(point[1] * (height - 1))

    distances = []
    for primitive in primitives:
        if primitive["type"] not in {"line", "polyline"}:
            raise ValueError("Chopstick primitives must be lines")
        image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(image)
        points = [to_pixel(point) for point in primitive["points"]]
        line_width = max(1, round(float(primitive["width"]) * min(width, height)))
        draw.line(points, fill=255, width=line_width)
        radius = line_width // 2
        for x, y in (points[0], points[-1]):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
        primitive_mask = np.asarray(image) > 127
        distances.append(
            cv2.distanceTransform((~primitive_mask).astype(np.uint8), cv2.DIST_L2, 5)
        )
    if len(distances) != 2:
        raise ValueError("Chopstick renderer requires exactly two primitives")
    rear = np.logical_and(rigid, distances[0] <= distances[1])
    front = np.logical_and(rigid, np.logical_not(rear))
    if not rear.any() or not front.any() or not np.array_equal(rear | front, rigid):
        raise ValueError("Chopstick z-order partition failed")
    return rear, front


def compose_case(
    proxy_dir: Path,
    anchor_spec: dict[str, object],
    config: dict[str, object],
    output_dir: Path,
    code_commit: str,
) -> dict[str, object]:
    import cv2
    import numpy as np

    proxy_manifest = verify_proxy(proxy_dir)
    source = read_image(proxy_dir / "first_frame.png", cv2.IMREAD_COLOR)
    proxy = read_image(proxy_dir / "motion_signal.png", cv2.IMREAD_COLOR)
    masks = {
        name: read_image(proxy_dir / filename, cv2.IMREAD_GRAYSCALE)
        for name, filename in MASK_FILES.items()
    }
    binary = {name: masks[name] > 127 for name in ("rigid", "contact", "material", "hole")}
    support = masks["edit_alpha"] > 0
    support_alpha = masks["edit_alpha"].astype(np.float32) / 255.0
    source_f = source.astype(np.float32)
    proxy_f = proxy.astype(np.float32)

    hole_alpha = soft_outer_alpha(
        binary["hole"], float(config["hole_outer_feather_sigma"]), support_alpha
    )
    result = composite(source_f, proxy_f, hole_alpha)

    object_mask = binary["rigid"] | binary["material"]
    offset_x, offset_y = config["cast_shadow_offset_xy"]
    transform = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    cast_shadow = cv2.warpAffine(
        object_mask.astype(np.float32),
        transform,
        (source.shape[1], source.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    cast_shadow = cv2.GaussianBlur(
        cast_shadow, (0, 0), sigmaX=float(config["cast_shadow_sigma"])
    )
    contact_shadow = cv2.GaussianBlur(
        binary["contact"].astype(np.float32),
        (0, 0),
        sigmaX=float(config["contact_shadow_sigma"]),
    )
    cast_shadow[object_mask] = 0.0
    contact_shadow[object_mask] = 0.0
    shadow = (
        cast_shadow * float(config["cast_shadow_opacity"])
        + contact_shadow * float(config["contact_shadow_opacity"])
    ) * support_alpha
    result *= 1.0 - np.clip(shadow, 0.0, 0.45)[..., None]

    utensil = anchor_spec["action"]["utensil"]
    rigid_render = rigid_texture(source, binary["rigid"], utensil, config)
    food_render = enhance_food(proxy, config)
    feather_sigma = float(config["foreground_outer_feather_sigma"])
    material_alpha = soft_outer_alpha(binary["material"], feather_sigma, support_alpha)
    partition_exact = True
    if utensil == "chopsticks":
        rear, front = split_chopsticks(
            binary["rigid"],
            anchor_spec["layers"]["rigid"]["primitives"],
            source.shape[1],
            source.shape[0],
        )
        rear_alpha = soft_outer_alpha(rear, feather_sigma, support_alpha)
        front_alpha = soft_outer_alpha(front, feather_sigma, support_alpha)
        result = composite(result, rigid_render, rear_alpha)
        result = composite(result, food_render, material_alpha)
        result = composite(result, rigid_render, front_alpha)
        partition_exact = bool(np.array_equal(rear | front, binary["rigid"]))
    else:
        rigid_alpha = soft_outer_alpha(binary["rigid"], feather_sigma, support_alpha)
        result = composite(result, rigid_render, rigid_alpha)
        result = composite(result, food_render, material_alpha)

    result = np.clip(np.rint(result), 0, 255).astype(np.uint8)
    result[~support] = source[~support]
    outside_difference = np.abs(
        result.astype(np.int16) - source.astype(np.int16)
    )[~support]
    outside_max = int(outside_difference.max(initial=0))
    if outside_max != 0 or not partition_exact:
        raise ValueError(f"Material-render invariant failed: {proxy_dir.name}")

    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "edited_2d.png"
    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError(f"Cannot write output: {output_path}")
    shader_preview = source.copy()
    shader_preview[binary["rigid"]] = np.clip(
        np.rint(rigid_render[binary["rigid"]]), 0, 255
    ).astype(np.uint8)
    review = np.concatenate([
        np.concatenate([source, proxy], axis=1),
        np.concatenate([shader_preview, result], axis=1),
    ], axis=0)
    review_path = output_dir / "review_board.png"
    if not cv2.imwrite(str(review_path), review):
        raise RuntimeError(f"Cannot write review board: {review_path}")

    manifest = {
        "schema_version": "foodstateedit.material_render_run.v1",
        "anchor_id": proxy_dir.name,
        "method": METHOD,
        "deterministic": True,
        "code_commit": code_commit,
        "config_sha256": sha256_file(Path(config["_config_path"])),
        "anchor_specs_sha256": sha256_file(Path(config["_anchor_specs_path"])),
        "proxy_manifest_sha256": sha256_file(proxy_dir / "proxy_manifest.json"),
        "proxy_status": proxy_manifest["status"],
        "utensil": utensil,
        "utensil_material": (
            "wood" if utensil in config["utensil_material_rule"]["wood"] else "metal"
        ),
        "invariants": {
            "rigid_mask_sha256": sha256_file(proxy_dir / "mask_rigid.png"),
            "material_mask_sha256": sha256_file(proxy_dir / "mask_material.png"),
            "rigid_pixel_count": int(binary["rigid"].sum()),
            "material_pixel_count": int(binary["material"].sum()),
            "chopstick_partition_exact": partition_exact,
            "outside_edit_alpha_max_pixel_difference": outside_max,
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
    parser.add_argument("--anchor-specs", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to reuse output root: {args.output_root}")

    config_path = args.config.resolve()
    config = load_json(config_path)
    if config.get("schema_version") != "foodstateedit.material_render.v1":
        raise ValueError(f"Unexpected config schema: {config_path}")
    if config.get("method") != METHOD:
        raise ValueError(f"Config method mismatch: {config_path}")
    config["_config_path"] = str(config_path)
    anchor_specs_path = args.anchor_specs.resolve()
    specs_document = load_json(anchor_specs_path)
    if specs_document.get("schema_version") != "foodstateedit.anchor_specs.v1":
        raise ValueError(f"Unexpected anchor spec schema: {anchor_specs_path}")
    specs = {item["anchor_id"]: item for item in specs_document["anchors"]}
    if set(specs) != set(ANCHORS):
        raise ValueError(f"Anchor spec mismatch: {sorted(specs)}")
    config["_anchor_specs_path"] = str(anchor_specs_path)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    cases = [
        compose_case(
            args.proxy_root.resolve() / anchor,
            specs[anchor],
            config,
            output_root / anchor,
            args.code_commit,
        )
        for anchor in ANCHORS
    ]
    summary = {
        "schema_version": "foodstateedit.material_render_summary.v1",
        "method": METHOD,
        "case_count": len(cases),
        "complete_count": sum(case["status"] == "complete" for case in cases),
        "all_protected_pixels_exact": all(
            case["invariants"]["outside_edit_alpha_max_pixel_difference"] == 0
            for case in cases
        ),
        "all_topology_partitions_exact": all(
            case["invariants"]["chopstick_partition_exact"] for case in cases
        ),
        "cases": cases,
    }
    (output_root / "material_render_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
