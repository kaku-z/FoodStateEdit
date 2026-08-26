#!/usr/bin/env python3
"""Build a geometry-aware GeoEdit case for lifting one noodle strand.

The source centreline is lifted with monocular VGGT depth, manipulated as an
inextensible 3-D discrete rod, and rendered back through the recovered camera.
The output directory follows the public GeoEdit input convention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--strand-mask", required=True, type=Path)
    parser.add_argument("--vggt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lift-up-px", type=float, default=500.0)
    parser.add_argument("--lift-right-px", type=float, default=170.0)
    parser.add_argument(
        "--toward-camera",
        type=float,
        default=0.12,
        help="Target reduction in VGGT camera-space depth.",
    )
    parser.add_argument("--root-fraction", type=float, default=0.075)
    parser.add_argument("--solver-iterations", type=int, default=1400)
    parser.add_argument("--no-chopsticks", action="store_true")
    return parser.parse_args()


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return mask.astype(bool)
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == index


def load_strand_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    data = np.load(path)
    if "masks" not in data.files:
        raise KeyError(f"{path} does not contain 'masks'")
    masks = np.asarray(data["masks"])
    mask = masks[0] if masks.ndim == 3 else masks
    if mask.shape != shape:
        mask = cv2.resize(
            mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
        )
    mask = largest_component(mask > 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0


def moving_average(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return values.copy()
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.ones(2 * radius + 1, dtype=np.float64) / (2 * radius + 1)
    return np.convolve(padded, kernel, mode="valid")


def sample_curve_depth(
    depth: np.ndarray,
    confidence: np.ndarray,
    mask: np.ndarray,
    points: np.ndarray,
    radius: int = 8,
) -> np.ndarray:
    height, width = depth.shape
    conf_floor = float(np.percentile(confidence[mask], 20.0))
    samples: list[float] = []
    for u, v in points:
        x = int(round(float(u)))
        y = int(round(float(v)))
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
        valid = disk & mask[y0:y1, x0:x1]
        local_conf = confidence[y0:y1, x0:x1]
        valid &= local_conf >= conf_floor
        values = depth[y0:y1, x0:x1][valid]
        if values.size == 0:
            values = depth[y0:y1, x0:x1][disk]
        samples.append(float(np.median(values)))
    sampled = np.asarray(samples, dtype=np.float64)
    median = moving_average(sampled, 7)
    residual = sampled - median
    mad = max(float(np.median(np.abs(residual))), 1e-5)
    sampled = np.clip(sampled, median - 3.0 * mad, median + 3.0 * mad)
    return moving_average(moving_average(sampled, 4), 4)


def backproject(points: np.ndarray, z: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    xyz = np.empty((len(points), 3), dtype=np.float64)
    xyz[:, 0] = (points[:, 0] - cx) * z / fx
    xyz[:, 1] = (points[:, 1] - cy) * z / fy
    xyz[:, 2] = z
    return xyz


def project(points: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.maximum(points[:, 2], 1e-6)
    uv = np.empty((len(points), 2), dtype=np.float64)
    uv[:, 0] = intrinsic[0, 0] * points[:, 0] / z + intrinsic[0, 2]
    uv[:, 1] = intrinsic[1, 1] * points[:, 1] / z + intrinsic[1, 2]
    return uv, z


def curve_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def smoothstep(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def solve_inextensible_rod(
    source: np.ndarray,
    target_grip: np.ndarray,
    root_count: int,
    iterations: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Quasi-static PBD solve with fixed root patch and one fixed grip node."""

    point_count = len(source)
    root_count = int(np.clip(root_count, 2, point_count - 3))
    rest = np.linalg.norm(np.diff(source, axis=0), axis=1)
    if np.any(rest <= 1e-8):
        raise ValueError("source 3-D centreline contains a zero-length segment")

    anchor = source[root_count - 1]
    free_length = float(rest[root_count - 1 :].sum())
    grip_delta = target_grip - anchor
    grip_distance = float(np.linalg.norm(grip_delta))
    reachable_ratio = 0.94
    if grip_distance > reachable_ratio * free_length:
        target_grip = anchor + grip_delta * (reachable_ratio * free_length / grip_distance)

    arc = np.concatenate(([0.0], np.cumsum(rest)))
    free_s = np.clip(
        (arc - arc[root_count - 1]) / max(arc[-1] - arc[root_count - 1], 1e-8),
        0.0,
        1.0,
    )
    weight = smoothstep(free_s)[:, None]
    endpoint_delta = target_grip - source[-1]
    guide = source + weight * endpoint_delta
    # Gravity-induced sag is expressed in camera Y, while the negative Z bow
    # makes the lift genuinely out of the source image plane.
    guide[:, 1] += 0.030 * np.sin(np.pi * free_s) ** 2
    guide[:, 2] -= 0.018 * np.sin(np.pi * free_s) ** 2
    positions = guide.copy()

    fixed = np.zeros(point_count, dtype=bool)
    fixed[:root_count] = True
    fixed[-1] = True
    fixed_values = source.copy()
    fixed_values[-1] = target_grip
    positions[fixed] = fixed_values[fixed]

    free_internal = np.arange(root_count, point_count - 1)
    for iteration in range(iterations):
        # A weak bending/shape prior resolves the many length-preserving
        # solutions without overriding the hard metric constraints.
        laplacian = 0.5 * (positions[:-2] + positions[2:]) - positions[1:-1]
        positions[1:-1] += 0.025 * laplacian
        positions[free_internal] += 0.004 * (guide[free_internal] - positions[free_internal])
        positions[fixed] = fixed_values[fixed]

        # Alternating sweep avoids a persistent directional bias at the bowl
        # exit. Each projection enforces the original 3-D segment length.
        order = range(point_count - 1)
        if iteration % 2:
            order = range(point_count - 2, -1, -1)
        for edge in order:
            i, j = edge, edge + 1
            delta = positions[j] - positions[i]
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-10:
                continue
            wi = 0.0 if fixed[i] else 1.0
            wj = 0.0 if fixed[j] else 1.0
            total = wi + wj
            if total == 0.0:
                continue
            correction = ((distance - rest[edge]) / distance) * delta / total
            positions[i] += wi * correction
            positions[j] -= wj * correction
        positions[fixed] = fixed_values[fixed]

    # Finish with a two-ended FABRIK solve. The PBD passes above choose a
    # smooth, gravity-consistent basin; FABRIK then removes their residual
    # stretch while satisfying both the bowl anchor and the gripper target.
    anchor_index = root_count - 1
    fabrik_iterations = max(2500, 5 * iterations)
    for _ in range(fabrik_iterations):
        positions[-1] = target_grip
        for i in range(point_count - 2, anchor_index - 1, -1):
            direction = positions[i] - positions[i + 1]
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                direction = source[i] - source[i + 1]
                norm = float(np.linalg.norm(direction))
            positions[i] = positions[i + 1] + rest[i] * direction / norm
        positions[anchor_index] = source[anchor_index]
        for i in range(anchor_index + 1, point_count):
            direction = positions[i] - positions[i - 1]
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                direction = source[i] - source[i - 1]
                norm = float(np.linalg.norm(direction))
            positions[i] = positions[i - 1] + rest[i - 1] * direction / norm
        positions[:root_count] = source[:root_count]
        if float(np.linalg.norm(positions[-1] - target_grip)) < 1e-8:
            break

    solved_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    diagnostics = {
        "max_segment_relative_error": float(np.max(np.abs(solved_lengths / rest - 1.0))),
        "mean_segment_relative_error": float(np.mean(np.abs(solved_lengths / rest - 1.0))),
        "grip_position_error": float(np.linalg.norm(positions[-1] - target_grip)),
        "requested_grip_distance": grip_distance,
        "free_rod_length": free_length,
    }
    return positions, diagnostics


def interpolate_curve(
    points: np.ndarray, colors: np.ndarray, subdivisions: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    dense_points: list[np.ndarray] = []
    dense_colors: list[np.ndarray] = []
    for index in range(len(points) - 1):
        for step in range(subdivisions):
            alpha = step / subdivisions
            dense_points.append((1.0 - alpha) * points[index] + alpha * points[index + 1])
            dense_colors.append((1.0 - alpha) * colors[index] + alpha * colors[index + 1])
    dense_points.append(points[-1])
    dense_colors.append(colors[-1])
    return np.asarray(dense_points), np.asarray(dense_colors)


def sample_source_colors(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    colors = []
    for u, v in points:
        x = int(np.clip(round(float(u)), 0, width - 1))
        y = int(np.clip(round(float(v)), 0, height - 1))
        patch = image[max(0, y - 4) : min(height, y + 5), max(0, x - 4) : min(width, x + 5)]
        colors.append(np.percentile(patch.reshape(-1, 3), 65.0, axis=0))
    colors_array = np.asarray(colors, dtype=np.float64)
    for channel in range(3):
        colors_array[:, channel] = moving_average(colors_array[:, channel], 3)
    return colors_array


def splat_tube(
    canvas: np.ndarray,
    z_buffer: np.ndarray,
    object_mask: np.ndarray,
    points_3d: np.ndarray,
    colors: np.ndarray,
    physical_radius: float,
    intrinsic: np.ndarray,
    scene_depth: np.ndarray | None,
    old_mask: np.ndarray | None = None,
) -> tuple[int, int]:
    height, width = object_mask.shape
    uv, z = project(points_3d, intrinsic)
    visible_pixels = 0
    tested_pixels = 0
    # Far-to-near ordering plus an explicit z-buffer gives correct tube
    # self-occlusion even where the lifted strand crosses itself.
    for index in np.argsort(z)[::-1]:
        u, v = uv[index]
        radius_px = max(1.5, intrinsic[0, 0] * physical_radius / z[index])
        x0 = max(0, int(np.floor(u - radius_px - 1)))
        x1 = min(width, int(np.ceil(u + radius_px + 2)))
        y0 = max(0, int(np.floor(v - radius_px - 1)))
        y1 = min(height, int(np.ceil(v + radius_px + 2)))
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        radial2 = ((xx - u) / radius_px) ** 2 + ((yy - v) / radius_px) ** 2
        inside = radial2 <= 1.0
        # Spherical front surface approximation supplies local depth curvature
        # instead of a flat painted ribbon.
        surface = z[index] - physical_radius * np.sqrt(np.clip(1.0 - radial2, 0.0, 1.0))
        local_z = z_buffer[y0:y1, x0:x1]
        visible = inside & (surface < local_z)
        if scene_depth is not None:
            local_scene = scene_depth[y0:y1, x0:x1]
            scene_visible = surface <= local_scene + 0.014
            if old_mask is not None:
                scene_visible |= old_mask[y0:y1, x0:x1]
            visible &= scene_visible
        tested_pixels += int(inside.sum())
        visible_pixels += int(visible.sum())
        if not np.any(visible):
            continue
        local_z[visible] = surface[visible]
        local_mask = object_mask[y0:y1, x0:x1]
        local_mask[visible] = True
        # A coarse cylindrical highlight is intentional: GeoEdit uses this as
        # geometric evidence and the diffusion branch restores photorealism.
        shade = 0.82 + 0.25 * np.sqrt(np.clip(1.0 - radial2, 0.0, 1.0))
        shaded = np.clip(colors[index][None, None, :] * shade[..., None], 0, 255)
        local_canvas = canvas[y0:y1, x0:x1]
        local_canvas[visible] = shaded[visible].astype(np.uint8)
    return visible_pixels, tested_pixels


def chopstick_geometry(
    contact: np.ndarray,
    intrinsic: np.ndarray,
    noodle_radius: float,
) -> list[np.ndarray]:
    z = float(contact[2])
    image_direction = np.asarray((0.72, -0.69), dtype=np.float64)
    direction = np.asarray(
        (
            image_direction[0] * z / intrinsic[0, 0],
            image_direction[1] * z / intrinsic[1, 1],
            -0.025,
        )
    )
    direction /= np.linalg.norm(direction)
    normal = np.asarray((-direction[1], direction[0], 0.0), dtype=np.float64)
    normal /= np.linalg.norm(normal)
    rod_radius = 0.0048
    separation = 2.0 * noodle_radius + 1.2 * rod_radius
    lines = []
    for side, z_offset in ((-1.0, 0.004), (1.0, -0.004)):
        tip = contact + side * 0.5 * separation * normal
        tip[2] += z_offset
        tail = tip + 0.48 * direction
        alpha = np.linspace(0.0, 1.0, 480)[:, None]
        lines.append((1.0 - alpha) * tip + alpha * tail)
    return lines


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    near, far = np.percentile(depth[valid], (1.0, 99.0))
    normalized = (far - depth) / max(float(far - near), 1e-6)
    normalized = np.clip(normalized, 0.0, 1.0)
    return np.rint(normalized * 255.0).astype(np.uint8)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = read_bgr(args.source)
    height, width = source.shape[:2]

    geometry = np.load(args.geometry)
    source_uv = np.asarray(geometry["source_points"], dtype=np.float64)
    vggt = np.load(args.vggt)
    depth = np.asarray(vggt["depth"], dtype=np.float64)
    confidence = np.asarray(vggt["confidence"], dtype=np.float64)
    intrinsic = np.asarray(vggt["intrinsic"], dtype=np.float64)
    if depth.shape != (height, width):
        raise ValueError(f"VGGT depth shape {depth.shape} != source shape {(height, width)}")
    old_mask = load_strand_mask(args.strand_mask, (height, width))

    source_z = sample_curve_depth(depth, confidence, old_mask, source_uv)
    source_3d = backproject(source_uv, source_z, intrinsic)
    root_count = max(3, int(round(args.root_fraction * len(source_3d))))

    target_uv_requested = source_uv[-1] + np.asarray(
        (args.lift_right_px, -args.lift_up_px), dtype=np.float64
    )
    target_uv_requested[0] = np.clip(target_uv_requested[0], 0.05 * width, 0.95 * width)
    target_uv_requested[1] = np.clip(target_uv_requested[1], 0.05 * height, 0.95 * height)
    target_z_requested = max(float(source_z[-1] - args.toward_camera), 0.60 * float(source_z[-1]))
    target_grip_requested = backproject(
        target_uv_requested[None, :], np.asarray([target_z_requested]), intrinsic
    )[0]
    target_3d, solve_report = solve_inextensible_rod(
        source_3d, target_grip_requested, root_count, args.solver_iterations
    )
    target_uv, target_z = project(target_3d, intrinsic)

    distance = cv2.distanceTransform(old_mask.astype(np.uint8), cv2.DIST_L2, 5)
    rounded = np.rint(source_uv).astype(np.int32)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, width - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, height - 1)
    radius_px = float(np.clip(np.median(distance[rounded[:, 1], rounded[:, 0]]), 8.0, 38.0))
    physical_radius = radius_px * float(np.median(source_z)) / float(
        0.5 * (intrinsic[0, 0] + intrinsic[1, 1])
    )

    old_mask_dilated = cv2.dilate(
        old_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    coarse_background = cv2.inpaint(
        source, old_mask_dilated * 255, max(5.0, 0.45 * radius_px), cv2.INPAINT_TELEA
    )
    motion_signal = coarse_background.copy()

    # Remove the old object from structural depth before inserting the new
    # geometry. OpenCV's inpainting is sufficient here because this branch is
    # not used as final RGB appearance.
    depth32 = depth.astype(np.float32)
    target_scene_depth = cv2.inpaint(
        depth32, old_mask_dilated * 255, max(5.0, 0.45 * radius_px), cv2.INPAINT_TELEA
    ).astype(np.float64)
    z_buffer = target_scene_depth.copy()
    new_mask = np.zeros((height, width), dtype=bool)

    source_colors = sample_source_colors(source, source_uv)
    dense_curve, dense_colors = interpolate_curve(target_3d, source_colors, subdivisions=5)
    visible, tested = splat_tube(
        motion_signal,
        z_buffer,
        new_mask,
        dense_curve,
        dense_colors,
        physical_radius,
        intrinsic,
        target_scene_depth,
        old_mask,
    )

    if not args.no_chopsticks:
        wood_bgr = np.asarray((72.0, 128.0, 178.0), dtype=np.float64)
        for line in chopstick_geometry(target_3d[-1], intrinsic, physical_radius):
            colors = np.repeat(wood_bgr[None, :], len(line), axis=0)
            v, t = splat_tube(
                motion_signal,
                z_buffer,
                new_mask,
                line,
                colors,
                0.0048,
                intrinsic,
                target_scene_depth,
                old_mask,
            )
            visible += v
            tested += t

    # A tiny closing operation removes pinholes from discrete 3-D splats but
    # does not expand the semantic editing region.
    new_mask = cv2.morphologyEx(
        new_mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ) > 0
    depth_control = normalize_depth(z_buffer)

    cv2.imwrite(str(args.output_dir / "first_frame.png"), source)
    cv2.imwrite(str(args.output_dir / "motion_signal.png"), motion_signal)
    cv2.imwrite(str(args.output_dir / "depth.png"), depth_control)
    cv2.imwrite(str(args.output_dir / "mask.png"), new_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(args.output_dir / "mask_old.png"), old_mask.astype(np.uint8) * 255)

    overlay = source.copy()
    cv2.polylines(overlay, [np.rint(source_uv).astype(np.int32)], False, (40, 220, 40), 5, cv2.LINE_AA)
    cv2.polylines(overlay, [np.rint(target_uv).astype(np.int32)], False, (40, 40, 240), 6, cv2.LINE_AA)
    cv2.circle(overlay, tuple(np.rint(target_uv[-1]).astype(int)), 13, (255, 80, 40), -1, cv2.LINE_AA)
    cv2.imwrite(str(args.output_dir / "diagnostic_3d_projection.png"), overlay)
    depth_color = cv2.applyColorMap(depth_control, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(args.output_dir / "target_depth_color.png"), depth_color)

    prompt = (
        "A photorealistic close-up food photograph of one thick glossy udon noodle "
        "being naturally lifted from the bowl by a pair of wooden chopsticks. Preserve "
        "the original bowl, broth, surrounding noodles, scallions, lighting, camera, and "
        "food identity. The lifted noodle is continuous, smooth, moist, and physically "
        "connected at the bowl exit, with plausible occlusion and contact shadows."
    )
    negative = (
        "broken noodle, disconnected noodle, fused chopsticks, extra chopsticks, duplicate "
        "objects, floating food, melted geometry, painted ribbon, hard seam, blurry detail, "
        "changed bowl, changed background, text, watermark"
    )
    (args.output_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    (args.output_dir / "negative_prompt.txt").write_text(negative + "\n", encoding="utf-8")

    np.savez_compressed(
        args.output_dir / "geometry_3d.npz",
        source_uv=source_uv.astype(np.float32),
        target_uv=target_uv.astype(np.float32),
        source_3d=source_3d.astype(np.float32),
        target_3d=target_3d.astype(np.float32),
        intrinsic=intrinsic.astype(np.float32),
        source_depth=source_z.astype(np.float32),
        target_depth=target_z.astype(np.float32),
    )
    source_length = curve_length(source_3d)
    target_length = curve_length(target_3d)
    report = {
        "algorithm": "VGGT lift -> 3D inextensible rod -> z-buffer render -> GeoEdit dual-branch VACE",
        "source_size": [width, height],
        "centreline_points": len(source_3d),
        "root_locked_points": root_count,
        "source_3d_length": source_length,
        "target_3d_length": target_length,
        "length_ratio": target_length / source_length,
        "source_grip_uv": source_uv[-1].tolist(),
        "target_grip_uv_requested": target_uv_requested.tolist(),
        "target_grip_uv_solved": target_uv[-1].tolist(),
        "source_grip_depth": float(source_z[-1]),
        "target_grip_depth": float(target_z[-1]),
        "out_of_plane_depth_displacement": float(target_z[-1] - source_z[-1]),
        "tube_radius_pixels_at_source": radius_px,
        "tube_radius_camera_units": physical_radius,
        "old_mask_ratio": float(old_mask.mean()),
        "new_mask_ratio": float(new_mask.mean()),
        "render_visibility_ratio": float(visible / max(tested, 1)),
        "solver": solve_report,
        "geoedit_mode": "non_hole",
        "recommended_tweak_index": 3,
        "recommended_tstrong_index": 15,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
