"""Finite synthetic counterexample: one observed front, two hidden solids."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mld2_scene import camera_basis


def rotation_y(degrees):
    c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def camera(degrees):
    r = rotation_y(degrees)
    return r[:, 0], r[:, 1], r[:, 2]


def material(xyz):
    u, v, d = xyz.T
    grain = .03 * np.sin(31 * u + 6 * np.sin(17 * v))
    fleck = .055 * ((np.sin(33 * u) * np.cos(29 * v)) > .72)
    layer = .05 * np.sin(25 * d)
    return np.clip(np.stack([.78 + grain - fleck + layer,
                            .53 + grain - fleck + layer,
                            .23 + grain - fleck + .5 * layer], -1), 0, 1)


def ray_box(origin, direction, lower, upper):
    parallel = np.abs(direction) < 1e-12
    divisor = np.where(parallel, 1, direction)
    a, b = (lower - origin) / divisor, (upper - origin) / divisor
    inside_parallel = (origin >= lower) & (origin <= upper)
    near_axis = np.where(parallel, -np.inf, np.minimum(a, b))
    far_axis = np.where(parallel, np.inf, np.maximum(a, b))
    near, far = near_axis.max(-1), far_axis.min(-1)
    hit = (far >= np.maximum(near, 0)) & np.all(~parallel | inside_parallel, -1)
    xyz = origin + np.where(hit, near, 0)[:, None] * direction
    axis = near_axis.argmax(-1)
    normal = np.zeros_like(origin)
    normal[np.arange(len(origin)), axis] = -np.sign(direction[np.arange(len(origin)), axis])
    return np.where(hit, near, np.inf), xyz, normal


def render(boxes, angle=0, size=256, scale=1.0, center=(0, 0, 0)):
    right, up, toward = camera(angle)
    px = (np.arange(size) + .5) * 2 / size - 1
    u, v = np.meshgrid(px, -px)
    origin = (np.asarray(center) + scale * (u[..., None] * right + v[..., None] * up)
              + 4 * toward).reshape(-1, 3)
    direction = np.broadcast_to(-toward, origin.shape)
    depth = np.full(len(origin), np.inf)
    rgb = np.full((len(origin), 3), .96)
    light = np.array([-.4, .6, 1.0]); light /= np.linalg.norm(light)
    for lower, upper, rotation, translation in boxes:
        local_origin = (origin - translation) @ rotation
        local_direction = direction @ rotation
        t, point, normal = ray_box(local_origin, local_direction, lower, upper)
        update = t < depth
        shade = .65 + .35 * np.maximum((normal @ rotation.T) @ light, 0)
        rgb[update] = (material(point) * shade[:, None])[update]
        depth[update] = t[update]
    srgb = np.where(rgb <= .0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - .055)
    return Image.fromarray(np.round(np.clip(srgb, 0, 1) * 255).astype(np.uint8).reshape(size, size, 3))


def font(size):
    for name in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pivot = np.array([.1, 0, .45])
    rotation = rotation_y(50)
    lift = np.array([.35, 1.10, .10])
    translation = pivot - rotation @ pivot + lift
    width, height, front, cut_u = 1.1, .84, .45, .1
    rows, results, source_arrays = [], [], []
    for name, back in (("Thin backing", -.2), ("Deep backing", -.8)):
        lower, upper = np.array([-.55, -.42, back]), np.array([.55, .42, front])
        source = [(lower, upper, np.eye(3), np.zeros(3))]
        remaining_upper = upper.copy(); remaining_upper[0] = cut_u
        carried_lower = lower.copy(); carried_lower[0] = cut_u
        action = [(lower, remaining_upper, np.eye(3), np.zeros(3)),
                  (carried_lower, upper, rotation, translation)]
        source64 = render(source, size=64, scale=1.45)
        source_arrays.append(np.asarray(source64).astype(np.int16))
        slug = name.lower().replace(" ", "_")
        source64.save(args.out / f"{slug}_source64.png")
        views = [source64.resize((256, 256), Image.Resampling.NEAREST),
                 render(source, scale=.8),
                 render(source, angle=55, scale=1.05, center=(0, 0, -.1)),
                 render(action, scale=1.22, center=(.10, .55, .05))]
        for key, view in zip(("source", "front", "side", "cut_lift"), views):
            view.save(args.out / f"{slug}_{key}.png")
        thickness = front - back
        volume = width * height * thickness
        carried_volume = (.55 - cut_u) * height * thickness
        results.append({"name": name, "back_cap_d": back, "front_cap_d": front,
                        "hidden_thickness": thickness, "true_volume": volume,
                        "unit_density_source_mass": volume,
                        "unit_density_carried_mass": carried_volume,
                        "unit_density_remaining_mass": volume - carried_volume,
                        "source64_sha256": hashlib.sha256(source64.tobytes()).hexdigest()})
        rows.append(views)
    diff = np.abs(source_arrays[0] - source_arrays[1])
    stats = {"probe": "synthetic single-view hidden-volume ambiguity",
             "data_kind": "Analytic synthetic solids; no real photos, model outputs or learned physics.",
             "source_camera": {"projection": "orthographic", "resolution": [64, 64],
                               "scale": 1.45, "world_basis_right_up_toward": camera_basis().tolist()},
             "coordinates": "u,v,d are camera-aligned; d increases toward the source camera.",
             "visible_front": {"u": [-.55, .55], "v": [-.42, .42], "d": front,
                               "material": "same deterministic canonical material function"},
             "cut": {"plane": "u > 0.1 is carried", "threshold": cut_u},
             "action": {"rotation_about_camera_up_degrees": 50, "fixed_pivot_uvd": pivot.tolist(),
                        "translation_of_pivot_uvd": lift.tolist(),
                        "rotation_matrix_uvd": rotation.tolist(), "SE3_translation_uvd": translation.tolist()},
             "source_pixel_max_abs_diff_8bit": int(diff.max()),
             "source_pixel_mean_abs_diff_8bit": float(diff.mean()),
             "source_different_pixel_count": int(np.any(diff != 0, -1).sum()),
             "solids": results,
             "true_volume_difference": results[1]["true_volume"] - results[0]["true_volume"],
             "true_carried_mass_difference": (results[1]["unit_density_carried_mass"]
                                             - results[0]["unit_density_carried_mass"]),
             "mass_units": "Unit density in arbitrary canonical length units cubed; no food-density claim.",
             "interpretation": "The exact same observed image admits both solids. A consistent hidden hypothesis can be chosen, but its true hidden volume cannot be identified from this source view alone."}
    canvas = Image.new("RGB", (1320, 870), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 22), "SYNTHETIC PROBE: same observed front, different hidden solids", font=font(27), fill="#182230")
    draw.text((30, 64), "Unit-density proxy. Identical source camera, material and cut; identical rigid lift action.", font=font(19), fill="#435166")
    headings = ("Observed source 64 x 64", "Common front surface", "Hidden side revealed", "Same cut + lift + rotation")
    for column, heading in enumerate(headings):
        x = 245 + column * 266
        draw.text((x, 112), heading, font=font(17), fill="#182230")
    for row, (views, result) in enumerate(zip(rows, results)):
        y = 149 + row * 312
        draw.text((30, y + 39), result["name"], font=font(23), fill="#182230")
        draw.text((30, y + 82), f"Back cap: {result['back_cap_d']:.1f}", font=font(18), fill="#435166")
        draw.text((30, y + 117), f"Volume: {result['true_volume']:.4f}", font=font(18), fill="#435166")
        draw.text((30, y + 147), f"Carried: {result['unit_density_carried_mass']:.4f}", font=font(18), fill="#435166")
        for column, view in enumerate(views):
            x = 245 + column * 266
            canvas.paste(view, (x, y))
            draw.rectangle((x, y, x + 255, y + 255), outline="#d2d8df")
    draw.text((30, 790), f"Source pixel difference: {int(diff.max())}/255    |    True volume difference: {stats['true_volume_difference']:.4f}    |    Carried mass difference: {stats['true_carried_mass_difference']:.4f}", font=font(21), fill="#182230")
    draw.text((30, 833), "Single-view consistency does not identify the true hidden volume.", font=font(20), fill="#9e3b23")
    canvas.save(args.out / "single_view_ambiguity.png")
    (args.out / "single_view_ambiguity.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps({key: stats[key] for key in ("source_pixel_max_abs_diff_8bit", "source_different_pixel_count", "true_volume_difference", "true_carried_mass_difference")}))


if __name__ == "__main__":
    main()
