"""Generate source-only synthetic supervision for the first learned MLD model.

These are programmatic solids, internal material fields, and cuts. Their labels
are synthetic generator truth, never measured food geometry or real before/after
truth. Every query is on a fixed, shared bbox lattice; the condition contains
only a rendered source photograph, its known projection, and a user action.
Generator parameters, occupancy, visibility, and action outcomes are targets.

The output uses separate memory-mappable NPY arrays. A trainer must use only the
manifest's ``inputs`` as condition and must not concatenate any target mask.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np


FORMAT_VERSION = "mld-synthetic-v1"
DEFAULT_SEED = 20261004
CAMERA_YAW_DEGREES = 30.0
CAMERA_ELEVATION_DEGREES = 35.0
CAMERA_SCALE = 1.45
INPUT_KEYS = ("source_rgb", "canonical_xyz", "source_uv", "actions")
TARGET_KEYS = ("joint", "occupancy", "carried_fraction", "cell_occupancy",
               "action_validmask")
SHAPE_FAMILIES = ("roundbox", "ellipsoid", "cylinder")
PALETTES = np.asarray([
    [.78, .67, .43], [.70, .38, .18], [.26, .46, .19], [.63, .13, .08],
    [.73, .76, .67], [.40, .22, .10], [.90, .70, .12], [.54, .28, .35],
], dtype=np.float64)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode("ascii"))
    h.update(json.dumps(list(value.shape)).encode("ascii"))
    h.update(value.tobytes())
    return h.hexdigest()


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    temporary.replace(path)


def canonical_grid(grid_size: int = 16) -> np.ndarray:
    if not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError("grid_size must be an integer of at least 2")
    centers = (np.arange(grid_size, dtype=np.float64) + .5) * (2 / grid_size) - 1
    return np.stack(np.meshgrid(centers, centers, centers, indexing="ij"),
                    axis=-1).reshape(-1, 3).astype(np.float32)


def camera_basis() -> np.ndarray:
    yaw = math.radians(CAMERA_YAW_DEGREES)
    elev = math.radians(CAMERA_ELEVATION_DEGREES)
    right = [-math.sin(yaw), math.cos(yaw), 0]
    up = [-math.sin(elev) * math.cos(yaw),
          -math.sin(elev) * math.sin(yaw), math.cos(elev)]
    toward_camera = [math.cos(elev) * math.cos(yaw),
                     math.cos(elev) * math.sin(yaw), math.sin(elev)]
    return np.asarray([right, up, toward_camera], dtype=np.float64)


def known_camera_projection(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("xyz must be a finite N x 3 array")
    projected = xyz @ camera_basis()[:2].T / CAMERA_SCALE
    projected[:, 1] *= -1
    # Out-of-image queries stay outside; do not replace them using GT visibility.
    return projected.astype(np.float32)


def scene_identity(index: int, seed: int = DEFAULT_SEED) -> str:
    if index < 0:
        raise ValueError("index must be nonnegative")
    return hashlib.sha256(f"mld-v1:{seed}:{index}".encode("ascii")).hexdigest()


def _rng(identity: str, stream: str) -> np.random.Generator:
    digest = hashlib.sha256((identity + ":" + stream).encode("ascii")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def axis_angle_matrix(axis_angle: np.ndarray) -> np.ndarray:
    a = np.asarray(axis_angle, dtype=np.float64)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError("axis_angle must be a finite 3-vector")
    theta = float(np.linalg.norm(a))
    if theta < 1e-12:
        return np.eye(3)
    x, y, z = a / theta
    skew = np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(theta) * skew + (1 - math.cos(theta)) * (skew @ skew)


def action_delta(xyz: np.ndarray, command: np.ndarray) -> np.ndarray:
    """Rigid carried displacement about canonical origin, never scene center."""
    xyz = np.asarray(xyz, dtype=np.float64)
    command = np.asarray(command, dtype=np.float64)
    if command.shape != (10,) or not np.isfinite(command).all():
        raise ValueError("command must be a finite action10 vector")
    return (xyz @ axis_angle_matrix(command[7:10]).T + command[4:7] - xyz
            ).astype(np.float32)


def _scene_parameters(identity: str) -> dict:
    rng = _rng(identity, "generator-targets")
    family = SHAPE_FAMILIES[int(rng.integers(len(SHAPE_FAMILIES)))]
    axis_angle = rng.uniform(-.25, .25, 3)
    axis_angle[2] = rng.uniform(-math.pi, math.pi)
    half_extent = rng.uniform(.37, .68, 3)
    if family == "roundbox":
        half_extent = rng.uniform(.31, .57, 3)
    elif family == "cylinder":
        half_extent[1] = half_extent[0]
    base = np.clip(PALETTES[int(rng.integers(len(PALETTES)))]
                   + rng.normal(0, .035, 3), .03, .94)
    layer_axis = rng.normal(size=3)
    layer_axis /= np.linalg.norm(layer_axis)
    inclusion_count = int(rng.integers(4, 10))
    return {
        "family": family,
        "center": rng.uniform(-.065, .065, 3).tolist(),
        "half_extent": half_extent.tolist(),
        "rounding": float(rng.uniform(.04, .12)),
        "axis_angle": axis_angle.tolist(),
        "base_linear_rgb": base.tolist(),
        "layer_axis": layer_axis.tolist(),
        "layer_frequency": float(rng.uniform(2.0, 6.0)),
        "layer_phase": float(rng.uniform(-math.pi, math.pi)),
        "layer_contrast": rng.uniform(-.13, .13, 3).tolist(),
        "inclusion_centers": rng.uniform(-.55, .55, (inclusion_count, 3)).tolist(),
        "inclusion_radii": rng.uniform(.055, .16, inclusion_count).tolist(),
        "inclusion_colors": np.clip(base + rng.uniform(-.32, .32, (inclusion_count, 3)),
                                     .02, .98).tolist(),
        "noise_vectors": rng.uniform(-2.7, 2.7, (4, 3)).tolist(),
        "noise_phases": rng.uniform(-math.pi, math.pi, 4).tolist(),
        "noise_color_weights": rng.uniform(-.045, .045, (4, 3)).tolist(),
    }


def shape_sdf(xyz: np.ndarray, parameters: dict) -> np.ndarray:
    """Analytic solid SDF; ellipsoid uses a conservative radial distance proxy."""
    xyz = np.asarray(xyz, dtype=np.float64)
    q = ((xyz - np.asarray(parameters["center"]))
         @ axis_angle_matrix(np.asarray(parameters["axis_angle"])))
    extent = np.asarray(parameters["half_extent"], dtype=np.float64)
    family = parameters["family"]
    if family == "ellipsoid":
        return (np.linalg.norm(q / extent, axis=-1) - 1) * extent.min()
    if family == "roundbox":
        d = np.abs(q) - extent
        return (np.linalg.norm(np.maximum(d, 0), axis=-1)
                + np.minimum(np.max(d, axis=-1), 0) - parameters["rounding"])
    if family == "cylinder":
        d = np.stack([np.linalg.norm(q[..., :2], axis=-1) - extent[0],
                      np.abs(q[..., 2]) - extent[2]], axis=-1)
        return np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(axis=-1), 0)
    raise ValueError("Unknown shape family")


def material_rgb(xyz: np.ndarray, parameters: dict) -> np.ndarray:
    """Persistent 3-D generator material, including unobserved internal structure."""
    xyz = np.asarray(xyz, dtype=np.float64)
    flat = xyz.reshape(-1, 3)
    phase = (flat @ np.asarray(parameters["layer_axis"])
             * parameters["layer_frequency"] * math.pi + parameters["layer_phase"])
    layer = np.tanh(4 * np.sin(phase))
    color = np.tile(np.asarray(parameters["base_linear_rgb"]), (len(flat), 1))
    color += layer[:, None] * np.asarray(parameters["layer_contrast"])
    modes = np.sin(flat @ np.asarray(parameters["noise_vectors"]).T * math.pi
                   + np.asarray(parameters["noise_phases"]))
    color += modes @ np.asarray(parameters["noise_color_weights"])
    for center, radius, inclusion_color in zip(
        parameters["inclusion_centers"], parameters["inclusion_radii"],
        parameters["inclusion_colors"],
    ):
        distance = np.linalg.norm(flat - np.asarray(center), axis=1)
        blend = np.clip((radius - distance) / max(.2 * radius, .01), 0, 1)
        color = color * (1 - blend[:, None]) + np.asarray(inclusion_color) * blend[:, None]
    return np.clip(color, .005, .995).reshape(xyz.shape)


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0, 1)
    srgb = np.where(rgb <= .0031308, rgb * 12.92,
                    1.055 * np.power(rgb, 1 / 2.4) - .055)
    return np.rint(np.clip(srgb, 0, 1) * 255).astype(np.uint8)


def render_source_rgb(parameters: dict, identity: str, image_size: int = 64) -> np.ndarray:
    """Only the first visible surface is rendered; no hidden target channels."""
    if image_size < 8:
        raise ValueError("image_size must be at least 8")
    basis = camera_basis()
    pixels = (np.arange(image_size, dtype=np.float64) + .5) / image_size * 2 - 1
    u, v = np.meshgrid(pixels, -pixels, indexing="xy")
    origins = (u[..., None] * CAMERA_SCALE * basis[0]
               + v[..., None] * CAMERA_SCALE * basis[1] + 2.5 * basis[2])
    origins = origins.reshape(-1, 3)
    direction = -basis[2]
    # A fixed sphere bounds every generated shape; it contains no scene-dependent
    # geometry information and simply shortens the raymarch.
    projection = origins @ direction
    discriminant = projection ** 2 - (np.sum(origins ** 2, axis=1) - 1.3 ** 2)
    eligible = discriminant >= 0
    distances = np.zeros(len(origins), dtype=np.float64)
    distances[eligible] = (-projection[eligible]
                           - np.sqrt(discriminant[eligible]))
    ray_end = np.zeros(len(origins), dtype=np.float64)
    ray_end[eligible] = -projection[eligible] + np.sqrt(discriminant[eligible])
    active = eligible.copy()
    hit = np.zeros(len(origins), dtype=bool)
    for _ in range(64):
        selected = np.flatnonzero(active)
        if not len(selected):
            break
        p = origins[selected] + distances[selected, None] * direction
        sd = shape_sdf(p, parameters)
        arrived = sd < .0012
        hit[selected[arrived]] = True
        active[selected[arrived]] = False
        marching = selected[~arrived]
        distances[marching] += np.maximum(sd[~arrived] * .95, .0005)
        active[marching] &= distances[marching] <= ray_end[marching]
    rng = _rng(identity, "source-light-background")
    base_background = rng.uniform(.09, .67, 3)
    gradient = rng.uniform(-.08, .08, 3)
    output = np.clip(base_background + v.reshape(-1, 1) * gradient, .015, .90)
    selected = np.flatnonzero(hit)
    if len(selected):
        p = origins[selected] + distances[selected, None] * direction
        eps = .0015
        normals = np.stack([
            shape_sdf(p + np.eye(3)[i] * eps, parameters)
            - shape_sdf(p - np.eye(3)[i] * eps, parameters)
            for i in range(3)
        ], axis=-1)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        light = basis[2] + rng.uniform(-.6, .6, 3)
        light /= np.linalg.norm(light)
        diffuse = .32 + .62 * np.maximum(normals @ light, 0)
        output[selected] = np.clip(material_rgb(p, parameters) * diffuse[:, None], 0, 1)
    return _linear_to_srgb(output.reshape(image_size, image_size, 3))


def frozen_actions(identity: str) -> np.ndarray:
    """Four commands from an independent RNG stream; they do not read GT shape."""
    rng = _rng(identity, "user-action-commands")
    actions = np.zeros((4, 10), dtype=np.float32)
    normals = np.asarray([[.7, .5, .6], [-.6, .7, .7], [.3, -.7, .7]])
    for row in range(1, 4):
        normal = normals[row - 1] + rng.uniform(-.15, .15, 3)
        normal /= np.linalg.norm(normal)
        actions[row, :3] = normal
        actions[row, 3] = rng.uniform(.24, .37) if row == 1 else rng.uniform(.12, .29)
        actions[row, 4:7] = rng.uniform(-.12, .12, 3)
        actions[row, 6] = rng.uniform(.32, .48) if row != 3 else rng.uniform(.65, .85)
        actions[row, 7:10] = rng.uniform(-.2, .2, 3) if row == 1 else rng.uniform(-.55, .55, 3)
    return actions


def subcell_partition(parameters: dict, xyz: np.ndarray, actions: np.ndarray,
                      grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exclusive eight-point source-cell budgets, not exact continuum volumes."""
    offsets = np.stack(np.meshgrid(*[[-.5 / grid_size, .5 / grid_size]] * 3,
                                   indexing="ij"), axis=-1).reshape(8, 3)
    samples = np.asarray(xyz, dtype=np.float64)[:, None, :] + offsets[None, :, :]
    inside = shape_sdf(samples, parameters) <= 0
    count = inside.sum(axis=1)
    fractions = np.zeros((len(actions), len(xyz)), dtype=np.float64)
    for row, command in enumerate(actions):
        if not np.any(command):
            continue
        cut = np.sum(samples * command[:3], axis=-1) > command[3]
        selected = np.sum(inside & cut, axis=1)
        fractions[row] = np.divide(selected, count, out=np.zeros(len(count), dtype=float),
                                   where=count > 0)
    return ((count / 8).astype(np.float16), fractions.astype(np.float16), count > 0)


def generate_scene(index: int, seed: int = DEFAULT_SEED, grid_size: int = 16,
                   image_size: int = 64) -> tuple[dict[str, np.ndarray], dict]:
    identity = scene_identity(index, seed)
    parameters = _scene_parameters(identity)
    xyz = canonical_grid(grid_size)
    sdf = shape_sdf(xyz, parameters)
    rgb = material_rgb(xyz, parameters)
    actions = frozen_actions(identity)
    cell_occupancy, fractions, validmask = subcell_partition(parameters, xyz, actions, grid_size)
    joint = np.concatenate([np.clip(sdf[:, None], -.5, .5) / .5,
                            rgb * 2 - 1], axis=1).astype(np.float16)
    record = {
        "source_rgb": render_source_rgb(parameters, identity, image_size),
        "joint": joint,
        "occupancy": sdf <= 0,
        "actions": actions,
        "carried_fraction": fractions,
        "cell_occupancy": cell_occupancy,
        "action_validmask": validmask,
    }
    metadata = {
        "index": index, "scene_id": identity, "generator_seed": seed,
        "shape_family": parameters["family"], "target_generator_parameters": parameters,
        "source_rgb_sha256": array_sha256(record["source_rgb"]),
        "joint_sha256": array_sha256(joint),
        "actions_sha256": array_sha256(actions),
        "action_semantics": "Null, small plane cut/lift, different plane cut/rotation, high lift; commands independent of target shape.",
        "target_scope": "Programmatic solid/material truth; not measured food or a real before/after pair.",
    }
    return record, metadata


def split_assignment(scene_count: int) -> np.ndarray:
    if scene_count < 8:
        raise ValueError("At least 8 independent scenes are required for three splits")
    train_count = scene_count * 3 // 4
    validation_count = scene_count // 8
    labels = np.full(scene_count, 2, dtype=np.uint8)
    labels[:train_count] = 0
    labels[train_count:train_count + validation_count] = 1
    return labels


def _worker(args):
    return generate_scene(*args)


def prepare_dataset(output: str | Path, scenes: int = 8192, seed: int = DEFAULT_SEED,
                    grid_size: int = 16, image_size: int = 64, workers: int = 1) -> dict:
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output must be empty; never overwrite a frozen dataset")
    if workers < 1:
        raise ValueError("workers must be positive")
    split = split_assignment(scenes)
    xyz = canonical_grid(grid_size)
    tokens = len(xyz)
    output.mkdir(parents=True, exist_ok=True)
    specs = {
        "source_rgb": ((scenes, image_size, image_size, 3), np.uint8),
        "joint": ((scenes, tokens, 4), np.float16),
        "occupancy": ((scenes, tokens), np.bool_),
        "actions": ((scenes, 4, 10), np.float32),
        "carried_fraction": ((scenes, 4, tokens), np.float16),
        "cell_occupancy": ((scenes, tokens), np.float16),
        "action_validmask": ((scenes, tokens), np.bool_),
    }
    frozen_recipe = {
        "format_version": FORMAT_VERSION, "status": "frozen_before_generation",
        "seed": seed, "scene_count": scenes, "grid_size": grid_size,
        "image_size": image_size, "workers": workers, "inputs": list(INPUT_KEYS),
        "targets": list(TARGET_KEYS), "shape_families": list(SHAPE_FAMILIES),
        "split_policy": "Independent scene identities in fixed seed order, 75% train / 12.5% validation / remainder test; all actions stay with their scene.",
        "truth_scope": "Synthetic supervision only; no real image texture bank, no real after-image or measured geometry truth.",
        "condition_policy": "Single first-hit source RGB, fixed full-box queries/projection and command; no true shape, material coefficients, target masks, target visibility, or after-image as input.",
        "source_hash": file_sha256(__file__),
        "started_unix": time.time(),
    }
    write_json(output / "recipe.json", frozen_recipe)
    np.save(output / "canonical_xyz.npy", xyz, allow_pickle=False)
    np.save(output / "source_uv.npy", known_camera_projection(xyz), allow_pickle=False)
    np.save(output / "splits.npy", split, allow_pickle=False)
    arrays = {key: np.lib.format.open_memmap(output / (key + ".npy"), mode="w+",
                                            shape=shape, dtype=dtype)
              for key, (shape, dtype) in specs.items()}
    arguments = ((i, seed, grid_size, image_size) for i in range(scenes))
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    iterator = pool.map(_worker, arguments, chunksize=8) if pool else map(_worker, arguments)
    started = time.time()
    try:
        with (output / "scenes.jsonl").open("w", encoding="utf-8") as stream:
            for index, (record, metadata) in enumerate(iterator):
                if metadata["index"] != index:
                    raise RuntimeError("Worker output order changed identity assignment")
                for key, array in arrays.items():
                    array[index] = record[key]
                metadata["split"] = ["train", "validation", "test"][int(split[index])]
                stream.write(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")) + "\n")
                if index % 128 == 0 or index == scenes - 1:
                    print(json.dumps({"generated": index + 1, "total": scenes,
                                      "elapsed_seconds": round(time.time() - started, 2)}), flush=True)
        for array in arrays.values():
            array.flush()
    finally:
        if pool:
            pool.shutdown(wait=True)
    files = []
    for path in sorted(output.iterdir()):
        if path.suffix not in (".npy", ".jsonl", ".json"):
            continue
        entry = {"path": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size}
        if path.suffix == ".npy":
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            entry.update(shape=list(array.shape), dtype=str(array.dtype))
        files.append(entry)
    manifest = {
        "format_version": FORMAT_VERSION, "status": "complete",
        "seed": seed, "scene_count": scenes, "grid_size": grid_size,
        "image_size": image_size, "tokens": tokens,
        "inputs": list(INPUT_KEYS), "targets": list(TARGET_KEYS),
        "split_codes": {"train": 0, "validation": 1, "test": 2},
        "split_counts": {name: int(np.sum(split == code)) for code, name in
                         enumerate(["train", "validation", "test"])},
        "camera": {"projection": "known orthographic", "yaw_degrees": CAMERA_YAW_DEGREES,
                   "elevation_degrees": CAMERA_ELEVATION_DEGREES,
                   "scale": CAMERA_SCALE, "basis_right_up_toward_camera": camera_basis().tolist(),
                   "uv_rule": "xyz @ basis[:2].T / scale; negate vertical; all fixed queries, no GT visibility"},
        "joint_channels": ["clamped_sdf_divided_by_0.5", "linear_red_times2_minus1",
                           "linear_green_times2_minus1", "linear_blue_times2_minus1"],
        "action_channels": ["cut_normal_x", "cut_normal_y", "cut_normal_z", "cut_offset",
                            "translation_x", "translation_y", "translation_z",
                            "axis_angle_x", "axis_angle_y", "axis_angle_z"],
        "action_semantics": "Carried if x dot cutnormal > offset; axis-angle rigid rotation about canonical origin, then translation. action0 exactly zero, carriedfraction0 and delta0.",
        "mass_scope": "Cell reference volume = (occupied subcell count / 8)*(2/grid_size)^3 under unit density. Child fractions conserve each discrete budget; not exact continuum cuts or physical grams.",
        "validmask_policy": "action_validmask is a target/loss mask for occupied subcells, forbidden as input; center occupancy is a separate target.",
        "shape_families": list(SHAPE_FAMILIES),
        "scope": "First MLD synthetic pretraining; single-source visual posterior, supervised synthetic cut/lift, no real paired or hidden-food truth.",
        "elapsed_seconds": time.time() - started, "files": files,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    result = prepare_dataset(args.output, args.scenes, args.seed, args.grid_size,
                             args.image_size, args.workers)
    print(json.dumps({"status": result["status"], "scene_count": result["scene_count"],
                      "split_counts": result["split_counts"]}), flush=True)


if __name__ == "__main__":
    main()
