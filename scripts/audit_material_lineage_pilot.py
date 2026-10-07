"""Independent artifact audit for the material-lineage pilot.

This verifies discrete identities, provenance, checkpoint execution, and pixel
invariants. None of these measurements establishes photographic realism,
measured food mass, complete 3-D reconstruction, novelty, or generalization.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from foodstateedit.material_lineage import MaterialLedger, MaterialState, material_noise


SCOPE = (
    "Artifact and internal-consistency audit of a discrete material-lineage "
    "prototype with a source-conditioned color denoiser. No independent "
    "photographic realism, measured mass, complete new model, or held-out-food "
    "generalization is established by these metrics."
)


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def array_sha(array):
    array = np.ascontiguousarray(array)
    value = hashlib.sha256()
    value.update(str(array.dtype).encode("ascii"))
    value.update(json.dumps(list(array.shape)).encode("ascii"))
    value.update(array.tobytes())
    return value.hexdigest()


def linear_rgb(rgb):
    rgb = np.asarray(rgb, dtype=np.float64) / 255
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rgb_image(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def mask_image(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


class Checks:
    def __init__(self):
        self.rows = []

    def check(self, name, condition, **evidence):
        self.rows.append(dict(name=name, passed=bool(condition), evidence=evidence))
        return bool(condition)

    def failure(self, name, error):
        self.check(name, False, error=f"{type(error).__name__}: {error}")

    @property
    def passed(self):
        return bool(self.rows) and all(row["passed"] for row in self.rows)


def checkpoint_audit(checkpoint_path, receipt, samples_path, checks, prefix):
    """Read actual saved tensors with weights_only, not an executable pickle."""
    import torch

    checks.check(prefix + "/checkpoint_file_sha",
                 file_sha(checkpoint_path) == receipt["checkpoint_sha256"])
    checks.check(prefix + "/training_samples_file_sha",
                 file_sha(samples_path) == receipt["samples_sha256"])
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    actual_steps = int(checkpoint["actual_optimizer_steps"])
    parameters = sum(int(t.numel()) for name, t in checkpoint["state_dict"].items()
                     if name.endswith(("weight", "bias")))
    finite = all(bool(torch.isfinite(t).all()) for t in checkpoint["state_dict"].values())
    checks.check(prefix + "/trained_checkpoint_content",
                 actual_steps > 0 and actual_steps == receipt["actual_optimizer_steps"]
                 == receipt["requested_optimizer_steps"]
                 and checkpoint["input_samples_sha256"] == file_sha(samples_path)
                 and checkpoint["model_config"] == receipt["model_config"]
                 and parameters == receipt["parameter_count"] and finite,
                 actual_optimizer_steps=actual_steps, tensor_parameter_count=parameters,
                 finite_tensors=finite)
    loss_steps = [int(row["step"]) for row in receipt["losses"]]
    checks.check(prefix + "/training_execution_trace",
                 receipt["status"] == "complete" and bool(loss_steps)
                 and loss_steps == sorted(set(loss_steps)) and loss_steps[-1] == actual_steps
                 and all(np.isfinite(row["epsilon_mse"]) for row in receipt["losses"]),
                 logged_steps=loss_steps)
    checks.check(prefix + "/sampling_internal_identity",
                 receipt["sampling_validation"]["repeat_bitwise_equal"]
                 and receipt["sampling_validation"]["independent_batch_bitwise_equal"],
                 caveat="Receipt alone is not an independent realism test")


def source_sample_audit(samples_path, source_path, receipt, checks, prefix,
                        pixel_key="source_pixel_yx", bare_mask_path=None):
    """Reproduce the source-only appearance proxy; it is not measured albedo."""
    with np.load(samples_path, allow_pickle=False) as data:
        colors = np.asarray(data["colors"], dtype=np.float32)
        coords = np.asarray(data["coords"], dtype=np.float32)
        base = np.asarray(data["base_color"], dtype=np.float32)
        pixels = np.asarray(data[pixel_key])
        source_linear = np.asarray(data["source_linear_rgb"])
        directional = np.asarray(data["source_directional_rgb"])
    source = rgb_image(source_path)
    valid_pixels = (pixels.dtype.kind in "iu" and pixels.shape == (len(colors), 2)
                    and np.all(pixels >= 0)
                    and np.all(pixels[:, 0] < source.shape[0])
                    and np.all(pixels[:, 1] < source.shape[1]))
    checks.check(prefix + "/source_pixel_coordinates", valid_pixels,
                 samples=len(colors), source_shape=list(source.shape))
    if valid_pixels:
        actual = linear_rgb(source)[pixels[:, 0], pixels[:, 1]]
        error = float(np.max(np.abs(actual - source_linear), initial=0))
        checks.check(prefix + "/source_only_observed_linear_rgb", error <= 2e-7,
                     max_linear_rgb_error=error,
                     limitation="Same original photograph, not independent held-out food")
        proxy = np.clip(actual / np.maximum(directional, .05) * base, 0, 1)
        proxy_error = float(np.max(np.abs(proxy - colors), initial=0))
        checks.check(prefix + "/source_only_proxy_training_colors", proxy_error <= 2e-7,
                     max_linear_rgb_error=proxy_error,
                     limitation="Source-derived angular correction; not calibrated intrinsic albedo")
        if bare_mask_path is not None:
            bare = mask_image(bare_mask_path)
            checks.check(prefix + "/source_bare_pixel_membership",
                         bare.shape == source.shape[:2]
                         and bool(np.all(bare[pixels[:, 0], pixels[:, 1]])),
                         limitation="Bare mask is a source-only approximation")
    checks.check(prefix + "/bounded_source_samples",
                 coords.shape == colors.shape and coords.shape[1:] == (3,)
                 and len(coords) >= 32 and base.shape == (3,)
                 and all(np.isfinite(a).all() and np.all(a >= 0) and np.all(a <= 1)
                         for a in (colors, coords, base)))
    for name, array in (("coords", coords), ("colors", colors), ("base_color", base)):
        checks.check(prefix + "/training_array_sha/" + name,
                     array_sha(array) == receipt["samples_arrays_sha256"][name])
    checks.check(prefix + "/source_sample_count",
                 len(coords) == receipt["source_observed_sample_count"]
                 and receipt["train_source_pixels"] + receipt["heldout_source_pixels"] == len(coords))
    permutation = np.random.default_rng(receipt["seed"]).permutation(len(coords))
    nval = min(len(coords) - 16, max(8, int(round(len(coords) * receipt["training"]["validation_fraction"]))))
    for name, array in (("train_indices", permutation[nval:]),
                        ("heldout_indices", permutation[:nval]),
                        ("evaluated_heldout_indices", permutation[:receipt["evaluated_heldout_source_pixels"]])):
        checks.check(prefix + "/same_photo_split_sha/" + name,
                     array_sha(array) == receipt["split_hashes"][name])


def original_outside_edit_audit(source_path, final_path, edit_path, checks, prefix):
    source, final, edit = rgb_image(source_path), rgb_image(final_path), mask_image(edit_path)
    shape_valid = source.shape == final.shape and edit.shape == source.shape[:2]
    checks.check(prefix + "/native_pixel_shapes", shape_valid,
                 source_shape=list(source.shape), final_shape=list(final.shape), edit_shape=list(edit.shape))
    if shape_valid:
        changed = np.any(source != final, axis=2)
        violations = int(np.count_nonzero(changed & ~edit))
        checks.check(prefix + "/exact_original_outside_edit", violations == 0,
                     violating_pixels=violations, changed_pixels=int(np.count_nonzero(changed)))


def material_budget_audit(path, checks, prefix):
    with np.load(path, allow_pickle=False) as data:
        ids = tuple(str(i) for i in data["material_ids"])
        coords, masses = data["coords"], data["root_mass"]
        remaining, carried = data["remaining_fraction"], data["carried_fraction"]
    source = MaterialState(ids, coords, masses, np.ones(len(ids)), coords)
    branches = []
    for name, fraction in (("remaining", remaining), ("carried", carried)):
        take = fraction > 0
        branches.append(MaterialState(tuple(i for i, yes in zip(ids, take) if yes),
                                      coords[take], masses[take], fraction[take], coords[take], name))
    result = MaterialLedger(source).validate(branches)
    checks.check(prefix + "/per_cell_budget", True, **result)
    checks.check(prefix + "/independent_fraction_sum",
                 np.all(remaining >= 0) and np.all(carried >= 0)
                 and np.allclose(remaining + carried, 1, rtol=0, atol=1e-12)
                 and np.all(np.isfinite(masses)) and np.all(masses > 0),
                 max_cell_fraction_error=float(np.max(np.abs(remaining + carried - 1))),
                 limitation="Exact discrete source-cell budget, not measured real-food mass")
    result["carried_mass_fraction"] = float(np.sum(masses * carried) / np.sum(masses))
    return result


def independent_source_geometry_audit(directory, source_inputs, checks, prefix):
    """Moller-Trumbore source rays, independent of the pilot rasterizer."""
    import trimesh
    input_paths = {Path(row["path"]).name: Path(row["path"]) for row in source_inputs["inputs"]}
    mesh = trimesh.load(input_paths["full.ply"], process=False)
    triangles = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.face_normals)
    with np.load(directory / "train_samples.npz", allow_pickle=False) as data:
        yx, coords = data["source_pixel_yx"], data["coords"]
        stored_directional = data["source_directional_rgb"]
        base = data["base_color"]
    K = np.asarray(source_inputs["intrinsics_canvas"])
    R = np.asarray(source_inputs["R_food_to_camera"])
    low, high = np.asarray(source_inputs["low"]), np.asarray(source_inputs["high"])
    rays = np.c_[yx[:, 1] + .5, yx[:, 0] + .5, np.ones(len(yx))] @ np.linalg.inv(K).T
    depth = np.full(len(yx), np.inf)
    face = np.full(len(yx), -1, dtype=int)
    for fid, triangle in enumerate(triangles):
        edge1, edge2 = triangle[1] - triangle[0], triangle[2] - triangle[0]
        p = np.cross(rays, edge2)
        det = p @ edge1
        valid_det = np.abs(det) > 1e-14
        inv = np.divide(1, det, out=np.zeros_like(det), where=valid_det)
        origin_offset = -triangle[0]
        u = (p @ origin_offset) * inv
        q = np.cross(origin_offset, edge1)
        v = (rays @ q) * inv
        distance = float(edge2 @ q) * inv
        take = valid_det & (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7) & (distance > 0) & (distance < depth)
        depth[take], face[take] = distance[take], fid
    all_hit = bool(np.all(face >= 0))
    checks.check(prefix + "/source_training_rays_hit_full_mesh", all_hit)
    if not all_hit:
        return
    world_food = (rays * depth[:, None]) @ R
    canonical = np.clip((world_food - low) / (high - low), 0, 1).astype(np.float32)
    coord_error = float(np.max(np.abs(canonical - coords), initial=0))
    checks.check(prefix + "/source_pixel_material_coordinates", coord_error <= 2e-6,
                 max_canonical_coordinate_error=coord_error)
    cal = read_json(input_paths["appearance_calibration.json"])
    plane_normals = np.asarray(cal["observed_plane_normals_camera"])
    plane_rgb = linear_rgb(cal["plane_median_rgb"])
    weights = 8 * normals[face] @ plane_normals.T
    weights = np.exp(weights - weights.max(axis=1, keepdims=True))
    predicted = weights @ plane_rgb / weights.sum(axis=1, keepdims=True)
    direction_error = float(np.max(np.abs(predicted - stored_directional), initial=0))
    checks.check(prefix + "/independent_source_angular_appearance", direction_error <= 2e-6,
                 max_linear_rgb_error=direction_error,
                 limitation="Positive source-plane interpolation, not recovered illumination")
    x0, y0, x1, y1 = cal["source_material_box_canvas"]
    predicted_base = np.median(linear_rgb(rgb_image(directory / "source.png")[y0:y1, x0:x1]), axis=(0, 1))
    checks.check(prefix + "/source_only_base_patch",
                 np.allclose(base, predicted_base, rtol=0, atol=2e-7))


def fields_audit(directory, config, receipt, checks, prefix, sampling_device="cpu"):
    import torch

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    trainer_path = directory.parent / "scripts" / "train_lineage_material_denoiser.py"
    checks.check(prefix + "/executed_trainer_script_hash", file_sha(trainer_path) == receipt["script_sha256"])
    spec = importlib.util.spec_from_file_location("audited_lineage_trainer", trainer_path)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    field = trainer.load_field(directory / "training" / "material_denoiser.pt", device=sampling_device)
    with np.load(directory / "train_samples.npz", allow_pickle=False) as data:
        colors = data["colors"]
    floor, ceiling = np.quantile(colors, .005, axis=0), np.quantile(colors, .995, axis=0)
    size = config["grid_size"]
    nodes = np.stack(np.meshgrid(*[np.linspace(0, 1, size)] * 3, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
    ids = np.asarray([f"{directory.name}:canonical-grid:{i}" for i in range(len(nodes))])
    streams = ["shared", "independent_remaining"] + [f"independent_carried_pose{p}" for p in range(config["poses"])]
    selected = np.unique(np.linspace(0, len(nodes) - 1, 17).round().astype(int))
    expected = {f"s{seed}_{stream}.npz" for seed in config["seeds"] for stream in streams}
    actual = {path.name for path in (directory / "fields").glob("*.npz")}
    checks.check(prefix + "/full_field_matrix", actual == expected, expected=len(expected), actual=len(actual))
    for seed in config["seeds"]:
        for stream in streams:
            path = directory / "fields" / f"s{seed}_{stream}.npz"
            with np.load(path, allow_pickle=False) as data:
                raw, bounded, noise = data["raw_linear_rgb"], data["bounded_linear_rgb"], data["noise"]
                stored_nodes, stored_ids = data["canonical_coords"], data["persistent_ids"]
                lower, upper = data["source_floor"], data["source_ceiling"]
            base_name = prefix + f"/field/{seed}/{stream}"
            checks.check(base_name + "/canonical_grid_identity",
                         np.array_equal(stored_nodes, nodes) and np.array_equal(stored_ids, ids))
            predicted_noise = material_noise(tuple(ids), seed, 3, stream).astype(np.float32)
            checks.check(base_name + "/persistent_noise_key", np.array_equal(noise, predicted_noise))
            checks.check(base_name + "/source_frozen_bounds",
                         np.array_equal(lower, floor) and np.array_equal(upper, ceiling)
                         and raw.shape == (len(nodes), 3) and np.isfinite(raw).all()
                         and np.all(raw >= 0) and np.all(raw <= 1)
                         and np.array_equal(bounded, np.clip(raw, floor, ceiling)))
            resampled = field.sample_material(nodes[selected], noise[selected], steps=config["ddim_steps"])
            sample_error = float(np.max(np.abs(resampled - raw[selected]), initial=0))
            is_cuda = str(sampling_device).startswith("cuda")
            tolerance = 0.0 if is_cuda else 5e-5
            checks.check(base_name + "/actual_checkpoint_resampling",
                         sample_error <= tolerance,
                         canonical_probe_points=len(selected), device=str(sampling_device),
                         max_linear_rgb_error=sample_error, numerical_tolerance=tolerance,
                         bitwise_equal=bool(np.array_equal(resampled, raw[selected])),
                         limitation="Checkpoint provenance probe, not physical accuracy or independent realism")


def closest_distances(mesh, points):
    """Plane/edge triangle distance without scale-dependent absolute tolerances.

    Tiny cutter faces trigger false edge-only distances in the legacy nearest
    point routine. Use cross-product area barycentrics, avoiding cancellation
    in d00*d11-d01*d01 and any fixed threshold on triangle size.
    """
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    normal2 = np.sum(normal * normal, axis=1)
    distances = []
    for start in range(0, len(points), 24):
        p = np.asarray(points[start:start + 24], dtype=np.float64)[:, None]
        delta = p - a[None]
        signed = np.einsum("pti,ti->pt", delta, normal)
        plane_squared = np.divide(signed * signed, normal2[None],
                                  out=np.full_like(signed, np.inf), where=normal2[None] > 0)
        v_numerator = np.einsum("pti,ti->pt", np.cross(delta, ac[None]), normal)
        w_numerator = np.einsum("pti,ti->pt", np.cross(ab[None], delta), normal)
        v = np.divide(v_numerator, normal2[None], out=np.full_like(signed, np.nan), where=normal2[None] > 0)
        w = np.divide(w_numerator, normal2[None], out=np.full_like(signed, np.nan), where=normal2[None] > 0)
        inside = (v >= 0) & (w >= 0) & (v + w <= 1)
        squared = np.where(inside, plane_squared, np.inf)
        for edge_start, edge_end in ((a, b), (b, c), (c, a)):
            edge = edge_end - edge_start
            edge2 = np.sum(edge * edge, axis=1)
            projection = np.einsum("pti,ti->pt", p - edge_start[None], edge)
            parameter = np.divide(projection, edge2[None], out=np.zeros_like(projection), where=edge2[None] > 0)
            nearest = edge_start[None] + np.clip(parameter, 0, 1)[..., None] * edge[None]
            squared = np.minimum(squared, np.sum((p - nearest) ** 2, axis=2))
        distances.extend(np.sqrt(squared.min(axis=1)))
    return np.asarray(distances)


def probes_audit(directory, config, source_inputs, checks, prefix):
    from scipy.ndimage import map_coordinates
    import trimesh

    low, high = np.asarray(source_inputs["low"]), np.asarray(source_inputs["high"])
    extent = float((high - low).max())
    R = np.asarray(source_inputs["R_food_to_camera"])
    input_paths = {Path(row["path"]).name: Path(row["path"]) for row in source_inputs["inputs"]}
    remaining_mesh = trimesh.load(input_paths["remaining.ply"], process=False)
    for seed in config["seeds"]:
        reference_ids, reference_coords, reference_color = None, None, None
        for pose in range(config["poses"]):
            name = prefix + f"/probe/{seed}/pose{pose}"
            with np.load(directory / f"pose{pose}" / f"probe_s{seed}.npz", allow_pickle=False) as data:
                ids, coords = data["material_ids"], data["coords"]
                remaining_world, carried_world = data["remaining_world"], data["carried_world"]
                rotation, translation = data["rotation"], data["translation"]
                recovered = data["recovered_coords"]
                shared_remaining, shared_carried = data["shared_remaining_rgb"], data["shared_carried_rgb"]
                independent_remaining, independent_carried = data["independent_remaining_rgb"], data["independent_carried_rgb"]
            if reference_ids is None:
                reference_ids, reference_coords, reference_color = ids, coords, shared_remaining
            checks.check(name + "/shared_sample_identity_across_poses",
                         np.array_equal(ids, reference_ids) and np.array_equal(coords, reference_coords)
                         and len(ids) > 0 and len(set(ids.tolist())) == len(ids))
            expected_carried = coords @ rotation.T + translation
            expected_recovered = (carried_world - translation) @ rotation
            coordinate_error = float(np.max(np.abs(expected_recovered - coords), initial=0)) / extent
            checks.check(name + "/actual_inverse_pose_correspondence",
                         np.allclose(remaining_world, coords, rtol=0, atol=extent * 1e-10)
                         and np.allclose(carried_world, expected_carried, rtol=0, atol=extent * 1e-10)
                         and np.allclose(recovered, expected_recovered, rtol=0, atol=extent * 1e-10)
                         and coordinate_error < 1e-10,
                         max_recovered_error_over_extent=coordinate_error)
            pair_error = float(np.max(np.abs(shared_remaining - shared_carried), initial=0))
            pose_error = float(np.max(np.abs(shared_carried - reference_color), initial=0))
            independent_delta = np.abs(independent_remaining - independent_carried)
            checks.check(name + "/shared_material_color_across_interface_and_poses",
                         pair_error <= 1e-10 and pose_error <= 1e-10,
                         shared_pair_max_linear_rgb_error=pair_error,
                         shared_pose_max_linear_rgb_error=pose_error,
                         independent_pair_mean_linear_rgb_difference=float(independent_delta.mean()),
                         independent_pair_max_linear_rgb_difference=float(independent_delta.max()),
                         limitation="Canonical unlit material correspondence, not real hidden anatomy or realism")
            for stream, expected in (("shared", shared_remaining), ("independent_remaining", independent_remaining),
                                     (f"independent_carried_pose{pose}", independent_carried)):
                with np.load(directory / "fields" / f"s{seed}_{stream}.npz", allow_pickle=False) as data:
                    grid = data["bounded_linear_rgb"].reshape(config["grid_size"], config["grid_size"], config["grid_size"], 3)
                ijk = np.clip((coords - low) / (high - low), 0, 1) * (config["grid_size"] - 1)
                expected_query = np.stack([map_coordinates(grid[..., c], ijk.T, order=1, mode="nearest", prefilter=False)
                                           for c in range(3)], axis=-1)
                checks.check(name + "/actual_grid_query/" + stream,
                             np.array_equal(expected, expected_query))
            if seed == config["seeds"][0] and pose == 0:
                distance = closest_distances(remaining_mesh, coords @ R.T)
                max_distance = float(distance.max(initial=0)) / extent
                checks.check(name + "/interface_belongs_to_remaining_mesh", max_distance < 1e-6,
                             all_samples=len(distance), max_distance_over_extent=max_distance)


def result_audit(directory, config, source_inputs, manifest, checks, prefix):
    import trimesh

    source_path = directory / "source.png"
    R = np.asarray(source_inputs["R_food_to_camera"])
    high, low = np.asarray(source_inputs["high"]), np.asarray(source_inputs["low"])
    extent = float((high - low).max())
    planned = {(pose, seed, variant) for pose in range(config["poses"])
               for seed in config["seeds"] for variant in config["variants"]}
    actual = [(row["pose"], row["seed"], row["variant"]) for row in manifest["results"]]
    checks.check(prefix + "/complete_result_matrix", set(actual) == planned and len(actual) == len(planned),
                 expected_images=len(planned), actual_images=len(actual))
    for pose in range(config["poses"]):
        pdir = directory / f"pose{pose}"
        record = read_json(pdir / "pose.json")
        with np.load(pdir / "geometry_channels.npz", allow_pickle=False) as data:
            labels, depth = data["labels"], data["depth"]
            fresh, sourcecoords, hit_yx = data["fresh"], data["sourcecoords"], data["hit_yx"]
        hit = labels > 0
        name = prefix + f"/pose{pose}"
        shape_valid = labels.shape == depth.shape == fresh.shape == (480, 640)
        checks.check(name + "/valid_labels_and_depth",
                     shape_valid and set(np.unique(labels)).issubset({0, 1, 2, 3})
                     and np.isfinite(depth[hit]).all() and np.all(depth[hit] > 0)
                     and np.array_equal(hit_yx, np.c_[np.where(hit)])
                     and sourcecoords.shape == (int(hit.sum()), 3) and np.isfinite(sourcecoords).all())
        food_pixels, spoon_pixels = int(np.sum(labels == 2)), int(np.sum(labels == 3))
        mesh = trimesh.load(pdir / "carried.ply", process=False)
        clearance = float(np.min(np.asarray(mesh.vertices) @ R[:, 2]) - high[2]) / extent
        clearance_rounding = abs(clearance - record["lift_clearance_over_extent"])
        # float32 PLY stores camera-space coordinates. Bound each component's
        # half-ULP error before the food-up projection; do not relax geometry.
        camera_vertices = np.asarray(mesh.vertices, dtype=np.float32)
        component_half_ulp = np.max(np.abs(np.spacing(camera_vertices)), axis=0).astype(np.float64) / 2
        serialization_bound = float(component_half_ulp @ np.abs(R[:, 2])) / extent + 1e-12
        checks.check(name + "/visible_proxy_lift_and_utensil",
                     food_pixels > 100 and spoon_pixels > 100 and clearance > 0
                     and food_pixels == record["visible_food_pixels"]
                     and spoon_pixels == record["visible_spoon_pixels"]
                     and clearance_rounding <= serialization_bound and serialization_bound <= 5e-6,
                     food_pixels=food_pixels, spoon_pixels=spoon_pixels,
                     lift_clearance_over_proxy_extent=clearance,
                     ply_float32_clearance_receipt_rounding=clearance_rounding,
                     ply_float32_half_ulp_bound_over_extent=serialization_bound,
                     limitation="Visible labeled proxy and geometry clearance; not human realism or measured contact")
        checks.check(name + "/fresh_surface_nonempty", bool(np.any(fresh)))
        for seed in config["seeds"]:
            for variant in config["variants"]:
                resultdir = pdir / f"{variant}_s{seed}"
                row = read_json(resultdir / "record.json")
                cell = name + f"/{seed}/{variant}"
                checks.check(cell + "/result_identity_and_hash",
                             row["case_id"] == directory.name and row["pose"] == pose
                             and row["seed"] == seed and row["variant"] == variant
                             and row["final_sha256"] == file_sha(resultdir / "composited.png")
                             and row["source_sha256"] == file_sha(source_path))
                original_outside_edit_audit(source_path, resultdir / "composited.png", pdir / "edit_mask.png", checks, cell)
                with Image.open(resultdir / "composited.png") as final:
                    cropped = np.asarray(final.crop(tuple(row["native_rect"])))
                checks.check(cell + "/view_is_declared_original_content_crop",
                             np.array_equal(cropped, rgb_image(resultdir / "view.png")))


def audit(root, sampling_device="cpu"):
    root = Path(root)
    checks = Checks()
    config = read_json(root / "config.json")
    expected_cases = ["new_01_7442", "new_02_7496", "new_03_7443", "new_04_7459",
                      "prospective_01_7473", "prospective_02_7441", "prospective_03_11160", "prospective_04_7498"]
    checks.check("plan/full_declared_matrix",
                 config["cases"] == expected_cases and config["seeds"] == [41, 163, 907]
                 and config["poses"] == 3 and config["variants"] == ["shared", "independent"]
                 and config["expected_images"] == 144 and config["optimizer_steps"] > 0)
    checks.check("plan/executed_script_frozen_hash",
                 file_sha(root / "scripts" / "run_material_lineage_pilot.py") == config["script_sha256"])
    summaries = []
    for cid in config["cases"]:
        directory = root / cid
        try:
            source_inputs = read_json(directory / "source_inputs.json")
            checks.check(cid + "/source_input_case_identity", source_inputs["case_id"] == cid)
            allowed_names = {"original.jpg", "source.png", "geometry_report.json", "appearance_calibration.json",
                             "maps.npz", "food_mask.png", "full.ply", "remaining.ply", "bite_source.ply",
                             "bite_lifted.ply", "fork.ply"}
            actual_names = [Path(row["path"]).name for row in source_inputs["inputs"]]
            checks.check(cid + "/no_old_generated_rgb_inputs",
                         set(actual_names) == allowed_names and len(actual_names) == len(allowed_names)
                         and all("gate_v" not in row["path"] and "composited" not in row["path"]
                                 and "raw_material" not in row["path"] for row in source_inputs["inputs"]),
                         input_names=actual_names,
                         limitation="Declared consumed files and source-derived colors; not a security proof of arbitrary code")
            for row in source_inputs["inputs"]:
                checks.check(cid + "/input_hash/" + Path(row["path"]).name,
                             file_sha(row["path"]) == row["sha256"])
            old_geometry_source = next(Path(row["path"]) for row in source_inputs["inputs"] if Path(row["path"]).name == "source.png")
            canonical_source = old_geometry_source.parents[2] / "inputs" / cid / "source.png"
            checks.check(cid + "/canonical_real_source_pixels",
                         np.array_equal(rgb_image(directory / "source.png"), rgb_image(canonical_source))
                         and np.array_equal(rgb_image(old_geometry_source), rgb_image(canonical_source)))
            receipt = read_json(directory / "training" / "training_receipt.json")
            samples_path = directory / "train_samples.npz"
            checks.check(cid + "/frozen_before_training",
                         config["created_unix"] <= receipt["created_unix"]
                         and receipt["seed"] == config["training_seed"]
                         and receipt["requested_optimizer_steps"] == config["optimizer_steps"]
                         and source_inputs["train_samples_sha256"] == file_sha(samples_path))
            checkpoint_audit(directory / "training" / "material_denoiser.pt", receipt, samples_path, checks, cid)
            source_sample_audit(samples_path, directory / "source.png", receipt, checks, cid)
            independent_source_geometry_audit(directory, source_inputs, checks, cid)
            budget = material_budget_audit(directory / "material_budget.npz", checks, cid)
            reported_budget = read_json(directory / "budget.json")
            checks.check(cid + "/budget_receipt_recomputed",
                         abs(budget["source_mass"] - reported_budget["source_mass"]) <= budget["source_mass"] * 1e-12
                         and abs(budget["carried_mass_fraction"] - reported_budget["quadrature_convergence"][-1]["carried_mass_fraction"]) < 1e-12)
            fields_audit(directory, config, receipt, checks, cid, sampling_device=sampling_device)
            probes_audit(directory, config, source_inputs, checks, cid)
            manifest = read_json(directory / "manifest.json")
            checks.check(cid + "/case_execution_completed", manifest["status"] == "complete_unreviewed"
                         and manifest["training_checkpoint_sha256"] == receipt["checkpoint_sha256"])
            result_audit(directory, config, source_inputs, manifest, checks, cid)
            summaries.append(dict(case_id=cid, results=len(manifest["results"]),
                                  optimizer_steps=receipt["actual_optimizer_steps"],
                                  parameter_count=receipt["parameter_count"], budget=budget,
                                  checkpoint_sha256=receipt["checkpoint_sha256"]))
        except Exception as error:
            checks.failure(cid + "/artifact_audit_exception", error)
    failures = [row for row in checks.rows if not row["passed"]]
    return dict(status="verified_internal_artifacts" if checks.passed else "failed_or_incomplete",
                scope=SCOPE, source_photos=len(summaries), expected_source_photos=8,
                expected_images=144, audited_images=sum(row["results"] for row in summaries),
                checks=len(checks.rows), failed_checks=len(failures), failures=failures,
                case_summaries=summaries, all_checks=checks.rows,
                claims=dict(full_material_lineage_diffusion_model_trained=False,
                            real_mass_measured=False, real_contact_measured=False,
                            heldout_food_generalization_established=False,
                            photographic_realism_established=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/host/space0/guo-z/tf-ufi/material_lineage_pilot_20261003"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sampling-device", default="cpu", help="cuda gives a strict bitwise field probe; cpu allows 5e-5 cross-device error")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        result = audit(args.root, sampling_device=args.sampling_device)
    except Exception as error:
        result = dict(status="failed_or_incomplete", scope=SCOPE,
                      failed_checks=1, failures=[dict(name="root_audit_exception", passed=False,
                                                    evidence=dict(error=f"{type(error).__name__}: {error}"))])
    output = args.output or args.root / "artifact_audit.json"
    temporary = Path(str(output) + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({key: result.get(key) for key in ("status", "checks", "failed_checks", "audited_images")}), flush=True)
    if result["status"] != "verified_internal_artifacts":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

