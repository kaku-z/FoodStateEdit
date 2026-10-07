"""Independent artifact audit for the MLD synthetic-pretraining experiment.

This checks provenance and declared measurements, not photographic realism.  It
never starts training and never changes checkpoints or datasets.  Synthetic
ground truth and real unpaired source probes are deliberately reported apart.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ALLOWED_INPUTS = {"source_rgb", "canonical_xyz", "source_uv", "actions"}
TARGET_ONLY_KEYS = {"joint", "occupancy", "cell_occupancy", "carried_fraction", "action_validmask"}
EXPECTED_MODULES = (
    "image_encoder", "source_transformer", "source_geometry_head",
    "source_material_head", "transition_head", "denoiser", "coordinate_projection", "action_encoder",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def independent_material_noise(identities: list[str], seed: int, channels: int = 4) -> np.ndarray:
    """Reconstruct the published persistent-ID noise contract without core import."""
    rows = []
    for identity in identities:
        values = []
        for pair in range((channels + 1) // 2):
            encoded = b"material-lineage-noise-v1"
            for part in (str(seed), "joint-field", identity, str(pair)):
                data = part.encode("utf-8")
                encoded += len(data).to_bytes(8, "little") + data
            digest = hashlib.sha256(encoded).digest()
            uniform = [((int.from_bytes(digest[start:start + 8], "little") >> 11) + .5) / 2 ** 53 for start in (0, 8)]
            radius = math.sqrt(-2 * math.log(uniform[0]))
            values.extend((radius * math.cos(2 * math.pi * uniform[1]), radius * math.sin(2 * math.pi * uniform[1])))
        rows.append(values[:channels])
    return np.asarray(rows, dtype=np.float32)


def validate_input_keys(keys: list[str] | set[str]) -> None:
    actual = set(keys)
    if not actual or not actual <= ALLOWED_INPUTS:
        raise ValueError(f"Illegal conditioning inputs: {sorted(actual - ALLOWED_INPUTS)}")
    if actual & TARGET_ONLY_KEYS:
        raise ValueError("Synthetic target information was declared as model input")


def validate_split_identity(scene_ids: list[str], splits: np.ndarray, expected_counts: tuple[int, int, int] | None = None) -> dict[str, int]:
    split_array = np.asarray(splits)
    if split_array.shape != (len(scene_ids),) or not np.isin(split_array, (0, 1, 2)).all():
        raise ValueError("Split codes must be one code (0, 1, 2) per scene")
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError("A synthetic shape/material scene identity occurs more than once")
    counts = tuple(int(np.count_nonzero(split_array == code)) for code in range(3))
    if expected_counts is not None and counts != expected_counts:
        raise ValueError(f"Split counts changed: {counts} != {expected_counts}")
    if any(count == 0 for count in counts):
        raise ValueError("Train, validation and test must all have identities")
    return dict(zip(("train", "validation", "test"), counts))


def validate_known_grid(xyz: np.ndarray, uv: np.ndarray, grid_size: int) -> None:
    centers = (np.arange(grid_size, dtype=np.float32) + np.float32(.5)) * np.float32(2 / grid_size) - np.float32(1)
    expected = np.stack(np.meshgrid(centers, centers, centers, indexing="ij"), axis=-1).reshape(-1, 3)
    if np.asarray(xyz).shape != expected.shape or not np.array_equal(xyz, expected):
        raise ValueError("Canonical coordinates are not the frozen scene-independent grid")
    if np.asarray(uv).shape != (len(expected), 2) or not np.isfinite(uv).all():
        raise ValueError("Source UV must be one fixed, finite projection per canonical grid point")


def validate_fraction_budget(fraction: np.ndarray, reference_mass: np.ndarray) -> dict[str, float]:
    fraction = np.asarray(fraction, dtype=np.float64)
    reference = np.asarray(reference_mass, dtype=np.float64)
    if fraction.shape[-1] != reference.shape[-1] or not np.isfinite(fraction).all() or not np.isfinite(reference).all():
        raise ValueError("Invalid fraction/reference mass arrays")
    if np.any(fraction < 0) or np.any(fraction > 1) or np.any(reference < 0) or np.any(reference > 1):
        raise ValueError("Fractions and discrete reference occupancy must lie in [0, 1]")
    while reference.ndim < fraction.ndim:
        reference = np.expand_dims(reference, -2)
    remaining = reference * (1 - fraction)
    carried = reference * fraction
    maximum = float(np.max(np.abs(remaining + carried - reference)))
    if maximum > 4 * np.finfo(np.float64).eps:
        raise ValueError("Reference mass is not conservatively allocated")
    return {"maximum_reference_budget_residual": maximum}


def checkpoint_differences(initial: dict[str, Any], final: dict[str, Any], expected_modules: tuple[str, ...] = EXPECTED_MODULES) -> dict[str, dict[str, float | int]]:
    if set(initial) != set(final):
        raise ValueError("Initial and final checkpoints contain different state keys")
    stats = {name: {"parameter_count": 0, "changed_parameters": 0, "squared_delta": 0.0} for name in expected_modules}
    for key in initial:
        a = initial[key].detach().cpu().numpy() if hasattr(initial[key], "detach") else np.asarray(initial[key])
        b = final[key].detach().cpu().numpy() if hasattr(final[key], "detach") else np.asarray(final[key])
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"Non-finite or mismatched checkpoint tensor: {key}")
        for name in expected_modules:
            if key.startswith(name + "."):
                if np.issubdtype(a.dtype, np.floating):
                    if key.endswith("frequencies"):
                        # Positional frequency buffers are fixed representation
                        # constants, not trainable model parameters.
                        continue
                    delta = b.astype(np.float64) - a.astype(np.float64)
                    stats[name]["parameter_count"] += int(a.size)
                    stats[name]["changed_parameters"] += int(np.count_nonzero(delta))
                    stats[name]["squared_delta"] += float(np.sum(delta * delta))
                break
    for name, values in stats.items():
        if values["parameter_count"] == 0 or values["changed_parameters"] == 0:
            raise ValueError(f"Required learned module is missing or unchanged: {name}")
        values["delta_l2"] = math.sqrt(values.pop("squared_delta"))
    return stats


def expected_modules_for_config(config: dict[str, Any]) -> tuple[str, ...]:
    state_dim = int(config.get("state_dim", 0))
    if state_dim not in (0, 4):
        raise ValueError("Unknown transition state schema")
    return EXPECTED_MODULES + (("state_encoder",) if state_dim else ())


def validate_probe_exclusion(probes: list[dict[str, Any]], history: dict[str, Any]) -> None:
    excluded_ids = {str(x) for x in history.get("source_numeric_ids", [])}
    excluded_names = {Path(x).name for x in history.get("source_file_names", [])}
    excluded_hashes = set(history.get("known_source_sha256", []))
    observed = set()
    for probe in probes:
        name = Path(probe.get("source_file_name", probe.get("file_name", probe.get("source_path", "")))).name
        digest = probe.get("sha256", probe.get("source_sha256"))
        if not name or not digest or len(digest) != 64:
            raise ValueError("Every real source probe must identify its filename and SHA-256")
        if name in excluded_names or Path(name).stem in excluded_ids or digest in excluded_hashes:
            raise ValueError(f"Historical development source was labelled as a new probe: {name}")
        if digest in observed:
            raise ValueError("Duplicate source content occurs in real probes")
        observed.add(digest)


def audit_real_probes(audit, manifest_path: Path, history_path: Path | None) -> None:
    from PIL import Image, ImageOps
    manifest = read_json(manifest_path)
    root = manifest_path.parent
    records = manifest.get("cases", [])
    require(bool(records) and len(records) == manifest.get("selected_count"), "Real probe count changed")
    recipe_path = root / "selection_recipe.json"
    recipe = read_json(recipe_path)
    history_path = history_path or Path(recipe["history_path"])
    history = read_json(history_path)
    audit.check("real source identity exclusion", validate_probe_exclusion, records, history)
    audit.check("history exclusion hash frozen before selection", require, sha256_file(history_path) == manifest["history_sha256"] == recipe["history_sha256"], "Historical source exclusions changed after selection")
    audit.check("real selection recipe hash", require, sha256_file(recipe_path) == manifest["selection_recipe_sha256"], "Real source selection recipe changed")
    audit.check("real source probe inputs are only pixels", require, recipe.get("model_inputs") == ["source_rgb"] and "never" in recipe.get("captions_policy", "").lower(), "Real source metadata was declared as model input")
    for record in manifest["files"]:
        path = root / record["path"]
        audit.check(f"real probe artifact {record['path']}", require, path.is_file() and sha256_file(path) == record["sha256"], "Missing or changed real probe artifact")
    candidates = read_json(root / "candidate_order.json")["candidates"]
    rank = lambda name: hashlib.sha256(("mld-real-probe-20261004:" + name).encode()).hexdigest()
    names = [entry["file_name"] for entry in candidates]
    audit.check("real candidate deterministic rank rule", require, len(set(names)) == len(names) and names == sorted(names, key=lambda name: (rank(name), name)) and all(entry["rank_sha256"] == rank(entry["file_name"]) and entry["rank"] == i for i, entry in enumerate(candidates)), "Real candidate order differs from its frozen metadata-only rule")
    audit.check("real candidate order recipe hash", require, sha256_file(root / "candidate_order.json") == recipe["candidate_order_sha256"], "Candidate order changed")
    historical = read_json(root / "historical_dhash_reference.json")
    audit.check("all available historical sources were decoded for duplicate exclusion", require, not historical.get("unreadable") and len(historical["references"]) == len(history["source_file_names"]), "Historical near-duplicate exclusion has missing references")
    def open_rgb(path):
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB").copy()
    def independent_dhash(image):
        grey = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
        bit_string = "".join("1" if value else "0" for value in (grey[:, 1:] > grey[:, :-1]).ravel())
        return int(bit_string, 2)
    references = []
    source_directory = Path(recipe["source_directory"])
    for row in historical["references"]:
        path = source_directory / row["file_name"]
        dhash = independent_dhash(open_rgb(path))
        audit.check(f"historical duplicate-reference source {row['file_name']}", require, sha256_file(path) == row["source_sha256"] and dhash == int(row["dhash64"], 16), "Historical source hash or perceptual duplicate reference changed")
        references.append((row["file_name"], dhash))
    stack = np.load(root / "source_rgb.npy", allow_pickle=False)
    decision_rows = read_json(root / "selection_decisions.json")["decisions"]
    audit.check("real candidate decisions form a prefix without skipped ranking positions", require, all(entry["candidate_rank"] == i and entry["file_name"] == candidates[i]["file_name"] for i, entry in enumerate(decision_rows)), "A candidate was skipped or chosen out of frozen rank order")
    selected = [entry for entry in decision_rows if entry["status"] == "selected"]
    audit.check("real selected identities agree with decision prefix", require, [entry["file_name"] for entry in selected] == [entry["source_file_name"] for entry in records], "Selected probe identities differ from recorded decisions")
    minimum_distances = []
    for index, row in enumerate(records):
        original = root / row["copied_original_path"]
        rgb = open_rgb(original)
        dhash = independent_dhash(rgb)
        minimum = min(((dhash ^ reference).bit_count() for _, reference in references), default=65)
        minimum_distances.append(minimum)
        audit.check(f"real probe original provenance/{index}", require, sha256_file(original) == row["source_sha256"] == row["copied_original_sha256"], "Copied original no longer matches frozen source bytes")
        audit.check(f"real probe independent duplicate distance/{index}", require, dhash == int(row["dhash64"], 16) and minimum == row["nearest_historical_or_previously_selected_distance"] and minimum > recipe["dhash"]["max_distance_rejected"], "Selected source violates frozen near-duplicate exclusion")
        size = int(recipe["source_rgb_size"])
        width, height = rgb.size
        scale = min(size / width, size / height)
        resized = (max(1, min(size, round(width * scale))), max(1, min(size, round(height * scale))))
        canvas = Image.new("RGB", (size, size), (255, 255, 255))
        canvas.paste(rgb.resize(resized, Image.Resampling.LANCZOS), ((size - resized[0]) // 2, (size - resized[1]) // 2))
        pixels = np.asarray(canvas)
        per_source = np.load(root / row["source_rgb_path"], allow_pickle=False)
        audit.check(f"real probe independent preprocessing/{index}", require, bool(np.array_equal(pixels, per_source) and np.array_equal(pixels, stack[index])), "Model pixels differ from independent EXIF/letterbox replay")
        references.append((row["source_file_name"], dhash))
    audit.measurements["unpaired_real_probe_count"] = len(records)
    audit.measurements["minimum_historical_or_selected_dhash_distance"] = min(minimum_distances)
    audit.measurements["real_probe_scope"] = "Unpaired source identity/domain probes. No target edit images or measured 3D labels; not a real edit-success test."
    audit.measurements["real_probe_manifest_sha256"] = sha256_file(manifest_path)


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.measurements: dict[str, Any] = {}

    def check(self, name: str, function, *args, **kwargs):
        try:
            value = function(*args, **kwargs)
            self.checks.append({"check": name, "passed": True})
            return value
        except Exception as exc:
            self.checks.append({"check": name, "passed": False, "reason": f"{type(exc).__name__}: {exc}"})
            return None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_generator(path: Path):
    spec = importlib.util.spec_from_file_location("mld_audit_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import generator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def audit_dataset(audit: Audit, root: Path, generator_path: Path) -> dict[str, np.ndarray] | None:
    manifest = audit.check("dataset manifest readable", read_json, root / "manifest.json")
    if manifest is None:
        return None
    audit.measurements["dataset_manifest_sha256"] = sha256_file(root / "manifest.json")
    audit.measurements["dataset_seed"] = int(manifest["seed"])
    audit.check("dataset generated completely", require, "complete" in str(manifest.get("status", "")).lower(), "Dataset is incomplete")
    inputs = manifest.get("inputs", manifest.get("input_keys", []))
    if isinstance(inputs, dict):
        inputs = list(inputs)
    audit.check("source/action conditioning only", validate_input_keys, inputs)
    audit.check("declared experiment conditioning is complete", require, set(inputs) == ALLOWED_INPUTS, "Dataset input schema changed")
    for record in manifest.get("files", []):
        relative = Path(record["path"])
        path = root / relative
        audit.check(f"dataset artifact {relative.as_posix()}", require, path.is_file() and sha256_file(path) == record["sha256"], f"Missing or changed dataset file {relative}")
    required = ("canonical_xyz", "source_uv", "source_rgb", "joint", "occupancy", "actions", "carried_fraction", "cell_occupancy", "action_validmask", "splits")
    arrays: dict[str, np.ndarray] = {}
    for name in required:
        array = audit.check(f"load {name}", np.load, root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        if array is not None:
            arrays[name] = array
    if len(arrays) != len(required):
        return None
    count, grid, image = int(manifest["scene_count"]), int(manifest["grid_size"]), int(manifest["image_size"])
    tokens = grid ** 3
    shapes = {"canonical_xyz": (tokens, 3), "source_uv": (tokens, 2), "source_rgb": (count, image, image, 3), "joint": (count, tokens, 4), "occupancy": (count, tokens), "actions": (count, 4, 10), "carried_fraction": (count, 4, tokens), "cell_occupancy": (count, tokens), "action_validmask": (count, tokens), "splits": (count,)}
    for name, shape in shapes.items():
        audit.check(f"shape {name}", require, arrays[name].shape == shape, f"{arrays[name].shape} != {shape}")
        audit.check(f"finite {name}", require, bool(np.isfinite(arrays[name]).all()), f"Non-finite {name}")
    audit.check("frozen canonical grid and shared UV", validate_known_grid, arrays["canonical_xyz"], arrays["source_uv"], grid)
    camera = manifest.get("camera", {})
    if "basis_right_up_toward_camera" in camera and "scale" in camera:
        basis = np.asarray(camera["basis_right_up_toward_camera"], dtype=np.float64)
        projected = np.asarray(arrays["canonical_xyz"], dtype=np.float64) @ basis[:2].T / float(camera["scale"])
        projected[:, 1] *= -1
        audit.check("independent camera equation agrees with shared UV", require, bool(np.array_equal(projected.astype(np.float32), arrays["source_uv"])), "Projection differs from the declared fixed camera")
    else:
        audit.check("known camera equation recorded", require, False, "No independent camera basis/scale in dataset manifest")
    audit.check("joint field channel normalization", require, bool(np.all((arrays["joint"] >= -1) & (arrays["joint"] <= 1))), "Joint target channels exceed signed normalization bounds")
    scenes_path = root / "scenes.jsonl"
    scenes = audit.check("scene metadata readable", lambda: [json.loads(line) for line in scenes_path.read_text(encoding="utf-8").splitlines() if line.strip()])
    if scenes is not None:
        identities = [scene.get("scene_id", scene.get("identity")) for scene in scenes]
        audit.check("all scene identities have metadata", require, len(scenes) == count and all(identities), "Missing scene metadata or identity")
        expected_counts = (int(count * .75), int(count * .125), count - int(count * .75) - int(count * .125))
        counts = audit.check("identity-group disjoint train/validation/test", validate_split_identity, identities, arrays["splits"], expected_counts)
        audit.measurements["synthetic_identity_counts"] = counts
        seed = int(manifest["seed"])
        audit.check("scene identity generator committed before sampling", require, all(identity == hashlib.sha256(f"mld-v1:{seed}:{index}".encode()).hexdigest() for index, identity in enumerate(identities)), "Scene identity differs from frozen hash rule")
    for start in range(0, count, 128):
        stop = min(start + 128, count)
        audit.check(f"fraction budgets scenes {start}:{stop}", validate_fraction_budget, arrays["carried_fraction"][start:stop], arrays["cell_occupancy"][start:stop])
    audit.check("zero action is actually zero", require, bool(np.all(arrays["actions"][:, 0] == 0)), "Action zero column is nonzero")
    audit.check("zero-action material stays in source branch", require, bool(np.all(arrays["carried_fraction"][:, 0] == 0)), "Zero action has carried material")
    audit.check("cell support mask is target-only occupancy support", require, bool(np.array_equal(arrays["action_validmask"], arrays["cell_occupancy"] > 0)), "Action validity mask does not match reference cell support")
    generator = audit.check("procedural generator import", load_generator, generator_path)
    if generator is not None:
        audit.check("UV depends only on fixed grid and known camera", require, bool(np.array_equal(generator.known_camera_projection(np.asarray(arrays["canonical_xyz"])), arrays["source_uv"])), "Saved UV differs from the known-camera projection")
        replay_indices = sorted(set([0, count - 1] + [int(np.flatnonzero(arrays["splits"] == code)[i]) for code in (0, 1, 2) for i in (0, -1)]))
        for index in replay_indices:
            replay = audit.check(f"procedural replay scene {index}", generator.generate_scene, index=index, seed=int(manifest["seed"]), grid_size=grid, image_size=image)
            if replay is None:
                continue
            record, metadata = replay
            if scenes is not None:
                for key in ("scene_id", "source_rgb_sha256", "joint_sha256", "actions_sha256"):
                    audit.check(f"replay scene {index}/metadata {key}", require, metadata.get(key) == scenes[index].get(key), f"Replayed metadata does not agree for {key}")
            for name in required:
                if name not in record:
                    continue
                actual = arrays[name][index]
                audit.check(f"replay scene {index}/{name}", require, bool(np.array_equal(np.asarray(record[name]).astype(actual.dtype), actual)), f"Deterministic replay differs for scene {index}/{name}")
        audit.measurements["replayed_scene_indices"] = replay_indices
        audit.measurements["generator_script_sha256"] = sha256_file(generator_path)
    return arrays


def _checkpoint_state(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(payload.get(key), dict):
            return payload[key]
    raise ValueError("Checkpoint does not contain a model state dict")


def audit_training(audit: Audit, root: Path, initial_path: Path, final_path: Path, receipt_path: Path) -> None:
    import torch
    receipt = audit.check("training receipt readable", read_json, receipt_path)
    initial = audit.check("initial checkpoint readable", torch.load, initial_path, map_location="cpu", weights_only=True)
    final = audit.check("final checkpoint readable", torch.load, final_path, map_location="cpu", weights_only=True)
    if initial is None or final is None or receipt is None:
        return
    state_initial = audit.check("initial checkpoint state", _checkpoint_state, initial)
    state_final = audit.check("final checkpoint state", _checkpoint_state, final)
    expected_modules = expected_modules_for_config(final.get("model_config", {}))
    audit.measurements["transition_state_dim"] = int(final.get("model_config", {}).get("state_dim", 0))
    if state_initial is not None and state_final is not None:
        changes = audit.check("every required learned module changed", checkpoint_differences, state_initial, state_final, expected_modules)
        audit.measurements["learned_module_weight_changes"] = changes
    actual_steps = receipt.get("actual_optimizer_steps", receipt.get("optimizer_steps", receipt.get("steps")))
    planned = receipt.get("planned_optimizer_steps", receipt.get("planned_steps", actual_steps))
    final_steps = final.get("optimizer_steps", final.get("step", final.get("steps")))
    audit.check("actual optimizer step count is positive and completed", require, isinstance(actual_steps, int) and actual_steps > 0 and actual_steps == planned == final_steps, f"Step count mismatch receipt={actual_steps}, planned={planned}, checkpoint={final_steps}")
    audit.measurements["actual_optimizer_steps"] = actual_steps
    audit.check("training and evaluation finished", require, receipt.get("status") == "complete_pretraining", f"Receipt status is {receipt.get('status')}; running or failed jobs cannot pass a final audit")
    audit.check("receipt dataset hash matches frozen dataset", require, receipt.get("dataset_manifest_sha256") == audit.measurements.get("dataset_manifest_sha256"), "Training receipt does not identify the audited dataset")
    if "input_keys" in receipt:
        audit.check("actual training input keys", validate_input_keys, receipt["input_keys"])
    gradients = receipt.get("per_module_gradient_nonzero_observations", receipt.get("gradient_receipts", {}))
    for name in expected_modules:
        value = gradients.get(name)
        if isinstance(value, dict):
            observed = value.get("nonzero_observations", value.get("nonzero_count", value.get("nonzero", 0)))
            finite = value.get("all_finite", value.get("finite", False))
        else:
            observed, finite = value or 0, value is not None
        audit.check(f"actual gradient observation/{name}", require, int(observed) > 0 and bool(finite), f"No finite nonzero actual gradient receipt for {name}")
    for label, path in (("initial", initial_path), ("final", final_path)):
        audit.measurements[f"{label}_checkpoint_sha256"] = sha256_file(path)
    if receipt.get("final_checkpoint_sha256"):
        audit.check("final checkpoint matches receipt hash", require, receipt["final_checkpoint_sha256"] == audit.measurements["final_checkpoint_sha256"], "Final checkpoint differs from receipt")
    if (root / "config.json").is_file():
        config = read_json(root / "config.json")
        audit.check("frozen training config hash", require, receipt.get("config_sha256") == sha256_file(root / "config.json"), "Training config changed")
        audit.check("heldout action excluded from optimizer updates", require, config.get("training_action_indices") == [0, 1, 2] and config.get("heldout_action_index") == 3, "Action split differs from frozen recipe")
    optimizer = final.get("optimizer_state_dict", {}).get("state", {})
    optimizer_steps = [int(value["step"].item() if hasattr(value.get("step"), "item") else value["step"]) for value in optimizer.values() if "step" in value]
    audit.check("optimizer moments confirm actual terminal updates", require, bool(optimizer_steps) and max(optimizer_steps) == actual_steps and min(optimizer_steps) > 0, "Saved optimizer state does not substantiate update count")
    audit.measurements["optimizer_parameter_step_range"] = [min(optimizer_steps), max(optimizer_steps)] if optimizer_steps else None
    if initial.get("model_config") and final.get("model_config"):
        audit.check("initial/final model architecture identical", require, initial["model_config"] == final["model_config"], "Architecture changed between checkpoints")
    audit.measurements["training_receipt_sha256"] = sha256_file(receipt_path)


def audit_evaluation(audit: Audit, path: Path, arrays: dict[str, np.ndarray] | None) -> None:
    evaluation = audit.check("evaluation artifact readable", read_json, path)
    if evaluation is None:
        return
    audit.measurements["evaluation_sha256"] = sha256_file(path)
    keys = evaluation.get("input_keys", evaluation.get("allowed_conditioning_inputs", []))
    audit.check("evaluation uses source/action inputs only", validate_input_keys, keys)
    audit.check("synthetic truth and real unpaired scope separated", require, evaluation.get("scope") is not None, "Evaluation must declare what its targets establish")
    # Trainer may report many metrics; the independent NPZ audit below establishes
    # their indexed target provenance rather than trusting a JSON score alone.
    prediction_path = path.parent / "evaluation_predictions.npz"
    if not prediction_path.is_file():
        audit.check("saved evaluation predictions independently available", require, False, "JSON-only evaluation cannot establish recomputed prediction metrics")
        return
    predictions = audit.check("evaluation predictions load", np.load, prediction_path, allow_pickle=False)
    if predictions is None or arrays is None:
        return
    audit.measurements["evaluation_predictions_sha256"] = sha256_file(prediction_path)
    if "scene_indices" not in predictions or "token_indices" not in predictions:
        audit.check("evaluation indexes identify frozen targets", require, False, "Missing scene/token indices")
        return
    scenes = predictions["scene_indices"].astype(np.int64)
    tokens = predictions["token_indices"].astype(np.int64)
    if tokens.ndim == 1:
        tokens = np.broadcast_to(tokens, (len(scenes), len(tokens)))
    audit.check("evaluation indices in bounds", require, bool(np.all((scenes >= 0) & (scenes < len(arrays["splits"]))) and np.all((tokens >= 0) & (tokens < len(arrays["canonical_xyz"])))), "Evaluation target index outside dataset")
    audit.measurements["evaluation_identity_counts"] = {label: int(np.count_nonzero(arrays["splits"][scenes] == code)) for code, label in enumerate(("train", "validation", "test"))}
    declared_code = {"validation": 1, "test": 2}.get(evaluation.get("split"))
    audit.check("declared heldout inference split", require, declared_code is not None and bool(np.all(arrays["splits"][scenes] == declared_code)), "Prediction identities do not belong entirely to their declared heldout split")
    audit.check("unique heldout scene identities", require, len(np.unique(scenes)) == len(scenes), "An identity was counted repeatedly in heldout evaluation")
    audit.check("saved split labels agree with frozen identity split", require, "split_codes" in predictions and bool(np.array_equal(predictions["split_codes"], arrays["splits"][scenes])), "Evaluation split labels changed")
    if "predictions_sha256" in evaluation:
        audit.check("evaluation prediction hash matches JSON", require, evaluation["predictions_sha256"] == audit.measurements["evaluation_predictions_sha256"], "Prediction archive differs from evaluated archive")
    target_joint = np.asarray(arrays["joint"][scenes[:, None], tokens])
    if audit.measurements.get("transition_state_dim", 0) == 4:
        audit.check("transition reads one shared sampled canonical state", require, "transition_state_joint" in predictions and bool(np.array_equal(predictions["transition_state_joint"], predictions["trained_joint"])), "State-conditioned transition does not document the same sampled field")
        audit.check("initial transition uses its own shared initial sample", require, "initial_transition_state_joint" in predictions and bool(np.array_equal(predictions["initial_transition_state_joint"], predictions["initial_joint"])), "Initial model transition state differs from initial sampled canonical field")
        invariants = evaluation.get("representation_invariants", {})
        audit.check("predicted training state and sampled inference state distinguished", require, invariants.get("transition_reads_same_sampled_source_state") is True and "Predicted posterior" in invariants.get("transition_training_state", "") and "never target" in invariants.get("transition_training_state", ""), "State coupling provenance is absent or confuses supervision with model inputs")
    # Identity formula is frozen by the dataset audit; all probes use the same
    # dataset generator seed, recorded independently of the training noise seed.
    dataset_seed = audit.measurements.get("dataset_seed")
    scene_ids = [hashlib.sha256(f"mld-v1:{dataset_seed}:{int(index)}".encode()).hexdigest() for index in scenes] if dataset_seed is not None else []
    if scene_ids and "material_id_noise" in predictions:
        ids = [f"{identity}:lattice:{int(token)}" for identity, token_row in zip(scene_ids, tokens) for token in token_row]
        noise = independent_material_noise(ids, int(evaluation["sampling_seed"])).reshape(target_joint.shape)
        audit.check("independent persistent material-ID sampling noise", require, bool(np.array_equal(noise, predictions["material_id_noise"])), "Sampling noise depends on row/order or differs from the scene/lattice material ID contract")
    else:
        audit.check("persistent material-ID noise provenance available", require, False, "Missing source identities or sampled noise array")
    if "target_joint" in predictions:
        audit.check("saved joint targets match frozen dataset", require, bool(np.array_equal(predictions["target_joint"], target_joint)), "Target joint values changed during evaluation")
    for name in ("trained_joint", "initial_joint", "source_shuffled_joint"):
        audit.check(f"counterfactual prediction exists/{name}", require, name in predictions, f"Missing required prediction {name}")
        if name in predictions:
            prediction = predictions[name]
            audit.check(f"finite prediction/{name}", require, prediction.shape == target_joint.shape and bool(np.isfinite(prediction).all()), "Invalid joint prediction")
            audit.measurements[f"independent_{name}_mse"] = float(np.mean((prediction.astype(np.float64) - target_joint.astype(np.float64)) ** 2))
    target_occupancy = np.asarray(arrays["occupancy"][scenes[:, None], tokens], dtype=np.float64)[..., None]
    target_valid = np.asarray(arrays["action_validmask"][scenes[:, None], tokens], dtype=np.float64)[..., None]
    target_mass = np.asarray(arrays["cell_occupancy"][scenes[:, None], tokens], dtype=np.float64)[..., None]
    for name, target in (("target_occupancy", target_occupancy), ("target_valid", target_valid), ("target_mass", target_mass)):
        audit.check(f"saved loss support target/{name}", require, name in predictions and bool(np.array_equal(predictions[name], target)), "Evaluation occupancy/reference mass target does not match the frozen dataset")
    recomputed: dict[str, Any] = {}
    def weighted_mean(value, weight):
        value = np.asarray(value, dtype=np.float64)
        weight = np.broadcast_to(np.asarray(weight, dtype=np.float64), value.shape)
        return float(np.sum(value * weight) / max(float(np.sum(weight)), 1.0))
    def field_metrics(sdf, rgb, logits):
        occupied = target_occupancy > .5
        predicted = np.asarray(logits) > 0
        return {"occupancy_iou": float(np.count_nonzero(predicted & occupied) / max(np.count_nonzero(predicted | occupied), 1)), "sdf_mae": float(np.mean(np.abs(np.asarray(sdf, dtype=np.float64) - target_joint[..., :1]))), "occupied_linear_rgb_mse": weighted_mean(((np.asarray(rgb, dtype=np.float64) - target_joint[..., 1:]) / 2) ** 2, target_occupancy)}
    for prefix, name in (("sampled_joint", "trained_joint"), ("initial_sampled_joint", "initial_joint"), ("source_shuffled_sampled_joint", "source_shuffled_joint")):
        if name in predictions:
            value = predictions[name]
            recomputed[prefix] = field_metrics(value[..., :1], value[..., 1:], -value[..., :1])
    if all(name in predictions for name in ("posterior_sdf", "posterior_rgb", "posterior_occupancy_logits")):
        recomputed["posterior"] = field_metrics(predictions["posterior_sdf"], predictions["posterior_rgb"], predictions["posterior_occupancy_logits"])
    if "action_indices" in predictions:
        actions = predictions["action_indices"].astype(np.int64)
        target_fraction = np.asarray(arrays["carried_fraction"][scenes[:, None], actions[:, None], tokens])[..., None]
        if "target_carried_fraction" in predictions:
            audit.check("saved material fractions match action-indexed dataset", require, bool(np.array_equal(predictions["target_carried_fraction"], target_fraction)), "Fraction target/action changed")
        for name in ("trained_fraction", "initial_fraction", "action_shuffled_fraction", "zero_fraction"):
            audit.check(f"counterfactual prediction exists/{name}", require, name in predictions, f"Missing {name}")
            if name in predictions:
                prediction = predictions[name]
                audit.check(f"valid bounded prediction/{name}", require, prediction.shape == target_fraction.shape and bool(np.isfinite(prediction).all()) and bool(np.all((prediction >= 0) & (prediction <= 1))), "Invalid carried fraction")
                target = np.zeros_like(target_fraction) if name == "zero_fraction" else target_fraction
                audit.measurements[f"independent_{name}_mse"] = float(np.mean((prediction.astype(np.float64) - target.astype(np.float64)) ** 2))
        def independent_rigid_delta(commands):
            xyz = np.asarray(arrays["canonical_xyz"])[tokens].astype(np.float64)
            values = np.empty_like(xyz)
            for i, command in enumerate(np.asarray(commands, dtype=np.float64)):
                angle = np.linalg.norm(command[7:10])
                if angle < 1e-12:
                    rotation = np.eye(3)
                else:
                    x, y, z = command[7:10] / angle
                    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                    rotation = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
                values[i] = xyz[i] @ rotation.T + command[4:7] - xyz[i]
            return values
        commands = arrays["actions"][scenes, actions]
        target_delta = independent_rigid_delta(commands)
        audit.check("independent Rodrigues rigid target", require, "target_carried_delta" in predictions and bool(np.allclose(predictions["target_carried_delta"], target_delta, atol=1e-6, rtol=0)), "Rigid displacement target differs from action-only Rodrigues computation")
        ood_fraction = np.asarray(arrays["carried_fraction"][scenes[:, None], np.full((len(scenes), 1), 3), tokens])[..., None]
        ood_delta = independent_rigid_delta(arrays["actions"][scenes, np.full(len(scenes), 3)])
        audit.check("heldout action3 target fraction", require, "ood_target_fraction" in predictions and bool(np.array_equal(predictions["ood_target_fraction"], ood_fraction)), "Heldout action target fractions changed")
        audit.check("heldout action3 independent rigid target", require, "ood_target_carried_delta" in predictions and bool(np.allclose(predictions["ood_target_carried_delta"], ood_delta, atol=1e-6, rtol=0)), "Heldout rigid displacement differs from action")
        names = ("trained_fraction", "initial_fraction", "action_shuffled_fraction", "zero_fraction", "trained_carried_delta", "zero_carried_delta", "ood_trained_fraction", "ood_trained_carried_delta", "trained_allocation")
        if all(name in predictions for name in names):
            allocation = np.asarray(predictions["trained_allocation"])
            audit.check("learned allocation pairs match predicted carried fraction", require, allocation.shape == (*target_fraction.shape[:2], 2) and bool(np.array_equal(allocation[..., 1:], predictions["trained_fraction"])) and bool(np.allclose(allocation[..., :1], 1 - predictions["trained_fraction"], atol=1e-7, rtol=0)), "Allocation does not preserve its branch fraction identity")
            recomputed["transition"] = {
                "fraction_mae": weighted_mean(np.abs(predictions["trained_fraction"] - target_fraction), target_valid),
                "initial_fraction_mae": weighted_mean(np.abs(predictions["initial_fraction"] - target_fraction), target_valid),
                "action_shuffled_fraction_mae": weighted_mean(np.abs(predictions["action_shuffled_fraction"] - target_fraction), target_valid),
                "carried_endpoint_rmse": math.sqrt(weighted_mean(np.sum((predictions["trained_carried_delta"] - predictions["target_carried_delta"]) ** 2, axis=-1, keepdims=True), target_mass * target_fraction)),
                "zero_fraction_mean": weighted_mean(predictions["zero_fraction"], target_valid),
                "zero_carried_delta_rms": math.sqrt(weighted_mean(predictions["zero_carried_delta"] ** 2, target_mass)),
                "ood_fraction_mae": weighted_mean(np.abs(predictions["ood_trained_fraction"] - ood_fraction), target_valid),
                "ood_carried_endpoint_rmse": math.sqrt(weighted_mean(np.sum((predictions["ood_trained_carried_delta"] - predictions["ood_target_carried_delta"]) ** 2, axis=-1, keepdims=True), target_mass * ood_fraction)),
                "max_allocation_budget_error": float(np.max(np.abs(allocation.sum(-1) - 1))),
            }
            if audit.measurements.get("transition_state_dim", 0) == 4:
                for key in ("posterior_state_fraction", "state_shuffled_fraction"):
                    audit.check(f"state coupling diagnostic saved/{key}", require, key in predictions and predictions[key].shape == target_fraction.shape and bool(np.isfinite(predictions[key]).all()), "Missing or invalid state ablation prediction")
                if "posterior_state_fraction" in predictions and "state_shuffled_fraction" in predictions:
                    recomputed["transition"].update({
                        "posterior_state_fraction_mae": weighted_mean(np.abs(predictions["posterior_state_fraction"] - target_fraction), target_valid),
                        "state_shuffled_fraction_mae": weighted_mean(np.abs(predictions["state_shuffled_fraction"] - target_fraction), target_valid),
                        "state_shuffle_output_change_mae": weighted_mean(np.abs(predictions["state_shuffled_fraction"] - predictions["trained_fraction"]), target_valid),
                    })
    for group, metrics in recomputed.items():
        for metric, value in metrics.items():
            actual = evaluation.get(group, {}).get(metric)
            audit.check(f"independent reported metric/{group}.{metric}", require, isinstance(actual, (int, float)) and math.isfinite(actual) and math.isclose(actual, value, rel_tol=2e-5, abs_tol=1e-7), f"Reported {actual} differs from independent {value}")
    audit.measurements["independently_recomputed_metrics"] = recomputed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--generator", type=Path, default=Path(__file__).with_name("prepare_mld_pretraining_data.py"))
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--final-checkpoint", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--history-sources", type=Path)
    parser.add_argument("--real-probes", type=Path)
    parser.add_argument("--dataset-only", action="store_true")
    args = parser.parse_args()
    audit = Audit()
    arrays = audit_dataset(audit, args.dataset_root, args.generator)
    if not args.dataset_only:
        audit.check("training audit runs", audit_training, audit, args.run_root, args.initial_checkpoint or args.run_root / "initial_checkpoint.pt", args.final_checkpoint or args.run_root / "final_checkpoint.pt", args.receipt or args.run_root / "training_receipt.json")
        audit.check("evaluation audit runs", audit_evaluation, audit, args.evaluation or args.run_root / "evaluation.json", arrays)
    if args.real_probes is not None:
        audit.check("independent real probe selection/preprocessing audit", audit_real_probes, audit, args.real_probes, args.history_sources)
    failures = sum(not check["passed"] for check in audit.checks)
    result = {"schema_version": "mld.training_audit.v1", "status": "passed" if failures == 0 else "failed", "scope": "Artifact provenance, deterministic synthetic targets, learned weight changes and reproducible counterfactual inference. No measured real 3D or photorealism claim.", "check_count": len(audit.checks), "failed_count": failures, "checks": audit.checks, "measurements": audit.measurements}
    args.run_root.mkdir(parents=True, exist_ok=True)
    output = args.run_root / ("dataset_audit.json" if args.dataset_only else "audit.json")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "checks": len(audit.checks), "failed": failures, "status": result["status"]}))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
