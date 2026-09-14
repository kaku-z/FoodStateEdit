"""Deterministic contracts for geometry-locked local appearance refinement.

The learned editor is deliberately kept outside this module.  It proposes a
local candidate; these helpers enforce exact support, measure structural
retention, and decide whether the proposal may replace the pre-refinement
frame.  A failed or incomplete gate always rolls back to the input frame.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


SCHEMA_VERSION = "foodstateedit.geometry_locked_refinement.v1"
REQUIRED_SEMANTIC_CHECKS = (
    "utensil_identity_preserved",
    "utensil_count_preserved",
    "contact_relation_preserved",
    "payload_identity_preserved",
    "source_correspondence_preserved",
    "photo_realism_improved",
)


def validate_refinement_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected refinement schema")
    if config.get("backend", {}).get("method") != "ChordEdit_SD_Turbo":
        raise ValueError("v1 freezes ChordEdit_SD_Turbo as the refinement backend")
    if config.get("candidate_policy", {}).get("candidate_count") != 1:
        raise ValueError("v1 forbids multi-candidate best-of selection")
    if not config.get("composition", {}).get("exact_outside_hard_support"):
        raise ValueError("exact outside-support compositing is required")
    gates = config.get("acceptance_gate", {})
    if tuple(gates.get("required_manual_checks", ())) != REQUIRED_SEMANTIC_CHECKS:
        raise ValueError("manual semantic gate is incomplete or reordered")
    if not gates.get("uncertain_is_failure") or not gates.get("rollback_on_any_failure"):
        raise ValueError("uncertain or failed refinement must roll back")


def _rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _mask_array(mask: Image.Image, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(mask.convert("L").resize(size, Image.Resampling.NEAREST), dtype=np.uint8)


def strict_local_composite(
    base: Image.Image,
    edited: Image.Image,
    soft_mask: Image.Image,
    hard_support: Image.Image,
    *,
    alpha: float,
) -> Image.Image:
    """Blend only inside ``hard_support`` and preserve every outside byte."""
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    base_rgb = _rgb_array(base)
    edited_rgb = _rgb_array(edited.resize(base.size, Image.Resampling.LANCZOS))
    hard = _mask_array(hard_support, base.size) > 0
    soft = _mask_array(soft_mask, base.size).astype(np.float32) / 255.0
    weight = np.clip(float(alpha) * soft, 0.0, 1.0)
    blended = np.rint(
        (1.0 - weight[..., None]) * base_rgb.astype(np.float32)
        + weight[..., None] * edited_rgb.astype(np.float32)
    ).astype(np.uint8)
    output = base_rgb.copy()
    output[hard] = blended[hard]
    return Image.fromarray(output, mode="RGB")


def _gradient_magnitude(rgb: np.ndarray) -> np.ndarray:
    gray = rgb.astype(np.float32).mean(axis=2) / 255.0
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = gray[:, 1:] - gray[:, :-1]
    gy[1:, :] = gray[1:, :] - gray[:-1, :]
    return np.sqrt(gx * gx + gy * gy)


def _edge_cosine(base: np.ndarray, candidate: np.ndarray, structure: np.ndarray) -> float:
    left = _gradient_magnitude(base)[structure].astype(np.float64)
    right = _gradient_magnitude(candidate)[structure].astype(np.float64)
    if left.size == 0:
        raise ValueError("structure mask is empty")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.dot(left, right) / denominator)


def audit_refinement_candidate(
    base: Image.Image,
    candidate: Image.Image,
    hard_support: Image.Image,
    structure_mask: Image.Image,
    thresholds: dict[str, float],
    manual_checks: dict[str, bool | None],
) -> dict[str, Any]:
    """Return an auditable accept/rollback decision for one fixed candidate."""
    base_rgb = _rgb_array(base)
    candidate_rgb = _rgb_array(candidate.resize(base.size, Image.Resampling.NEAREST))
    hard = _mask_array(hard_support, base.size) > 0
    structure = (_mask_array(structure_mask, base.size) > 0) & hard
    if not hard.any():
        raise ValueError("hard support is empty")
    outside = ~hard
    difference = np.abs(candidate_rgb.astype(np.int16) - base_rgb.astype(np.int16))
    outside_max = int(difference[outside].max()) if outside.any() else 0
    changed = np.any(difference > 0, axis=2)
    changed_fraction = float(changed[hard].mean())
    edge_cosine = _edge_cosine(base_rgb, candidate_rgb, structure)

    automatic = {
        "outside_support_exact": outside_max == 0,
        "minimum_local_change": changed_fraction >= float(thresholds["min_changed_fraction"]),
        "maximum_local_change": changed_fraction <= float(thresholds["max_changed_fraction"]),
        "structure_edge_retention": edge_cosine >= float(thresholds["min_structure_edge_cosine"]),
    }
    normalized_manual = {name: manual_checks.get(name) for name in REQUIRED_SEMANTIC_CHECKS}
    manual_complete = all(value is not None for value in normalized_manual.values())
    manual_pass = manual_complete and all(value is True for value in normalized_manual.values())
    accepted = all(automatic.values()) and manual_pass
    return {
        "metrics": {
            "outside_support_max_abs_difference": outside_max,
            "inside_support_changed_fraction": changed_fraction,
            "structure_edge_cosine": edge_cosine,
        },
        "automatic_checks": automatic,
        "manual_checks": normalized_manual,
        "manual_complete": manual_complete,
        "accepted": accepted,
        "disposition": "use_candidate" if accepted else "rollback_to_pre_refinement_frame",
    }


def apply_decision(base: Image.Image, candidate: Image.Image, audit: dict[str, Any]) -> Image.Image:
    """Materialize the conservative final image from an audit record."""
    return candidate.copy() if audit.get("accepted") is True else base.copy()
