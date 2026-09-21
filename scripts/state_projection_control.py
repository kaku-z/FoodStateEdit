"""Geometry-aware planning and denoising schedule helpers for E6.

The functions in this module are deliberately independent of torch so the
control contract can be unit-tested before a GPU run.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class VisibilityPlan:
    offset_xy: tuple[int, int]
    visible_fraction_before: float
    visible_fraction_after: float
    displacement_pixels: float


def visible_source_fraction(source_mask, payload_mask):
    """Return the fraction of a source projection not hidden by the payload."""
    source = np.asarray(source_mask, dtype=bool)
    payload = np.asarray(payload_mask, dtype=bool)
    if source.shape != payload.shape:
        raise ValueError("source and payload masks must have the same shape")
    count = int(source.sum())
    if count == 0:
        raise ValueError("source mask is empty")
    return float((source & ~payload).sum() / count)


def translate_mask(mask, offset_xy):
    """Translate a binary mask without wrapping pixels across image borders."""
    binary = np.asarray(mask, dtype=np.uint8)
    dx, dy = map(int, offset_xy)
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        binary,
        matrix,
        (binary.shape[1], binary.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0


def plan_visible_hold_offset(source_mask, payload_mask, candidate_offsets,
                             minimum_visible_fraction):
    """Choose the shortest offset that reveals the requested source fraction.

    The selection is deterministic: candidates are ranked first by whether the
    visibility constraint is met, then by displacement, then by higher visible
    fraction, and finally lexicographically by (dx, dy).  If no candidate meets
    the threshold, the most visible candidate is returned.
    """
    if not 0.0 <= minimum_visible_fraction <= 1.0:
        raise ValueError("minimum_visible_fraction must be in [0, 1]")
    candidates = [tuple(map(int, item)) for item in candidate_offsets]
    if not candidates:
        raise ValueError("candidate_offsets is empty")
    before = visible_source_fraction(source_mask, payload_mask)
    rows = []
    for offset in candidates:
        shifted = translate_mask(payload_mask, offset)
        visible = visible_source_fraction(source_mask, shifted)
        distance = float(np.hypot(*offset))
        feasible = visible >= minimum_visible_fraction
        rows.append((offset, visible, distance, feasible))
    feasible_rows = [row for row in rows if row[3]]
    if feasible_rows:
        chosen = min(feasible_rows, key=lambda row: (row[2], -row[1], row[0]))
    else:
        chosen = min(rows, key=lambda row: (-row[1], row[2], row[0]))
    return VisibilityPlan(
        offset_xy=chosen[0],
        visible_fraction_before=before,
        visible_fraction_after=chosen[1],
        displacement_pixels=chosen[2],
    )


def validate_state_projection_schedule(total_steps, replace_start, rigid_end,
                                       cavity_end):
    """Validate the staged latent-projection schedule used by the Wan runtime."""
    values = [total_steps, replace_start, rigid_end, cavity_end]
    if any(int(value) != value for value in values):
        raise ValueError("schedule values must be integers")
    total_steps, replace_start, rigid_end, cavity_end = map(int, values)
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0 <= replace_start <= total_steps:
        raise ValueError("replace_start is outside the denoising schedule")
    if not replace_start <= rigid_end <= total_steps:
        raise ValueError("rigid_end is outside the denoising schedule")
    if not replace_start <= cavity_end <= total_steps:
        raise ValueError("cavity_end is outside the denoising schedule")
    return {
        "total_steps": total_steps,
        "replace_start": replace_start,
        "rigid_end": rigid_end,
        "cavity_end": cavity_end,
        "cavity_projection_steps": list(range(replace_start, cavity_end)),
    }
