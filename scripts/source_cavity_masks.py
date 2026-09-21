"""Geometry-bounded source completion; payload pixels never enter repair support."""
import cv2
import numpy as np


def polygon_mask(points, shape):
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
    return mask > 0


def dilate(mask, radius):
    if radius < 0:
        raise ValueError("Negative dilation radius")
    return cv2.dilate(mask.astype(np.uint8), np.ones((2 * radius + 1,) * 2, np.uint8)) > 0


def source_reveal_mask(source_body, moving_payload, *, lifted, source_radius, exclusion_radius):
    if source_body.shape != moving_payload.shape:
        raise ValueError("Mask shapes differ")
    if not lifted:
        return np.zeros_like(source_body, dtype=bool)
    return dilate(source_body, source_radius) & ~dilate(moving_payload, exclusion_radius)


def inward_alpha(support, feather):
    """Feather exclusively inside support; never spill into protected pixels."""
    if feather <= 0:
        raise ValueError("Feather width must be positive")
    distance = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.rint(255 * np.clip(distance / feather, 0, 1)).astype(np.uint8)
    alpha[~support] = 0
    return alpha


def hybrid_condition(base_frame, structure, support):
    if structure.shape != base_frame.shape or support.shape != base_frame.shape[:2]:
        raise ValueError("Condition shapes differ")
    result = base_frame.copy()
    result[support] = structure[support]
    return result


def composite_repair(base_frame, candidate, alpha):
    if base_frame.shape != candidate.shape or alpha.shape != base_frame.shape[:2]:
        raise ValueError("Composite shapes differ")
    w = alpha[..., None].astype(np.float32) / 255.0
    return np.rint(candidate.astype(np.float32) * w + base_frame.astype(np.float32) * (1 - w)).clip(0, 255).astype(np.uint8)
