#!/usr/bin/env python3
"""Deterministic root-locked, length-preserving noodle deformation.

The solver treats the audited source centerline as an inextensible chain.  A
short bowl-side prefix remains exactly fixed, the grip endpoint follows the
requested chopstick target, and the free chain is relaxed with position-based
length constraints, mild bending regularization and a downward sag guide.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ElasticStrandResult:
    points: np.ndarray
    requested_grip: np.ndarray
    solved_grip: np.ndarray
    root_lock_count: int
    rest_length: float
    solved_length: float
    maximum_segment_relative_error: float
    grip_was_clipped: bool
    slack_amplitude: float
    continuation_steps: int
    iterations_per_step: int
    contact_start_index: int = -1
    contact_end_index: int = -1

    @property
    def length_ratio(self) -> float:
        return self.solved_length / max(self.rest_length, 1e-6)


def _curve_length(points: np.ndarray) -> float:
    return float(
        np.linalg.norm(
            np.diff(np.asarray(points, dtype=np.float32), axis=0), axis=1
        ).sum()
    )


def _smoothstep(value: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _project_lengths(
    points: np.ndarray,
    rest: np.ndarray,
    pinned: np.ndarray,
    *,
    reverse: bool,
) -> None:
    order = range(len(rest) - 1, -1, -1) if reverse else range(len(rest))
    for index in order:
        left = index
        right = index + 1
        delta = points[right] - points[left]
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-7:
            continue
        error_vector = (distance - float(rest[index])) * delta / distance
        left_free = not bool(pinned[left])
        right_free = not bool(pinned[right])
        if left_free and right_free:
            points[left] += 0.5 * error_vector
            points[right] -= 0.5 * error_vector
        elif left_free:
            points[left] += error_vector
        elif right_free:
            points[right] -= error_vector


def _fabrik_finish(
    points: np.ndarray,
    rest: np.ndarray,
    *,
    root_index: int,
    root_position: np.ndarray,
    grip_position: np.ndarray,
    maximum_iterations: int = 2400,
    tolerance: float = 0.015,
) -> float:
    """Converge both endpoints without leaving accumulated segment stretch."""

    root_position = np.asarray(root_position, dtype=np.float32)
    grip_position = np.asarray(grip_position, dtype=np.float32)
    endpoint_error = float("inf")
    for _ in range(maximum_iterations):
        points[-1] = grip_position
        for index in range(len(points) - 2, root_index - 1, -1):
            direction = points[index] - points[index + 1]
            distance = float(np.linalg.norm(direction))
            if distance <= 1e-7:
                continue
            points[index] = (
                points[index + 1]
                + float(rest[index]) * direction / distance
            )

        points[root_index] = root_position
        for index in range(root_index, len(points) - 1):
            direction = points[index + 1] - points[index]
            distance = float(np.linalg.norm(direction))
            if distance <= 1e-7:
                continue
            points[index + 1] = (
                points[index]
                + float(rest[index]) * direction / distance
            )
        endpoint_error = float(np.linalg.norm(points[-1] - grip_position))
        if endpoint_error <= tolerance:
            break
    return endpoint_error


def _smooth_gravity_slack_arc(
    rest: np.ndarray,
    *,
    root_index: int,
    root_position: np.ndarray,
    grip_position: np.ndarray,
    start_tangent: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Build a tangent-matched, fold-free inextensible lifting arc.

    A cubic Hermite-like Bezier leaves the bowl along the source tangent and
    gradually turns toward the grip.  Its handle length is solved from the
    available rest length, so slack is distributed across the free suffix
    rather than accumulating as a kink at either endpoint.
    """

    suffix_rest = np.asarray(rest[root_index:], dtype=np.float32)
    free_length = float(suffix_rest.sum())
    root = np.asarray(root_position, dtype=np.float32).reshape(2)
    grip = np.asarray(grip_position, dtype=np.float32).reshape(2)
    chord = grip - root
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= 1e-6:
        raise ValueError("gravity slack arc needs separated endpoints")
    chord_tangent = chord / chord_length
    source_tangent = np.asarray(start_tangent, dtype=np.float32).reshape(2)
    source_tangent /= max(float(np.linalg.norm(source_tangent)), 1e-6)
    upward = np.asarray((0.0, -1.0), dtype=np.float32)
    end_tangent = 0.75 * chord_tangent + 0.25 * upward
    end_tangent /= max(float(np.linalg.norm(end_tangent)), 1e-6)

    dense_count = max(2048, 16 * (len(suffix_rest) + 1))
    parameter = np.linspace(0.0, 1.0, dense_count, dtype=np.float32)
    one_minus = 1.0 - parameter

    def dense_curve(handle: float) -> np.ndarray:
        control_1 = root + float(handle) * source_tangent
        control_2 = grip - 0.72 * float(handle) * end_tangent
        return (
            (one_minus**3)[:, None] * root[None, :]
            + (3.0 * one_minus**2 * parameter)[:, None] * control_1[None, :]
            + (3.0 * one_minus * parameter**2)[:, None] * control_2[None, :]
            + (parameter**3)[:, None] * grip[None, :]
        )

    def dense_length(handle: float) -> float:
        curve = dense_curve(handle)
        return float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())

    low = 0.0
    high = max(1.0, 0.25 * free_length)
    while dense_length(high) < free_length and high < 4.0 * free_length:
        high *= 2.0
    for _ in range(56):
        middle = 0.5 * (low + high)
        if dense_length(middle) < free_length:
            low = middle
        else:
            high = middle
    handle = 0.5 * (low + high)
    dense = dense_curve(handle)
    dense_segment = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    dense_arc = np.concatenate(
        (np.zeros(1, dtype=np.float32), np.cumsum(dense_segment, dtype=np.float32))
    )
    requested_arc = np.concatenate(
        (np.zeros(1, dtype=np.float32), np.cumsum(suffix_rest, dtype=np.float32))
    )
    suffix = np.stack(
        (
            np.interp(requested_arc, dense_arc, dense[:, 0]),
            np.interp(requested_arc, dense_arc, dense[:, 1]),
        ),
        axis=1,
    ).astype(np.float32)
    suffix[0] = root
    suffix[-1] = grip
    return suffix, float(handle)


def solve_root_locked_strand(
    source_points: np.ndarray,
    requested_grip: np.ndarray,
    *,
    image_shape: tuple[int, int],
    root_lock_fraction: float = 0.18,
    sag_ratio: float = 0.020,
    continuation_steps: int = 10,
    iterations_per_step: int = 70,
    constraint_passes: int = 3,
    guide_stiffness: float = 0.030,
    bending_stiffness: float = 0.055,
) -> ElasticStrandResult:
    """Move one endpoint while preserving root pixels and segment lengths.

    Coordinates are ``(x, y)`` and positive y points downward.  If the
    requested grip is farther from the locked prefix than the remaining chain
    can reach, it is clipped to 92% of that reach instead of stretching.
    """

    source = np.asarray(source_points, dtype=np.float32).reshape(-1, 2)
    if len(source) < 12:
        raise ValueError("elastic strand needs at least twelve samples")
    requested = np.asarray(requested_grip, dtype=np.float32).reshape(2)
    height, width = (int(image_shape[0]), int(image_shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("invalid image shape")

    rest = np.linalg.norm(np.diff(source, axis=0), axis=1).astype(np.float32)
    if np.any(rest <= 1e-6):
        raise ValueError("source centerline contains a zero-length segment")
    rest_length = float(rest.sum())
    lock_last = int(round(float(root_lock_fraction) * (len(source) - 1)))
    lock_last = int(np.clip(lock_last, 2, len(source) - 4))
    root_lock_count = lock_last + 1

    remaining_length = float(rest[lock_last:].sum())
    locked_root = source[lock_last]
    reach_vector = requested - locked_root
    reach_distance = float(np.linalg.norm(reach_vector))
    # Avoid an almost taut state: it looks artificial and makes numerical
    # endpoint convergence ill-conditioned.  The reserved 8% becomes visible
    # gravity slack rather than hidden stretch.
    maximum_reach = 0.92 * remaining_length
    grip_was_clipped = reach_distance > maximum_reach
    if grip_was_clipped:
        requested_direction = reach_vector / max(reach_distance, 1e-6)
        solved_grip = locked_root + maximum_reach * requested_direction
    else:
        solved_grip = requested.copy()

    cumulative = np.concatenate(
        (np.zeros(1, dtype=np.float32), np.cumsum(rest, dtype=np.float32))
    )
    free_arc = cumulative - cumulative[lock_last]
    free_arc /= max(float(cumulative[-1] - cumulative[lock_last]), 1e-6)
    free_arc = np.clip(free_arc, 0.0, 1.0)
    deformation_weight = np.asarray(_smoothstep(free_arc), dtype=np.float32)
    deformation_weight[:root_lock_count] = 0.0

    pinned = np.zeros(len(source), dtype=bool)
    pinned[:root_lock_count] = True
    pinned[-1] = True
    points = source.copy()
    source_curvature = np.zeros_like(source)
    source_curvature[1:-1] = source[1:-1] - 0.5 * (
        source[:-2] + source[2:]
    )
    sag_pixels = float(sag_ratio) * float(height)

    continuation_steps = max(1, int(continuation_steps))
    iterations_per_step = max(1, int(iterations_per_step))
    constraint_passes = max(1, int(constraint_passes))
    free_indices = np.arange(root_lock_count, len(source) - 1, dtype=np.int32)

    for step in range(1, continuation_steps + 1):
        progress = float(_smoothstep(step / continuation_steps))
        grip = source[-1] * (1.0 - progress) + solved_grip * progress
        displacement = grip - source[-1]
        guide = source + deformation_weight[:, None] * displacement[None, :]
        sag_envelope = np.sin(np.pi * free_arc) * deformation_weight
        guide[:, 1] += sag_pixels * progress * sag_envelope
        guide[:root_lock_count] = source[:root_lock_count]
        guide[-1] = grip

        points[-1] = grip
        for iteration in range(iterations_per_step):
            if len(free_indices):
                points[free_indices] += float(guide_stiffness) * (
                    guide[free_indices] - points[free_indices]
                )

            if len(free_indices) > 1 and bending_stiffness > 0.0:
                bend_indices = free_indices[free_indices < len(source) - 1]
                midpoint = 0.5 * (
                    points[bend_indices - 1] + points[bend_indices + 1]
                )
                curvature_scale = (1.0 - progress) + 0.28 * progress
                bend_target = midpoint + curvature_scale * source_curvature[bend_indices]
                points[bend_indices] += float(bending_stiffness) * (
                    bend_target - points[bend_indices]
                )

            points[:root_lock_count] = source[:root_lock_count]
            points[-1] = grip
            for pass_index in range(constraint_passes):
                _project_lengths(
                    points,
                    rest,
                    pinned,
                    reverse=bool((iteration + pass_index) % 2),
                )
                points[:root_lock_count] = source[:root_lock_count]
                points[-1] = grip

    # Replace any locally accumulated PBD slack with a globally fair,
    # gravity-oriented arc.  FABRIK then only needs to remove the minute chord
    # error introduced by arc-length resampling, so it cannot form endpoint
    # folds while converging the two pinned ends.
    smooth_suffix, slack_amplitude = _smooth_gravity_slack_arc(
        rest,
        root_index=lock_last,
        root_position=source[lock_last],
        grip_position=solved_grip,
        start_tangent=source[lock_last] - source[lock_last - 1],
    )
    points[lock_last:] = smooth_suffix
    _fabrik_finish(
        points,
        rest,
        root_index=lock_last,
        root_position=source[lock_last],
        grip_position=solved_grip,
    )
    points[:root_lock_count] = source[:root_lock_count]

    solved_segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    maximum_error = float(
        np.max(np.abs(solved_segments - rest) / np.maximum(rest, 1e-6))
    )
    solved_length = _curve_length(points)
    return ElasticStrandResult(
        points=points.astype(np.float32),
        requested_grip=requested.astype(np.float32),
        solved_grip=solved_grip.astype(np.float32),
        root_lock_count=root_lock_count,
        rest_length=rest_length,
        solved_length=solved_length,
        maximum_segment_relative_error=maximum_error,
        grip_was_clipped=bool(grip_was_clipped),
        slack_amplitude=slack_amplitude,
        continuation_steps=continuation_steps,
        iterations_per_step=iterations_per_step,
    )


def solve_mid_gripped_strand(
    source_points: np.ndarray,
    requested_grip: np.ndarray,
    *,
    image_shape: tuple[int, int],
    grip_index: int,
    root_lock_fraction: float = 0.45,
    tail_turn_degrees: float = 70.0,
) -> ElasticStrandResult:
    """Lift one noodle at an interior point and retain a hanging free tail.

    The pre-grip branch is solved as a root-locked inextensible strand.  The
    remaining samples continue through the pinch with matched tangent and bend
    back toward the bowl.  It is still one source identity; the second visible
    branch is the continuation of that same noodle, not an added strand.
    """

    source = np.asarray(source_points, dtype=np.float32).reshape(-1, 2)
    if len(source) < 16:
        raise ValueError("mid-gripped strand needs at least sixteen samples")
    grip_index = int(np.clip(grip_index, 8, len(source) - 5))
    full_lock_last = int(round(float(root_lock_fraction) * (len(source) - 1)))
    full_lock_last = int(np.clip(full_lock_last, 2, grip_index - 3))
    prefix_lock_fraction = float(full_lock_last) / float(grip_index)
    prefix = solve_root_locked_strand(
        source[: grip_index + 1],
        requested_grip,
        image_shape=image_shape,
        root_lock_fraction=prefix_lock_fraction,
        sag_ratio=0.020,
    )

    tail_rest = np.linalg.norm(
        np.diff(source[grip_index:], axis=0), axis=1
    ).astype(np.float32)
    tail_length = float(tail_rest.sum())
    height, width = int(image_shape[0]), int(image_shape[1])
    start_tangent = prefix.points[-1] - prefix.points[-2]
    # A real pinch redirects the free side of the noodle.  Mirror the incoming
    # upward tangent toward gravity; the rods occlude this short contact bend,
    # while the visible tail immediately hangs back into the bowl.
    start_tangent[1] = abs(float(start_tangent[1]))
    start_angle = float(np.arctan2(start_tangent[1], start_tangent[0]))
    cumulative = np.concatenate(
        (np.zeros(1, dtype=np.float32), np.cumsum(tail_rest, dtype=np.float32))
    )

    def constant_turn_candidate(turn_radians: float) -> np.ndarray:
        # Chord directions are evaluated at segment midpoints.  This makes
        # every source segment length exact while distributing the pinch turn
        # uniformly instead of allowing a cubic Bezier cusp.
        midpoint_arc = cumulative[:-1] + 0.5 * tail_rest
        direction_angle = (
            start_angle + float(turn_radians) * midpoint_arc / max(tail_length, 1e-6)
        )
        delta = tail_rest[:, None] * np.stack(
            (np.cos(direction_angle), np.sin(direction_angle)), axis=1
        )
        result = np.empty((len(tail_rest) + 1, 2), dtype=np.float32)
        result[0] = prefix.solved_grip
        result[1:] = prefix.solved_grip + np.cumsum(delta, axis=0)
        return result

    turn = np.deg2rad(abs(float(tail_turn_degrees)))
    candidates = (constant_turn_candidate(turn), constant_turn_candidate(-turn))
    image_center = np.asarray((0.5 * width, 0.5 * height), dtype=np.float32)
    inward = image_center - prefix.solved_grip
    inward /= max(float(np.linalg.norm(inward)), 1e-6)

    def tail_score(candidate: np.ndarray) -> float:
        inside = (
            (candidate[:, 0] >= 4.0)
            & (candidate[:, 0] <= width - 5.0)
            & (candidate[:, 1] >= 4.0)
            & (candidate[:, 1] <= height - 5.0)
        )
        inward_progress = float(np.dot(candidate[-1] - candidate[0], inward))
        downward_progress = float(candidate[-1, 1] - candidate[0, 1])
        gravity_preference = 200.0 if downward_progress >= 0.0 else 0.0
        return (
            1000.0 * float(np.mean(inside))
            + gravity_preference
            + 0.50 * inward_progress
            + downward_progress
        )

    tail = max(candidates, key=tail_score)

    points = np.empty_like(source)
    points[: grip_index + 1] = prefix.points
    points[grip_index:] = tail
    points[: prefix.root_lock_count] = source[: prefix.root_lock_count]
    rest = np.linalg.norm(np.diff(source, axis=0), axis=1)
    solved_segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    maximum_error = float(
        np.max(np.abs(solved_segments - rest) / np.maximum(rest, 1e-6))
    )
    rest_length = float(rest.sum())
    solved_length = _curve_length(points)
    return ElasticStrandResult(
        points=points.astype(np.float32),
        requested_grip=np.asarray(requested_grip, dtype=np.float32).reshape(2),
        solved_grip=prefix.solved_grip.astype(np.float32),
        root_lock_count=prefix.root_lock_count,
        rest_length=rest_length,
        solved_length=solved_length,
        maximum_segment_relative_error=maximum_error,
        grip_was_clipped=prefix.grip_was_clipped,
        slack_amplitude=max(
            prefix.slack_amplitude,
            tail_length / max(turn, 1e-6),
        ),
        continuation_steps=prefix.continuation_steps,
        iterations_per_step=prefix.iterations_per_step,
    )


def _wrapped_angle(value: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(value), np.cos(value))


def solve_contact_aware_strand(
    source_points: np.ndarray,
    requested_grip: np.ndarray,
    *,
    image_shape: tuple[int, int],
    grip_index: int,
    root_lock_fraction: float = 0.45,
    collision_radius: float | None = None,
) -> ElasticStrandResult:
    """Solve a single mid-gripped strand as an inextensible elastic rod.

    Segment angles, rather than point coordinates, are optimized.  Therefore
    every source segment keeps its exact rest length by construction.  The
    complete chain is solved at once with a fixed bowl-side prefix and an
    interior grip constraint.  Bending energy, gravity, source-curvature
    retention, image bounds and non-local self-collision jointly determine the
    free tail; no prescribed tail angle or hand-authored Bezier is used.
    """

    try:
        from scipy.optimize import minimize
    except ImportError as error:  # pragma: no cover - target env has SciPy
        raise RuntimeError("contact-aware solver requires scipy") from error

    source = np.asarray(source_points, dtype=np.float64).reshape(-1, 2)
    requested = np.asarray(requested_grip, dtype=np.float64).reshape(2)
    if len(source) < 16:
        raise ValueError("contact-aware strand needs at least sixteen samples")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("invalid image shape")
    grip_index = int(np.clip(grip_index, 8, len(source) - 5))
    lock_last = int(round(float(root_lock_fraction) * (len(source) - 1)))
    lock_last = int(np.clip(lock_last, 2, grip_index - 3))
    root_lock_count = lock_last + 1

    rest = np.linalg.norm(np.diff(source, axis=0), axis=1)
    if np.any(rest <= 1e-7):
        raise ValueError("source centerline contains a zero-length segment")
    pre_grip_length = float(rest[lock_last:grip_index].sum())
    locked_root = source[lock_last]
    reach = requested - locked_root
    reach_distance = float(np.linalg.norm(reach))
    maximum_reach = 0.92 * pre_grip_length
    grip_was_clipped = reach_distance > maximum_reach
    if grip_was_clipped:
        solved_grip = locked_root + maximum_reach * reach / max(reach_distance, 1e-7)
    else:
        solved_grip = requested.copy()

    # A feasible tangent-matched prefix supplies only the optimizer start.
    # The original post-grip source samples are translated rigidly, retaining
    # their observed curvature instead of replacing them with a synthetic hook.
    prefix = solve_root_locked_strand(
        source[: grip_index + 1].astype(np.float32),
        solved_grip.astype(np.float32),
        image_shape=image_shape,
        root_lock_fraction=float(lock_last) / float(grip_index),
        sag_ratio=0.020,
    )
    initial = source.copy()
    initial[: grip_index + 1] = prefix.points
    tail_translation = solved_grip - source[grip_index]
    initial[grip_index:] = source[grip_index:] + tail_translation
    initial[:root_lock_count] = source[:root_lock_count]

    variable_rest = rest[lock_last:]
    initial_delta = np.diff(initial, axis=0)[lock_last:]
    initial_angles = np.unwrap(
        np.arctan2(initial_delta[:, 1], initial_delta[:, 0])
    )
    source_delta = np.diff(source, axis=0)[lock_last:]
    source_angles = np.unwrap(
        np.arctan2(source_delta[:, 1], source_delta[:, 0])
    )
    kernel_coordinate = np.arange(-7, 8, dtype=np.float64)
    angle_kernel = np.exp(-0.5 * (kernel_coordinate / 3.0) ** 2)
    angle_kernel /= float(angle_kernel.sum())
    padded_source_angles = np.pad(source_angles, 7, mode="edge")
    smooth_source_angles = np.convolve(
        padded_source_angles, angle_kernel, mode="valid"
    )
    pre_segment_count = grip_index - lock_last
    root_position = source[lock_last].copy()
    guide = initial[lock_last:].copy()
    total_free_length = max(float(variable_rest.sum()), 1e-6)
    radius = (
        float(collision_radius)
        if collision_radius is not None
        else 0.012 * float(min(height, width))
    )
    radius = max(radius, 2.0)

    def reconstruct(angles: np.ndarray) -> np.ndarray:
        segment = variable_rest[:, None] * np.stack(
            (np.cos(angles), np.sin(angles)), axis=1
        )
        result = np.empty((len(angles) + 1, 2), dtype=np.float64)
        result[0] = root_position
        result[1:] = root_position + np.cumsum(segment, axis=0)
        return result

    adjacent = np.arange(len(initial_angles) - 1)
    joint_full_indices = lock_last + adjacent + 1
    full_arc = np.concatenate(([0.0], np.cumsum(rest)))
    contact_half_arc = 1.35 * radius
    contact_joint_mask = (
        np.abs(full_arc[joint_full_indices] - full_arc[grip_index])
        <= contact_half_arc
    )
    # Keep a genuine finite contact patch even when centerline sampling is
    # coarse.  These are joints, not pixels: the window scales with noodle
    # radius and therefore transfers to other image resolutions.
    if np.count_nonzero(contact_joint_mask) < 7:
        contact_joint_mask = np.abs(
            joint_full_indices - grip_index
        ) <= 3
    contact_joint_indices = np.flatnonzero(contact_joint_mask)
    contact_start_index = int(joint_full_indices[contact_joint_indices[0]])
    contact_end_index = int(joint_full_indices[contact_joint_indices[-1]])
    curvature_weight = np.ones(len(adjacent), dtype=np.float64)
    curvature_weight[adjacent < max(pre_segment_count - 1, 0)] = 1.40
    curvature_weight[adjacent >= pre_segment_count] = 0.15
    # Source curvature is not a valid target inside a newly created clamp.
    # Let the elastic energy distribute the turn over the complete jaw contact
    # patch instead of concentrating it at one sample.
    curvature_weight[contact_joint_mask] = 0.0
    joint_rest = 0.5 * (variable_rest[:-1] + variable_rest[1:])
    minimum_bend_radius = max(0.80 * radius, 2.0)
    maximum_joint_turn = np.clip(
        joint_rest / minimum_bend_radius,
        np.deg2rad(6.0),
        np.deg2rad(16.0),
    )
    tail_joint_mask = adjacent >= pre_segment_count
    tail_segment_rest = variable_rest[pre_segment_count:]
    tail_length = max(float(tail_segment_rest.sum()), 1e-6)
    tail_joint_rest_fraction = (
        joint_rest[tail_joint_mask] / tail_length
    )
    # Cooked udon is a flexible rod.  Its scale-independent material behavior
    # is governed by the elastogravity number q L^3 / B, not by a pixel-space
    # downward offset.  A value of six gives a compliant hanging branch while
    # retaining a finite bend radius at the clamp.
    tail_elastogravity_number = 6.0

    point_indices = np.arange(len(initial_angles) + 1)
    pair_left, pair_right = np.triu_indices(len(point_indices), k=10)
    contact_point = pre_segment_count
    contact_neighborhood = (
        (np.abs(pair_left - contact_point) <= 18)
        | (np.abs(pair_right - contact_point) <= 18)
    )
    pair_left = pair_left[~contact_neighborhood]
    pair_right = pair_right[~contact_neighborhood]
    collision_distance = 1.55 * radius

    previous_locked_angle = float(
        np.arctan2(
            source[lock_last, 1] - source[lock_last - 1, 1],
            source[lock_last, 0] - source[lock_last - 1, 0],
        )
    )

    def objective(angles: np.ndarray) -> float:
        points = reconstruct(angles)
        angle_change = _wrapped_angle(np.diff(angles))
        source_change = _wrapped_angle(np.diff(smooth_source_angles))
        curvature_error = _wrapped_angle(angle_change - source_change)
        bending = float(np.sum(curvature_weight * curvature_error**2))
        fairing_weight = np.full(len(angle_change), 0.35, dtype=np.float64)
        fairing_weight[tail_joint_mask] = 0.0
        fairing_weight[contact_joint_mask] = 4.0
        fairing = float(np.sum(fairing_weight * angle_change**2))
        tail_elastic_bending = 0.5 * float(
            np.sum(
                angle_change[tail_joint_mask] ** 2
                / np.maximum(tail_joint_rest_fraction, 1e-6)
            )
        )
        # A noodle cannot form a mathematical corner.  This soft curvature
        # barrier gives the strand a radius-dependent minimum bend radius and
        # turns the former point pinch into a C1-like finite contact arc.
        turn_excess = np.maximum(
            np.abs(angle_change) - maximum_joint_turn,
            0.0,
        )
        curvature_barrier = 80.0 * float(np.sum(turn_excess**2))
        root_turn = float(_wrapped_angle(angles[0] - previous_locked_angle))

        displacement = (points - guide) / radius
        guide_weight = np.full(len(points), 0.015, dtype=np.float64)
        # The guide is only a numerical initializer on the hanging side.  A
        # strong positional tether here was the actual source of the sharp V:
        # it forced the translated source tail to meet the lifted branch at one
        # point.  Tail orientation must instead emerge from bending + gravity.
        guide_weight[pre_segment_count + 1 :] = 0.00010
        guide_energy = float(
            np.sum(guide_weight[:, None] * displacement**2)
        )

        # Positive image y points down, hence negative potential encourages a
        # hanging equilibrium.  Tail nodes receive more mass because they are
        # supported only at the interior pinch.
        normalized_y = points[:, 1] / max(float(height), 1.0)
        gravity_weight = np.full(len(points), 0.018, dtype=np.float64)
        gravity_weight[pre_segment_count + 1 :] = 0.0
        supported_gravity = -float(np.sum(gravity_weight * normalized_y))
        tail_midpoint_y = 0.5 * (
            points[pre_segment_count:-1, 1]
            + points[pre_segment_count + 1 :, 1]
        )
        tail_vertical = (
            tail_midpoint_y - points[pre_segment_count, 1]
        ) / tail_length
        tail_gravity = -tail_elastogravity_number * float(
            np.sum((tail_segment_rest / tail_length) * tail_vertical)
        )

        x, y = points[:, 0], points[:, 1]
        outside = (
            np.maximum(4.0 - x, 0.0) ** 2
            + np.maximum(x - (width - 5.0), 0.0) ** 2
            + np.maximum(4.0 - y, 0.0) ** 2
            + np.maximum(y - (height - 5.0), 0.0) ** 2
        )
        bounds_energy = 4.0 * float(np.sum(outside)) / total_free_length**2

        if len(pair_left):
            separation = np.linalg.norm(
                points[pair_left] - points[pair_right], axis=1
            )
            overlap = np.maximum(collision_distance - separation, 0.0)
            collision_energy = 0.65 * float(
                np.sum((overlap / radius) ** 2)
            )
        else:
            collision_energy = 0.0
        return (
            1.00 * bending
            + fairing
            + tail_elastic_bending
            + curvature_barrier
            + 2.5 * root_turn**2
            + guide_energy
            + supported_gravity
            + tail_gravity
            + bounds_energy
            + collision_energy
        )

    def grip_constraint(angles: np.ndarray) -> np.ndarray:
        return reconstruct(angles)[pre_segment_count] - solved_grip

    optimization = minimize(
        objective,
        initial_angles,
        method="SLSQP",
        constraints=({"type": "eq", "fun": grip_constraint},),
        options={"maxiter": 800, "ftol": 1e-10, "disp": False},
    )
    constraint_error = float(np.linalg.norm(grip_constraint(optimization.x)))
    if not optimization.success and constraint_error > 0.20:
        raise RuntimeError(
            "contact-aware elastic solve failed: "
            f"{optimization.message}; grip error={constraint_error:.4f}"
        )

    points = source.copy()
    points[lock_last:] = reconstruct(optimization.x)
    points[:root_lock_count] = source[:root_lock_count]
    solved_segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    maximum_error = float(
        np.max(np.abs(solved_segments - rest) / np.maximum(rest, 1e-7))
    )
    rest_length = float(rest.sum())
    solved_length = _curve_length(points)
    vertical_relaxation = float(
        np.max(np.abs(points[lock_last:, 1] - guide[:, 1]))
    )
    return ElasticStrandResult(
        points=points.astype(np.float32),
        requested_grip=requested.astype(np.float32),
        solved_grip=solved_grip.astype(np.float32),
        root_lock_count=root_lock_count,
        rest_length=rest_length,
        solved_length=solved_length,
        maximum_segment_relative_error=maximum_error,
        grip_was_clipped=bool(grip_was_clipped),
        slack_amplitude=vertical_relaxation,
        continuation_steps=int(getattr(optimization, "nit", 0)),
        iterations_per_step=1,
        contact_start_index=contact_start_index,
        contact_end_index=contact_end_index,
    )
