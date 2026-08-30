"""Deterministic 3-D geometry helpers for FoodStateEdit motion projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PinholeCamera:
    width: int
    height: int
    intrinsic: np.ndarray
    geometry_source: str

    @classmethod
    def normalized_relative(
        cls,
        width: int,
        height: int,
        focal_scale: float = 1.2,
    ) -> "PinholeCamera":
        if width <= 0 or height <= 0 or focal_scale <= 0:
            raise ValueError("Camera dimensions and focal scale must be positive")
        focal = focal_scale * max(width, height)
        intrinsic = np.asarray(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return cls(width, height, intrinsic, "relative_3d_normalized_pinhole")

    @classmethod
    def reconstructed(
        cls,
        width: int,
        height: int,
        intrinsic: np.ndarray,
        geometry_source: str,
    ) -> "PinholeCamera":
        matrix = np.asarray(intrinsic, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("Intrinsic matrix must be finite 3x3")
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValueError("Camera focal lengths must be positive")
        return cls(width, height, matrix, geometry_source)

    def backproject(self, uv: np.ndarray, depth: np.ndarray | float) -> np.ndarray:
        pixels = np.asarray(uv, dtype=np.float64)
        if pixels.ndim != 2 or pixels.shape[1] != 2:
            raise ValueError("uv must have shape [N, 2]")
        z = np.broadcast_to(np.asarray(depth, dtype=np.float64), (len(pixels),))
        if not np.isfinite(pixels).all() or not np.isfinite(z).all() or np.any(z <= 0):
            raise ValueError("Pixels and positive depths must be finite")
        homogeneous = np.column_stack([pixels, np.ones(len(pixels))])
        rays = homogeneous @ np.linalg.inv(self.intrinsic).T
        return rays * z[:, None]

    def project(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("xyz must have shape [N, 3]")
        if not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
            raise ValueError("3-D points must be finite and in front of the camera")
        projected = points @ self.intrinsic.T
        uv = projected[:, :2] / projected[:, 2:3]
        return uv, points[:, 2].copy()


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def interpolate_points(source: np.ndarray, target: np.ndarray, amount: float) -> np.ndarray:
    source_points = np.asarray(source, dtype=np.float64)
    target_points = np.asarray(target, dtype=np.float64)
    if source_points.shape != target_points.shape:
        raise ValueError("Source and target point arrays must match")
    weight = smoothstep(amount)
    return source_points * (1.0 - weight) + target_points * weight


def orthonormal_frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direction = np.asarray(axis, dtype=np.float64)
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError("Axis must be a finite 3-vector")
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("Axis must be nonzero")
    tangent = direction / norm
    reference = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(tangent, reference))) > 0.95:
        reference = np.asarray([0.0, 1.0, 0.0])
    normal = np.cross(tangent, reference)
    normal /= np.linalg.norm(normal)
    binormal = np.cross(tangent, normal)
    binormal /= np.linalg.norm(binormal)
    return tangent, normal, binormal


def helix_about_axis(
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    pitch: float,
    turns: float,
    samples: int,
    phase: float = 0.0,
) -> np.ndarray:
    if radius <= 0 or turns <= 0 or samples < 2:
        raise ValueError("Helix radius/turns must be positive and samples >= 2")
    tangent, normal, binormal = orthonormal_frame(axis)
    center_point = np.asarray(center, dtype=np.float64)
    if center_point.shape != (3,):
        raise ValueError("Center must be a 3-vector")
    parameter = np.linspace(0.0, 1.0, samples)
    angle = phase + parameter * turns * 2.0 * np.pi
    axial = (parameter - 0.5) * pitch * turns
    return (
        center_point[None, :]
        + axial[:, None] * tangent[None, :]
        + radius * np.cos(angle)[:, None] * normal[None, :]
        + radius * np.sin(angle)[:, None] * binormal[None, :]
    )


def cubic_bezier(
    start: np.ndarray,
    control_a: np.ndarray,
    control_b: np.ndarray,
    end: np.ndarray,
    samples: int,
) -> np.ndarray:
    if samples < 2:
        raise ValueError("Bezier samples must be >= 2")
    points = [np.asarray(value, dtype=np.float64) for value in (start, control_a, control_b, end)]
    if any(point.shape != (3,) for point in points):
        raise ValueError("Bezier controls must be 3-vectors")
    parameter = np.linspace(0.0, 1.0, samples)[:, None]
    one_minus = 1.0 - parameter
    return (
        one_minus**3 * points[0]
        + 3.0 * one_minus**2 * parameter * points[1]
        + 3.0 * one_minus * parameter**2 * points[2]
        + parameter**3 * points[3]
    )


def polyline_length(points: np.ndarray) -> float:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError("Polyline must have shape [N>=2, 3]")
    return float(np.linalg.norm(np.diff(values, axis=0), axis=1).sum())


def zbuffer_splat(
    canvas: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    colors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-splat nearest point per integer pixel and return visibility."""
    image = np.asarray(canvas).copy()
    pixels = np.rint(np.asarray(uv, dtype=np.float64)).astype(np.int64)
    z = np.asarray(depth, dtype=np.float64)
    rgb = np.asarray(colors, dtype=np.uint8)
    if pixels.shape != (len(z), 2) or rgb.shape != (len(z), 3):
        raise ValueError("uv/depth/colors shapes do not agree")
    height, width = image.shape[:2]
    valid = (
        np.isfinite(z)
        & (z > 0)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    visible = np.zeros(len(z), dtype=bool)
    if not np.any(valid):
        return image, visible
    indices = np.flatnonzero(valid)
    flat = pixels[indices, 1] * width + pixels[indices, 0]
    order = np.lexsort((z[indices], flat))
    sorted_indices = indices[order]
    sorted_flat = flat[order]
    first = np.r_[True, sorted_flat[1:] != sorted_flat[:-1]]
    winners = sorted_indices[first]
    image[pixels[winners, 1], pixels[winners, 0]] = rgb[winners]
    visible[winners] = True
    return image, visible
