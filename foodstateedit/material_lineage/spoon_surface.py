"""One exact analytic spoon surface for geometry, food contact and measurement.

Coordinates are absolute spoon-local XY (camera vertices @ spoon_axes_camera).
The joined surface follows mld2_real_spoon_v3.utensil, including its shallow
ellipse, smooth neck blend, curved/tapered handle and small central metal ridge.
"""
from __future__ import annotations

import numpy as np


def _handle_profile(x, radius):
    a, b = radius
    u = np.clip((x - a) / (7 * a), 0., 1.)
    du = np.where((x > a) & (x < 8 * a), 1. / (7 * a), 0.)
    center_y = .055 * b * np.sin(np.pi * u)
    center_y_dx = .055 * b * np.pi * np.cos(np.pi * u) * du
    tail = np.clip((u - .90) / .10, 0., 1.)
    taper = np.sqrt(np.maximum(0., 1. - tail ** 2))
    base_width = b * (.11 + .13 * u ** .7)
    width = base_width * taper
    base_derivative = b * .13 * .7 * np.maximum(u, 1.e-12) ** -.3
    taper_derivative = np.where((u > .90) & (u < 1.), -10 * tail / np.maximum(taper, 1.e-12), 0.)
    width_dx = (base_derivative * taper + base_width * taper_derivative) * du
    return u, du, center_y, center_y_dx, width, width_dx


def surface_height_gradient(local_xy, spoon_info):
    """Return exact surface heights and XY derivatives in spoon-local units."""
    local = np.asarray(local_xy, dtype=float) - np.asarray(spoon_info["center_local"], dtype=float)[:2]
    radius = np.asarray(spoon_info["radius_ab"], dtype=float)
    a, b = radius
    floor = float(spoon_info["floor_local"])
    curvature = float(spoon_info["bowl_curvature_height"])
    x, y = local[..., 0], local[..., 1]
    if not spoon_info.get("bowl_handle_blend", False):
        height = floor + curvature * ((x / a) ** 2 + (y / b) ** 2)
        gradient = 2 * curvature * local / radius ** 2
        return height, gradient
    bowl_x = np.minimum(x, a)
    bowl = floor + curvature * ((bowl_x / a) ** 2 + (y / b) ** 2)
    bowl_gradient = np.stack([2 * curvature * bowl_x / a ** 2 * (x < a), 2 * curvature * y / b ** 2], axis=-1)
    u, du, cy, cy_dx, width, width_dx = _handle_profile(x, radius)
    local_width = np.maximum(width, .001 * b)
    local_width_dx = np.where(width > .001 * b, width_dx, 0.)
    normalized_y = (y - cy) / local_width
    normalized_y_dx = -cy_dx / local_width - normalized_y * local_width_dx / local_width
    ridge = np.maximum(0., 1. - normalized_y ** 2)
    ridge_dx = np.where(ridge > 0., -2 * normalized_y * normalized_y_dx, 0.)
    ridge_dy = np.where(ridge > 0., -2 * normalized_y / local_width, 0.)
    handle = floor + curvature + .06 * b * np.sin(np.pi * u) + .06 * b * u + .018 * b * ridge
    handle_dx = .06 * b * (np.pi * np.cos(np.pi * u) + 1.) * du + .018 * b * ridge_dx
    handle_dy = .018 * b * ridge_dy
    t = np.clip((x / a - .85) / .25, 0., 1.)
    blend = t * t * (3. - 2. * t)
    blend_dx = 6 * t * (1 - t) / (.25 * a)
    height = bowl * (1. - blend) + handle * blend
    dx = bowl_gradient[..., 0] * (1. - blend) + handle_dx * blend + (handle - bowl) * blend_dx
    dy = bowl_gradient[..., 1] * (1. - blend) + handle_dy * blend
    return height, np.stack([dx, dy], axis=-1)


def surface_height(local_xy, spoon_info):
    """Height only; supports arbitrary leading array dimensions."""
    return surface_height_gradient(local_xy, spoon_info)[0]


def surface_domain(local_xy, spoon_info):
    """Valid projected metal footprint; includes the V3 handle when requested."""
    local = np.asarray(local_xy, dtype=float) - np.asarray(spoon_info["center_local"], dtype=float)[:2]
    radius = np.asarray(spoon_info["radius_ab"], dtype=float)
    domain = np.sum((local / radius) ** 2, axis=-1) <= 1.
    if spoon_info.get("bowl_handle_blend", False):
        _, _, cy, _, width, _ = _handle_profile(local[..., 0], radius)
        domain |= (local[..., 0] >= .85 * radius[0]) & (local[..., 0] <= 8 * radius[0]) & (np.abs(local[..., 1] - cy) <= width)
    return domain
