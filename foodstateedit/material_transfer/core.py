"""Geometry-authoritative oracle on unit cubes with persistent material IDs.

This first experiment supports integer translations and an orthographic camera.
Volume means the volume of the inferred cubes, never image area. There is no
learned reconstruction, tool contact, real mass estimate, or dynamics claim.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MaterialState:
    # grid[x,y,z] stores the original material ID, or -1 for empty space.
    grid: np.ndarray
    canonical: np.ndarray
    selected_ids: np.ndarray

    def __post_init__(self):
        for name in ("grid", "canonical", "selected_ids"):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.grid.ndim != 3 or self.canonical.ndim != 2 or self.canonical.shape[1] != 3:
            raise ValueError("Expected a 3D grid and Nx3 canonical coordinates")
        ids = self.grid[self.grid >= 0]
        if np.any(ids >= len(self.canonical)) or len(np.unique(ids)) != len(ids):
            raise ValueError("Material IDs must be valid and occur exactly once")
        if not np.isin(self.selected_ids, ids).all():
            raise ValueError("Selected IDs must exist in the source")


def make_cake(seed=0, size=56):
    """Generate a source-only pre-cut component and exact oracle volume."""
    if size < 48:
        raise ValueError("The fixture needs a grid of at least 48 cells")
    rng = np.random.default_rng(seed)
    grid = np.full((size, size, size), -1, dtype=np.int32)
    height = int(rng.integers(9, 14))
    xyz = np.stack(np.meshgrid(np.arange(8, 26), np.arange(13, 33),
                              np.arange(3, 3 + height), indexing="ij"), -1).reshape(-1, 3)
    grid[tuple(xyz.T)] = np.arange(len(xyz))
    # This label exists before a destination is supplied. No target-shaped mask.
    selected = np.flatnonzero((xyz[:, 0] >= 20) & (xyz[:, 1] >= 23))
    return MaterialState(grid, xyz, selected)


def transport(source, delta):
    """Remove selected cubes once, then transport them; reject invalid edits.

    Checks every rounded lattice placement along the straight path. This is a
    discrete path check, not certified continuous collision detection.
    """
    delta = np.asarray(delta)
    if delta.shape != (3,) or not np.isfinite(delta).all() or not np.equal(delta, np.round(delta)).all():
        raise ValueError("Oracle transport requires a finite integer xyz translation")
    delta = delta.astype(int)
    moving = np.argwhere(np.isin(source.grid, source.selected_ids))
    ids = source.grid[tuple(moving.T)]
    remaining = source.grid.copy()
    remaining[tuple(moving.T)] = -1
    count = max(1, int(np.max(np.abs(delta))) * 4)
    for offset in np.unique(np.round(np.linspace(0, 1, count + 1)[:, None] * delta).astype(int), axis=0):
        destination = moving + offset
        if np.any(destination < 0) or np.any(destination >= np.asarray(source.grid.shape)):
            raise ValueError("Transport leaves the scene bounds")
        if np.any(remaining[tuple(destination.T)] >= 0):
            raise ValueError("Transport intersects remaining food on the discrete path")
    destination = moving + delta
    remaining[tuple(destination.T)] = ids
    return MaterialState(remaining, source.canonical, source.selected_ids)


def material_rgb(q, normal):
    """One deterministic material field shared by all views and edit states."""
    layer = np.floor((q[:, 2] - 3) / 2).astype(int) % 3
    palette = np.array([[0.72, 0.38, 0.16], [0.98, 0.88, 0.65], [0.62, 0.19, 0.15]])
    rgb = palette[layer]
    grain = 0.035 * np.sin(q[:, 0] * 2.3 + np.sin(q[:, 1] * 1.7) + q[:, 2] * 3.1)
    light = np.array([-0.35, -0.45, 0.82]); light /= np.linalg.norm(light)
    shading = 0.55 + 0.45 * np.maximum(0, normal @ light)
    return np.clip((rgb + grain[:, None]) * shading[:, None], 0, 1)


def render(state, resolution=160, material_fn=None):
    """Ray/cube traversal: first occupied cube owns visibility and material.

    Returns RGB, linear ray depth, original material IDs, and canonical hit
    coordinates. Internal voxel faces are never rendered through occupied cells.
    Background is an analytic checker plane; it is not copied from source RGB.
    """
    if resolution < 8:
        raise ValueError("Resolution must be >= 8")
    shape = np.asarray(state.grid.shape)
    direction = np.array([0.32, 0.42, -1.0]); direction /= np.linalg.norm(direction)
    right = np.cross(direction, [0, 1, 0]); right /= np.linalg.norm(right)
    up = np.cross(right, direction)
    xx, yy = np.meshgrid(np.linspace(-34, 34, resolution), np.linspace(34, -34, resolution))
    center = np.array([27., 24., 7.])
    origin = (center + xx[..., None] * right + yy[..., None] * up - 100 * direction).reshape(-1, 3)
    n = len(origin)
    slab_a = -origin / direction
    slab_b = (shape - origin) / direction
    near = np.minimum(slab_a, slab_b); far = np.maximum(slab_a, slab_b)
    entry = near.max(axis=1); leave = far.min(axis=1)
    active = (entry <= leave) & (leave >= 0)
    distance = np.maximum(entry, 0)
    voxel = np.floor(origin + (distance[:, None] + 1e-7) * direction).astype(int)
    normal = np.zeros((n, 3))
    axis = near.argmax(axis=1)
    step = np.sign(direction).astype(int)
    normal[np.arange(n), axis] = -step[axis]
    ids = np.full(n, -1, dtype=np.int32)
    q = np.full((n, 3), np.nan)
    depth = np.full(n, np.inf)
    hit_normal = np.zeros((n, 3))
    # Each iteration crosses at least one cell; grid dimension sum bounds steps.
    for _ in range(int(shape.sum()) + 3):
        valid = active & (voxel >= 0).all(axis=1) & (voxel < shape).all(axis=1) & (distance <= leave + 1e-7)
        active &= valid
        indices = np.flatnonzero(active)
        if not len(indices):
            break
        cell_ids = state.grid[tuple(voxel[indices].T)]
        hits = indices[cell_ids >= 0]
        if len(hits):
            ids[hits] = state.grid[tuple(voxel[hits].T)]
            depth[hits] = distance[hits]
            hit_normal[hits] = normal[hits]
            local = origin[hits] + distance[hits, None] * direction - voxel[hits]
            q[hits] = state.canonical[ids[hits]] + np.clip(local, 0, 1)
            active[hits] = False
        indices = np.flatnonzero(active)
        boundary = voxel[indices] + (step > 0)
        crossing = (boundary - origin[indices]) / direction
        next_distance = crossing.min(axis=1)
        crossed = np.isclose(crossing, next_distance[:, None], atol=1e-9, rtol=0)
        voxel[indices] += crossed * step
        distance[indices] = next_distance
        # Ties pick a deterministic face normal at edges.
        axis = crossed.argmax(axis=1)
        normal[indices] = 0
        normal[indices, axis] = -step[axis]
    plane_t = -origin[:, 2] / direction[2]
    plane = origin + plane_t[:, None] * direction
    checker = (np.floor(plane[:, 0] / 5) + np.floor(plane[:, 1] / 5)).astype(int) % 2
    rgb = np.repeat((0.90 + 0.04 * checker)[:, None], 3, axis=1)
    food = ids >= 0
    rgb[food] = (material_rgb if material_fn is None else material_fn)(q[food], hit_normal[food])
    return {"rgb": rgb.reshape(resolution, resolution, 3),
            "depth": depth.reshape(resolution, resolution),
            "material_ids": ids.reshape(resolution, resolution),
            "canonical_hits": q.reshape(resolution, resolution, 3),
            "surface_normals": hit_normal.reshape(resolution, resolution, 3)}
