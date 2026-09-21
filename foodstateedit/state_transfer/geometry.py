"""An explicit, auditable 2.5D planning contract, NOT a physics simulator.

Coordinates: camera x right, y down, z forward; image coordinates are pixel
centres. One observed surface parcel per food pixel. Hidden volume, cutting,
fluid evolution and newly exposed appearance are not inferred by this module.
"""
from dataclasses import dataclass, replace

import numpy as np

KEEP, MOVE, REARRANGE, EXPOSE = range(4)
REMAIN, CARRIED, DETACHED = range(3)
CHANNELS = ("keep", "move", "rearrange", "expose", "source_u", "source_v",
            "depth", "contact", "confidence", "utensil")


def _finite(a, shape, name):
    a = np.asarray(a)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name}: expected finite array of shape {shape}")
    return a


@dataclass(frozen=True)
class Camera:
    height: int
    width: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self):
        if (not isinstance(self.height, (int, np.integer)) or not isinstance(self.width, (int, np.integer))
                or self.height < 2 or self.width < 2):
            raise ValueError("camera dimensions must be integers >= 2")
        if not np.isfinite([self.fx, self.fy, self.cx, self.cy]).all() or min(self.fx, self.fy) <= 0:
            raise ValueError("invalid camera intrinsics")


@dataclass(frozen=True)
class FoodState:
    ids: np.ndarray
    source_xy: np.ndarray
    origin_xyz: np.ndarray
    xyz: np.ndarray
    owner: np.ndarray
    contact: np.ndarray
    confidence: np.ndarray
    edges: np.ndarray

    def validate(self, camera):
        n = len(self.ids)
        ids = _finite(self.ids, (n,), "ids")
        if ids.dtype.kind not in "iu" or len(np.unique(ids)) != n or (ids < 0).any():
            raise ValueError("source ids must be unique nonnegative integers")
        xy = _finite(self.source_xy, (n, 2), "source_xy")
        if (xy.dtype.kind not in "iu" or len(np.unique(xy, axis=0)) != n
                or (xy < 0).any() or (xy[:, 0] >= camera.width).any()
                or (xy[:, 1] >= camera.height).any()):
            raise ValueError("one valid source pixel per observed parcel is required")
        for name in ("origin_xyz", "xyz"):
            a = _finite(getattr(self, name), (n, 3), name)
            if (a[:, 2] <= 0).any():
                raise ValueError("nonpositive depth; use an explicit out-of-view plan instead")
        owner = _finite(self.owner, (n,), "owner")
        if not np.isin(owner, [REMAIN, CARRIED, DETACHED]).all():
            raise ValueError("invalid ownership")
        for name in ("contact", "confidence"):
            a = _finite(getattr(self, name), (n,), name)
            if ((a < 0) | (a > 1)).any():
                raise ValueError(f"{name} must lie in [0, 1]")
        edges = np.asarray(self.edges)
        if edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in "iu":
            raise ValueError("edges must be integer [E,2] row indices")
        if edges.size and ((edges < 0).any() or (edges >= n).any() or (edges[:, 0] == edges[:, 1]).any()):
            raise ValueError("invalid graph edge")


def from_mask(food_mask, depth, camera, confidence=None):
    """Backproject observed food; 4-neighbour edges are bookkeeping, not bonds."""
    shape = (camera.height, camera.width)
    mask = _finite(food_mask, shape, "food_mask").astype(bool)
    depth = _finite(depth, shape, "depth")
    if (depth <= 0).any():
        raise ValueError("reference depth must be positive")
    conf = np.ones(shape) if confidence is None else _finite(confidence, shape, "confidence")
    y, x = np.nonzero(mask)
    xy = np.stack([x, y], axis=1)
    z = depth[y, x]
    xyz = np.stack([(x-camera.cx)*z/camera.fx, (y-camera.cy)*z/camera.fy, z], axis=1)
    lookup = np.full(shape, -1, dtype=np.int64)
    lookup[y, x] = np.arange(len(x))
    pairs = []
    for a, b in ((lookup[:, :-1], lookup[:, 1:]), (lookup[:-1], lookup[1:])):
        valid = (a >= 0) & (b >= 0)
        pairs.append(np.stack([a[valid], b[valid]], axis=1))
    state = FoodState(np.arange(len(x), dtype=np.int64), xy, xyz.copy(), xyz.copy(),
                      np.zeros(len(x), np.int64), np.zeros(len(x)), conf[y, x].copy(),
                      np.concatenate(pairs, axis=0))
    state.validate(camera)
    return state


@dataclass(frozen=True)
class Transition:
    """Joint per-parcel update supplied by an oracle or future trained planner.

    Displacements are incremental in camera units, not pixel offsets. All
    parcels, including the remaining food, participate in the SAME update.
    """
    delta_xyz: np.ndarray
    owner: np.ndarray
    contact: np.ndarray
    confidence: np.ndarray
    keep_edges: np.ndarray | None = None


def apply_transition(state, transition, camera):
    state.validate(camera)
    delta = _finite(transition.delta_xyz, state.xyz.shape, "delta_xyz")
    edges = state.edges.copy()
    if transition.keep_edges is not None:
        keep = np.asarray(transition.keep_edges)
        if keep.dtype != bool or keep.shape != (len(edges),):
            raise ValueError("keep_edges must be a boolean [E] array")
        edges = edges[keep]
    out = replace(state, ids=state.ids.copy(), source_xy=state.source_xy.copy(),
                  origin_xyz=state.origin_xyz.copy(), xyz=state.xyz + delta,
                  owner=np.asarray(transition.owner).copy(), contact=np.asarray(transition.contact).copy(),
                  confidence=np.asarray(transition.confidence).copy(), edges=edges)
    out.validate(camera)
    return out


def project(state, camera, reference_depth, *, utensil_depth=None, utensil_confidence=1.0):
    """Return H,W,10 controls + visible ids and visibility diagnostics.

    This is a nearest-pixel point z-buffer, not a surface/volume renderer.
    Uncovered source pixels become EXPOSE with confidence zero and invalid
    correspondence (2,2). They mean UNKNOWN surface, NOT an empty physical hole.
    Hidden parcels remain in the ledger. Static non-food geometry can occlude
    moving food. Utensil depth, if provided, is positive where visible geometry
    exists and +inf elsewhere. Equal depth favours static geometry/tool.
    """
    state.validate(camera)
    shape = (camera.height, camera.width)
    rd = _finite(reference_depth, shape, "reference_depth").astype(float)
    if (rd <= 0).any():
        raise ValueError("reference_depth must be positive")
    yy, xx = np.indices(shape)
    out = np.zeros((*shape, 10), np.float32)
    out[..., KEEP] = 1
    out[..., 4] = 2*xx/(camera.width-1)-1
    out[..., 5] = 2*yy/(camera.height-1)-1
    out[..., 6] = rd
    out[..., 8] = 1
    zbuffer = rd.copy()
    visible_ids = np.full(shape, -1, np.int64)
    sx, sy = state.source_xy.T
    out[sy, sx] = 0
    out[sy, sx, EXPOSE] = 1
    out[sy, sx, 4:6] = 2  # invalid source, explicitly gated in routing
    zbuffer[sy, sx] = np.inf
    xyz = state.xyz
    u = camera.fx*xyz[:, 0]/xyz[:, 2]+camera.cx
    v = camera.fy*xyz[:, 1]/xyz[:, 2]+camera.cy
    inside = (u >= -.5) & (u < camera.width-.5) & (v >= -.5) & (v < camera.height-.5)
    candidates = np.flatnonzero(inside)
    # Stable deterministic tie-breaking by source id, independent of row order.
    order = candidates[np.lexsort((state.ids[candidates], xyz[candidates, 2]))]
    moved = np.linalg.norm(xyz-state.origin_xyz, axis=1) > 1e-8
    for i in order:
        x, y = int(np.floor(u[i]+.5)), int(np.floor(v[i]+.5))
        if xyz[i, 2] >= zbuffer[y, x]:
            continue
        role = MOVE if state.owner[i] in (CARRIED, DETACHED) else REARRANGE if moved[i] else KEEP
        out[y, x] = 0
        out[y, x, role] = 1
        out[y, x, 4:6] = [2*sx[i]/(camera.width-1)-1, 2*sy[i]/(camera.height-1)-1]
        out[y, x, 6:9] = [xyz[i, 2], state.contact[i], state.confidence[i]]
        zbuffer[y, x] = xyz[i, 2]
        visible_ids[y, x] = state.ids[i]
    if utensil_depth is not None:
        td = np.asarray(utensil_depth, float)
        if td.shape != shape or np.isnan(td).any() or (td <= 0).any():
            raise ValueError("utensil_depth must be positive or +inf, shape H,W")
        if not np.isfinite(utensil_confidence) or not 0 <= utensil_confidence <= 1:
            raise ValueError("invalid utensil confidence")
        tool = np.isfinite(td) & (td <= zbuffer)
        out[tool] = 0
        out[tool, EXPOSE] = 1
        out[tool, 4:6] = 2
        out[tool, 6] = td[tool]
        out[tool, 8] = utensil_confidence
        out[tool, 9] = 1
        visible_ids[tool] = -1
    visible = np.unique(visible_ids[visible_ids >= 0])
    return {"condition": out, "visible_ids": visible_ids,
            "diagnostics": {"parcels": len(state.ids), "visible_parcels": len(visible),
                            "out_of_view": int((~inside).sum()),
                            "occluded_parcels": int(inside.sum()-len(visible)),
                            "unknown_exposed_pixels": int(((out[..., EXPOSE] > 0) & (out[..., 8] == 0)).sum())}}
