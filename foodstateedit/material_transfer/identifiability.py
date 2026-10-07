"""Source-observed appearance, hidden-surface counterexample, geometric tool checks."""
import numpy as np
from .core import material_rgb


def cells_at_hits(q, normals):
    # Inward epsilon picks the owning cell even at its positive face.
    return np.floor(q - normals * 1e-6).astype(int)


class CutSurfaceAlternative:
    """Modify only interfaces that are strictly internal before the edit.

    Geometry and source-visible appearance remain identical. This is a deliberate
    ambiguity construction, not an estimated distribution of real food textures.
    """
    def __init__(self, source):
        self.source = source
        self.selected = np.isin(source.grid, source.selected_ids)

    def hidden_faces(self, q, normals):
        cell = cells_at_hits(q, normals)
        neighbor = cell + np.round(normals).astype(int)
        shape = np.asarray(self.source.grid.shape)
        valid = ((cell >= 0) & (cell < shape) & (neighbor >= 0) & (neighbor < shape)).all(axis=1)
        result = np.zeros(len(q), dtype=bool)
        indices = np.flatnonzero(valid)
        a, b = tuple(cell[indices].T), tuple(neighbor[indices].T)
        result[indices] = ((self.source.grid[a] >= 0) & (self.source.grid[b] >= 0) &
                           (self.selected[a] != self.selected[b]))
        return result

    def __call__(self, q, normals):
        rgb = material_rgb(q, normals)
        mask = self.hidden_faces(q, normals)
        # A second admissible hidden material hypothesis. Outside these internal
        # interfaces the full field is unchanged, at any source resolution.
        rgb[mask] = np.array([.12, .66, .88])
        return rgb


class SourceAppearance:
    """Nearest source RGB samples in known canonical coordinates, by face normal.

    Geometry/camera/correspondence are still oracle. Only source-visible RGB is
    supplied; no full material field, hidden RGB or edited-frame RGB is accepted.
    Fixed illumination and translation let observed radiance be reused directly.
    """
    def __init__(self, source_render):
        mask = source_render["material_ids"] >= 0
        self.q = source_render["canonical_hits"][mask].copy()
        self.normals = source_render["surface_normals"][mask].copy()
        self.rgb = source_render["rgb"][mask].copy()
        if not len(self.q):
            raise ValueError("No source-visible food samples")

    def __call__(self, q, normals):
        result = np.empty((len(q), 3))
        for normal in np.unique(normals, axis=0):
            queries = np.flatnonzero(np.all(normals == normal, axis=1))
            sample = np.flatnonzero(np.all(self.normals == normal, axis=1))
            if not len(sample):
                sample = np.arange(len(self.q))
            for start in range(0, len(queries), 128):
                indices = queries[start:start + 128]
                distance = np.sum((q[indices, None] - self.q[sample][None]) ** 2, axis=2)
                result[indices] = self.rgb[sample[np.argmin(distance, axis=1)]]
        return result


def swept_box_hits(lo, hi, delta, obstacle_lo, obstacle_hi, epsilon=1e-9):
    """Exact positive-interior overlap for translating axis-aligned boxes.

    Continuous t in [0,1], fixed orientation. Boundary contact is allowed.
    Obstacles have shape (N,3). This is not deformable contact or force balance.
    """
    lo, hi, delta = np.asarray(lo), np.asarray(hi), np.asarray(delta)
    obstacle_lo, obstacle_hi = np.asarray(obstacle_lo), np.asarray(obstacle_hi)
    entry = np.full(len(obstacle_lo), -np.inf)
    leave = np.full(len(obstacle_lo), np.inf)
    possible = np.ones(len(obstacle_lo), dtype=bool)
    for axis in range(3):
        if abs(delta[axis]) < epsilon:
            possible &= (hi[axis] > obstacle_lo[:, axis] + epsilon) & (lo[axis] < obstacle_hi[:, axis] - epsilon)
        else:
            a = (obstacle_lo[:, axis] - hi[axis]) / delta[axis]
            b = (obstacle_hi[:, axis] - lo[axis]) / delta[axis]
            entry = np.maximum(entry, np.minimum(a, b))
            leave = np.minimum(leave, np.maximum(a, b))
    return possible & (np.maximum(entry, 0) < np.minimum(leave, 1) - epsilon)


def tool_probe(source, delta, gap, thickness, blocked=False):
    """Insert a finite-thickness flat blade from +x, then carry it with the piece.

    A full bounding-rectangle support blade is used; handle/curvature/force
    balance/friction are absent. Plate height varies to give the requested gap.
    The blocked control places a finite wall across this prescribed approach.
    Rejection means this path fails, not that all possible tool paths fail.
    """
    if gap < 0 or thickness <= 0:
        raise ValueError("Nonnegative gap and positive thickness required")
    cells = np.argwhere(source.grid >= 0).astype(float)
    moving = np.argwhere(np.isin(source.grid, source.selected_ids)).astype(float)
    base = moving[:, 2].min()
    lo = np.array([moving[:, 0].min(), moving[:, 1].min(), base - thickness])
    hi = np.array([moving[:, 0].max() + 1, moving[:, 1].max() + 1, base])
    start = lo.copy(); start[0] = cells[:, 0].max() + 4
    plate_lo = np.array([0., 0., -10.])
    plate_hi = np.array([56., 56., base - gap])
    obstacle_lo = np.vstack([cells, plate_lo])
    obstacle_hi = np.vstack([cells + 1, plate_hi])
    if blocked:
        wall_lo = np.array([cells[:, 0].max() + 2, lo[1] - 1, base - thickness - 1])
        wall_hi = wall_lo + [1, hi[1] - lo[1] + 2, thickness + 2]
        obstacle_lo = np.vstack([obstacle_lo, wall_lo])
        obstacle_hi = np.vstack([obstacle_hi, wall_hi])
    approach = swept_box_hits(start, start + hi - lo, lo - start, obstacle_lo, obstacle_hi)
    remaining = np.argwhere((source.grid >= 0) & ~np.isin(source.grid, source.selected_ids)).astype(float)
    carry = swept_box_hits(lo, hi, delta, np.vstack([remaining, plate_lo]), np.vstack([remaining + 1, plate_hi]))
    center = (moving + .5).mean(axis=0)
    supported = bool(np.all(center[:2] >= lo[:2]) and np.all(center[:2] <= hi[:2]))
    return {"feasible": bool(not approach.any() and not carry.any() and supported),
            "approach_collisions": int(approach.sum()), "carry_collisions": int(carry.sum()),
            "geometric_centroid_supported": supported, "gap": gap, "thickness": thickness,
            "blocked": blocked, "expected_for_fixture": bool(gap >= thickness and not blocked)}
