"""Source-only layers and a Dirichlet boundary compositor for the v3 repair.

No paired target, inferred volume, or physical albedo is used. The local neutral
plate fit is an explicit restricted background model, not a general inpainter.
"""
import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from .core import dilate


def poisson_merge(base, proposal, mask, screening=0.):
    """Preserve proposal gradients inside mask and base Dirichlet values outside.

    Base pixels *inside* the replacement domain never enter the right-hand side.
    This avoids blending the removed food back into the repaired background.
    """
    if not np.isfinite(screening) or screening < 0:
        raise ValueError('Nonnegative finite screening required')
    mask = np.asarray(mask, bool)
    if base.shape != proposal.shape or base.shape[:2] != mask.shape:
        raise ValueError('Aligned RGB images and mask required')
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        raise ValueError('Poisson region requires an observed image boundary')
    if not mask.any():
        return base.copy()
    yy, xx = np.nonzero(mask)
    n = len(yy)
    ids = np.full(mask.shape, -1, np.int32)
    ids[yy, xx] = np.arange(n)
    rows, cols, vals = [np.arange(n)], [np.arange(n)], [np.full(n, 4.+screening)]
    p, b = proposal.astype(np.float64), base.astype(np.float64)
    # screening>0 adds a data term, preventing the harmonic boundary correction
    # from shifting the whole generated hole's brightness. Default 0 preserves
    # the original frozen v3 ablation and its reproducibility.
    rhs = (4.+screening)*p[yy, xx]
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = yy+dy, xx+dx
        rhs -= p[ny, nx]
        neighbor = ids[ny, nx]
        inside = neighbor >= 0
        rows.append(np.flatnonzero(inside)); cols.append(neighbor[inside])
        vals.append(np.full(inside.sum(), -1.))
        rhs[~inside] += b[ny[~inside], nx[~inside]]
    matrix = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                        shape=(n, n)).tocsc()
    solution = splu(matrix).solve(rhs)
    out = base.copy()
    out[yy, xx] = np.clip(np.rint(solution), 0, 255).astype(np.uint8)
    return out


def neutral_plate(source, source_mask, protected_mask=None):
    """Robust quadratic RGB field fitted only to nearby observed neutral plate."""
    h, w = source_mask.shape
    yy, xx = np.mgrid[:h, :w]
    sy, sx = np.nonzero(source_mask)
    rgb = source.astype(float)
    protected = np.zeros_like(source_mask) if protected_mask is None else np.asarray(protected_mask, bool)
    if np.any(protected & source_mask):
        raise ValueError('Kept food and removed food overlap')
    brightness = rgb.max(axis=2)
    neutral = (brightness-rgb.min(axis=2))/np.maximum(brightness, 1) < .22
    # Exclude food core and its shadow/fringe from fitting.
    sample = dilate(source_mask, 65) & ~dilate(source_mask, 18) & ~protected & neutral & (brightness > 65)
    sample &= ((xx+3*yy) % 3 == 0)
    u, v = (xx-sx.mean())/100, (yy-sy.mean())/100
    basis = np.stack([np.ones_like(u), u, v, u*u, u*v, v*v], -1)
    X, Y = basis[sample], rgb[sample]
    if len(X) < 200:
        raise ValueError('Insufficient neutral plate observations; no silent fallback')
    keep = np.ones(len(X), bool)
    for _ in range(4):
        if keep.sum() < 100:
            raise ValueError('Unstable plate fit')
        coef = np.linalg.lstsq(X[keep], Y[keep], rcond=None)[0]
        residual = np.max(np.abs(X@coef-Y), axis=1)
        keep = residual < 25
    fit = (basis.reshape(-1, 6)@coef).reshape(source.shape)
    fit = np.clip(np.rint(fit), 0, 255).astype(np.uint8)
    # A removal boundary must lie OUTSIDE the food fringe/shadow. Harmonic
    # interpolation from a contaminated food boundary spreads its color inward.
    # Use fitted plate through the entire food core, feather only the outer rim.
    hole = dilate(source_mask, 28) & ~protected
    distance = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.minimum(distance/14, 1.)
    alpha[source_mask] = 1.  # Never mix original food into the erased footprint.
    background = np.rint(alpha[...,None]*fit+(1-alpha[...,None])*source).clip(0,255).astype(np.uint8)
    return background, hole, {'fit_samples': int(keep.sum()), 'hole_pixels': int(hole.sum())}


def build_layers(source, state, handle_xy, protected_mask=None):
    background, hole, metadata = neutral_plate(source, state['source_mask'], protected_mask)
    target = state['target_mask']
    yy, xx = np.nonzero(target)
    center = (int(round(xx.mean())), int(round(yy.mean())))
    tool = np.zeros(target.shape, np.uint8)
    cv2.line(tool, center, tuple(handle_xy), 255, 19, cv2.LINE_AA)
    axes = (int((xx.max()-xx.min())*.56)+8, int((yy.max()-yy.min())*.56)+8)
    cv2.ellipse(tool, center, axes, 0, 0, 360, 255, -1, cv2.LINE_AA)
    tool = tool > 0
    # The plate hidden by the original food is also unknown: allow the model
    # to repair this placeholder, while it never sees the original food there.
    free = (dilate(tool, 10) | hole) & ~target
    if protected_mask is not None:
        free &= ~protected_mask
    free[[0, -1], :] = False; free[:, [0, -1]] = False
    canvas = background.copy()
    canvas[tool & ~hole] = (168, 177, 186)
    canvas[target] = state['transported_rgb'][target]
    base = background.copy()
    base[target] = state['transported_rgb'][target]
    return {'background': background, 'reference': canvas, 'base': base,
            'free_mask': free, 'hole': hole, 'target_mask': target,
            'metadata': metadata}


def composite_layers(base, proposal, free_mask):
    hard = base.copy()
    hard[free_mask] = proposal[free_mask]
    smooth = poisson_merge(base, proposal, free_mask)
    return hard, smooth
