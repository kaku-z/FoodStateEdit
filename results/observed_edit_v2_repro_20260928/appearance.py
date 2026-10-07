"""Bounded image-space lighting adaptation, not estimated physical albedo."""
import numpy as np


def interior_alpha(mask, width=6):
    mask = np.asarray(mask, bool)
    current = mask.copy()
    distance = np.zeros(mask.shape, np.float32)
    for _ in range(width):
        distance += current
        p = np.pad(current, 1)
        current = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return distance/width


def relight_observed(payload, proposal, target_mask, core):
    """Fit only 6 color parameters, bounded a in [.85,1.15], b in [-12,12]."""
    x = payload[core].astype(float)
    y = proposal[core].astype(float)
    if len(x) < 20:
        raise ValueError('Insufficient observed core')
    parameters = []
    out = payload.astype(float).copy()
    for channel in range(3):
        X = np.column_stack([x[:, channel], np.ones(len(x))])
        keep = np.ones(len(x), bool)
        a, b = 1., 0.
        for _ in range(3):
            coef = np.linalg.lstsq(X[keep], y[keep, channel], rcond=None)[0]
            a, b = np.clip(coef[0], .85, 1.15), np.clip(coef[1], -12, 12)
            residual = np.abs(X @ [a, b] - y[:, channel])
            keep = residual <= np.quantile(residual, .8)
        out[:, :, channel] = a*out[:, :, channel]+b
        parameters.append([float(a), float(b)])
    alpha = interior_alpha(target_mask, 6)[..., None]
    final = np.rint(alpha*np.clip(out, 0, 255)+(1-alpha)*proposal).clip(0, 255).astype(np.uint8)
    return final, parameters
