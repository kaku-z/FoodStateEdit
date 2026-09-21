"""Explicit layout conversion for a future Wan model_fn integration.

No inferred temporal compression: callers supply the selected source frames.
Nearest-cell sampling is a prototype; it is not the VAE's true receptive-field
aggregation. Do not use naive averaging of source UVs across occlusion edges.
"""
import numpy as np


def pack_controls(conditions, frame_indices, grid_hw, *, prefix_tokens=0):
    """THWC -> [1,L,10], flattened in t,y,x order; prefix is inactive.

    Returns numpy arrays; wrap using torch.as_tensor on the chosen device.
    prefix_tokens counts tokens, NOT frames. The caller must compare L to the
    actual DiT tensor and choose a layout for both high/low-noise models.
    """
    c = np.asarray(conditions)
    if c.ndim != 4 or c.shape[-1] != 10 or not np.isfinite(c).all() or min(c.shape[:3]) < 1:
        raise ValueError("expected finite nonempty [T,H,W,10]")
    fi = np.asarray(frame_indices)
    if fi.ndim != 1 or not len(fi) or fi.dtype.kind not in "iu" or (fi < 0).any() or (fi >= len(c)).any():
        raise ValueError("explicit valid integer frame_indices required")
    if len(grid_hw) != 2 or any(int(v) != v or v < 1 for v in grid_hw):
        raise ValueError("grid_hw must be two positive integers")
    if int(prefix_tokens) != prefix_tokens or prefix_tokens < 0:
        raise ValueError("invalid prefix_tokens")
    gh, gw = map(int, grid_hw)
    h, w = c.shape[1:3]
    ys = np.minimum(((np.arange(gh)+.5)*h/gh).astype(int), h-1)
    xs = np.minimum(((np.arange(gw)+.5)*w/gw).astype(int), w-1)
    video = c[fi[:, None, None], ys[None, :, None], xs[None, None, :]].reshape(-1, 10)
    prefix = np.zeros((int(prefix_tokens), 10), np.float32)
    prefix[:, 0] = 1
    prefix[:, 4:6] = 2
    packed = np.concatenate([prefix, video], axis=0).astype(np.float32)[None]
    active = np.ones(packed.shape[:2], bool)
    active[:, :int(prefix_tokens)] = False
    return packed, active
