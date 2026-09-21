"""Build image-aligned SCFST roles and source UVs from frozen E7 controls."""
from __future__ import annotations

import numpy as np

from foodstateedit.state_transfer.bridge import pack_controls
from foodstateedit.state_transfer.geometry import KEEP, MOVE, EXPOSE


def _box(mask: np.ndarray) -> tuple[int, int, int, int]:
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError("empty correspondence mask")
    return int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1


def build_condition_video(generation_mask, planned_payload, removal_projection):
    """Return [T,H,W,10] controls without claiming recovered physical state.

    Payload pixels are mapped affinely back to the frozen removal projection.
    Cavity-support pixels outside the payload are unknown exposure. All other
    pixels keep identity correspondence. MOVE wins where masks overlap.
    """
    mask = np.asarray(generation_mask)
    payload = np.asarray(planned_payload)
    source = np.asarray(removal_projection)
    if mask.ndim != 3 or payload.shape != mask.shape or source.shape != mask.shape[1:]:
        raise ValueError("expected mask/payload [T,H,W] and source [H,W]")
    if not np.isin(mask, [0, 255]).all() or not np.isin(payload, [0, 255]).all():
        raise ValueError("masks must be binary uint8-style arrays")
    t, h, w = mask.shape
    yy, xx = np.indices((h, w))
    identity_u = 2 * xx / (w - 1) - 1
    identity_v = 2 * yy / (h - 1) - 1
    out = np.zeros((t, h, w, 10), np.float32)
    out[..., KEEP] = 1
    out[..., 4] = identity_u
    out[..., 5] = identity_v
    out[..., 6] = 1
    out[..., 8] = 1
    sx0, sy0, sx1, sy1 = _box(source > 0)
    for frame in range(t):
        moved = payload[frame] > 0
        exposed = (mask[frame] > 0) & ~moved
        out[frame, exposed, :4] = 0
        out[frame, exposed, EXPOSE] = 1
        out[frame, exposed, 4:6] = 2
        out[frame, exposed, 6] = 0
        out[frame, exposed, 8] = 0
        if moved.any():
            tx0, ty0, tx1, ty1 = _box(moved)
            mx = xx[moved]
            my = yy[moved]
            # Pixel-centre affine map; clamped to the observed source box.
            su = sx0 + ((mx - tx0 + .5) / max(1, tx1 - tx0)) * (sx1 - sx0) - .5
            sv = sy0 + ((my - ty0 + .5) / max(1, ty1 - ty0)) * (sy1 - sy0) - .5
            su = np.clip(su, sx0, sx1 - 1)
            sv = np.clip(sv, sy0, sy1 - 1)
            out[frame, moved, :4] = 0
            out[frame, moved, MOVE] = 1
            out[frame, moved, 4] = 2 * su / (w - 1) - 1
            out[frame, moved, 5] = 2 * sv / (h - 1) - 1
            out[frame, moved, 7] = 1
    if not np.allclose(out[..., :4].sum(-1), 1):
        raise AssertionError("roles are not exclusive")
    return out


def build_packed(generation_mask, planned_payload, removal_projection,
                 frame_indices, grid_hw, prefix_tokens):
    video = build_condition_video(generation_mask, planned_payload, removal_projection)
    packed, active = pack_controls(video, frame_indices, grid_hw, prefix_tokens=prefix_tokens)
    return packed, active
