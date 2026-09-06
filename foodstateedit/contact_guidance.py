"""Experimental RGB-observed strand constraints, with no VACE dependency.

The color observer is deliberately a small, falsifiable prototype. A passing
geometry/mask test does NOT validate its interpretation of natural RGB images.
Visibility is an externally supplied 3D prior, never inferred from a missing
generated strand. Frame time and diffusion time are separate inputs.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F


def image_tensor(array):
    return torch.as_tensor(array.copy(), dtype=torch.float32).permute(2, 0, 1)[None] / 255.0


def rgb_observer(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen analytic color likelihoods, NOT semantic segmentation.

    No target mask is multiplied into these outputs. This is important: an
    observer must actually respond to changed output pixels, not draw its answer
    from the intended control. Ranges intentionally remain broad for shading.
    """
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("RGB must have shape B,3,H,W")
    red, green, blue = rgb[:, :1], rgb[:, 1:2], rgb[:, 2:3]
    noodle = (torch.sigmoid((red - .64) * 18)
              * torch.sigmoid((green - .59) * 18)
              * torch.sigmoid((green - blue - .065) * 25))
    utensil = (torch.sigmoid((red - green - .075) * 28)
               * torch.sigmoid((green - blue - .025) * 28)
               * torch.sigmoid((.65 - green) * 18))
    return noodle, utensil


def sample_curve(field: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Bilinear samples in pixel coordinates; output B,N."""
    if field.ndim != 4 or field.shape[1] != 1:
        raise ValueError("Scalar image expected")
    if uv.ndim != 2 or uv.shape[1] != 2 or len(uv) < 2:
        raise ValueError("Curve must be N,2, with at least two points")
    height, width = field.shape[-2:]
    uv = uv.to(device=field.device, dtype=field.dtype)
    if not torch.isfinite(uv).all():
        raise ValueError("Curve contains invalid coordinates")
    if (uv[:, 0] < 0).any() or (uv[:, 0] > width-1).any() or (uv[:, 1] < 0).any() or (uv[:, 1] > height-1).any():
        raise ValueError("Curve extends outside measured image")
    normalized = uv / uv.new_tensor([width - 1, height - 1]) * 2 - 1
    grid = normalized[None, None].expand(field.shape[0], -1, -1, -1)
    return F.grid_sample(field, grid, align_corners=True, padding_mode="zeros")[:, 0, 0]


def path_reliability(samples: torch.Tensor, visible: torch.Tensor, temperature=.08):
    """Smooth weakest-link score on a fixed sampled path, not a topology proof."""
    if temperature <= 0:
        raise ValueError("Positive temperature required")
    selected = samples[..., visible.to(device=samples.device, dtype=torch.bool)]
    if selected.shape[-1] < 2:
        raise ValueError("At least two visible samples are needed")
    return (-temperature * torch.logsumexp(-selected / temperature, dim=-1)
            + temperature * math.log(selected.shape[-1]))


def structure_energy(noodle, utensil, uv, visible, tip_uv, *, phase, mode="phase_visibility"):
    """Observe generated probability fields under fixed path/contact constraints.

    Fixed guidance ignores phase and visibility for ablation. Proposed guidance
    activates only at contact/lift/hold and exempts truly occluded path samples.
    The loss is a path-evidence surrogate. It cannot certify semantic topology.
    """
    if mode not in {"fixed", "phase_visibility"}:
        raise ValueError("Unknown guidance mode")
    if phase not in {"source", "approach", "contact", "lift", "final_hold"}:
        raise ValueError("Unknown phase")
    enabled = mode == "fixed" or phase in {"contact", "lift", "final_hold"}
    zero = (noodle.sum() + utensil.sum()) * 0
    if not enabled:
        return {"total": zero, "connection": zero, "contact": zero, "active": False}
    visibility = visible if mode == "phase_visibility" else torch.ones_like(visible, dtype=torch.bool)
    samples = sample_curve(noodle, uv)
    connection = (1 - path_reliability(samples, visibility)).mean()
    # Both sampled utensil tips must be visible. A missing noodle endpoint is
    # exempted only when the fixed 3D visibility prior says it is occluded.
    tips = sample_curve(utensil, tip_uv)
    contact = (1 - tips).mean()
    if bool(visibility[-1]):
        contact = contact + (1 - samples[:, -3:].mean())
    return {"total": connection + .5 * contact,
            "connection": connection, "contact": contact, "active": True}


def project_points(xyz, intrinsic):
    projected = xyz @ intrinsic.T
    if not torch.isfinite(projected).all() or (projected[:, 2] <= 0).any():
        raise ValueError("Positive finite 3D depth is required")
    return projected[:, :2] / projected[:, 2:3]


def occlusion_visibility(uv, strand_depth, tips, handles, utensil_depths, radius):
    """Independent per-stick depth tests; a depth crossing alone isn't occlusion."""
    visible = torch.ones(len(uv), dtype=torch.bool, device=uv.device)
    for tip, handle, depth in zip(tips, handles, utensil_depths):
        direction = handle - tip
        alpha = ((uv-tip) * direction).sum(-1) / direction.square().sum().clamp_min(1e-8)
        closest = tip + alpha.clamp(0, 1)[:, None] * direction
        overlaps = (uv-closest).square().sum(-1) <= radius**2
        behind = strand_depth > depth + 1e-6
        visible &= ~(overlaps & behind)
    return visible


def geometry_frame(arrays, geometry, frame, width, height):
    if frame < 6:
        raise ValueError("Stored Day 13 strand geometry begins at contact")
    xyz = torch.as_tensor(arrays["strand_xyz"][frame].copy(), dtype=torch.float32)
    intrinsic = torch.as_tensor(arrays["intrinsic"].copy(), dtype=torch.float32)
    uv = project_points(xyz, intrinsic)
    pinch = project_points(torch.as_tensor(arrays["pinch_xyz"][frame:frame+1].copy()), intrinsic)[0]
    settings = geometry["chopsticks"]
    sep = settings["tip_separation_normalized"] * height
    hsep = settings["handle_separation_normalized"] * height
    offset = uv.new_tensor(settings["handle_offset_normalized"]) * uv.new_tensor([width, height])
    tips = torch.stack([pinch + uv.new_tensor([0., -sep/2]), pinch + uv.new_tensor([0., sep/2])])
    handles = torch.stack([pinch + offset + uv.new_tensor([0., -hsep/2]), pinch + offset + uv.new_tensor([0., hsep/2])])
    depths = torch.as_tensor(arrays["chopstick_depths"].copy())
    visible = occlusion_visibility(uv, xyz[:, 2], tips, handles, depths, settings["line_width_px"] / 2)
    return uv, visible, tips, handles
