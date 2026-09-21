"""Optional torch components for a trainable residual provenance branch.

Zero-initialized outputs make the untrained branch exactly identity. This does
not block the base model's other reference paths or guarantee visual success.
"""
import math

import torch
from torch import nn


def validate_conditions(c):
    if c.ndim != 3 or c.shape[-1] != 10 or not c.is_floating_point() or not torch.isfinite(c).all():
        raise ValueError("conditions must be finite floating [B,L,10]")
    roles = c[..., :4]
    if ((roles < 0) | (roles > 1)).any() or not torch.allclose(roles.sum(-1), torch.ones_like(roles[..., 0]), atol=1e-5):
        raise ValueError("four role probabilities must sum to one")
    if (c[..., 6] < 0).any() or ((c[..., 7:] < 0) | (c[..., 7:] > 1)).any():
        raise ValueError("invalid depth/contact/confidence/utensil channels")
    valid_uv = (c[..., 4:6].abs() <= 1).all(-1)
    invalid_uv = (c[..., 4:6] == 2).all(-1)
    if (~(valid_uv | invalid_uv)).any():
        raise ValueError("UV must be normalized [-1,1] or the invalid sentinel (2,2)")
    return valid_uv


def gather_reference(memory, uv):
    """Gather the four neighbouring source tokens, O(B*L*4*C), not O(L*HW).

    Returns [B,L,4,C] and bilinear positional prior [B,L,4]. Coordinates use
    align_corners=True. Invalid UVs are safely clamped here and MUST be gated
    by the caller; no image-colour similarity is used to invent correspondence.
    """
    if memory.ndim != 4 or not memory.is_floating_point() or not torch.isfinite(memory).all():
        raise ValueError("reference memory must be finite floating [B,C,H,W]")
    b, channels, height, width = memory.shape
    if min(b, channels, height, width) < 1:
        raise ValueError("reference memory dimensions must be nonzero")
    if uv.ndim != 3 or uv.shape[0] != b or uv.shape[-1] != 2 or not torch.isfinite(uv).all():
        raise ValueError("UV must be finite [B,L,2]")
    xy = uv.float().clamp(-1, 1)
    x = (xy[..., 0]+1)*(width-1)/2
    y = (xy[..., 1]+1)*(height-1)/2
    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = (x0+1).clamp(max=width-1), (y0+1).clamp(max=height-1)
    dx, dy = x-x0, y-y0
    indices = torch.stack([y0*width+x0, y0*width+x1, y1*width+x0, y1*width+x1], -1)
    prior = torch.stack([(1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy], -1)
    flat = memory.flatten(2).transpose(1, 2)
    selected = flat[torch.arange(b, device=memory.device)[:, None, None], indices]
    return selected, prior


class StateTransferAdapter(nn.Module):
    """Call after a chosen DiT block with aligned controls and cached memory.

    No trained weights or automatic hooks are supplied. Both projections are
    zero-initialized; training is required for any useful image effect.
    ``active`` protects reference-prefix tokens. Batch size must match exactly:
    the caller must explicitly duplicate conditions for classifier-free guidance.
    """
    def __init__(self, token_dim, reference_channels, hidden_dim=64):
        super().__init__()
        if min(token_dim, reference_channels, hidden_dim) <= 0:
            raise ValueError("dimensions must be positive")
        self.token_dim = token_dim
        self.reference_channels = reference_channels
        self.query = nn.Linear(token_dim, hidden_dim, bias=False)
        self.key = nn.Linear(reference_channels, hidden_dim, bias=False)
        self.value = nn.Linear(reference_channels, hidden_dim, bias=False)
        self.route_out = nn.Linear(hidden_dim, token_dim, bias=False)
        self.state_net = nn.Sequential(nn.Linear(10, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, token_dim))
        nn.init.zeros_(self.route_out.weight)
        nn.init.zeros_(self.state_net[-1].weight)
        nn.init.zeros_(self.state_net[-1].bias)

    def forward(self, tokens, memory, conditions, active=None, strength=1.0):
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("strength must be finite and in [0,1]")
        valid_uv = validate_conditions(conditions)
        if tokens.ndim != 3 or tokens.shape[:2] != conditions.shape[:2] or tokens.shape[-1] != self.token_dim:
            raise ValueError("token/condition layout mismatch")
        if not torch.isfinite(tokens).all():
            raise ValueError("nonfinite tokens")
        if memory.ndim != 4 or memory.shape[0] != tokens.shape[0] or memory.shape[1] != self.reference_channels:
            raise ValueError("reference memory layout mismatch")
        if any(x.device != tokens.device for x in (memory, conditions)):
            raise ValueError("tokens, memory and conditions must share a device")
        if tokens.dtype != self.query.weight.dtype or memory.dtype != tokens.dtype:
            raise ValueError("adapter, tokens and memory must share dtype")
        if active is None:
            active = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        if active.shape != tokens.shape[:2] or active.dtype != torch.bool or active.device != tokens.device:
            raise ValueError("active must be boolean [B,L] on token device")
        values, prior = gather_reference(memory, conditions[..., 4:6])
        q, k = self.query(tokens), self.key(values)
        logits = (q.float().unsqueeze(2)*k.float()).sum(-1)/math.sqrt(k.shape[-1])
        # Exact zero positional weights stay excluded; at least one is nonzero.
        logits = logits + prior.masked_fill(prior == 0, 1).log()
        logits = logits.masked_fill(prior == 0, -torch.inf)
        weights = logits.softmax(-1).to(tokens.dtype)
        routed = (weights.unsqueeze(-1)*self.value(values)).sum(2)
        gate = conditions[..., :3].sum(-1)*conditions[..., 8]*valid_uv*(1-conditions[..., 9])
        state = conditions.float().clone()
        state[..., 4:6] = torch.where(valid_uv.unsqueeze(-1), state[..., 4:6], 0)
        # Relative depth may vary in scale; log1p avoids an unbounded raw channel.
        state[..., 6] = torch.log1p(state[..., 6])
        delta = self.route_out(routed)*gate.to(tokens.dtype).unsqueeze(-1) + self.state_net(state.to(tokens.dtype))
        output = torch.where(active.unsqueeze(-1), tokens + strength*delta, tokens)
        return output, {"source_gate": gate.detach(), "active_tokens": int(active.sum().item())}
