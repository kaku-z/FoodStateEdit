"""Shared image-conditioned material-lineage pretraining architecture.

The input sites are a fixed canonical bounding-box lattice, not ground-truth
surface samples.  Image projection is known camera information, not a target
visibility mask.  The source field is sampled once and reused across actions:
the denoiser intentionally cannot receive an action or branch identity.

This module provides a learned field and transition, not measured 3-D truth,
calibrated albedo, or a guarantee of photographic realism.  The allocation
conserves an externally initialized reference mass for every persistent site;
it does not establish that the inferred source mass or shape is correct.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


MODEL_FORMAT_VERSION = 1
JOINT_CHANNELS = ("scaled_signed_distance", "linear_red", "linear_green", "linear_blue")
ACTION_CHANNELS = (
    "cut_normal_x", "cut_normal_y", "cut_normal_z", "cut_offset",
    "translation_x", "translation_y", "translation_z",
    "axis_angle_x", "axis_angle_y", "axis_angle_z",
)


@dataclass(frozen=True)
class MLDConfig:
    """Configuration serialized verbatim with trained checkpoints."""

    width: int = 128
    heads: int = 4
    source_blocks: int = 2
    diffusion_blocks: int = 2
    image_channels: int = 3
    action_dim: int = 10
    joint_dim: int = 4
    state_dim: int = 0
    coordinate_bands: int = 4
    time_bands: int = 8

    def __post_init__(self):
        integer_names = (
            "width", "heads", "source_blocks", "diffusion_blocks", "image_channels",
            "action_dim", "joint_dim", "coordinate_bands", "time_bands",
        )
        if any(isinstance(getattr(self, k), bool) or not isinstance(getattr(self, k), int)
               or getattr(self, k) < 1 for k in integer_names):
            raise ValueError("MLD configuration entries must be positive integers")
        if self.width < 8 or self.width % self.heads:
            raise ValueError("width must be at least eight and divisible by heads")
        if (self.image_channels, self.action_dim, self.joint_dim) != (3, 10, 4):
            raise ValueError("v1 schema is RGB image, action10, and joint SDF/RGB4")
        if isinstance(self.state_dim, bool) or not isinstance(self.state_dim, int) or self.state_dim not in (0, 4):
            raise ValueError("state_dim must be zero (legacy) or four (coupled canonical state)")


def _transformer(width: int, heads: int, blocks: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=width, nhead=heads, dim_feedforward=width * 4, dropout=0.0,
        activation="gelu", batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=blocks, norm=nn.LayerNorm(width),
                                 enable_nested_tensor=False)


def canonical_to_world(points: torch.Tensor, rotation: torch.Tensor,
                       translation: torch.Tensor) -> torch.Tensor:
    """Apply a supplied proper rigid frame to row-vector canonical points.

    Shapes are points[B,N,3], rotation[B,3,3] or [3,3], and translation[B,3]
    or [3].  This external transformation guarantees covariance of the
    representation under a joint world-frame change.  It is not a claim that
    the CNN/Transformer is equivariant to rotating its input photograph.
    The caller is responsible for supplying an orthonormal proper rotation.
    """
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape [B,N,3]")
    if rotation.shape not in ((3, 3), (points.shape[0], 3, 3)):
        raise ValueError("rotation must have shape [3,3] or [B,3,3]")
    if translation.shape not in ((3,), (points.shape[0], 3)):
        raise ValueError("translation must have shape [3] or [B,3]")
    offset = translation if translation.ndim == 1 else translation[:, None, :]
    return torch.matmul(points, rotation.transpose(-1, -2)) + offset


class JointFieldDenoiser(nn.Module):
    """Joint SDF/material epsilon predictor; no action or target input."""

    def __init__(self, config: MLDConfig):
        super().__init__()
        self.config = config
        self.register_buffer("time_frequencies", 2.0 ** torch.arange(config.time_bands))
        dimension = config.width + config.joint_dim + 1 + 2 * config.time_bands
        self.input_projection = nn.Sequential(nn.Linear(dimension, config.width), nn.SiLU())
        self.transformer = _transformer(config.width, config.heads, config.diffusion_blocks)
        self.output_projection = nn.Linear(config.width, config.joint_dim)

    def forward(self, features: torch.Tensor, noisy_joint: torch.Tensor,
                normalized_t: torch.Tensor) -> torch.Tensor:
        if (features.ndim != 3 or features.shape[-1] != self.config.width
                or noisy_joint.shape != (*features.shape[:2], self.config.joint_dim)):
            raise ValueError("features/noisy_joint must be matching [B,N,W]/[B,N,4]")
        batch, count = features.shape[:2]
        if normalized_t.shape not in ((batch,), (batch, 1)):
            raise ValueError("normalized_t must have shape [B] or [B,1]")
        time = normalized_t.reshape(batch, 1).to(dtype=features.dtype)
        phase = 2.0 * torch.pi * time * self.time_frequencies.to(features.dtype)
        embedding = torch.cat((time, phase.sin(), phase.cos()), dim=-1)
        embedding = embedding[:, None, :].expand(-1, count, -1)
        tokens = self.input_projection(torch.cat((features, noisy_joint, embedding), dim=-1))
        return self.output_projection(self.transformer(tokens))


class MLDModel(nn.Module):
    """One model shared across all source images, lattice sites and actions.

    ``encode_source`` accepts only the observed source and known camera/lattice.
    ``transition`` returns conservative two-branch weights and learned offsets.
    ``denoise`` accepts only source features, noisy canonical joint field and time.
    Neither source generation function accepts a cut action, so two branches
    cannot independently regenerate their material appearance.
    """

    def __init__(self, config: MLDConfig | None = None):
        super().__init__()
        self.config = config or MLDConfig()
        c = self.config
        groups = math.gcd(c.width, 8)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(c.image_channels, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, c.width, 3, stride=2, padding=1),
            nn.GroupNorm(groups, c.width), nn.SiLU(),
        )
        self.register_buffer("coordinate_frequencies", 2.0 ** torch.arange(c.coordinate_bands))
        coordinate_dim = 3 + 6 * c.coordinate_bands
        self.coordinate_projection = nn.Sequential(
            nn.Linear(coordinate_dim + 2 + 2 * c.width, c.width), nn.SiLU(),
            nn.Linear(c.width, c.width),
        )
        self.source_transformer = _transformer(c.width, c.heads, c.source_blocks)
        self.source_geometry_head = nn.Sequential(
            nn.Linear(c.width, c.width), nn.SiLU(), nn.Linear(c.width, 2),
        )
        self.source_material_head = nn.Sequential(
            nn.Linear(c.width, c.width), nn.SiLU(), nn.Linear(c.width, 3), nn.Tanh(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(c.action_dim, c.width), nn.SiLU(), nn.Linear(c.width, c.width),
        )
        self.state_encoder = (nn.Sequential(
            nn.Linear(c.state_dim, c.width), nn.SiLU(), nn.Linear(c.width, c.width),
        ) if c.state_dim else None)
        self.transition_head = nn.Sequential(
            nn.Linear(c.width * (3 if c.state_dim else 2), c.width), nn.SiLU(),
            nn.Linear(c.width, c.width), nn.SiLU(), nn.Linear(c.width, 7),
        )
        self.denoiser = JointFieldDenoiser(c)

    def encode_source(self, image: torch.Tensor, canonical_xyz: torch.Tensor,
                      source_uv: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict source SDF, normalized linear RGB and occupancy log odds.

        image is source sRGB[B,3,64,64] in [0,1]; lattice xyz is [B,N,3]
        normalized to [-1,1]; source_uv[B,N,2] is the known fixed-camera
        projection in grid_sample coordinates.  It must not encode target
        geometry, branch membership, or ground-truth visibility.
        """
        c = self.config
        if image.ndim != 4 or image.shape[1:] != (c.image_channels, 64, 64):
            raise ValueError("source image must have shape [B,3,64,64]")
        if (canonical_xyz.ndim != 3 or canonical_xyz.shape[0] != image.shape[0]
                or canonical_xyz.shape[-1] != 3 or canonical_xyz.shape[1] < 1):
            raise ValueError("canonical_xyz must be nonempty [B,N,3]")
        if source_uv.shape != (*canonical_xyz.shape[:2], 2):
            raise ValueError("source_uv must have shape [B,N,2]")
        spatial = self.image_encoder(image)
        local = F.grid_sample(spatial, source_uv[:, :, None, :], mode="bilinear",
                              padding_mode="zeros", align_corners=False)
        local = local.squeeze(-1).transpose(1, 2)
        global_feature = spatial.mean(dim=(-2, -1))[:, None, :].expand(-1, canonical_xyz.shape[1], -1)
        phase = torch.pi * canonical_xyz.unsqueeze(-1) * self.coordinate_frequencies.to(canonical_xyz.dtype)
        coordinate_feature = torch.cat(
            (canonical_xyz, phase.sin().flatten(2), phase.cos().flatten(2)), dim=-1,
        )
        feature = self.coordinate_projection(torch.cat(
            (coordinate_feature, source_uv, local, global_feature), dim=-1,
        ))
        feature = self.source_transformer(feature)
        geometry = self.source_geometry_head(feature)
        return {
            "features": feature,
            "sdf": geometry[..., :1],
            "rgb": self.source_material_head(feature),
            "occupancy_logits": geometry[..., 1:2],
        }

    def transition(self, features: torch.Tensor, action: torch.Tensor,
                   state_joint: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Predict fractions and offsets without making new material tokens.

        action is cut normal3, offset1, translation3, and axis-angle3.  All-zero
        action is the null command.  Null outputs are learned, not zeroed by a
        hard mask.  Allocation's last axis is [remaining, carried] and sums to
        one per reference site; offsets are canonical-space, not world-space.
        With state_dim=4, state_joint[B,N,4] must be the predicted or sampled
        canonical source SDF/RGB field.  It must never be supplied from a target
        label or noisy target.  The same sampled source field is reused for all
        actions.  Legacy state_dim=0 models retain their previous parameter
        shapes and reject a state argument instead of silently ignoring it.
        """
        if features.ndim != 3 or features.shape[-1] != self.config.width:
            raise ValueError("features must have shape [B,N,W]")
        if action.shape != (features.shape[0], self.config.action_dim):
            raise ValueError("action must have shape [B,10]")
        action_feature = self.action_encoder(action)[:, None, :].expand(-1, features.shape[1], -1)
        condition = [features, action_feature]
        if self.config.state_dim:
            if state_joint is None or state_joint.shape != (*features.shape[:2], self.config.state_dim):
                raise ValueError("coupled transition requires predicted/sampled state_joint [B,N,4]")
            condition.append(self.state_encoder(state_joint))
        elif state_joint is not None:
            raise ValueError("legacy state_dim=0 transition does not accept a source state")
        prediction = self.transition_head(torch.cat(condition, dim=-1))
        carried = prediction[..., :1].sigmoid()
        return {
            "carried_fraction": carried,
            "allocation": torch.cat((1.0 - carried, carried), dim=-1),
            "remaining_delta": prediction[..., 1:4],
            "carried_delta": prediction[..., 4:7],
        }

    def denoise(self, features: torch.Tensor, noisy_joint: torch.Tensor,
                normalized_t: torch.Tensor) -> torch.Tensor:
        return self.denoiser(features, noisy_joint, normalized_t)

    def forward(self, image: torch.Tensor, canonical_xyz: torch.Tensor,
                source_uv: torch.Tensor, action: torch.Tensor | None = None,
                noisy_joint: torch.Tensor | None = None,
                normalized_t: torch.Tensor | None = None,
                source_state: torch.Tensor | None = None) -> dict:
        source = self.encode_source(image, canonical_xyz, source_uv)
        output = {"source": source}
        if action is not None:
            state = source_state
            if self.config.state_dim and state is None:
                state = torch.cat((source["sdf"], source["rgb"]), dim=-1)
            output["transition"] = self.transition(source["features"], action, state)
        if (noisy_joint is None) != (normalized_t is None):
            raise ValueError("noisy_joint and normalized_t must be supplied together")
        if noisy_joint is not None:
            output["epsilon"] = self.denoise(source["features"], noisy_joint, normalized_t)
        return output
