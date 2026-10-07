"""Cached source fields, analytic allocation and motion, spatial residual priors.

Every query is evaluated independently against one cached image context. Actions
can redistribute an inferred source state, but cannot regenerate its geometry or
material. The canonical field is in [-1, 1]^3, SDF is divided by 0.5, and material
is linear RGB mapped to [-1, 1]. These units describe the synthetic training world.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MLD2Config:
    width: int = 384
    global_width: int = 512
    blocks: int = 5
    coordinate_bands: int = 6
    grid_size: int = 16
    sdf_scale: float = 0.5
    residual_scale: float = 0.12


def project_uv(xyz: torch.Tensor) -> torch.Tensor:
    yaw, elev = math.radians(30), math.radians(35)
    basis = xyz.new_tensor([[-math.sin(yaw), math.cos(yaw), 0],
        [math.sin(elev)*math.cos(yaw), math.sin(elev)*math.sin(yaw), -math.cos(elev)]])
    return xyz @ basis.T / 1.45


def srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    return torch.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055).pow(2.4))


class PointBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.network = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, x):
        return x + self.network(self.norm(x)) * .5


class MLD2Model(nn.Module):
    def __init__(self, config: MLD2Config | None = None):
        super().__init__()
        self.config = config or MLD2Config()
        c = self.config
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.SiLU())
        self.global_encoder = nn.Sequential(
            nn.Conv2d(128, 192, 3, 2, 1), nn.GroupNorm(8, 192), nn.SiLU(),
            nn.Conv2d(192, 256, 3, 2, 1), nn.GroupNorm(8, 256), nn.SiLU(),
            nn.Flatten(), nn.Linear(256 * 4 * 4, c.global_width), nn.SiLU(),
            nn.Linear(c.global_width, c.global_width), nn.SiLU())
        self.register_buffer('coordinate_frequencies', 2.0 ** torch.arange(c.coordinate_bands))
        self.coordinate_projection = nn.Sequential(
            nn.Linear(128 + c.global_width + 3 + 6*c.coordinate_bands + 2 + 3, c.width), nn.SiLU())
        self.source_decoder = nn.Sequential(*(PointBlock(c.width) for _ in range(c.blocks)))
        self.geometry_head = nn.Sequential(nn.LayerNorm(c.width), nn.Linear(c.width, c.width),
                                          nn.SiLU(), nn.Linear(c.width, 9))
        self.material_head = nn.Sequential(nn.LayerNorm(c.width), nn.Linear(c.width, c.width),
                                          nn.SiLU(), nn.Linear(c.width, 3))
        self.source_image_head = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.SiLU(),
                                              nn.Conv2d(64, 2, 1))
        self.residual_input = nn.Sequential(nn.Linear(c.width + 4 + 17, 192), nn.SiLU())
        self.residual_decoder = nn.Sequential(*(PointBlock(192) for _ in range(4)))
        self.residual_head = nn.Linear(192, 4)
        self.register_buffer('time_frequencies', 2.0 ** torch.arange(8))
        offsets = torch.tensor(np.stack(np.meshgrid(*([[-.5/c.grid_size, .5/c.grid_size]]*3),
                                                    indexing='ij'), -1).reshape(-1, 3), dtype=torch.float32)
        self.register_buffer('subcell_offsets', offsets)

    def encode_image(self, image):
        local = self.image_encoder(image)
        auxiliary = F.interpolate(self.source_image_head(local), image.shape[-2:],
                                  mode='bilinear', align_corners=False)
        return {'local': local, 'global': self.global_encoder(local), 'image': image,
                'mask_logits': auxiliary[:, :1], 'depth': auxiliary[:, 1:2]}

    def query(self, cache, xyz, uv=None):
        if uv is None:
            uv = project_uv(xyz)
        local = F.grid_sample(cache['local'], uv[:, :, None], align_corners=False,
                              padding_mode='border').squeeze(-1).transpose(1, 2)
        observed = F.grid_sample(cache['image'], uv[:, :, None], align_corners=False,
                                 padding_mode='border').squeeze(-1).transpose(1, 2)
        linear = srgb_to_linear(observed.float()) * 2 - 1
        phase = math.pi * xyz[..., None] * self.coordinate_frequencies.to(xyz.dtype)
        position = torch.cat((xyz, phase.sin().flatten(2), phase.cos().flatten(2)), -1)
        global_feature = cache['global'][:, None].expand(-1, xyz.shape[1], -1)
        feature = self.source_decoder(self.coordinate_projection(torch.cat(
            (position, uv, local, global_feature, linear.to(local.dtype)), -1)))
        geometry = self.geometry_head(feature).float()
        # The observed projected color is a source anchor. The learned correction
        # handles shading and hidden material; the target image is never an input.
        rgb = torch.tanh(torch.atanh(linear*.95) + self.material_head(feature).float())
        return {'features': feature, 'sdf': geometry[..., :1], 'corner_sdf': geometry[..., 1:],
                'occupancy_logits': -geometry[..., :1]/.025,
                'corner_occupancy_logits': -geometry[..., 1:]/.025, 'rgb': rgb}

    def encode_source(self, image, canonical_xyz, source_uv=None):
        return self.query(self.encode_image(image), canonical_xyz, source_uv)

    def query_state(self, cache, xyz, uv=None):
        """Evaluate the actual continuous field at all eight material sub-sites.

        ``query`` exposes a fast auxiliary eight-value head for supervision.
        Scientific allocation uses these direct continuous queries instead.
        """
        source = self.query(cache, xyz, uv)
        subxyz = (xyz[..., None, :]+self.subcell_offsets.to(xyz.dtype)).flatten(1, 2)
        sub_sdf = self.query(cache, subxyz)['sdf'].reshape(*xyz.shape[:2], 8)
        source['fast_corner_sdf'] = source['corner_sdf']
        source['corner_sdf'] = sub_sdf
        source['corner_occupancy_logits'] = -sub_sdf/.025
        return source

    def denoise_residual(self, features, noisy_residual, normalized_t):
        t = normalized_t.reshape(-1, 1)
        phase = math.pi * 2 * t * self.time_frequencies.to(t.dtype)
        te = torch.cat((t, phase.sin(), phase.cos()), -1)[:, None].expand(-1, features.shape[1], -1)
        hidden = self.residual_input(torch.cat((features, noisy_residual.to(features.dtype),
                                                te.to(features.dtype)), -1))
        return self.residual_head(self.residual_decoder(hidden)).float()

    def allocation(self, source, xyz, action, soft=True):
        """Return predicted material mass, not a coordinate-only cut fraction."""
        points = xyz[..., None, :] + self.subcell_offsets.to(xyz.dtype)
        cut = (points * action[:, None, None, :3]).sum(-1) > action[:, None, None, 3]
        is_cut = action[:, :4].abs().sum(-1) > 0
        cut = cut & is_cut[:, None, None]
        occupied = (source['corner_occupancy_logits'].sigmoid() if soft
                    else (source['corner_sdf'] <= 0).float())
        source_mass = occupied.mean(-1, keepdim=True)
        carried_mass = (occupied * cut).mean(-1, keepdim=True)
        return {'source_mass': source_mass, 'carried_mass': carried_mass,
                'remaining_mass': source_mass - carried_mass,
                'carried_fraction': carried_mass/source_mass.clamp_min(1e-8),
                'carried_xyz': rigid_transform(xyz, action), 'remaining_xyz': xyz,
                'remaining_rgb': source['rgb'], 'carried_rgb': source['rgb']}


def rigid_transform(xyz, action):
    a = action[:, 7:10]
    theta = a.norm(dim=-1, keepdim=True)
    axis = a / theta.clamp_min(1e-8)
    cross = torch.cross(axis[:, None].expand_as(xyz), xyz, dim=-1)
    dot = (xyz * axis[:, None]).sum(-1, keepdim=True)
    cos = theta.cos()[:, None]
    sin = theta.sin()[:, None]
    return xyz*cos + cross*sin + axis[:, None]*dot*(1-cos) + action[:, None, 4:7]


def spatial_noise(xyz, scene_ids, seed=20261004, channels=4, modes=32):
    """A replayable continuous Gaussian Fourier field, not independent query noise."""
    coordinates = np.asarray(xyz, dtype=np.float32)
    all_noise = []
    for scene_id in scene_ids:
        digest = hashlib.sha256(f'mld2-residual:{seed}:{scene_id}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
        frequency = rng.normal(0, 2.0, (3, modes)).astype(np.float32)
        sine = rng.normal(size=(modes, channels)).astype(np.float32)
        cosine = rng.normal(size=(modes, channels)).astype(np.float32)
        phase = coordinates @ frequency * math.pi
        noise = (np.sin(phase) @ sine + np.cos(phase) @ cosine) / math.sqrt(modes)
        all_noise.append(noise.astype(np.float32))
    return np.stack(all_noise)


@torch.inference_mode()
def sample_residual(model, source, noise, steps=50):
    alpha = torch.cumprod(1-torch.linspace(.0001, .02, 1000, device=noise.device), 0)
    schedule = np.rint(np.linspace(999, 0, steps)).astype(int)
    state = noise.clone()
    for index, t in enumerate(schedule):
        at = alpha[t]
        next_a = alpha[schedule[index+1]] if index+1 < len(schedule) else state.new_tensor(1)
        time = state.new_full((len(state),), t/999)
        epsilon = model.denoise_residual(source['features'], state, time)
        x0 = ((state-(1-at).sqrt()*epsilon)/at.sqrt()).clamp(-3, 3)
        epsilon = (state-at.sqrt()*x0)/(1-at).sqrt()
        state = next_a.sqrt()*x0 + (1-next_a).sqrt()*epsilon
    posterior = torch.cat((source['sdf'], source['rgb']), -1)
    return (posterior + model.config.residual_scale*state).clamp(-1, 1)
