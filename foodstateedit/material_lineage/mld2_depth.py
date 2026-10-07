"""Image-observed free-space constraints on the same continuous source field."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .mld2 import MLD2Model, project_uv


class DepthSurfaceMLD2(MLD2Model):
    def __init__(self, config=None):
        super().__init__(config)
        self.image_refiner = nn.Sequential(nn.Conv2d(131, 64, 3, padding=1), nn.SiLU(), nn.Conv2d(64, 2, 1))
        nn.init.zeros_(self.image_refiner[-1].weight)
        nn.init.zeros_(self.image_refiner[-1].bias)
        yaw, elevation = math.radians(30), math.radians(35)
        self.register_buffer('toward_camera', torch.tensor([math.cos(elevation)*math.cos(yaw),
                                                          math.cos(elevation)*math.sin(yaw), math.sin(elevation)]))

    def encode_image(self, image):
        cache = super().encode_image(image)
        features = F.interpolate(cache['local'], image.shape[-2:], mode='bilinear', align_corners=False)
        refined = self.image_refiner(torch.cat((features, image), 1)).float()
        cache['mask_logits'] = cache['mask_logits'].float()+refined[:, :1]
        cache['depth'] = cache['depth'].float()+.2*refined[:, 1:2]
        return cache

    def query(self, cache, xyz, uv=None):
        if uv is None:
            uv = project_uv(xyz)
        source = super().query(cache, xyz, uv)
        depth = F.grid_sample(cache['depth'], uv[:, :, None], align_corners=False,
                               padding_mode='border').squeeze(-1).transpose(1, 2)
        mask = F.grid_sample(cache['mask_logits'], uv[:, :, None], align_corners=False,
                              padding_mode='border').squeeze(-1).transpose(1, 2).sigmoid()
        toward_depth = (xyz*self.toward_camera).sum(-1, keepdim=True)
        free_space = (toward_depth-depth-.01)/self.config.sdf_scale
        silhouette = 2*(.5-mask)
        source['raw_sdf'] = source['sdf']
        source['sdf'] = torch.maximum(torch.maximum(source['sdf'], free_space), silhouette)
        source['occupancy_logits'] = -source['sdf']/.025
        return source


class DepthSurfaceDetachedMLD2(DepthSurfaceMLD2):
    def query(self, cache, xyz, uv=None):
        if uv is None:
            uv = project_uv(xyz)
        source = MLD2Model.query(self, cache, xyz, uv)
        depth = F.grid_sample(cache['depth'].detach(), uv[:, :, None], align_corners=False,
                               padding_mode='border').squeeze(-1).transpose(1, 2)
        mask = F.grid_sample(cache['mask_logits'].detach(), uv[:, :, None], align_corners=False,
                              padding_mode='border').squeeze(-1).transpose(1, 2).sigmoid()
        toward_depth = (xyz*self.toward_camera).sum(-1, keepdim=True)
        free_space = (toward_depth-depth-.01)/self.config.sdf_scale
        silhouette = 2*(.5-mask)
        source['raw_sdf'] = source['sdf']
        source['sdf'] = torch.maximum(torch.maximum(source['sdf'], free_space), silhouette)
        source['occupancy_logits'] = -source['sdf']/.025
        return source
