"""Detach geometry constraints from source-observation prediction targets."""
import argparse
import torch
from torch.nn import functional as F
import train_mld2 as base
import train_mld2_depth_surface as surface
from foodstateedit.material_lineage.mld2_depth import DepthSurfaceDetachedMLD2


def main():
    original_losses = surface.continuous.losses

    def supervised_observations(model, cache, source, batch):
        total, parts = original_losses(model, cache, source, batch)
        parts['observed_mask'] = F.binary_cross_entropy_with_logits(cache['mask_logits'].float(), batch['source_mask'])
        parts['observed_depth'] = base.weighted((cache['depth'].float()-batch['source_depth']).square(), batch['source_mask'])
        return total+1.65*parts['observed_mask']+3.8*parts['observed_depth'], parts

    surface.continuous.losses = supervised_observations
    surface.DepthSurfaceMLD2 = DepthSurfaceDetachedMLD2
    original_save = base.checkpoint

    def save(path, value):
        value['observation_maps_detached'] = True
        value['observation_losses'] = 'mask BCE weight2.0, depth MSE weight4.0; geometric free-space gradients cannot move maps.'
        original_save(path, value)

    base.checkpoint = save
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--supplement-root', required=True)
    p.add_argument('--surface-supplement', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--resume', required=True)
    p.add_argument('--seed', type=int, default=41)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--residual-start', type=int, default=1000000)
    p.add_argument('--batch-size', type=int, default=24)
    p.add_argument('--tokens', type=int, default=512)
    p.add_argument('--surface-tokens', type=int, default=256)
    p.add_argument('--checkpoint-every', type=int, default=1000)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--no-image', action='store_true')
    surface.main(p.parse_args())


if __name__ == '__main__':
    main()
