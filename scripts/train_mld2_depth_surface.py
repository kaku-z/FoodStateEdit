"""Observed front geometry plus full-object surface supervision, source-only."""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from torch.nn import functional as F
import train_mld2 as base
import train_mld2_continuous as continuous
from foodstateedit.material_lineage.mld2_depth import DepthSurfaceMLD2


class SurfaceData(continuous.ContinuousData):
    supplement_root = None

    def __init__(self, root, supplement, seed):
        super().__init__(root, supplement, seed)
        self.full_xyz = np.load(Path(self.supplement_root)/'all_surface_near_xyz.npy', mmap_mode='r')
        self.full_joint = np.load(Path(self.supplement_root)/'all_surface_near_joint.npy', mmap_mode='r')

    def batch(self, scenes, tokens, device, no_image=False, surface_tokens=None):
        batch = super().batch(scenes, tokens, device, no_image, surface_tokens)
        if surface_tokens is not None:
            query = self.rng.choice(self.full_xyz.shape[1], 256, replace=False)
            scene = np.asarray(scenes)
            for key, array in [('full_xyz', self.full_xyz), ('full_joint', self.full_joint)]:
                batch[key] = torch.from_numpy(np.asarray(array[scene[:, None], query[None]], dtype=np.float32).copy()).to(device)
        return batch


def main(args):
    original_losses = continuous.losses

    def losses(model, cache, source, batch):
        total, parts = original_losses(model, cache, source, batch)
        field = model.query(cache, batch['full_xyz'])
        target = batch['full_joint']
        occupied = (target[..., :1] <= 0).float()
        parts['full_surface_sdf'] = (field['sdf']-target[..., :1]).abs().mean()
        parts['full_surface_occupancy'] = F.binary_cross_entropy_with_logits(field['occupancy_logits'], occupied)
        parts['full_surface_material'] = base.weighted((field['rgb']-target[..., 1:]).square(), occupied)
        return total+4*parts['full_surface_sdf']+parts['full_surface_occupancy']+.5*parts['full_surface_material'], parts

    original_save = base.checkpoint

    def checkpoint(path, value):
        value['model_variant'] = 'depth_surface'
        value['observation_constraints'] = 'Predicted source mask/depth only, continuous max with raw hidden SDF; no target maps in inference.'
        original_save(path, value)

    SurfaceData.supplement_root = args.surface_supplement
    continuous.ContinuousData = SurfaceData
    continuous.MLD2Model = DepthSurfaceMLD2
    continuous.losses = losses
    base.checkpoint = checkpoint
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    (root/'variant_recipe.json').write_text(json.dumps({'model_variant': 'depth_surface',
        'source_constraints': 'max(raw_sdf, (query_toward_depth - predicted_front_depth -.01)/.5, 2*(.5 - predicted_source_mask))',
        'full_object_surface_supervision': args.surface_supplement,
        'surface_supervision_per_update': 256, 'labels_are_targets_only': True}, indent=2))
    continuous.main(args)


if __name__ == '__main__':
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
    main(p.parse_args())
