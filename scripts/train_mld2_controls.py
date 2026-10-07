"""Matched warm-start controls for spectral alias and sparse hidden positives."""
from pathlib import Path
import argparse
import json

import numpy as np
import torch
import train_mld2 as base
import train_mld2_continuous as continuous
from mld2_lowpass import attach_lowpass


class BalancedData(continuous.ContinuousData):
    def batch(self, scenes, tokens, device, no_image=False, surface_tokens=None):
        batch = base.Data.batch(self, scenes, tokens, device, no_image, surface_tokens)
        if surface_tokens is not None:
            coordinates, targets = [], []
            for scene in scenes:
                joint = np.asarray(self.uniform_joint[scene], dtype=np.float32)
                sdf = joint[:, 0]
                general = self.rng.choice(len(sdf), 128, replace=False)
                near_pool = np.argsort(np.abs(sdf))[:128]
                near = self.rng.choice(near_pool, 64, replace=False)
                occupied = np.flatnonzero(sdf <= 0)
                positive_pool = occupied if len(occupied) else np.argsort(sdf)[:32]
                positive = self.rng.choice(positive_pool, 64, replace=True)
                selected = np.concatenate((general, near, positive))
                coordinates.append(np.asarray(self.uniform_xyz[scene, selected], dtype=np.float32))
                targets.append(joint[selected])
            batch['uniform_xyz'] = torch.from_numpy(np.stack(coordinates)).to(device)
            batch['uniform_joint'] = torch.from_numpy(np.stack(targets)).to(device)
        return batch


def main(args):
    recipe = {'control': args.control, 'parent_checkpoint': args.resume,
              'sampling': ('128 uniform + 64 full-object near-SDF + 64 occupied queries; targets select training queries only'
                           if args.control == 'balanced_full' else '256 uniform queries, as main continuous revision'),
              'geometry_frequency_max': 4 if args.control == 'lowpass4' else 32,
              'evaluation': 'Full independent 1024 uniform queries and full4096 grid, unchanged across controls.'}
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    (output/'control_recipe.json').write_text(json.dumps(recipe, indent=2))
    original_checkpoint = base.checkpoint

    def save(path, value):
        value['control'] = args.control
        value['uniform_sampling'] = recipe['sampling']
        if args.control == 'lowpass4':
            value['coordinate_lowpass_max_frequency'] = 4
        original_checkpoint(path, value)

    base.checkpoint = save
    if args.control == 'lowpass4':
        original_model = continuous.MLD2Model

        def model(config):
            return attach_lowpass(original_model(config), 4)

        continuous.MLD2Model = model
    else:
        continuous.ContinuousData = BalancedData
    continuous.main(args)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--control', choices=['lowpass4', 'balanced_full'], required=True)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--supplement-root', required=True)
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
