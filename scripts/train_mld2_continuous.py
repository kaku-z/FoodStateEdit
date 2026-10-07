"""Continue MLD2 from a real checkpoint with global continuous-query supervision."""
from pathlib import Path
import argparse
import dataclasses
import json
import math
import os
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
import train_mld2 as base
from foodstateedit.material_lineage.mld2 import MLD2Config, MLD2Model


class ContinuousData(base.Data):
    def __init__(self, root, supplement, seed):
        super().__init__(root)
        self.uniform_xyz = np.load(Path(supplement)/'uniform_xyz.npy', mmap_mode='r')
        self.uniform_joint = np.load(Path(supplement)/'uniform_joint.npy', mmap_mode='r')
        self.rng = np.random.default_rng(seed+9971)

    def batch(self, scenes, tokens, device, no_image=False, surface_tokens=None):
        batch = super().batch(scenes, tokens, device, no_image, surface_tokens)
        if surface_tokens is not None:
            query = self.rng.choice(self.uniform_xyz.shape[1], 256, replace=False)
            scenes = np.asarray(scenes)
            for name, array in [('uniform_xyz', self.uniform_xyz), ('uniform_joint', self.uniform_joint)]:
                value = np.asarray(array[scenes[:, None], query[None]], dtype=np.float32).copy()
                batch[name] = torch.from_numpy(value).to(device)
        return batch


def losses(model, cache, source, batch):
    total, parts = base.source_losses(model, cache, source, batch)
    uniform = model.query(cache, batch['uniform_xyz'])
    target = batch['uniform_joint']
    occupied = (target[..., :1] <= 0).float()
    parts['uniform_sdf'] = (uniform['sdf']-target[..., :1]).abs().mean()
    parts['uniform_occupancy'] = F.binary_cross_entropy_with_logits(uniform['occupancy_logits'], occupied)
    parts['uniform_material'] = base.weighted((uniform['rgb']-target[..., 1:]).square(), occupied)
    total = total+3*parts['uniform_sdf']+.5*parts['uniform_occupancy']+.7*parts['uniform_material']
    return total, parts


@torch.inference_mode()
def uniform_metrics(model, data, device, split=1, no_image=False, limit=None, output=None):
    model.eval()
    scenes = data.groups[split]
    if limit is not None:
        scenes = scenes[:limit]
    statistics = base.empty_stats()
    per_scene = []
    for start in range(0, len(scenes), 16):
        selected = scenes[start:start+16]
        image = np.asarray(data.a['source_rgb'][selected], dtype=np.float32).copy().transpose(0, 3, 1, 2)/255
        if no_image:
            image[:] = .5
        image = torch.from_numpy(image).to(device)
        xyz = torch.from_numpy(np.asarray(data.uniform_xyz[selected], dtype=np.float32).copy()).to(device)
        joint = torch.from_numpy(np.asarray(data.uniform_joint[selected], dtype=np.float32).copy()).to(device)
        source = model.query(model.encode_image(image), xyz)
        truth = joint[..., 0] <= 0
        predicted = source['sdf'][..., 0] <= 0
        intersection, union = (truth & predicted).sum(1), (truth | predicted).sum(1)
        statistics['intersection'] += float(intersection.sum())
        statistics['union'] += float(union.sum())
        statistics['sdf_abs'] += float((source['sdf']-joint[..., :1]).abs().sum())
        statistics['color_sq'] += float((((source['rgb']-joint[..., 1:])/2).square()*truth[..., None]).sum())
        statistics['count'] += truth.numel()
        statistics['occupied'] += float(truth.sum())
        sdf_error = (source['sdf']-joint[..., :1]).abs().mean((1, 2))
        for index, scene in enumerate(selected):
            per_scene.append({'scene_index': int(scene), 'shape_family': data.records[int(scene)]['shape_family'],
                              'sdf_mae': float(sdf_error[index]),
                              'intersection': int(intersection[index]), 'union': int(union[index]),
                              'true_occupied_queries': int(truth[index].sum()),
                              'predicted_occupied_queries': int(predicted[index].sum())})
    values = {'scene_count': len(scenes), 'queries_per_scene': data.uniform_xyz.shape[1],
              'occupancy_iou': statistics['intersection']/max(statistics['union'], 1),
              'sdf_mae': statistics['sdf_abs']/statistics['count'],
              'occupied_linear_rgb_mse': statistics['color_sq']/max(statistics['occupied']*3, 1),
              'empty_gt_query_scene_count': sum(row['true_occupied_queries'] == 0 for row in per_scene),
              'per_family': {family: {'scene_count': sum(row['shape_family'] == family for row in per_scene),
                                     'sdf_mae': float(np.mean([row['sdf_mae'] for row in per_scene if row['shape_family'] == family])),
                                     'occupancy_iou': sum(row['intersection'] for row in per_scene if row['shape_family'] == family)/
                                                      max(sum(row['union'] for row in per_scene if row['shape_family'] == family), 1)}
                             for family in sorted({row['shape_family'] for row in per_scene})},
              'scope': 'Independent continuous canonical queries; no lattice interpolation or target occupancy input.'}
    if output:
        base.json_write(output, {'metrics': values, 'per_scene': per_scene})
    model.train()
    return values


def main(args):
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda')
    data = ContinuousData(args.dataset_root, args.supplement_root, args.seed)
    saved = torch.load(args.resume, map_location='cpu', weights_only=True)
    start_step = saved['optimizer_steps']
    model = MLD2Model(MLD2Config(**saved['model_config'])).to(device)
    model.load_state_dict(saved['state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    optimizer.load_state_dict(saved['optimizer_state_dict'])
    base.checkpoint(root/'resume_checkpoint.pt', saved)
    rng = np.random.default_rng(args.seed+4993)
    alpha = torch.cumprod(1-torch.linspace(.0001, .02, 1000, device=device), 0)
    receipt = {'status': 'training', 'pid': os.getpid(), 'seed': args.seed, 'no_image': args.no_image,
               'resume_checkpoint': str(args.resume), 'resume_optimizer_steps': start_step,
               'actual_steps': start_step, 'actual_additional_updates': 0, 'planned_steps': args.steps,
               'residual_start': args.residual_start, 'uniform_query_count_per_update': 256,
               'parameter_count': sum(p.numel() for p in model.parameters()),
               'model_sha256': base.sha(Path(__file__).resolve().parents[1]/'foodstateedit/material_lineage/mld2.py'),
               'trainer_sha256': base.sha(__file__), 'base_trainer_sha256': base.sha(base.__file__),
               'dataset_manifest_sha256': base.sha(data.root/'manifest.json'),
               'supplement_manifest_sha256': base.sha(Path(args.supplement_root)/'manifest.json')}
    base.json_write(root/'training_receipt.json', receipt)
    print(json.dumps(receipt), flush=True)
    started = time.monotonic()
    with (root/'metrics.jsonl').open('w', buffering=1) as log:
        for step in range(start_step+1, args.steps+1):
            scenes = rng.choice(data.groups[0], args.batch_size, replace=False)
            tokens = rng.choice(data.tokens, args.tokens, replace=False)
            surface_tokens = rng.choice(data.a['surface_xyz'].shape[1], args.surface_tokens, replace=False)
            batch = data.batch(scenes, tokens, device, args.no_image, surface_tokens)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                cache = model.encode_image(batch['image'])
                source = model.query(cache, batch['xyz'], batch['uv'])
                total, parts = losses(model, cache, source, batch)
                if step > args.residual_start:
                    parts['residual'] = base.residual_loss(model, source, batch, alpha)
                    total = total+.5*parts['residual']
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            schedule = .2+.8*(1+math.cos(math.pi*step/args.steps))/2
            for group in optimizer.param_groups:
                group['lr'] = args.lr*schedule
            optimizer.step()
            if step % 50 == 0 or step == start_step+1:
                log.write(json.dumps({'step': step, 'additional_updates': step-start_step,
                                      'elapsed_seconds': time.monotonic()-started, 'total_loss': float(total.detach()),
                                      'gradient_norm': float(norm), **{k: float(v.detach()) for k, v in parts.items()}})+'\n')
            if step % args.checkpoint_every == 0 or step == args.steps:
                val = base.validation(model, data, device, args.no_image)
                val['uniform'] = uniform_metrics(model, data, device, 1, args.no_image, 32)
                ck = {'model_config': saved['model_config'], 'state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(), 'optimizer_steps': step,
                      'resume_optimizer_steps': start_step, 'additional_updates': step-start_step,
                      'seed': args.seed, 'no_image': args.no_image, 'validation': val}
                base.checkpoint(root/'latest_checkpoint.pt', ck)
                if step == args.residual_start:
                    base.checkpoint(root/'posterior_checkpoint.pt', ck)
                receipt.update(actual_steps=step, actual_additional_updates=step-start_step,
                               elapsed_seconds=time.monotonic()-started, validation=val)
                base.json_write(root/'training_receipt.json', receipt)
                print(json.dumps({'step': step, 'seconds': receipt['elapsed_seconds'], 'validation': val}), flush=True)
    base.checkpoint(root/'final_checkpoint.pt', ck)
    receipt['status'] = 'evaluating'
    base.json_write(root/'training_receipt.json', receipt)
    evaluation = base.evaluate(model, data, root, device, args.seed, args.no_image, args.steps > args.residual_start)
    evaluation['uniform_test'] = uniform_metrics(model, data, device, 2, args.no_image,
                                                output=root/'uniform_evaluation.json')
    base.json_write(root/'evaluation.json', evaluation)
    receipt.update(status='complete', elapsed_seconds=time.monotonic()-started,
                   final_checkpoint_sha256=base.sha(root/'final_checkpoint.pt'), evaluation=evaluation)
    base.json_write(root/'training_receipt.json', receipt)
    print(json.dumps({'status': 'complete', 'step': args.steps, 'additional_updates': args.steps-start_step,
                      'seconds': receipt['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--supplement-root', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--resume', required=True)
    p.add_argument('--seed', type=int, default=41)
    p.add_argument('--steps', type=int, default=65000)
    p.add_argument('--residual-start', type=int, default=55000)
    p.add_argument('--batch-size', type=int, default=24)
    p.add_argument('--tokens', type=int, default=512)
    p.add_argument('--surface-tokens', type=int, default=256)
    p.add_argument('--checkpoint-every', type=int, default=1000)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--no-image', action='store_true')
    main(p.parse_args())
