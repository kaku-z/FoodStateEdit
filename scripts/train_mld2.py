"""Train MLD2 source fields and an optional residual prior; evaluate whole grids."""
from pathlib import Path
import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from foodstateedit.material_lineage.mld2 import MLD2Config, MLD2Model, project_uv, spatial_noise, sample_residual


def json_write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')
    temp.replace(path)


def checkpoint(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


class Data:
    def __init__(self, root):
        self.root = Path(root)
        keys = ['canonical_xyz', 'source_uv', 'source_rgb', 'joint', 'corner_sdf', 'cell_occupancy',
                'occupancy', 'source_mask', 'source_depth', 'surface_xyz', 'surface_rgb', 'surface_valid', 'splits']
        self.a = {k: np.load(self.root/(k+'.npy'), mmap_mode='r') for k in keys}
        if (self.root/'near_xyz.npy').exists():
            self.a['near_xyz'] = np.load(self.root/'near_xyz.npy', mmap_mode='r')
            self.a['near_joint'] = np.load(self.root/'near_joint.npy', mmap_mode='r')
            self.a['near_corner_sdf'] = np.load(self.root/'near_corner_sdf.npy', mmap_mode='r')
        self.groups = [np.flatnonzero(self.a['splits'] == s) for s in range(3)]
        self.records = [json.loads(x) for x in (self.root/'scenes.jsonl').read_text().splitlines()]
        self.tokens = len(self.a['canonical_xyz'])

    def batch(self, scenes, tokens, device, no_image=False, surface_tokens=None):
        scenes = np.asarray(scenes)
        tokens = np.asarray(tokens)
        image = np.asarray(self.a['source_rgb'][scenes], dtype=np.float32).copy().transpose(0, 3, 1, 2)/255
        if no_image:
            image[:] = .5
        values = {'image': image,
                  'xyz': np.broadcast_to(self.a['canonical_xyz'][tokens], (len(scenes), len(tokens), 3)).copy(),
                  'uv': np.broadcast_to(self.a['source_uv'][tokens], (len(scenes), len(tokens), 2)).copy()}
        for key in ['joint', 'corner_sdf', 'cell_occupancy', 'occupancy']:
            values[key] = np.asarray(self.a[key][scenes[:, None], tokens[None]], dtype=np.float32).copy()
        for key in ['source_mask', 'source_depth']:
            values[key] = np.asarray(self.a[key][scenes], dtype=np.float32).copy()[:, None]
        if surface_tokens is not None:
            for key in ['surface_xyz', 'surface_rgb', 'surface_valid']:
                values[key] = np.asarray(self.a[key][scenes[:, None], surface_tokens[None]], dtype=np.float32).copy()
            if 'near_xyz' in self.a:
                for key in ['near_xyz', 'near_joint', 'near_corner_sdf']:
                    values[key] = np.asarray(self.a[key][scenes[:, None], surface_tokens[None]], dtype=np.float32).copy()
        return {key: torch.from_numpy(np.ascontiguousarray(value)).to(device) for key, value in values.items()}


def weighted(value, weight):
    return (value*weight).sum()/weight.expand_as(value).sum().clamp_min(1)


def source_losses(model, cache, source, batch):
    true_sdf = batch['joint'][..., :1]
    sdf_weight = 1 + 4*torch.exp(-true_sdf.abs()*6)
    losses = {'sdf': weighted((source['sdf']-true_sdf).abs(), sdf_weight),
              'occupancy': F.binary_cross_entropy_with_logits(source['occupancy_logits'], batch['occupancy'][..., None]),
              'corner_sdf': weighted((source['corner_sdf']-batch['corner_sdf']).abs(),
                                     1+4*torch.exp(-batch['corner_sdf'].abs()*6)),
              'corner_occupancy': F.binary_cross_entropy_with_logits(source['corner_occupancy_logits'],
                                                                     (batch['corner_sdf'] <= 0).float()),
              'material': weighted((source['rgb']-batch['joint'][..., 1:]).square(), batch['occupancy'][..., None]),
              'mask': F.binary_cross_entropy_with_logits(cache['mask_logits'].float(), batch['source_mask']),
              'depth': weighted((cache['depth'].float()-batch['source_depth']).square(), batch['source_mask'])}
    surface = model.query(cache, batch['surface_xyz'])
    valid = batch['surface_valid'][..., None]
    losses['surface_sdf'] = weighted(surface['sdf'].abs(), valid)
    losses['surface_material'] = weighted((surface['rgb']-(2*batch['surface_rgb']-1)).square(), valid)
    # Couple the fast subcell head to the continuous field used by the renderer.
    corner_count = min(64, batch['xyz'].shape[1])
    corner_xyz = (batch['xyz'][:, :corner_count, None]+model.subcell_offsets).flatten(1, 2)
    continuous = model.query(cache, corner_xyz)['sdf'].reshape(-1, corner_count, 8)
    losses['corner_consistency'] = F.smooth_l1_loss(continuous, source['corner_sdf'][:, :corner_count])
    if 'near_xyz' in batch:
        near = model.query(cache, batch['near_xyz'])
        losses['near_sdf'] = weighted((near['sdf']-batch['near_joint'][..., :1]).abs(), valid)
        losses['near_occupancy'] = weighted(F.binary_cross_entropy_with_logits(
            near['occupancy_logits'], (batch['near_joint'][..., :1] <= 0).float(), reduction='none'), valid)
        losses['near_material'] = weighted((near['rgb']-batch['near_joint'][..., 1:]).square(),
                                            valid*(batch['near_joint'][..., :1] <= 0))
        losses['near_corner_sdf'] = weighted((near['corner_sdf']-batch['near_corner_sdf']).abs(), valid)
        losses['near_corner_occupancy'] = weighted(F.binary_cross_entropy_with_logits(
            near['corner_occupancy_logits'], (batch['near_corner_sdf'] <= 0).float(), reduction='none'), valid)
    coefficients = {'sdf': 2, 'occupancy': .3, 'corner_sdf': 1.5, 'corner_occupancy': .3,
                    'material': 1, 'mask': .35, 'depth': .2, 'surface_sdf': 3,
                    'surface_material': 1.5, 'near_sdf': 2, 'near_occupancy': .3, 'near_material': .5,
                    'near_corner_sdf': 1, 'near_corner_occupancy': .2}
    coefficients['corner_consistency'] = .4
    return sum(value*coefficients[key] for key, value in losses.items()), losses


def correlated_noise(xyz, channels=4, modes=32):
    batch = len(xyz)
    frequency = torch.randn(batch, 3, modes, device=xyz.device)*2
    sine = torch.randn(batch, modes, channels, device=xyz.device)
    cosine = torch.randn(batch, modes, channels, device=xyz.device)
    phase = torch.bmm(xyz, frequency)*math.pi
    return (torch.bmm(phase.sin(), sine)+torch.bmm(phase.cos(), cosine))/math.sqrt(modes)


def residual_loss(model, source, batch, alpha):
    posterior = torch.cat((source['sdf'], source['rgb']), -1).detach()
    residual = ((batch['joint']-posterior)/model.config.residual_scale).clamp(-3, 3)
    noise = correlated_noise(batch['xyz'])
    ts = torch.randint(0, len(alpha), (len(posterior),), device=posterior.device)
    at = alpha[ts, None, None]
    noisy = at.sqrt()*residual+(1-at).sqrt()*noise
    predicted = model.denoise_residual(source['features'].detach(), noisy, ts.float()/999)
    return (F.mse_loss(predicted[..., :1], noise[..., :1]) +
            weighted((predicted[..., 1:]-noise[..., 1:]).square(), batch['occupancy'][..., None]))


def make_actions(scene_ids, mode='id'):
    if mode == 'combined_ood':
        return np.concatenate((make_actions(scene_ids, 'normal_ood')[:, :4],
                               make_actions(scene_ids, 'height_ood')[:, 4:7],
                               make_actions(scene_ids, 'rotation_ood')[:, 7:10]), -1)
    actions = []
    for sid in scene_ids:
        key = hashlib.sha256(('mld2-actions:20261004:'+str(sid)).encode()).digest()
        rng = np.random.default_rng(int.from_bytes(key[:8], 'little'))
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        while n[0] > .65:
            n = rng.normal(size=3)
            n /= np.linalg.norm(n)
        offset = rng.uniform(-.35, .35)
        translation = rng.uniform(-.12, .12, 3)
        translation[2] = rng.uniform(.25, .5)
        aa = rng.normal(size=3)
        aa = aa/np.linalg.norm(aa)*rng.uniform(0, .45)
        if mode == 'normal_ood':
            n = np.array([rng.uniform(.65, .98), 0., 0.])
            phase = rng.uniform(-math.pi, math.pi)
            radius = math.sqrt(1-n[0]**2)
            n[1:] = [radius*math.cos(phase), radius*math.sin(phase)]
        elif mode == 'height_ood':
            translation[2] = rng.uniform(.8, 1.2)
        elif mode == 'rotation_ood':
            aa = aa/np.linalg.norm(aa)*rng.uniform(1.0, 1.6)
        elif mode == 'null':
            n[:] = 0
            offset = 0
            translation[:] = 0
            aa[:] = 0
        actions.append(np.concatenate((n, [offset], translation, aa)))
    return np.asarray(actions, dtype=np.float32)


def add_field_statistics(stats, source, batch):
    predicted = source['sdf'][..., 0] <= 0
    truth = batch['occupancy'] > .5
    stats['intersection'] += float((predicted & truth).sum())
    stats['union'] += float((predicted | truth).sum())
    stats['sdf_abs'] += float((source['sdf']-batch['joint'][..., :1]).abs().sum())
    stats['corner_abs'] += float((source['corner_sdf']-batch['corner_sdf']).abs().sum())
    stats['color_sq'] += float((((source['rgb']-batch['joint'][..., 1:])/2).square()*truth[..., None]).sum())
    stats['count'] += truth.numel()
    stats['occupied'] += float(truth.sum())


def empty_stats():
    return dict.fromkeys(['intersection', 'union', 'sdf_abs', 'corner_abs', 'color_sq', 'count', 'occupied'], 0.)


def field_statistics(stats):
    return {'occupancy_iou': stats['intersection']/max(stats['union'], 1),
            'sdf_mae': stats['sdf_abs']/max(stats['count'], 1),
            'corner_sdf_mae': stats['corner_abs']/max(stats['count']*8, 1),
            'occupied_linear_rgb_mse': stats['color_sq']/max(stats['occupied']*3, 1)}


@torch.inference_mode()
def validation(model, data, device, no_image=False):
    model.eval()
    batch = data.batch(data.groups[1][:32], np.arange(data.tokens), device, no_image,
                       np.arange(512))
    cache = model.encode_image(batch['image'])
    source = model.query(cache, batch['xyz'], batch['uv'])
    stats = empty_stats()
    add_field_statistics(stats, source, batch)
    values = field_statistics(stats)
    surface = model.query(cache, batch['surface_xyz'])
    values['visible_linear_rgb_mse'] = float(weighted(((surface['rgb']-(2*batch['surface_rgb']-1))/2).square(),
                                                      batch['surface_valid'][..., None]))
    values['visible_sdf_mae'] = float(weighted(surface['sdf'].abs(), batch['surface_valid'][..., None]))
    model.train()
    return values


@torch.inference_mode()
def evaluate(model, data, root, device, seed, no_image=False, residual=True):
    model.eval()
    test = data.groups[2]
    tokens = np.arange(data.tokens)
    count = len(test)
    arrays = {'scene_indices': test, 'xyz': np.asarray(data.a['canonical_xyz']).copy(),
              'sdf': np.empty((count, data.tokens, 1), dtype=np.float16),
              'rgb': np.empty((count, data.tokens, 3), dtype=np.float16),
              'corner_sdf': np.empty((count, data.tokens, 8), dtype=np.float16),
              'source_mass_soft': np.empty((count, data.tokens, 1), dtype=np.float16),
              'source_mass_hard': np.empty((count, data.tokens, 1), dtype=np.float16)}
    modes = ['id', 'normal_ood', 'height_ood', 'rotation_ood', 'combined_ood', 'null']
    per_scene = {key: np.zeros(count, dtype=np.float64) for key in
                 ['source_iou', 'source_soft_mass_l1_normalized', 'source_hard_mass_l1_normalized',
                  'source_prior_mass_l1_normalized', 'sdf_mae', 'occupied_linear_rgb_mse']}
    for mode in modes:
        for method in ['soft', 'hard', 'prior', 'shuffle', 'all_solid', 'zero']:
            per_scene[f'{mode}_{method}_carried_l1_normalized'] = np.zeros(count, dtype=np.float64)
        arrays[f'actions_{mode}'] = np.empty((count, 10), dtype=np.float32)
    stats = empty_stats()
    prior = np.zeros((data.tokens, 8), dtype=np.float64)
    for start in range(0, len(data.groups[0]), 128):
        prior += np.asarray(data.a['corner_sdf'][data.groups[0][start:start+128]] <= 0).sum(0)
    prior = torch.from_numpy((prior/len(data.groups[0])).astype(np.float32)).to(device)
    masses = {mode: dict.fromkeys(['soft_error', 'hard_error', 'prior_error', 'shuffle_error',
                                'target_mass', 'predicted_soft_mass', 'predicted_hard_mass', 'n',
                                'soft_total_error', 'hard_total_error', 'prior_total_error', 'scenes'], 0.) for mode in modes}
    visible_error, visible_count, visible_sdf = 0., 0., 0.
    for start in range(0, count, 16):
        scene = test[start:start+16]
        batch = data.batch(scene, tokens, device, no_image, np.arange(512))
        cache = model.encode_image(batch['image'])
        source = model.query_state(cache, batch['xyz'], batch['uv'])
        add_field_statistics(stats, source, batch)
        surface = model.query(cache, batch['surface_xyz'])
        valid = batch['surface_valid'][..., None]
        visible_error += float((((surface['rgb']-(2*batch['surface_rgb']-1))/2).square()*valid).sum())
        visible_sdf += float((surface['sdf'].abs()*valid).sum())
        visible_count += float(valid.sum())
        for key in ['sdf', 'rgb', 'corner_sdf']:
            arrays[key][start:start+len(scene)] = source[key].cpu().numpy()
        occupancy = (batch['corner_sdf'] <= 0).float()
        true_mass = occupancy.mean(-1, keepdim=True)
        true_volume = true_mass.sum((1, 2)).clamp_min(1)
        predicted_mass = source['corner_occupancy_logits'].sigmoid().mean(-1, keepdim=True)
        hard_mass = (source['corner_sdf'] <= 0).float().mean(-1, keepdim=True)
        pp, tt = source['sdf'][..., 0] <= 0, batch['occupancy'] > .5
        sl = slice(start, start+len(scene))
        per_scene['source_iou'][sl] = ((pp & tt).sum(1)/(pp | tt).sum(1).clamp_min(1)).cpu().numpy()
        per_scene['sdf_mae'][sl] = (source['sdf']-batch['joint'][..., :1]).abs().mean((1, 2)).cpu().numpy()
        per_scene['occupied_linear_rgb_mse'][sl] = ((((source['rgb']-batch['joint'][..., 1:])/2).square()*tt[..., None]).sum((1, 2))/
                                                   (tt.sum(1)*3).clamp_min(1)).cpu().numpy()
        for method, pm in [('soft', predicted_mass), ('hard', hard_mass), ('prior', prior.mean(-1)[None, :, None])]:
            per_scene[f'source_{method}_mass_l1_normalized'][sl] = ((pm-true_mass).abs().sum((1, 2))/true_volume).cpu().numpy()
        ids = [data.records[int(index)]['scene_id'] for index in scene]
        for mode in modes:
            action = torch.from_numpy(make_actions(ids, mode)).to(device)
            arrays[f'actions_{mode}'][sl] = action.cpu().numpy()
            predicted = model.allocation(source, batch['xyz'], action, soft=True)
            hard = model.allocation(source, batch['xyz'], action, soft=False)
            points = batch['xyz'][..., None, :]+model.subcell_offsets
            cut = (points*action[:, None, None, :3]).sum(-1) > action[:, None, None, 3]
            cut = cut & (action[:, :4].abs().sum(-1) > 0)[:, None, None]
            target = (occupancy*cut).mean(-1, keepdim=True)
            prior_mass = (prior[None]*cut).mean(-1, keepdim=True)
            shuffled_occ = source['corner_occupancy_logits'].sigmoid().roll(1, 0)
            shuffle_mass = (shuffled_occ*cut).mean(-1, keepdim=True)
            all_solid = cut.float().mean(-1, keepdim=True)
            for method, pm in [('soft', predicted['carried_mass']), ('hard', hard['carried_mass']),
                               ('prior', prior_mass), ('shuffle', shuffle_mass), ('all_solid', all_solid),
                               ('zero', torch.zeros_like(target))]:
                per_scene[f'{mode}_{method}_carried_l1_normalized'][sl] = ((pm-target).abs().sum((1, 2))/true_volume).cpu().numpy()
            s = masses[mode]
            for key, value in [('soft_error', predicted['carried_mass']), ('hard_error', hard['carried_mass']),
                               ('prior_error', prior_mass), ('shuffle_error', shuffle_mass)]:
                s[key] += float((value-target).abs().sum())
            for key, value in [('soft_total_error', predicted['carried_mass']),
                               ('hard_total_error', hard['carried_mass']), ('prior_total_error', prior_mass)]:
                s[key] += float((value.sum(1)-target.sum(1)).abs().sum())
            s['target_mass'] += float(target.sum())
            s['predicted_soft_mass'] += float(predicted['carried_mass'].sum())
            s['predicted_hard_mass'] += float(hard['carried_mass'].sum())
            s['n'] += target.numel()
            s['scenes'] += len(scene)
            if mode == 'id':
                arrays['source_mass_soft'][start:start+len(scene)] = predicted['source_mass'].cpu().numpy()
                arrays['source_mass_hard'][start:start+len(scene)] = hard['source_mass'].cpu().numpy()
    values = {mode: {'carried_soft_mae': s['soft_error']/s['n'],
                     'carried_hard_mae': s['hard_error']/s['n'],
                     'state_independent_prior_mae': s['prior_error']/s['n'],
                     'shuffled_state_mae': s['shuffle_error']/s['n'],
                     'carried_soft_total_scene_mae': s['soft_total_error']/s['scenes'],
                     'carried_hard_total_scene_mae': s['hard_total_error']/s['scenes'],
                     'prior_total_scene_mae': s['prior_total_error']/s['scenes'],
                     'total_target_cells': s['target_mass'],
                     'total_predicted_soft_cells': s['predicted_soft_mass'],
                     'total_predicted_hard_cells': s['predicted_hard_mass']} for mode, s in masses.items()}
    result = {'scope': 'Synthetic geometry/material supervision; real unpaired images require separate validation.',
              'test_scene_count': count, 'queries_per_scene': data.tokens, 'seed': seed,
              'source': field_statistics(stats), 'allocation': values,
              'visible_linear_rgb_mse': visible_error/max(visible_count*3, 1),
              'visible_sdf_mae': visible_sdf/max(visible_count, 1),
              'allocation_field': 'The same continuous SDF decoder queried at actual eight sub-sites; auxiliary fast corner head is not used for final mass.',
              'rigid_motion': 'Analytic supplied SE3; not a learned motion generalization claim.'}
    families = np.asarray([data.records[int(index)]['shape_family'] for index in test])
    result['macro_per_scene'] = {key: float(value.mean()) for key, value in per_scene.items()}
    result['per_family'] = {family: {'scene_count': int((families == family).sum()),
                                    **{key: float(value[families == family].mean()) for key, value in per_scene.items()}}
                            for family in sorted(set(families))}
    result['normalized_mass_metric'] = 'Sum absolute per-cell material mass error / GT source volume, then mean across scenes; same cell volume cancels.'
    arrays.update(per_scene)
    arrays['shape_families'] = families
    np.savez_compressed(root/'evaluation_predictions.npz', **arrays)
    if residual:
        subset = test[:32]
        batch = data.batch(subset, tokens, device, no_image)
        source = model.encode_source(batch['image'], batch['xyz'], batch['uv'])
        ids = [data.records[int(index)]['scene_id'] for index in subset]
        noise = torch.from_numpy(spatial_noise(np.asarray(data.a['canonical_xyz']), ids)).to(device)
        sampled = sample_residual(model, source, noise, 50)
        residual_stats = empty_stats()
        proxy = {'sdf': sampled[..., :1], 'rgb': sampled[..., 1:], 'corner_sdf': source['corner_sdf']}
        add_field_statistics(residual_stats, proxy, batch)
        result['residual_prior_32_scene_diagnostic'] = field_statistics(residual_stats)
        result['residual_prior_32_scene_diagnostic']['corner_sdf_mae_scope'] = 'Unchanged posterior corner head; residual only sampled SDF/RGB.'
        np.savez_compressed(root/'residual_predictions.npz', scene_indices=subset,
                            sampled_joint=sampled.cpu().numpy().astype(np.float16), noise=noise.cpu().numpy().astype(np.float16))
    json_write(root/'evaluation.json', result)
    return result


def main(args):
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    if (root/'final_checkpoint.pt').exists():
        raise RuntimeError('Output already has a final checkpoint; use another output directory.')
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = Data(args.dataset_root)
    config = MLD2Config()
    model = MLD2Model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    rng = np.random.default_rng(args.seed)
    cfg = dataclasses.asdict(config)
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    checkpoint(root/'initial_checkpoint.pt', {'model_config': cfg, 'state_dict': initial, 'optimizer_steps': 0})
    alpha = torch.cumprod(1-torch.linspace(.0001, .02, 1000, device=device), 0)
    receipt = {'status': 'training', 'pid': os.getpid(), 'seed': args.seed,
               'parameter_count': sum(p.numel() for p in model.parameters()), 'planned_steps': args.steps,
               'actual_steps': 0, 'no_image': args.no_image, 'residual_start': args.residual_start,
               'no_image_input': 'constant RGB0.5, containing no scene information',
               'model_sha256': sha(Path(__file__).resolve().parents[1]/'foodstateedit/material_lineage/mld2.py'),
               'trainer_sha256': sha(__file__), 'dataset_manifest_sha256': sha(data.root/'manifest.json')}
    json_write(root/'training_receipt.json', receipt)
    print(json.dumps(receipt), flush=True)
    started = time.monotonic()
    with (root/'metrics.jsonl').open('w', buffering=1) as log:
        for step in range(1, args.steps+1):
            scenes = rng.choice(data.groups[0], args.batch_size, replace=False)
            tokens = rng.choice(data.tokens, args.tokens, replace=False)
            surface_tokens = rng.choice(data.a['surface_xyz'].shape[1], args.surface_tokens, replace=False)
            batch = data.batch(scenes, tokens, device, args.no_image, surface_tokens)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                cache = model.encode_image(batch['image'])
                source = model.query(cache, batch['xyz'], batch['uv'])
                total, losses = source_losses(model, cache, source, batch)
                if step > args.residual_start:
                    losses['residual'] = residual_loss(model, source, batch, alpha)
                    total = total+losses['residual']*.5
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            schedule = min(step/200, 1)*(.2+.8*(1+math.cos(math.pi*step/args.steps))/2)
            for group in optimizer.param_groups:
                group['lr'] = args.lr*schedule
            optimizer.step()
            if step % 50 == 0 or step == 1:
                row = {'step': step, 'elapsed_seconds': time.monotonic()-started,
                       'total_loss': float(total.detach()), 'gradient_norm': float(norm),
                       **{k: float(v.detach()) for k, v in losses.items()}}
                log.write(json.dumps(row)+'\n')
            if step % args.checkpoint_every == 0 or step == args.steps:
                val = validation(model, data, device, args.no_image)
                print(json.dumps({'step': step, 'seconds': time.monotonic()-started, 'validation': val}), flush=True)
                ck = {'model_config': cfg, 'state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(), 'optimizer_steps': step,
                      'seed': args.seed, 'no_image': args.no_image, 'validation': val}
                checkpoint(root/'latest_checkpoint.pt', ck)
                if step == args.residual_start:
                    checkpoint(root/'posterior_checkpoint.pt', ck)
                receipt.update(actual_steps=step, elapsed_seconds=time.monotonic()-started, validation=val)
                json_write(root/'training_receipt.json', receipt)
    checkpoint(root/'final_checkpoint.pt', ck)
    receipt['status'] = 'evaluating'
    json_write(root/'training_receipt.json', receipt)
    result = evaluate(model, data, root, device, args.seed, args.no_image, args.steps > args.residual_start)
    receipt.update(status='complete', elapsed_seconds=time.monotonic()-started,
                   final_checkpoint_sha256=sha(root/'final_checkpoint.pt'), evaluation=result)
    json_write(root/'training_receipt.json', receipt)
    print(json.dumps({'status': 'complete', 'seconds': receipt['elapsed_seconds'], 'evaluation': result}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--seed', type=int, default=41)
    p.add_argument('--steps', type=int, default=50000)
    p.add_argument('--residual-start', type=int, default=40000)
    p.add_argument('--batch-size', type=int, default=24)
    p.add_argument('--tokens', type=int, default=512)
    p.add_argument('--surface-tokens', type=int, default=256)
    p.add_argument('--checkpoint-every', type=int, default=1000)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--no-image', action='store_true')
    main(p.parse_args())
