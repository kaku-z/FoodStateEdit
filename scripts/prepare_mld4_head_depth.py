"""Separate head-depth protocol with full foreground range and explicit validity.

The original dense guide files are copied unchanged. New depth control files
use the transported food raster and visible spoon geometry, never photograph
depth accidentally left behind at uncovered target-food pixels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mask(path):
    return np.asarray(Image.open(path).convert('L')) > 127


def normalize_depth(depth, valid, foreground):
    """Near=bright; one monotonic depth function for every valid surface.

    Foreground min/max map to245/70. Strictly farther surfaces map to60..0;
    strictly nearer surfaces map to255..245. Scene surfaces whose physical
    depth overlaps foreground keep the same mapping as foreground.
    """
    valid = np.asarray(valid, bool) & np.isfinite(depth) & (depth > 0)
    foreground = np.asarray(foreground, bool) & valid
    if not foreground.any():
        raise ValueError('No finite head food or spoon depth')
    near, far = float(depth[foreground].min()), float(depth[foreground].max())
    scene_near, scene_far = float(depth[valid].min()), float(depth[valid].max())
    control = np.full(depth.shape, 127., dtype=np.float32)
    middle = valid & (depth >= near) & (depth <= far)
    front = valid & (depth < near)
    back = valid & (depth > far)
    if far > near:
        control[middle] = 245 - 175 * (depth[middle] - near) / (far - near)
    else:
        control[middle] = 157.5
    if front.any():
        control[front] = 245 + 10 * (near - depth[front]) / (near - scene_near)
    if back.any():
        control[back] = 60 * (scene_far - depth[back]) / (scene_far - far)
    record = dict(foreground_near=near, foreground_far=far, scene_near=scene_near, scene_far=scene_far,
                  foreground_intensity_range=[70, 245], nearer_intensity_range=[245, 255],
                  farther_intensity_range=[0, 60], invalid_intensity=127,
                  degenerate_foreground_range=far == near,
                  normalization='Piecewise monotonic camera-z mapping, near bright. The same depth maps to the same intensity regardless of food/spoon/background label. No foreground quantile clipping.',
                  caveat='Scene pixels at foreground depths cannot all be forced dark while preserving physical depth ordering. Such pixels keep the foreground-range mapping. A 10-level jump separates strictly farther surfaces; uint8 quantization can produce ties.')
    return control, valid, foreground, record


def prepare_case(parent, output):
    if parent.resolve() == output.resolve():
        raise ValueError('Depth guides require a separate output directory')
    shutil.copytree(parent, output, dirs_exist_ok=True)
    head = output / 'head'
    original_hashes = {str(p.relative_to(parent)): sha(p) for p in parent.rglob('*') if p.is_file()}
    with np.load(head / 'guide_channels.npz') as payload:
        channels = {key: payload[key] for key in payload.files}
    food = mask(head / 'target_food_mask.png')
    spoon = mask(head / 'spoon_mask.png') & ~food
    depth = channels['depth'].copy()
    valid = channels['depth_valid'].copy() & np.isfinite(depth) & (depth > 0)
    # The transported raster is authoritative. Source-scene depth underneath a
    # missing transported sample is not a surface observation of the bite.
    target = channels['target_depth']
    target_valid = np.isfinite(target) & (target > 0)
    valid[food] = target_valid[food]
    depth[food] = target[food]
    depth[~valid] = np.nan
    control, valid, foreground, normal = normalize_depth(depth, valid, food | spoon)
    quantized = np.rint(np.clip(control, 0, 255)).astype(np.uint8)
    Image.fromarray(quantized).save(head / 'depth_control.png')
    Image.fromarray(valid.astype(np.uint8) * 255).save(head / 'depth_control_valid_mask.png')
    Image.fromarray(foreground.astype(np.uint8) * 255).save(head / 'depth_control_foreground_mask.png')
    np.savez_compressed(head / 'depth_control_channels.npz', depth=depth, valid=valid,
                        foreground=foreground, target_food=food, visible_spoon=spoon,
                        control_float=control / 255.)
    order = np.argsort(depth[valid], kind='stable')
    monotonic = bool(np.all(np.diff(control[valid][order]) <= 1.e-4))
    stage = json.loads((head / 'stage_manifest.json').read_text())
    old = np.asarray(Image.open(head / 'depth.png').convert('L'))
    record = dict(case_id=parent.name, variant='head_relative_physical_depth_v1',
                  parent=str(parent.resolve()), parent_stage_manifest_sha256=sha(parent / 'head' / 'stage_manifest.json'),
                  bbox_xyxy=stage['bbox_xyxy'], native_size=stage['native_size'], generation_size=stage['generation_size'],
                  control_image='head/depth_control.png', control_valid_mask='head/depth_control_valid_mask.png',
                  normalization=normal, foreground_pixels=int(foreground.sum()),
                  target_food_missing_depth_pixels=int((food & ~target_valid).sum()),
                  foreground_depth_range=[normal['foreground_near'], normal['foreground_far']],
                  food_depth_pixels=int((food & valid).sum()), visible_spoon_depth_pixels=int((spoon & valid).sum()),
                  foreground_control_uint8_range=[int(quantized[foreground].min()), int(quantized[foreground].max())],
                  foreground_unique_uint8_levels=int(len(np.unique(quantized[foreground]))),
                  old_foreground_white_fraction=float(np.mean(old[foreground] == 255)),
                  new_foreground_saturated_fraction=float(np.mean((quantized[foreground] == 0) | (quantized[foreground] == 255))),
                  valid_scene_pixels_within_foreground_depth_range=int((valid & ~foreground & (depth >= normal['foreground_near']) & (depth <= normal['foreground_far'])).sum()),
                  physical_depth_order_preserved=monotonic,
                  invalid_scope='Transported raster misses and source uncertainty are explicitly invalid. No food depth is borrowed from the original photograph, and no hidden geometry is reconstructed by this builder.',
                  spoon_scope='Visible spoon uses the existing guide depth from projected spoon mesh vertices and local interpolation; no new spoon surface is inferred here.',
                  condition_scope='Only select the new depth control as the head control image. Dense source material condition, masks, stage boxes and all original files remain byte-identical.',
                  original_files_unchanged=all(sha(output / name) == digest for name, digest in original_hashes.items()),
                  original_file_hashes=original_hashes, builder_sha256=sha(Path(__file__)))
    assert monotonic and record['original_files_unchanged'] and record['new_foreground_saturated_fraction'] == 0
    record['file_hashes'] = {p.name: sha(p) for p in head.glob('depth_control*') if p.is_file() and p.suffix != '.json'}
    for path in (head / 'depth_control.json', output / 'depth_protocol.json'):
        path.write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--indices', type=int, nargs='+')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for parent in sorted(args.input_root.glob('real_*')):
        if args.indices is not None and int(parent.name.split('_')[1]) not in args.indices:
            continue
        if not (parent / 'factorization.json').exists():
            continue
        row = prepare_case(parent, args.output / parent.name)
        rows.append(row)
        print(json.dumps({key: row[key] for key in ('case_id', 'foreground_depth_range', 'foreground_unique_uint8_levels', 'old_foreground_white_fraction', 'target_food_missing_depth_pixels')}), flush=True)
    (args.output / 'depth_manifest.json').write_text(json.dumps(dict(cases=rows, generated=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
