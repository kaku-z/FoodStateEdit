"""Audit high-lift coverage and freeze a swept-support inference-only package."""
import argparse
import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from build_flexible_completion_dataset import (
    draw_segment, make_review, phase_amount, render_strand, write_video,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path, root):
    return dict(path=path.relative_to(root).as_posix(), size_bytes=path.stat().st_size,
                sha256=digest(path))


def write_json(path, data):
    with path.open('x', encoding='utf-8') as out:
        json.dump(data, out, indent=2)
        out.write('\n')


def decode_rgb(path, width, height):
    result = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path),
                             '-vf', f'scale={width}:{height}:flags=lanczos',
                             '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                            check=True, capture_output=True)
    return np.frombuffer(result.stdout, np.uint8).reshape(-1, height, width, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    old_root = repo / 'artifacts/day17_high_lift_relative3d_udon_v1'
    geom_path = repo / 'configs/flexible_completion_udon_relative3d_geometry_high_lift_v1.json'
    old_config_path = repo / 'configs/day17_high_lift_vace_pilot_v1.json'
    old_config = json.loads(old_config_path.read_text())
    old_manifest = json.loads((old_root / 'dataset_manifest.json').read_text())
    for rec in old_config['dataset']['files'].values():
        assert digest(old_root / rec['path']) == rec['sha256']
    assert digest(geom_path) == '5166512f019449f1ac23d8a762505398356a95be4e9c7e81a8d2f2b84d58408c'
    geometry = json.loads(geom_path.read_text())
    source = np.asarray(Image.open(old_root / old_config['dataset']['files']['reference']['path']).convert('RGB'))
    old_alpha = np.asarray(Image.open(old_root / old_config['dataset']['files']['edit_alpha']['path']).convert('L'))
    height, width = old_alpha.shape
    xyz = np.load(old_root / old_config['dataset']['files']['geometry']['path'])
    K = xyz['intrinsic']
    sticks = geometry['chopsticks']
    all_support = np.ones((height, width), bool)
    frames, audits, cores = [], [], []
    for i in range(21):
        frame = source.copy()
        phase, _ = phase_amount(i, geometry['timeline'])
        if i == 0:
            frames.append(frame)
            cores.append(np.zeros_like(old_alpha, bool))
            continue
        pinch = np.asarray(old_manifest['frame_records'][i]['pinch_uv'])
        strand_xyz = xyz['strand_xyz'][i].astype(float)
        if phase in ('contact', 'lift', 'final_hold'):
            uvh = strand_xyz @ K.T
            uv = uvh[:, :2] / uvh[:, 2:]
            render_strand(frame, uv, strand_xyz[:, 2], (1.0, 1.06), all_support,
                          geometry['strand'], front=False)
        lines = []
        body = np.zeros_like(source)
        for side, color_key in ((1.0, 'far_color_rgb'), (-1.0, 'near_color_rgb')):
            start = pinch + [0, side * sticks['tip_separation_normalized'] * height / 2]
            end = pinch + np.asarray(sticks['handle_offset_normalized']) * [width, height] + [0, side * sticks['handle_separation_normalized'] * height / 2]
            lines.append((start, end))
            draw_segment(body, start, end, (255, 255, 255), sticks['line_width_px'], all_support)
            draw_segment(frame, start + [2, 3], end + [2, 3], (40, 24, 17), sticks['shadow_width_px'], all_support)
            draw_segment(frame, start, end, tuple(sticks[color_key]), sticks['line_width_px'], all_support)
            draw_segment(frame, start - [1, 1], end - [1, 1], tuple(sticks['highlight_rgb']), max(2, sticks['line_width_px'] // 4), all_support)
        if phase in ('contact', 'lift', 'final_hold'):
            render_strand(frame, uv, strand_xyz[:, 2], (1.0, 1.06), all_support,
                          geometry['strand'], front=True)
        core = np.any(frame != source, axis=2)
        stick_core = np.any(body > 0, axis=2)
        audits.append(dict(frame=i, phase=phase, stick_pixels=int(stick_core.sum()),
                           stick_outside_old_support=int((stick_core & (old_alpha == 0)).sum()),
                           stick_below_full_alpha=int((stick_core & (old_alpha < 255)).sum()),
                           mean_old_alpha_on_sticks=float(old_alpha[stick_core].mean()),
                           total_render_pixels=int(core.sum()),
                           render_outside_old_support=int((core & (old_alpha == 0)).sum())))
        frames.append(frame)
        cores.append(core)
    swept = np.logical_or.reduce(cores)
    # Fixed 8-pixel full-opacity margin and an 8-pixel feather ramp; no output tuning.
    core_image = Image.fromarray(np.uint8(swept) * 255)
    full = np.asarray(core_image.filter(ImageFilter.MaxFilter(17)))
    expanded = full.copy()
    for radius in range(9, 17):
        dilated = np.asarray(core_image.filter(ImageFilter.MaxFilter(radius * 2 + 1))) > 0
        expanded = np.maximum(expanded, np.uint8(dilated) * round(255 * (17 - radius) / 9))
    new_alpha = np.maximum(old_alpha, expanded)
    assert np.all(new_alpha >= old_alpha)
    assert np.all(new_alpha[swept] == 255)
    assert np.array_equal(frames[0], source)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for key in ('reference', 'strand_mask', 'geometry'):
        rel = old_config['dataset']['files'][key]['path']
        dest = output / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old_root / rel, dest)
    alpha_path = output / old_config['dataset']['files']['edit_alpha']['path']
    alpha_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(new_alpha).save(alpha_path)
    control_path = output / old_config['dataset']['files']['relative3d_control']['path']
    control_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(control_path, frames, 8)
    review_dir = output / 'review'
    review_dir.mkdir()
    make_review(frames, [0, 3, 6, 10, 15, 20], 'full swept control').save(review_dir / 'control.png')
    Image.fromarray(new_alpha).save(review_dir / 'expanded_alpha.png')
    Image.fromarray(old_alpha).save(review_dir / 'old_alpha.png')
    # A CPU-only projection ablation on the existing raw generator output.
    raw_path = repo / 'artifacts/day17_high_lift_vace_gp38_20260907_v1/raw.mp4'
    assert digest(raw_path) == '7199ad64468d7af47ce72cccb58922f93fe2b3360c55e630d6b5ec76a927cc99'
    raw = decode_rgb(raw_path, width, height)
    assert len(raw) == 21
    for name, alpha in [('old_projection', old_alpha), ('expanded_projection', new_alpha)]:
        weight = alpha.astype(float)[..., None] / 255
        projected = np.clip(np.rint(raw * weight + source * (1 - weight)), 0, 255).astype(np.uint8)
        make_review(projected, [0, 3, 6, 10, 15, 20], name).save(review_dir / f'{name}.png')
    audit = dict(schema_version='foodstateedit.high_lift_support_audit.v1',
                 old_support_fraction=float((old_alpha > 0).mean()),
                 new_support_fraction=float((new_alpha > 0).mean()),
                 swept_render_pixels=int(swept.sum()),
                 uncovered_render_pixels_after_fix=int((swept & (new_alpha < 255)).sum()),
                 full_margin_px=8, feather_margin_px=8, frames=audits,
                 projection_diagnostic='same existing raw video, only final alpha changes',
                 raw_sha256=digest(raw_path), geometry_sha256=digest(geom_path))
    write_json(output / 'support_audit.json', audit)
    dataset_files = {key: record(output / value['path'], output)
                     for key, value in old_config['dataset']['files'].items()}
    manifest = dict(schema_version='foodstateedit.high_lift_inference_dataset.v1',
                    scope='inference only; no high-lift target or training labels',
                    source_day17_manifest_sha256=digest(old_root / 'dataset_manifest.json'),
                    geometry_sha256=digest(geom_path), builder_sha256=digest(Path(__file__)),
                    files=dataset_files, support_audit=audit)
    write_json(output / 'dataset_manifest.json', manifest)
    config = copy.deepcopy(old_config)
    config['freeze_date'] = '2026-09-08'
    config['scientific_status'] = 'one_seen_synthetic_high_lift_swept_support_repair_pilot'
    config['question'] = 'Does full swept support preserve raised sticks and connected noodle under the same Day17 prompt, seed, geometry and base model?'
    config['dataset']['remote_root'] = '/host/space0/guo-z/tf-ufi/outputs/' + output.name
    config['dataset']['manifest'] = record(output / 'dataset_manifest.json', output)
    config['dataset']['files'] = dataset_files
    config['output_root'] = '/host/space0/guo-z/tf-ufi/outputs/day18_swept_support_vace_gp38_20260908_v1'
    config['support_repair'] = dict(previous_config_sha256=digest(old_config_path),
                                  changes=['control raster support', 'VACE mask', 'final projection alpha'],
                                  unchanged=['prompt', 'negative_prompt', 'seed', 'geometry', 'model', 'steps'],
                                  not_a_mask_stage_isolation=True)
    config_path = repo / 'configs/day18_high_lift_swept_support_v1.json'
    write_json(config_path, config)
    print(json.dumps(dict(output=str(output), final_hold_audit=audits[-1],
                          old_support=audit['old_support_fraction'], new_support=audit['new_support_fraction'],
                          config_sha256=digest(config_path)), indent=2))


if __name__ == '__main__':
    main()
