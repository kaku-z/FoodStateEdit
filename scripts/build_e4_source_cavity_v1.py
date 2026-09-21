"""Prepare an explicitly labelled source-only local-completion experiment."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from source_cavity_masks import polygon_mask, source_reveal_mask, inward_alpha, hybrid_condition

ROOT = Path(__file__).resolve().parents[1]
FACES = ([0, 1, 2, 3], [3, 2, 6, 7], [2, 1, 5, 6])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def sheet(frames, indices, title):
    h, w = frames.shape[1:3]
    image = Image.new('RGB', (w * 3, (h + 30) * 2), 'white')
    draw = ImageDraw.Draw(image)
    for k, i in enumerate(indices):
        x, y = (k % 3) * w, (k // 3) * (h + 30)
        draw.text((x + 4, y + 5), f'{title} / f{i}', fill='black')
        image.paste(Image.fromarray(frames[i]), (x, y + 30))
    return image


def projected_payload(geometry, frame, width, height):
    # Reproduce the frozen cake transform without changing the frozen renderer.
    vertices = np.asarray(geometry['vertices_uv'], dtype=float)
    depths = np.asarray(geometry['vertex_relative_depths'], dtype=float)
    K = np.asarray(geometry['camera_intrinsic'])
    trace = next(r for r in geometry['trace'] if r['frame'] == frame)
    xyz = (np.column_stack([vertices, np.ones(8)]) @ np.linalg.inv(K).T) * depths[:, None]
    contact = np.array([.525 * width, .545 * height, 1.0])
    origin = np.linalg.inv(K) @ contact * 1.1
    angle = -.12 * trace['lift']
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    positions = (xyz - origin) @ rotation.T + np.asarray(trace['anchor_xyz'])
    pixels = positions @ K.T
    return pixels[:, :2] / pixels[:, 2:]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text(encoding='utf-8'))
    controls = ROOT / cfg['source_controls']
    baseline = ROOT / cfg['baseline_condition']
    for path, digest in [(controls / 'manifest.json', cfg['source_manifest_sha256']),
                         (baseline / 'condition_manifest.json', cfg['baseline_manifest_sha256']),
                         (ROOT / cfg['base_config'], cfg['base_config_sha256'])]:
        if sha(path) != digest:
            raise ValueError(f'Frozen dependency hash mismatch: {path}')
    source_manifest = json.loads((controls / 'manifest.json').read_text())
    case = next(c for c in source_manifest['cases'] if c['case_id'] == 'cake')
    for record in case['files'].values():
        if sha(controls / record['path']) != record['sha256']:
            raise ValueError('Controls changed')
    baseline_manifest = json.loads((baseline / 'condition_manifest.json').read_text())
    reference = np.asarray(Image.open(controls / 'cake/reference.png').convert('RGB'))
    geometry = json.loads((controls / 'cake/geometry.json').read_text())
    h, w = reference.shape[:2]
    vertices = np.asarray(geometry['vertices_uv'])
    source_body = np.logical_or.reduce([polygon_mask(vertices[f], (h, w)) for f in FACES])

    # Boundaries of two exposed cut faces. Do not draw an artificial foreground
    # edge around the newly exposed plate floor or paste any donor cake texture.
    structure = np.zeros_like(reference)
    for pair in [(3, 0), (0, 1), (0, 4), (3, 7), (1, 5), (7, 4), (4, 5)]:
        pts = np.rint(vertices[list(pair)]).astype(int)
        cv2.line(structure, tuple(pts[0]), tuple(pts[1]), (255, 255, 255), 2, cv2.LINE_AA)

    bases, conditions, masks, alphas, payloads, overlays = [], [], [], [], [], []
    for i in range(cfg['num_frames']):
        path = baseline / f'projected_frames/{i:02d}.png'
        if sha(path) != baseline_manifest['files'][f'projected_frames/{i:02d}.png']['sha256']:
            raise ValueError(f'Baseline frame changed: {i}')
        base = np.asarray(Image.open(path).convert('RGB'))
        if i == 0:
            payload = source_body.copy()
        else:
            v = projected_payload(geometry, i, w, h)
            payload = np.logical_or.reduce([polygon_mask(v[f], (h, w)) for f in FACES])
        support = source_reveal_mask(source_body, payload,
            lifted=i >= cfg['first_lift_frame'], source_radius=cfg['source_dilation_pixels'],
            exclusion_radius=cfg['payload_exclusion_pixels'])
        alpha = inward_alpha(support, cfg['inside_feather_pixels'])
        condition = hybrid_condition(base, structure, support)
        overlay = base.copy()
        overlay[support] = np.rint(.5 * overlay[support] + .5 * np.array([255, 30, 60])).astype(np.uint8)
        bases.append(base); conditions.append(condition); masks.append(support.astype(np.uint8) * 255)
        alphas.append(alpha); payloads.append(payload.astype(np.uint8) * 255); overlays.append(overlay)

    a.output_root.mkdir(parents=True, exist_ok=False)
    arrays = dict(baseline=np.stack(bases), condition=np.stack(conditions), generation_mask=np.stack(masks),
                  projection_alpha=np.stack(alphas), planned_payload=np.stack(payloads))
    np.savez_compressed(a.output_root / 'controls.npz', **arrays)
    Image.fromarray(reference).save(a.output_root / 'reference.png')
    sheet(np.stack(overlays), [0, 6, 9, 11, 15, 20], 'SOURCE REPAIR MASK - NOT RESULT').save(a.output_root / 'mask_review.png')
    sheet(arrays['condition'], [0, 6, 9, 11, 15, 20], 'HYBRID CONTROL - NOT RESULT').save(a.output_root / 'condition_review.png')
    manifest = dict(schema_version='foodstateedit.source_cavity_controls.v1', config=cfg,
        config_sha256=sha(a.config), builder_sha256=sha(Path(__file__)),
        mask_helper_sha256=sha(Path(__file__).with_name('source_cavity_masks.py')),
        support_pixels_by_frame=[int((m > 0).sum()) for m in masks],
        source_geometry='frozen manual relative 3D cake cuboid, not reconstructed truth',
        files={f.name:dict(sha256=sha(f), size_bytes=f.stat().st_size) for f in a.output_root.iterdir() if f.is_file()})
    write_json(a.output_root / 'manifest.json', manifest)
    print(json.dumps({'manifest_sha256':sha(a.output_root / 'manifest.json'),
                      'support_pixels_by_frame':manifest['support_pixels_by_frame']}))


if __name__ == '__main__':
    main()
