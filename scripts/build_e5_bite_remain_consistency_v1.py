"""Freeze E5 shared-volume cake controls without modifying earlier evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from bite_remain_consistency import (build_cavity_anchor, dilate, inward_alpha,
    polygon_mask, visible_repair_support)

ROOT = Path(__file__).resolve().parents[1]
VISIBLE_FACES = ([0, 1, 2, 3], [3, 2, 6, 7], [2, 1, 5, 6])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2); f.write('\n')


def projected_payload(geometry, frame):
    vertices = np.asarray(geometry['vertices_uv'], float)
    depths = np.asarray(geometry['vertex_relative_depths'], float)
    K = np.asarray(geometry['camera_intrinsic'], float)
    trace = next(row for row in geometry['trace'] if row['frame'] == frame)
    xyz = (np.column_stack([vertices, np.ones(8)]) @ np.linalg.inv(K).T) * depths[:, None]
    origin = np.linalg.inv(K) @ np.array([.525 * 688, .545 * 512, 1.0]) * 1.1
    angle = -.12 * trace['lift']
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    positions = (xyz - origin) @ rotation.T + np.asarray(trace['anchor_xyz'])
    pixels = positions @ K.T
    return pixels[:, :2] / pixels[:, 2:], float(trace['lift'])


def sheet(frames, title):
    ids = [0, 8, 9, 12, 15, 20]; h, w = frames.shape[1:3]
    out = Image.new('RGB', (3 * w, 2 * (h + 30)), 'white'); draw = ImageDraw.Draw(out)
    for k, i in enumerate(ids):
        x, y = (k % 3) * w, (k // 3) * (h + 30)
        draw.text((x + 4, y + 5), f'{title} / f{i}', fill='black')
        out.paste(Image.fromarray(frames[i]), (x, y + 30))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    a = p.parse_args(); cfg = json.loads(a.config.read_text(encoding='utf-8'))
    controls = ROOT / cfg['source_controls']; baseline = ROOT / cfg['baseline_condition']
    frozen = [(controls / 'manifest.json', cfg['source_manifest_sha256']),
              (baseline / 'condition_manifest.json', cfg['baseline_manifest_sha256']),
              (ROOT / cfg['base_config'], cfg['base_config_sha256'])]
    for path, digest in frozen:
        if sha(path) != digest: raise ValueError(f'frozen dependency mismatch: {path}')
    source_manifest = json.loads((controls / 'manifest.json').read_text())
    case = next(c for c in source_manifest['cases'] if c['case_id'] == 'cake')
    for record in case['files'].values():
        if sha(controls / record['path']) != record['sha256']: raise ValueError('source controls changed')
    baseline_manifest = json.loads((baseline / 'condition_manifest.json').read_text())
    reference = np.asarray(Image.open(controls / 'cake/reference.png').convert('RGB'))
    geometry = json.loads((controls / 'cake/geometry.json').read_text())
    vertices = np.asarray(geometry['vertices_uv'], float); h, w = reference.shape[:2]
    removal = np.logical_or.reduce([polygon_mask(vertices[list(f)], (h, w)) for f in VISIBLE_FACES])

    # Real material donors: the large cake's visible crumb face and neutral plate pixels.
    crumb_patch = reference[250:350, 115:285].copy()
    yy, xx = np.indices((h, w)); hsv = cv2.cvtColor(reference, cv2.COLOR_RGB2HSV)
    plate_roi = ((xx > 265) & (xx < 510) & (yy > 300) & (yy < 440))
    plate_donor = plate_roi & (hsv[..., 1] < 42) & ~dilate(removal, 18)
    anchor, surfaces = build_cavity_anchor(reference, vertices, crumb_patch, plate_donor)

    shared_alpha = np.asarray(Image.open(controls / 'cake/shared_alpha.png').convert('L')) > 0
    local_roi = (xx >= 270) & (xx <= 485) & (yy >= 205) & (yy <= 435)
    cleanup_domain = (shared_alpha & local_roi) | dilate(removal, 8)
    bases=[]; conditions=[]; masks=[]; alphas=[]; payloads=[]; previews=[]
    for i in range(cfg['num_frames']):
        rel = f'projected_frames/{i:02d}.png'; path = baseline / rel
        if sha(path) != baseline_manifest['files'][rel]['sha256']: raise ValueError(f'baseline frame changed: {i}')
        base = np.asarray(Image.open(path).convert('RGB'))
        if i == 0:
            payload, lift = removal.copy(), 0.0
        else:
            moved, lift = projected_payload(geometry, i)
            payload = np.logical_or.reduce([polygon_mask(moved[list(f)], (h, w)) for f in VISIBLE_FACES])
        support = visible_repair_support(removal, cleanup_domain, payload, lift,
            base_radius=cfg['base_reveal_radius_pixels'], growth_radius=cfg['growth_radius_pixels'],
            exclusion_radius=cfg['payload_exclusion_pixels'])
        alpha = inward_alpha(support, cfg['inside_feather_pixels'])
        target = reference.copy()
        target[removal] = anchor[removal]
        condition = base.copy(); condition[support] = target[support]
        preview = base.copy()
        preview[support] = np.rint(.55 * preview[support] + .45 * np.array([250, 35, 70])).astype(np.uint8)
        bases.append(base); conditions.append(condition); masks.append(support.astype(np.uint8)*255)
        alphas.append(alpha); payloads.append(payload.astype(np.uint8)*255); previews.append(preview)

    a.output_root.mkdir(parents=True, exist_ok=False)
    arrays = dict(baseline=np.stack(bases), condition=np.stack(conditions),
        generation_mask=np.stack(masks), projection_alpha=np.stack(alphas),
        planned_payload=np.stack(payloads), removal_projection=removal.astype(np.uint8)*255,
        cavity_anchor=anchor, wall_back=surfaces['wall_back'].astype(np.uint8)*255,
        wall_left=surfaces['wall_left'].astype(np.uint8)*255, floor=surfaces['floor'].astype(np.uint8)*255)
    np.savez_compressed(a.output_root / 'controls.npz', **arrays)
    Image.fromarray(reference).save(a.output_root / 'reference.png')
    Image.fromarray(anchor).save(a.output_root / 'cavity_anchor.png')
    sheet(np.stack(previews), 'E5 REPAIR SUPPORT - NOT RESULT').save(a.output_root / 'mask_review.png')
    sheet(arrays['condition'], 'E5 SHARED-VOLUME CONDITION - NOT RESULT').save(a.output_root / 'condition_review.png')
    manifest = dict(schema_version='foodstateedit.bite_remain_controls.v1', config=cfg,
        config_sha256=sha(a.config), builder_sha256=sha(Path(__file__)),
        helper_sha256=sha(Path(__file__).with_name('bite_remain_consistency.py')),
        source_volume_pixels=int(removal.sum()),
        surface_pixels={k:int(v.sum()) for k,v in surfaces.items()},
        support_pixels_by_frame=[int((m>0).sum()) for m in masks],
        invariants=['one frozen removal volume produces payload and cavity',
                    'moving payload excluded from local regeneration',
                    'outside inward alpha preserved exactly'],
        files={f.name:dict(sha256=sha(f), size_bytes=f.stat().st_size)
               for f in a.output_root.iterdir() if f.is_file()})
    write_json(a.output_root / 'manifest.json', manifest)
    print(json.dumps({'manifest_sha256':sha(a.output_root/'manifest.json'),
                      'support_pixels_by_frame':manifest['support_pixels_by_frame']}))


if __name__ == '__main__': main()
