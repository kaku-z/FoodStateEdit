"""Verify local E4 evidence and assemble a labelled before/after research panel."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--controls', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.results / 'files_manifest.json').read_text())
    for rel, rec in manifest.items():
        path = a.results / rel
        if path.stat().st_size != rec['size_bytes'] or sha(path) != rec['sha256']:
            raise ValueError('Result integrity failure: ' + rel)
    run = json.loads((a.results / 'run_manifest.json').read_text())
    if run['status'] != 'complete_requires_visual_review':
        raise ValueError('Run did not complete')
    if sha(a.controls / 'manifest.json') != run['controls_manifest_sha256']:
        raise ValueError('Control manifest mismatch')
    cfg_manifest = json.loads((a.controls / 'manifest.json').read_text())
    for rel, rec in cfg_manifest['files'].items():
        if sha(a.controls / rel) != rec['sha256']:
            raise ValueError('Control file integrity failure')
    with np.load(a.controls / 'controls.npz', allow_pickle=False) as data:
        base, alpha, payload = [data[k] for k in ['baseline', 'projection_alpha', 'planned_payload']]
    expected = {f'{i:02d}.png' for i in range(len(base))}
    for name in ['raw_frames', 'projected_frames']:
        if {f.name for f in (a.results / name).glob('*.png')} != expected:
            raise ValueError('Frame coverage mismatch: ' + name)
    frames, records = [], []
    for i in range(len(base)):
        final = np.asarray(Image.open(a.results / f'projected_frames/{i:02d}.png').convert('RGB'))
        diff = np.abs(final.astype(np.int16) - base[i].astype(np.int16))
        active = alpha[i] > 0
        outside_max = int(diff[~active].max(initial=0))
        payload_max = int(diff[payload[i] > 0].max(initial=0))
        if outside_max or payload_max:
            raise ValueError('Protected baseline pixels changed')
        records.append(dict(frame=i, active_pixels=int(active.sum()),
            inside_change_mae=float(diff[active].mean()) if active.any() else None,
            outside_repair_max_difference=outside_max, planned_payload_max_difference=payload_max))
        frames.append(final)
    a.output_root.mkdir(parents=True, exist_ok=False)
    # All panels below are untouched recorded output pixels, merely arranged.
    h, w = base.shape[1:3]
    panel = Image.new('RGB', (2 * w, 2 * h + 64), 'white')
    draw = ImageDraw.Draw(panel)
    for col, (title, frame) in enumerate([('E3 baseline', base[-1]), ('E4 source-only repair', frames[-1])]):
        x = col * w
        draw.text((x + 10, 8), title, fill='black')
        panel.paste(Image.fromarray(frame), (x, 28))
        # Crop is fixed by the planned final repair support, with equal margin.
        ys, xs = np.where(alpha[-1] > 0)
        box = (max(0, int(xs.min()) - 28), max(0, int(ys.min()) - 28),
               min(w, int(xs.max()) + 29), min(h, int(ys.max()) + 29))
        crop = Image.fromarray(frame).crop(box)
        crop.thumbnail((w - 20, h - 10))
        # Scale the diagnostic crop identically in both columns.
        scale = min((w - 20) / crop.width, (h - 10) / crop.height)
        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.NEAREST)
        panel.paste(crop, (x + (w - crop.width) // 2, h + 60))
    panel.save(a.output_root / 'before_after.png')
    summary = dict(status='technical_validation_only_requires_visual_review', run=run,
        verified_result_records=len(manifest), frames=records,
        note='MAE measures modification only. No ground truth or automatic naturalness score is available.')
    with (a.output_root / 'validation.json').open('x', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(dict(verified_records=len(manifest), frames=len(records),
                         final_inside_change_mae=records[-1]['inside_change_mae'], protected_max=0)))


if __name__ == '__main__':
    main()
