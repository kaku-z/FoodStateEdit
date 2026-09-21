"""Validate collected evidence and build complete, labelled internal review sheets.

Image differences are diagnostics, never action/photorealism success scores.
No model output, seed, or frame is omitted from the review grids.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sheet(frames, labels, output, columns=7, cell_width=384):
    h, w = frames[0].shape[:2]
    cell_height = round(h * cell_width / w)
    rows = (len(frames) + columns - 1) // columns
    canvas = Image.new('RGB', (columns * cell_width, rows * (cell_height + 24)), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (frame, label) in enumerate(zip(frames, labels)):
        x, y = i % columns * cell_width, i // columns * (cell_height + 24)
        draw.text((x + 6, y + 5), label, fill='black')
        canvas.paste(Image.fromarray(frame).resize((cell_width, cell_height)), (x, y + 24))
    canvas.save(output)


def masked_mae(left, right, mask):
    difference = np.abs(left.astype(np.int16) - right.astype(np.int16))
    return float(difference[mask].mean()) if mask.any() else None


def run(evidence, controls, out):
    if out.exists():
        raise FileExistsError(out)
    verified = json.loads(evidence.with_name(evidence.name + '_verified.json').read_text())
    assert verified['all_files_sha256_verified']
    for name, expected in verified['files'].items():
        assert digest(evidence / name) == expected, name
    manifest = json.loads((controls / 'manifest.json').read_text())
    spec = manifest['config']
    out.mkdir(parents=True)
    result = dict(schema='foodstateedit.spatial_stress_review.v1',
                  evidence=str(evidence.resolve()), controls=str(controls.resolve()),
                  evidence_sha256_verified=True,
                  semantic_review_status='pending_internal_nonblind_visual_review',
                  diagnostic_limit='Pixel differences are not semantic success, physical correctness, or statistical superiority. Composite preservation is enforced by the shared mask.',
                  jobs={})
    for job in manifest['jobs']:
        name = job['id']
        state = json.loads((evidence / name / 'run_manifest.json').read_text())
        rec = dict(status=state['status'], pipeline_load_count=state['pipeline_load_count'], arms={})
        result['jobs'][name] = rec
        if state['status'] != 'complete_requires_visual_review':
            rec['error'] = state.get('error', 'incomplete evidence preserved')
            continue
        assert state['pipeline_load_count'] == 1
        assert state['seed'] == spec['seed'] and state['frames'] == 21 and state['steps'] == 20
        assert not state['lora'] and not state['ttm']
        assert {c['condition'] for c in state['completed']} == set(spec['arms'])
        assert state['manifest_sha256'] == digest(controls / 'manifest.json')
        for path, expected in job['files'].items():
            assert digest(controls / name / path) == expected['sha256']
        reference = np.array(Image.open(controls / name / 'reference.png').convert('RGB'))
        alpha = np.array(Image.open(controls / name / 'shared_alpha.png').convert('L'))
        assert set(np.unique(alpha)).issubset({0, 255})
        inside, outside = alpha > 0, alpha == 0
        control_arrays = np.load(controls / name / 'controls.npz', allow_pickle=False)
        arm_frames = {}
        for arm in spec['arms']:
            folder = evidence / name / arm
            paths = sorted((folder / 'raw_frames').glob('*.png'))
            assert [p.name for p in paths] == [f'{i:02d}.png' for i in range(21)]
            frames = [np.array(Image.open(p).convert('RGB')) for p in paths]
            assert all(f.shape == reference.shape for f in frames)
            composite = [np.where(inside[..., None], f, reference) for f in frames]
            assert all(np.array_equal(f[outside], reference[outside]) for f in composite)
            final = np.array(Image.open(folder / 'projected_final_hold.png').convert('RGB'))
            assert np.array_equal(final[outside], reference[outside])
            arm_frames[arm] = frames
            metrics = dict(raw_frame_count=21,
                           shape=list(reference.shape),
                           raw_outside_mae_0_255=[masked_mae(f, reference, outside) for f in frames],
                           raw_inside_mae_0_255=[masked_mae(f, reference, inside) for f in frames],
                           raw_vs_control_inside_mae_0_255=[masked_mae(f, c, inside) for f, c in zip(frames, control_arrays[arm])],
                           lossless_composite_outside_max=0,
                           delivered_final_outside_max=0,
                           raw_frame_sha256=[digest(p) for p in paths])
            rec['arms'][arm] = metrics
            labels = [f'{arm} | RAW frame {i:02d}' for i in range(21)]
            sheet(frames, labels, out / f'{name}_{arm}_raw_all21.png')
            labels = [f'{arm} | COMPOSITE frame {i:02d}' for i in range(21)]
            sheet(composite, labels, out / f'{name}_{arm}_composite_all21.png')
        rec['pairwise_raw_inside_mae_0_255'] = {}
        for baseline in spec['arms'][:2]:
            rec['pairwise_raw_inside_mae_0_255'][baseline + '_vs_relative3d'] = [
                masked_mae(a, b, inside) for a, b in zip(arm_frames[baseline], arm_frames['relative3d'])]
        sheet([reference] + [arm_frames[a][20] for a in spec['arms']],
              ['Input'] + [a + ' | RAW frame20' for a in spec['arms']],
              out / f'{name}_final_comparison.png', columns=4, cell_width=512)
        sheet([f for a in spec['arms'] for f in [control_arrays[a][20], arm_frames[a][20]]],
              [label for a in spec['arms'] for label in [a + ' CONTROL20', a + ' RAW20']],
              out / f'{name}_control_vs_raw.png', columns=2, cell_width=688)
    (out / 'diagnostics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence', type=Path, required=True)
    ap.add_argument('--controls', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    res = run(args.evidence, args.controls, args.out)
    print(json.dumps({j:dict(status=r['status'], reviewed_cells=len(r['arms'])) for j,r in res['jobs'].items()}, indent=2))
