"""Explicitly compose and collect development outputs, never prospective views.

Generated crops and raw full-frame generations are preserved independently.
All composite masks and crop transforms were derived from source geometry.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

ROOT = Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')


def read(path):
    for attempt in range(8):
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            if attempt == 7:
                raise
            time.sleep(.25)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compose(gate):
    folder = ROOT / gate
    config = read(folder / 'config.json')
    manifest = read(ROOT / 'inputs/manifest.json')
    cases = {c['case_id']: c for c in manifest['cases']}
    out = folder / 'compositions_v1'
    out.mkdir(exist_ok=True)
    rows = []
    for job in config['jobs']:
        cid = job['case_id']
        assert cid in manifest['development_ids']
        candidates = list(folder.glob('worker_*/' + job['id'] + '/raw.png'))
        if not candidates:
            continue
        assert len(candidates) == 1
        patch_path = candidates[0]
        transform = read(ROOT / 'gate_v7' / cid / 'transform.json')
        x0, y0, x1, y1 = transform['box']
        patch = np.asarray(Image.open(patch_path).convert('RGB').resize(
            (x1-x0, y1-y0), Image.Resampling.LANCZOS), float)
        original = np.asarray(Image.open(ROOT / 'inputs' / cid / 'source.png').convert('RGB'), float)
        mask = np.asarray(Image.open(ROOT / 'fracture_controls_v1' / cid / 'edit_mask.png')) > 0
        local = np.asarray(Image.open(ROOT / 'gate_v7' / cid / 'full_mask.png')) > 0
        alpha = np.clip(distance_transform_edt(mask)/5, 0, 1)[..., None]
        local_alpha = np.clip(distance_transform_edt(local)/4, 0, 1)[..., None]
        bases = [('local', ROOT / f'gate_v4/worker_1/{cid}__canny__41/composited.png')]
        global_path = ROOT / f'gate_v5/worker_1/{cid}__controlled_refine__41/raw.png'
        if global_path.exists():
            bases.append(('global_local', global_path))
        for kind, base_path in bases:
            generated = np.asarray(Image.open(base_path).convert('RGB'), float)
            base = original*(1-alpha) + generated*alpha
            canvas = base.copy()
            canvas[y0:y1, x0:x1] = patch
            final = np.uint8(np.clip(np.rint(base*(1-local_alpha)+canvas*local_alpha), 0, 255))
            assert np.array_equal(final[~(mask | local)], original[~(mask | local)])
            dest = out / (job['id'] + '__' + kind)
            dest.mkdir(exist_ok=True)
            Image.fromarray(final).save(dest / 'composited.png')
            prep = cases[cid]['preprocessing']
            left, top = prep['pad_left_top']
            width, height = prep['resized']
            Image.fromarray(final).crop((left, top, left+width, top+height)).save(dest / 'view.png')
            row = dict(id=dest.name, raw_crop=str(patch_path), raw_crop_sha256=sha(patch_path),
                       base=str(base_path), base_sha256=sha(base_path), transform=transform,
                       outside_edit_source_exact=True, raw_output=False,
                       source_only_crop=True, view_crop='Fixed original content rectangle, no output-dependent crop')
            (dest / 'result.json').write_text(json.dumps(row, indent=2)+'\n')
            rows.append(row)
    (out / 'manifest.json').write_text(json.dumps(dict(
        status='complete' if len(rows) == 2*len(config['jobs']) else 'partial',
        expected=2*len(config['jobs']), rows=rows,
        script_sha256=sha(Path(__file__))), indent=2)+'\n')
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gates', nargs='+', default=['gate_v7', 'gate_v8'])
    parser.add_argument('--archive', required=True)
    args = parser.parse_args()
    for gate in args.gates:
        assert gate in ['gate_v7', 'gate_v8', 'gate_v9']
        if (ROOT / gate / 'config.json').exists():
            print(gate, 'composites', len(compose(gate)), flush=True)
    archive = ROOT / args.archive
    assert archive.suffix == '.zip' and archive.parent == ROOT
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for gate in ['gate_v5', 'gate_v7', 'gate_v8', 'gate_v9']:
            for path in (ROOT / gate).rglob('*'):
                if path.is_file() and path.suffix in ['.json', '.png', '.py', '.txt']:
                    z.write(path, path.relative_to(ROOT))
    print(dict(archive=str(archive), sha256=sha(archive)), flush=True)


if __name__ == '__main__':
    main()
