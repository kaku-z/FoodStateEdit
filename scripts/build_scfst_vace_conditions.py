"""Freeze SCFST token controls derived from an existing E7 control bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.scfst_vace_conditions import build_packed


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-controls', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--frame-indices', type=int, nargs='+', required=True)
    p.add_argument('--grid-height', type=int, required=True)
    p.add_argument('--grid-width', type=int, required=True)
    p.add_argument('--prefix-tokens', type=int, default=0,
                   help='Explicit DiT reference-prefix length; VACE-only runs use zero.')
    a = p.parse_args()
    source = a.source_controls.resolve(); output = a.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest_path = source/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    record = manifest['files']['controls.npz']
    if sha256(source/'controls.npz') != record['sha256']:
        raise ValueError('source controls hash mismatch')
    with np.load(source/'controls.npz', allow_pickle=False) as z:
        packed, active = build_packed(z['generation_mask'], z['planned_payload'],
                                      z['removal_projection'], a.frame_indices,
                                      (a.grid_height, a.grid_width),
                                      a.prefix_tokens)
    if packed.shape[1] != a.prefix_tokens+len(a.frame_indices)*a.grid_height*a.grid_width:
        raise AssertionError('packed token count mismatch')
    output.mkdir(parents=True)
    np.savez_compressed(output/'state_transfer_tokens.npz', condition=packed, active=active)
    roles = np.argmax(packed[0, :, :4], axis=-1)
    result = {
        'schema_version':'foodstateedit.scfst_vace_tokens.v1',
        'status':'wiring_control_not_learned_state',
        'source_manifest_sha256':sha256(manifest_path),
        'source_controls_sha256':record['sha256'],
        'builder_sha256':sha256(Path(__file__)),
        'frame_indices':a.frame_indices,
        'grid_hw':[a.grid_height,a.grid_width],
        'prefix_tokens':a.prefix_tokens,
        'shape':list(packed.shape),
        'active_tokens':int(active.sum()),
        'role_counts':{name:int((roles==idx).sum()) for idx,name in enumerate(('keep','move','rearrange','expose'))},
        'limitations':['affine source correspondence from frozen payload boxes','no material dynamics',
                       'no rearrange labels in this cake wiring control','explicit provisional VAE frame indices'],
    }
    result['files']={'state_transfer_tokens.npz':{'sha256':sha256(output/'state_transfer_tokens.npz'),
                                                   'size_bytes':(output/'state_transfer_tokens.npz').stat().st_size}}
    (output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
