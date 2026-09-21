"""CPU-only synthetic contract demo. Produces controls, NEVER edited photos."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodstateedit.state_transfer import Camera, Transition, apply_transition, from_mask, project
from foodstateedit.state_transfer.geometry import CHANNELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True, type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)  # never overwrite evidence
    cam = Camera(48, 64, 64, 64, 31.5, 23.5)
    depth = np.full((48, 64), 2.)
    mask = np.zeros((48, 64), bool); mask[26:38, 18:46] = True
    origin = from_mask(mask, depth, cam)
    xy = origin.source_xy
    selected = (xy[:, 0] >= 27) & (xy[:, 0] <= 34) & (xy[:, 1] < 32)
    adjacent = (xy[:, 0] >= 35) & (xy[:, 0] <= 38) & (xy[:, 1] < 32)
    conditions, visible, positions, diagnostics = [], [], [], []
    for t in range(21):
        phase = max(0., (t-4)/16)
        delta = np.zeros_like(origin.xyz)
        delta[selected] = phase*np.array([.16, -.48, -.3])
        delta[adjacent, 0] = -.08*phase
        owner = np.zeros(len(xy), np.int64)
        if phase > 0:
            owner[selected] = 1
        state = apply_transition(origin, Transition(delta, owner, selected.astype(float)*(phase > 0),
                                                   np.ones(len(xy))), cam)
        result = project(state, cam, depth)
        conditions.append(result['condition']); visible.append(result['visible_ids'])
        positions.append(state.xyz); diagnostics.append(result['diagnostics'])
    np.savez_compressed(root/'controls.npz', condition=np.stack(conditions), visible_ids=np.stack(visible),
                        parcel_xyz=np.stack(positions), source_ids=origin.ids, source_xy=origin.source_xy)
    data = root/'controls.npz'
    manifest = {'status': 'synthetic_contract_demo_not_generation', 'trained': False,
                'channels': CHANNELS, 'frames': 21, 'camera': vars(cam),
                'shape': list(np.stack(conditions).shape), 'diagnostics': diagnostics,
                'files': [{'path': data.name, 'sha256': hashlib.sha256(data.read_bytes()).hexdigest()}],
                'limitations': ['oracle displacements, not material dynamics', 'point z-buffer, not volume renderer',
                                'unknown exposure has no reconstructed depth or appearance', 'no GPU or VACE run']}
    (root/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(root), 'status': manifest['status'], 'frames': 21}))


if __name__ == '__main__':
    main()
