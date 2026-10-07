"""Freeze source-caption-based geometry routing without case-ID exceptions.

This records a topology hypothesis, not a verified food identity. Existing
strand geometry is retained only for a consistent noodle-family component.
Other strand selections use a source-owned cohesive packet and require a new
caption probe if that packet changes source ownership.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def packet_status(folder, source_sha256):
    if not (folder / 'state.json').exists():
        return None
    state = json.loads((folder / 'state.json').read_text())
    if state['mode'] != 'cohesive' or sha(folder / 'source.png') != source_sha256:
        return None
    with np.load(folder / 'transport.npz') as data:
        faces = data['faces']
        _, counts = np.unique(np.sort(np.r_[faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1), axis=0, return_counts=True)
        closed = bool(np.all(counts == 2))
        ids = bool(np.array_equal(data['carried_ids'], data['target_ids']))
        finite = bool(np.isfinite(data['rest_xyz']).all() and np.isfinite(data['target_xyz']).all())
    solve = state['solver']['full']
    if not (closed and ids and finite and solve['preserve_volume']):
        return None
    return dict(geometry_case=str(folder.resolve()), state_sha256=sha(folder / 'state.json'),
                transport_sha256=sha(folder / 'transport.npz'), source_sha256=source_sha256,
                actual_closed_edge_incidence_all_two=closed, material_ids_preserved=ids,
                finite_geometry=finite, volume_constraint_enabled=True,
                volume_relative_error=solve['volume_relative_error'],
                source_topology=state.get('source_topology'))


def build(captions_path, guide_roots, packet_roots):
    captions = json.loads(captions_path.read_text(encoding='utf-8'))
    rows = []
    found = set()
    for root in guide_roots:
        for folder in sorted(root.glob('real_*')):
            path = folder / 'guide_manifest.json'
            if not path.exists():
                continue
            if folder.name in found:
                raise ValueError('Duplicate baseline case: ' + folder.name)
            found.add(folder.name)
            guide = json.loads(path.read_text())
            caption = captions.get(folder.name, {})
            consistent_noodle = caption.get('component_consistency') is True and caption.get('component_family') == 'noodle'
            packet = guide['mode'] == 'strand' and not consistent_noodle
            existing = []
            if packet:
                for candidate in packet_roots:
                    result = packet_status(candidate / 'real' / folder.name, guide['source_sha256'])
                    if result is not None:
                        existing.append(result)
            if len(existing) > 1:
                raise ValueError('Multiple reusable packets; provide one authoritative packet root for ' + folder.name)
            rows.append(dict(case_id=folder.name, index=int(folder.name.split('_')[1]),
                original_mode=guide['mode'], route='cohesive_packet' if packet else 'retain_existing',
                reason=('Existing strand mode lacks consistent noodle-family component evidence.' if packet else
                        'Consistent noodle-family component retains strand hypothesis.' if guide['mode'] == 'strand' else
                        'Existing non-strand geometry is retained; caption ambiguity alone does not change its geometry.'),
                consistent_noodle_component=consistent_noodle,
                component_label=caption.get('component_label'), component_family=caption.get('component_family'),
                component_consistency=caption.get('component_consistency'), raw_labels=caption.get('raw_labels'),
                original_guide_case=str(folder.resolve()), original_guide_manifest_sha256=sha(path),
                source_sha256=guide['source_sha256'], guide_source_image_sha256=sha(folder / 'source.png'),
                selected_packet=existing[0] if existing else None,
                needs_packet_build=packet and not existing,
                source_ownership_recheck_required=packet,
                scope='Material-packet closure is an explicit unobserved-topology hypothesis, not a ground-truth ingredient or true cohesion/volume assertion.' if packet else 'Original source-derived geometry unchanged.'))
    return dict(policy_version='source_caption_geometry_routing_v1',
                rule='if original guide mode is strand and not (component_consistency is true and component_family equals noodle), use a cohesive source-owned packet; otherwise retain existing geometry',
                cases=rows, input_captions=str(captions_path.resolve()), input_captions_sha256=sha(captions_path),
                caption_metadata=captions.get('_metadata'), source_only=True, case_id_exceptions=False,
                generated_image_feedback_used=False, model_consistency_is_ground_truth=False,
                missing_caption_policy='Conservative packet route for an original strand mode without consistent source evidence.',
                routing_is_frozen_before_geometry_recaption=True,
                subsequent_caption_scope='After changed source ownership, a fresh source-only caption may describe the retained packet. It does not retrospectively change this frozen routing decision.',
                builder_sha256=sha(Path(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captions', type=Path, required=True)
    parser.add_argument('--guide-roots', type=Path, nargs='+', required=True)
    parser.add_argument('--packet-roots', type=Path, nargs='*', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.captions, args.guide_roots, args.packet_roots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(cases=len(result['cases']), packet_routes=[c['case_id'] for c in result['cases'] if c['route']=='cohesive_packet'],
                          needs_build=[c['case_id'] for c in result['cases'] if c['needs_packet_build']])))


if __name__ == '__main__':
    main()
