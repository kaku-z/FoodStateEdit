"""Freeze a uniform routed MLD4 protocol, then execute explicit GPU shards.

Preparation never loads a model. Run each shard once, using this script's saved
copy and the protocol it produced. Images are selected by one protocol-wide
variant; the runner never scores candidates or chooses per-case winners.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


MODULES = ['run_mld4_pipeline.py', 'run_mld4_reconstruction.py',
           'run_mld4_factorized.py', 'probe_mld3_metal_conditioning.py',
           'refine_mld4_source_seam.py', 'mld4_latent_transport.py', 'mld4_binding_adapter.py']
MANIFESTS = {'boundary': 'guide_manifest.json', 'anchored': 'guide_manifest.json',
             'depth': 'depth_protocol.json'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def pixels(path, mode='RGB'):
    import numpy as np
    from PIL import Image
    return np.asarray(Image.open(path).convert(mode))


def ownership(folder):
    import numpy as np
    folder = Path(folder)
    image = pixels(folder/'source.png')
    owned = np.zeros(image.shape[:2], dtype=bool)
    x0, y0, x1, y1 = read(folder/'guide_manifest.json')['source_reference_bbox']
    mask = pixels(folder/'source_reference_mask.png', 'L') > 127
    reference = pixels(folder/'source_reference.png')
    if not np.array_equal(reference[mask], image[y0:y1, x0:x1][mask]):
        raise ValueError(f'Selected source RGB changed: {folder}')
    owned[y0:y1, x0:x1] = mask
    return image, owned


def validate_case(row, caption):
    import numpy as np
    guides = {key: Path(row['production_guides'][key]).resolve() for key in MANIFESTS}
    hashes = {}
    for key, folder in guides.items():
        manifest = folder/MANIFESTS[key]
        if sha(manifest) != row['production_guide_manifest_hashes'][key]:
            raise ValueError(f'Routing manifest mismatch: {manifest}')
        hashes.update({str(p): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()})
    if not caption.get('ownership_provenance_verified') or not caption.get('source_only') or caption.get('oracle'):
        raise ValueError(f'Unverified automatic source ownership: {row["case_id"]}')
    for key in ['head', 'source', 'source_surface']:
        if not isinstance(caption.get(key), str) or not caption[key].strip():
            raise ValueError(f'Missing caption {key}: {row["case_id"]}')
    semantic = Path(caption['resolved_source_guide']['folder'])
    for name, digest in caption['resolved_source_guide']['sha256'].items():
        path = semantic/name
        if sha(path) != digest:
            raise ValueError(f'Caption source input changed: {path}')
        hashes[str(path)] = digest
    source, selected = ownership(guides['boundary'])
    semantic_source, semantic_selected = ownership(semantic)
    if not np.array_equal(source, semantic_source) or not np.array_equal(selected, semantic_selected):
        raise ValueError(f'Caption does not describe routed source material: {row["case_id"]}')
    for key, source_name, mask_name in [('anchored', 'source.png', 'inpaint_mask.png'),
                                         ('depth', 'full_source.png', 'full_inpaint_mask.png')]:
        if not np.array_equal(source, pixels(guides[key]/source_name)):
            raise ValueError(f'Guide source mismatch: {row["case_id"]}/{key}')
        if not np.array_equal(pixels(guides['boundary']/'inpaint_mask.png', 'L'), pixels(guides[key]/mask_name, 'L')):
            raise ValueError(f'Frozen edit region mismatch: {row["case_id"]}/{key}')
    anchored_source, anchored_selected = ownership(guides['anchored'])
    if not np.array_equal(anchored_source, source) or not np.array_equal(anchored_selected, selected):
        raise ValueError(f'Anchored ownership mismatch: {row["case_id"]}')
    return guides, hashes


def prepare(args):
    if args.output.exists():
        raise FileExistsError(f'Choose a fresh experiment output: {args.output}')
    if args.shards < 1 or args.steps < 1 or args.resolution < 256 or args.cavity_seam < 0:
        raise ValueError('Invalid shard count, steps, resolution, or cavity seam')
    route = read(args.routing)
    captions = read(args.captions)
    rows = sorted(route['cases'], key=lambda r: r['index'])
    if args.indices is not None:
        indices = set(args.indices)
        rows = [r for r in rows if r['index'] in indices]
        if {r['index'] for r in rows} != indices:
            raise ValueError('Requested index is absent from routing manifest')
    if not rows or len({r['case_id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate case selection')
    if args.shards > len(rows):
        raise ValueError('Shard count exceeds selected case count')
    validated = []
    input_hashes = {}
    for row in rows:
        if row.get('needs_packet_build'):
            raise ValueError(f'Unfinished routed geometry: {row["case_id"]}')
        guides, hashes = validate_case(row, captions[row['case_id']])
        input_hashes.update(hashes)
        validated.append((row, guides))

    # All validation precedes output creation; every run gets its own snapshots.
    output = args.output.resolve()
    output.mkdir(parents=True)
    snapshots = output/'snapshot'
    scripts = snapshots/'scripts'
    scripts.mkdir(parents=True)
    source_scripts = Path(__file__).resolve().parent
    for name in MODULES:
        shutil.copy2(source_scripts/name, scripts/name)
    shutil.copy2(args.routing, snapshots/'routing.json')
    shutil.copy2(args.captions, snapshots/'captions.json')
    binding_path = None
    if args.binding_adapter:
        binding_dir = snapshots/'material_binding'
        binding_dir.mkdir()
        binding_path = binding_dir/args.binding_adapter.name
        shutil.copy2(args.binding_adapter, binding_path)
        shutil.copy2(args.binding_adapter.parent/'frozen_config.json', binding_dir/'frozen_config.json')
    for key in MANIFESTS:
        (output/'guides'/key).mkdir(parents=True)
    cases = []
    for row, guides in validated:
        for key, folder in guides.items():
            (output/'guides'/key/row['case_id']).symlink_to(folder, target_is_directory=True)
        cases.append(dict(case_id=row['case_id'], index=row['index'], route=row['route'],
            source_guides={k: str(v) for k, v in guides.items()},
            frozen_edit_mask_sha256=sha(guides['boundary']/'inpaint_mask.png'),
            source_sha256=sha(guides['boundary']/'source.png'),
            resolved_geometry_case=row.get('resolved_geometry_case'),
            semantic_target=captions[row['case_id']]['component_label']))
    settings = dict(base_cfg=args.base_cfg, refinement_cfg=args.cfg, head_control='depth',
        source_mode=args.source_mode, cavity_seam=args.cavity_seam, seed=args.seed,
        refinement_stages=args.refinement_stages,
        steps=args.steps, resolution=args.resolution, control_scale=args.control_scale,
        base_variant='owned_anchors', refinement_variant='production', selection=args.selection,
        binding_adapter=str(binding_path) if binding_path else None,
        generated_food_pixels_allowed=True, per_case_selection=False)
    shards = []
    for shard_id in range(args.shards):
        indices = [c['index'] for c in cases[shard_id::args.shards]]
        common = ['--output', str(output), '--indices', *map(str, indices),
            '--steps', str(args.steps), '--resolution', str(args.resolution),
            '--seed', str(args.seed), '--control-scale', str(args.control_scale)]
        base = [str(args.python), str(scripts/'run_mld4_reconstruction.py'),
            '--guides', str(output/'guides/anchored'), '--variant', 'owned_anchors',
            '--cfg', str(args.base_cfg), '--worker-name', f'base_shard_{shard_id}', *common]
        refinement = [str(args.python), str(scripts/'run_mld4_factorized.py'),
            '--guides', str(output/'guides/depth'), '--material-guides', str(output/'guides/boundary'),
            '--variant', 'production', '--base-variant', 'owned_anchors', '--cfg', str(args.cfg),
            '--head-control', 'depth', '--source-mode', args.source_mode,
            '--cavity-seam', str(args.cavity_seam), '--captions', str(snapshots/'captions.json'),
            '--worker-name', f'refine_shard_{shard_id}', '--stages', *args.refinement_stages, *common]
        if binding_path:
            refinement.extend(['--binding-adapter', str(binding_path)])
        shards.append(dict(shard_id=shard_id, indices=indices, commands=[base, refinement]))
    snapshot_hashes = {str(p): sha(p) for p in sorted(snapshots.rglob('*')) if p.is_file()}
    protocol = dict(version='mld4_uniform_routed_inference_v1', prepared_utc=utc(),
        status_at_freeze='prepared_not_executed', output=str(output), python=str(args.python),
        settings=settings, cases=cases, shards=shards, input_hashes=input_hashes,
        snapshot_hashes=snapshot_hashes, source_routing_sha256=sha(args.routing),
        source_captions_sha256=sha(args.captions),
        interpretation='Generated appearance is a hypothesis; source correspondence and frozen exterior are audited, not real paired ground truth.')
    write(output/'protocol.json', protocol)
    (output/'protocol.sha256').write_text(sha(output/'protocol.json')+'\n', encoding='ascii')
    print(json.dumps(dict(protocol=str(output/'protocol.json'), cases=len(cases), shards=args.shards,
        sha256=sha(output/'protocol.json'), model_loaded=False)))


def verify_hashes(hashes):
    for name, digest in hashes.items():
        if sha(name) != digest:
            raise ValueError(f'Frozen execution input changed: {name}')


def finalize_case(protocol, case):
    import numpy as np
    root = Path(protocol['output'])/'real'/case['case_id']
    guide = Path(case['source_guides']['boundary'])
    variant = protocol['settings']['refinement_variant']
    selected = root/'candidates'/variant/(protocol['settings']['selection']+'.png')
    source = pixels(guide/'source.png')
    final = pixels(selected)
    edit = pixels(guide/'inpaint_mask.png', 'L') > 127
    if final.shape != source.shape or not np.array_equal(final[~edit], source[~edit]):
        raise ValueError(f'Final changes frozen source exterior: {case["case_id"]}')
    assets = {'source.png': guide/'source.png', 'guide.png': guide/'coarse.png',
        'final.png': selected, 'edit_mask.png': guide/'inpaint_mask.png',
        'moved_food_mask.png': guide/'target_food_mask.png', 'spoon_mask.png': guide/'spoon_mask.png',
        'source_removed_mask.png': guide/'source_removal_mask.png'}
    if any((root/name).exists() for name in [*assets, 'reform_state.json']):
        raise FileExistsError(f'Final delivery already exists: {root}')
    for name, path in assets.items():
        shutil.copy2(path, root/name)
    food = pixels(guide/'target_food_mask.png', 'L') > 127
    baseline = Path(case['resolved_geometry_case'])/'final.png' if case.get('resolved_geometry_case') else None
    state = dict(case_id=case['case_id'], selected_variant=variant,
        selected_candidate=protocol['settings']['selection'], selection_is_uniform=True,
        protocol_sha256=sha(Path(protocol['output'])/'protocol.json'),
        frozen_edit_mask_sha256=case['frozen_edit_mask_sha256'], source_sha256=case['source_sha256'],
        baseline_final_path=str(baseline) if baseline and baseline.exists() else None,
        semantic_target=case['semantic_target'], generated_food_pixels=int((np.any(final != source, axis=2)&food).sum()),
        changed_outside_edit_pixels=0, timestamp=utc(),
        artifact_sha256={name: sha(root/name) for name in assets},
        material_identity_verified=False, photorealism_verified=False)
    write(root/'reform_state.json', state)
    return state


def run(args):
    path = args.protocol.resolve()
    if sha(path) != path.with_suffix('.sha256').read_text(encoding='ascii').strip():
        raise ValueError('Protocol hash mismatch')
    protocol = read(path)
    root = Path(protocol['output'])
    own_snapshot = root/'snapshot/scripts/run_mld4_pipeline.py'
    if Path(__file__).resolve() != own_snapshot.resolve():
        raise ValueError(f'Execute the frozen runner: {own_snapshot}')
    shard = next(s for s in protocol['shards'] if s['shard_id'] == args.shard_id)
    cases = [c for c in protocol['cases'] if c['index'] in shard['indices']]
    if any((root/'real'/c['case_id']/'candidates').exists() for c in cases):
        raise FileExistsError('A selected case already has candidates; use a fresh output to retain all attempts')
    verify_hashes(protocol['snapshot_hashes'])
    verify_hashes(protocol['input_hashes'])
    status_path = root/f'shard_{args.shard_id}_status.json'
    with status_path.open('x', encoding='utf-8') as stream:
        json.dump(dict(status='starting', gpu=args.gpu, started_utc=utc()), stream)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONDONTWRITEBYTECODE='1')
    completed = []
    try:
        for stage, command in zip(['base', 'refinement'], shard['commands']):
            verify_hashes(protocol['snapshot_hashes'])
            verify_hashes(protocol['input_hashes'])
            write(status_path, dict(status='running', stage=stage, gpu=args.gpu, completed=completed, updated_utc=utc()))
            with (root/f'shard_{args.shard_id}_{stage}.log').open('x', encoding='utf-8') as log:
                subprocess.run(command, env=env, cwd=own_snapshot.parent, stdout=log, stderr=subprocess.STDOUT, check=True)
            completed.append(stage)
        verify_hashes(protocol['input_hashes'])
        finals = [finalize_case(protocol, case) for case in cases]
        write(status_path, dict(status='complete', gpu=args.gpu, completed=completed,
            cases=[f['case_id'] for f in finals], finished_utc=utc(), protocol_sha256=sha(path)))
    except Exception as error:
        write(status_path, dict(status='failed', gpu=args.gpu, completed=completed,
            error=f'{type(error).__name__}: {error}', finished_utc=utc()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare', help='Validate inputs and freeze a new protocol; no model execution')
    p.add_argument('--routing', type=Path, required=True)
    p.add_argument('--captions', type=Path, required=True)
    p.add_argument('--binding-adapter', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--python', type=Path, default=Path(sys.executable))
    p.add_argument('--indices', nargs='+', type=int)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--base-cfg', type=float, default=3.)
    p.add_argument('--cfg', type=float, default=1.)
    p.add_argument('--source-mode', choices=['reconstruct', 'seam', 'inpaint', 'cavity'], default='cavity')
    p.add_argument('--refinement-stages', nargs='+', choices=['head', 'source'], default=['head', 'source'])
    p.add_argument('--cavity-seam', type=float, default=1.)
    p.add_argument('--seed', type=int, default=41)
    p.add_argument('--steps', type=int, default=40)
    p.add_argument('--resolution', type=int, default=1024)
    p.add_argument('--control-scale', type=float, default=.7)
    p.add_argument('--selection', choices=['unprojected', 'projected'], default='unprojected')
    p.set_defaults(function=prepare)
    p = sub.add_parser('run', help='Execute exactly one frozen shard on its assigned GPU')
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--shard-id', type=int, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.set_defaults(function=run)
    args = parser.parse_args()
    args.function(args)


if __name__ == '__main__':
    main()
