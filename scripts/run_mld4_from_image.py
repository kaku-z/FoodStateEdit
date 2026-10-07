"""One RGB (+ optional coarse food text) to a frozen MLD4 head-only result.

Run in the installed geometry Python environment. Every invocation owns a fresh
output directory; model processes run sequentially on one physical GPU. No
generated image is used for source semantics, routing, or case selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


QWEN_PYTHON = '/mnt/tmp/guo-z_first_bite_structure_20260930_runtime/venv/bin/python'
CHECKPOINT = '/host/space0/guo-z/tf-ufi/material_lineage_full_20261004/runs_continuous/shared_s41/inference_checkpoint.pt'
GLM_ROOT = '/host/space0/guo-z/FluxSAM-Seg/models/GLM-4.6V-Flash'
SCRIPTS = [
    'run_mld4_from_image.py', 'run_mld3.py', 'mld2_real_geometry.py',
    'mld2_real_geometry_v2.py', 'mld2_real_spoon_v3.py', 'prepare_mld3_ingredients.py',
    'prepare_mld4_guides.py', 'prepare_mld4_anchors.py', 'prepare_mld4_factorized.py',
    'prepare_mld4_dense_guides.py', 'prepare_mld4_head_depth.py',
    'diagnose_mld4_semantics.py', 'compile_mld4_captions.py', 'prepare_mld4_routing.py',
    'run_mld4_pipeline.py', 'run_mld4_reconstruction.py', 'run_mld4_factorized.py',
    'probe_mld3_metal_conditioning.py', 'refine_mld4_source_seam.py',
    'mld4_latent_transport.py', 'mld4_binding_adapter.py',
    'project_mld4_visible_material.py', 'probe_mld4_locked_material.py',
]
CASE = 'real_00_input'
SOURCE_FILES = ['source.png', 'source_reference.png', 'source_reference_mask.png', 'guide_manifest.json']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def utc():
    return datetime.now(timezone.utc).isoformat()


def freeze(args):
    args.output.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).resolve().parents[1]
    snapshot = args.output / 'snapshot'
    (snapshot / 'scripts').mkdir(parents=True)
    for name in SCRIPTS:
        shutil.copy2(code / 'scripts' / name, snapshot / 'scripts' / name)
    for source in (code / 'foodstateedit').rglob('*.py'):
        target = snapshot / source.relative_to(code)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    source = args.output / 'input' / ('original' + args.image.suffix.lower())
    source.parent.mkdir()
    shutil.copy2(args.image, source)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write(args.output / 'input_protocol.json', dict(
        version='mld4_from_image_v3', created_utc=utc(), original_image=str(args.image),
        frozen_image=str(source), input_sha256=sha(source), settings=settings,
        code_sha256={str(p.relative_to(snapshot)): sha(p) for p in sorted(snapshot.rglob('*.py'))},
        checkpoint_sha256=sha(args.checkpoint),
        models=dict(SAM3='/host/space0/guo-z/Evol-SAM3/sam3/sam3.pt',
                    MoGe2='/host/space0/guo-z/tf-ufi/food3d_repair_20260928/models/model.pt',
                    MLD2=str(args.checkpoint), GLM=str(args.glm_root),
                    Qwen_Image_2_1='/mnt/tmp/guo-z_first_bite_structure_20260930_models/qwen-image-2.1'),
        fresh_perception=True, image_selection=False, manual_case_override=False,
        routing_policy='Initial auto mode; original strand without paired noodle agreement becomes one cohesive packet; recaption once without rerouting.',
        generated_target_feedback=False, refinement_stages=['head'], binding_adapter=None,
        handle_extension_integrated=not args.legacy_handle,
        observed_material_projection=dict(enabled=not args.free_material, confidence_threshold=.75,
            core_distance_pixels=2, scalar_gain_bounds=[.72, 1.4],
            source='Transported observed source RGB; no generated chromatic content in bound core'),
        final_visual_success='unverified'))
    command = [sys.executable, '-u', str(snapshot / 'scripts' / Path(__file__).name),
               '--execute-snapshot', '--image', str(source), '--output', str(args.output),
               '--food-prompt', args.food_prompt, '--checkpoint', str(args.checkpoint),
               '--glm-root', str(args.glm_root), '--qwen-python', str(args.qwen_python),
               '--vlm-python', str(args.vlm_python), '--gpu', str(args.gpu),
               '--steps', str(args.steps), '--seed', str(args.seed),
               '--resolution', str(args.resolution), '--iterations', str(args.iterations)]
    if args.legacy_handle:
        command.append('--legacy-handle')
    if args.free_material:
        command.append('--free-material')
    subprocess.run(command, check=True)


class Execution:
    def __init__(self, args):
        self.args = args
        self.root = args.output
        self.scripts = Path(__file__).resolve().parent
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
                        EGL_DEVICE_ID=str(args.gpu), PYOPENGL_PLATFORM='egl',
                        OMP_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
        self.status = dict(status='running', started_utc=utc(), physical_gpu=args.gpu, stages=[])
        (self.root / 'logs').mkdir()
        self.save()

    def save(self):
        write(self.root / 'status.json', self.status)

    def call(self, name, script, *arguments, python=None):
        command = [str(python or sys.executable), '-u', str(self.scripts / script), *map(str, arguments)]
        if script == 'run_mld3.py':
            command = [str(python or sys.executable), '-u', str(self.scripts / Path(__file__).name),
                       '_geometry', str(self.args.seed), *map(str, arguments)]
            if not self.args.legacy_handle:
                command.append('--edge-connected-handle')
        log = self.root / 'logs' / (name + '.log')
        record = dict(name=name, command=command, log=str(log), status='running', started_utc=utc())
        self.status['stages'].append(record)
        self.status['stage'] = name
        self.save()
        print(name, flush=True)
        start = time.monotonic()
        with log.open('x', encoding='utf-8') as stream:
            result = subprocess.run(command, env=self.env, cwd=self.scripts, stdout=stream, stderr=subprocess.STDOUT)
        record.update(returncode=result.returncode, seconds=time.monotonic()-start,
                      status='complete' if result.returncode == 0 else 'failed', finished_utc=utc())
        self.save()
        result.check_returncode()

    def boundary(self, geometry, destination, prefix):
        self.call(prefix + '_boundary', 'prepare_mld4_guides.py', '--input-root', geometry,
                  '--output', destination, '--source-shape-control')

    def describe(self, guides, destination, prefix):
        self.call(prefix + '_source_descriptions', 'diagnose_mld4_semantics.py', '--root', guides,
                  '--output', destination / 'queries', '--indices', 0, '--mode', 'context_material',
                  '--model-root', self.args.glm_root, python=self.args.vlm_python)
        self.call(prefix + '_source_measurements', 'diagnose_mld4_semantics.py', '--root', guides,
                  '--output', destination / 'measurements', '--indices', 0, '--mode', 'measure')
        folder = guides / CASE
        provenance = destination / 'queries' / CASE / 'input_provenance.json'
        # Context preparation verified exact source RGB ownership; the compiler
        # repeats its source hashes and measured-color checks before publication.
        manifest = dict(expected_cases=1, source_only=True, cases=[dict(
            case_id=CASE,
            reply_paths=[str(destination / 'queries' / CASE / ('reply_' + t + '.json')) for t in ('A', 'B')],
            source_input_provenance=str(provenance),
            measured_material_path=str(destination / 'measurements' / CASE / 'measured_material.json'),
            resolved_guide_folder=str(folder), resolved_guide_sha256={name: sha(folder / name) for name in SOURCE_FILES},
            ownership_equivalence=dict(exact_source_RGB_and_full_ownership_mask=True,
                                       reason='The query and resolved guide use the identical directory and byte-verified source artifacts.'))])
        write(destination / 'pairs_manifest.json', manifest)
        captions = destination / 'captions.json'
        self.call(prefix + '_compile_captions', 'compile_mld4_captions.py', '--manifest',
                  destination / 'pairs_manifest.json', '--output', captions)
        return captions

    def execute(self):
        args = self.args
        frozen = read(self.root / 'input_protocol.json')
        if sha(args.image) != frozen['input_sha256']:
            raise ValueError('Frozen source input changed')
        for name, digest in frozen['code_sha256'].items():
            if sha(self.root / 'snapshot' / name) != digest:
                raise ValueError('Frozen code changed: ' + name)
        geometry = self.root / 'geometry_initial'
        self.call('fresh_perception_geometry', 'run_mld3.py', '--image', args.image,
                  '--food-prompt', args.food_prompt, '--output', geometry,
                  '--checkpoint', args.checkpoint, '--iterations', args.iterations)
        boundary = self.root / 'guides_initial_boundary'
        self.boundary(geometry, boundary, 'initial')
        captions = self.describe(boundary, self.root / 'semantics_initial', 'initial')
        route_path = self.root / 'routing_initial.json'
        self.call('freeze_source_routing', 'prepare_mld4_routing.py', '--captions', captions,
                  '--guide-roots', boundary, '--output', route_path)
        route = read(route_path)
        row = route['cases'][0]
        if row['needs_packet_build']:
            geometry = self.root / 'geometry_packet'
            self.call('source_owned_cohesive_packet', 'run_mld3.py', '--input-root',
                      self.root / 'geometry_initial' / 'observations', '--output', geometry,
                      '--checkpoint', args.checkpoint, '--iterations', args.iterations,
                      '--material-mode', 'cohesive')
            from prepare_mld4_routing import packet_status
            row['selected_packet'] = packet_status(geometry / 'real' / CASE, row['source_sha256'])
            if row['selected_packet'] is None:
                raise ValueError('Routed packet failed topology/material-ID/finite-geometry checks')
            boundary = self.root / 'guides_packet_boundary'
            self.boundary(geometry, boundary, 'packet')
            captions = self.describe(boundary, self.root / 'semantics_packet', 'packet')
            row['needs_packet_build'] = False
        anchored = self.root / 'guides_anchored'
        factorized = self.root / 'guides_factorized'
        dense = self.root / 'guides_dense'
        depth = self.root / 'guides_depth'
        for name, script, parent, target in [
            ('anchors', 'prepare_mld4_anchors.py', boundary, anchored),
            ('factorization', 'prepare_mld4_factorized.py', anchored, factorized),
            ('dense_core', 'prepare_mld4_dense_guides.py', factorized, dense),
            ('depth_control', 'prepare_mld4_head_depth.py', dense, depth),
        ]:
            extra = ['--resolution', args.resolution] if name == 'factorization' else []
            self.call(name, script, '--input-root', parent, '--output', target, *extra)
        row.update(resolved_geometry_case=str(geometry / 'real' / CASE),
                   production_guides={key: str(path / CASE) for key, path in
                                      [('boundary', boundary), ('anchored', anchored), ('depth', depth)]},
                   production_guide_manifest_hashes=dict(
                       boundary=sha(boundary / CASE / 'guide_manifest.json'),
                       anchored=sha(anchored / CASE / 'guide_manifest.json'),
                       depth=sha(depth / CASE / 'depth_protocol.json')),
                   source_ownership_recheck_completed=True)
        route.update(resolved_captions=str(captions), resolved_captions_sha256=sha(captions),
                     initial_routing_sha256=sha(route_path), source_recaption_did_not_change_route=True)
        resolved = self.root / 'routing_resolved_manifest.json'
        write(resolved, route)
        production = self.root / 'production'
        self.call('freeze_render_protocol', 'run_mld4_pipeline.py', 'prepare',
                  '--routing', resolved, '--captions', captions, '--output', production,
                  '--python', args.qwen_python, '--shards', 1, '--base-cfg', 3, '--cfg', 1,
                  '--refinement-stages', 'head', '--source-mode', 'cavity', '--cavity-seam', 0,
                  '--steps', args.steps, '--seed', args.seed, '--resolution', args.resolution,
                  '--control-scale', .7, '--selection', 'unprojected')
        self.call('full_frame_and_head', str(production / 'snapshot' / 'scripts' / 'run_mld4_pipeline.py'),
                  'run', '--protocol', production / 'protocol.json', '--shard-id', 0,
                  '--gpu', args.gpu, python=args.qwen_python)
        generative = production / 'real' / CASE / 'final.png'
        final = generative
        material = None
        if not args.free_material:
            bound = self.root / 'material_bound'
            self.call('observed_material_projection', 'project_mld4_visible_material.py',
                      '--production', production, '--output', bound)
            final = bound / 'real' / CASE / 'final.png'
            material = read(bound / 'real' / CASE / 'result.json')
        import numpy as np
        from PIL import Image
        source = np.asarray(Image.open(production / 'real' / CASE / 'source.png').convert('RGB'))
        pixels = np.asarray(Image.open(final).convert('RGB'))
        editable = np.asarray(Image.open(production / 'real' / CASE / 'edit_mask.png').convert('L')) > 127
        changed_outside = int(np.any(pixels != source, axis=2)[~editable].sum())
        if changed_outside:
            raise ValueError('Final material projection changed the frozen source exterior')
        self.status.update(status='complete', stage='complete', finished_utc=utc(),
                           final=str(final), final_sha256=sha(final),
                           generative_A_final=str(generative), observed_material_contract=material,
                           changed_outside_frozen_edit_pixels=changed_outside,
                           production_protocol_sha256=sha(production / 'protocol.json'),
                           fresh_perception=True, initial_mode=row['original_mode'], route=row['route'],
                           source_only_routing=True,
                           output_selection='uniform_unprojected_A' if args.free_material else 'uniform_observed_material_B',
                           visual_success='unverified_requires_review')
        self.save()
        print(json.dumps(self.status, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--food-prompt', default='food')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--checkpoint', type=Path, default=Path(CHECKPOINT))
    parser.add_argument('--glm-root', type=Path, default=Path(GLM_ROOT))
    parser.add_argument('--qwen-python', type=Path, default=Path(QWEN_PYTHON))
    parser.add_argument('--vlm-python', type=Path, default=Path(QWEN_PYTHON))
    parser.add_argument('--iterations', type=int, default=180)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--legacy-handle', action='store_true',
                        help='Retain the old shaft that may end inside the image; default extends only the distal shaft beyond the frame.')
    parser.add_argument('--free-material', action='store_true',
                        help='Return generative A without observed-material projection; default returns source-bound B.')
    parser.add_argument('--execute-snapshot', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.image = args.image.resolve()
    args.output = args.output.resolve()
    if not args.execute_snapshot:
        freeze(args)
        return
    execution = Execution(args)
    try:
        execution.execute()
    except Exception as error:
        execution.status.update(status='failed', error=f'{type(error).__name__}: {error}', finished_utc=utc())
        execution.save()
        raise


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '_geometry':
        import runpy
        import numpy as np
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(int(sys.argv[2]))
        np.random.seed(int(sys.argv[2]))
        script = Path(__file__).resolve().parent / 'run_mld3.py'
        sys.argv = [str(script), *sys.argv[3:]]
        runpy.run_path(str(script), run_name='__main__')
    else:
        main()
