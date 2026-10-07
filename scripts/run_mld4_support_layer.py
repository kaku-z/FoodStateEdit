"""Source-layer ablation: inpaint a whole SAM component, accept only its bite.

The SAM mask and its complement are hypotheses, never true food/background
labels. The broad completion is diagnostic; only the exact removed-material
footprint can alter the final source region.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image


VARIANT = 'source_support_layer_v1'
NEGATIVE = 'CGI, polygon, plastic, flat pasted patch, invented toppings, hand, person, fingers, face, duplicate bite, floating food, deep cup'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, record):
    Path(path).write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def rgb(path):
    return np.asarray(Image.open(path).convert('RGB'))


def mask(path):
    return np.asarray(Image.open(path).convert('L')) > 127


def prepare(args):
    root = args.root.resolve()
    probe = root/args.probe_name
    if (probe/'protocol.json').exists():
        raise FileExistsError('Keep the frozen support-layer probe unchanged')
    captions = read(args.captions)
    cases = sorted(p for p in args.geometry.glob('real/real_*') if int(p.name.split('_')[1]) in args.indices)
    if {int(p.name.split('_')[1]) for p in cases} != set(args.indices):
        raise ValueError('Missing selected source case')
    probe.mkdir(parents=True, exist_ok=True)
    scripts = probe/'snapshot/scripts'
    scripts.mkdir(parents=True, exist_ok=True)
    for name in ['run_mld4_support_layer.py', 'probe_mld3_metal_conditioning.py']:
        shutil.copy2(Path(__file__).parent/name, scripts/name)
    shutil.copy2(args.captions, probe/'snapshot/captions.json')
    requests = []
    for source_case in cases:
        case = source_case.name
        guide = args.guides/case
        destination = root/'real'/case/'candidates'/args.variant
        if destination.exists():
            raise FileExistsError(f'Candidate already exists: {destination}')
        inputs = dict(source=source_case/'source.png', whole_food_mask=source_case/'source_food_mask.png',
            removed_mask=source_case/'source_removed_mask.png',
            base=root/'real'/case/'candidates/factorized_auto_depth/unprojected.png',
            source_stage_mask=guide/'source_stage_full_mask.png', head_stage_mask=guide/'head_stage_full_mask.png',
            edit_mask=guide/'full_inpaint_mask.png')
        source = rgb(inputs['source']); whole = mask(inputs['whole_food_mask']); removed = mask(inputs['removed_mask'])
        if not np.array_equal(source, rgb(guide/'full_source.png')):
            raise ValueError(f'Source mismatch: {case}')
        source_stage = mask(inputs['source_stage_mask']); head_stage = mask(inputs['head_stage_mask'])
        editable = mask(inputs['edit_mask'])
        if np.any(removed & ~whole) or np.any(removed & ~source_stage) or np.any(source_stage & head_stage):
            raise ValueError(f'Incompatible source/target masks: {case}')
        if np.any(removed & ~editable):
            raise ValueError(f'Removal outside frozen edit mask: {case}')
        folder = probe/case
        folder.mkdir()
        saved = {}
        for name, path in inputs.items():
            target = folder/(name+'.png'); shutil.copy2(path, target)
            saved[name] = dict(original_path=str(path), snapshot_path=str(target), sha256=sha(target))
        scale = args.resolution/max(source.shape[:2])
        width, height = [max(256, round(v*scale/32)*32) for v in [source.shape[1], source.shape[0]]]
        prompt = captions[case]['source_surface']
        request = dict(case_id=case, variant=args.variant, destination=str(destination), inputs=saved,
            prompt=prompt, caption_key='source_surface', captions_sha256=sha(args.captions),
            negative_prompt=NEGATIVE, source_only_prompt=True, manual_ingredient_names=False,
            full_original_scene_input=True, mask_rule='Exact E3 source_food_mask; no dilation, union, erosion, or relabeling',
            whole_food_mask_pixels=int(whole.sum()), exact_acceptance_pixels=int(removed.sum()),
            mask_complement_is_verified_background=False, hidden_support_is_ground_truth=False,
            final_rule='Keep factorized_auto_depth target; restore original source stage; accept generated support only on exact source_removed_mask',
            width=width, height=height, seed=41, steps=40, true_cfg_scale=1.,
            control_image=None, reference_images=None, control_context_scale=1.)
        write(folder/'request.json', request)
        requests.append(dict(case_id=case, request_path=str(folder/'request.json'), sha256=sha(folder/'request.json')))
    protocol = dict(method='whole_component_support_completion_exact_bite_acceptance',
        case_requests=requests, variant=args.variant, snapshot_sha256={str(p): sha(p) for p in sorted((probe/'snapshot').rglob('*')) if p.is_file()},
        model='Qwen-Image-2.1 + Fun ControlNet-Union inpainting path without control/reference',
        prepared_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), generated_targets_used_for_caption=False,
        whole_food_mask_manual_corrections=False, final_exterior_exact=True)
    write(probe/'protocol.json', protocol)
    (probe/'protocol.sha256').write_text(sha(probe/'protocol.json')+'\n', encoding='ascii')
    print(json.dumps(dict(protocol=str(probe/'protocol.json'), sha256=sha(probe/'protocol.json'), cases=len(requests))), flush=True)


def run(args):
    probe = args.root.resolve()/args.probe_name
    protocol_path = probe/'protocol.json'
    if sha(protocol_path) != (probe/'protocol.sha256').read_text().strip():
        raise ValueError('Protocol changed')
    protocol = read(protocol_path)
    for path, digest in protocol['snapshot_sha256'].items():
        if sha(path) != digest:
            raise ValueError(f'Snapshot changed: {path}')
    if Path(__file__).resolve() != (probe/'snapshot/scripts/run_mld4_support_layer.py').resolve():
        raise ValueError('Invoke the frozen snapshot script')
    requests = []
    for row in protocol['case_requests']:
        if sha(row['request_path']) != row['sha256']:
            raise ValueError('Request changed')
        request = read(row['request_path'])
        for asset in request['inputs'].values():
            if sha(asset['snapshot_path']) != asset['sha256']:
                raise ValueError('Input snapshot changed')
        if Path(request['destination']).exists():
            raise FileExistsError(request['destination'])
        requests.append(request)
    # Every request and input was frozen before loading any neural model.
    import torch
    from probe_mld3_metal_conditioning import load_pipeline, VENDOR
    pipe, adapter = load_pipeline()
    completed = []
    for request in requests:
        case = request['case_id']; folder = probe/case; out = Path(request['destination']); out.mkdir(parents=True)
        inputs = {key: Path(value['snapshot_path']) for key, value in request['inputs'].items()}
        source = rgb(inputs['source']); removed = mask(inputs['removed_mask'])
        base = rgb(inputs['base']); full = base.copy()
        source_stage = mask(inputs['source_stage_mask']); head_stage = mask(inputs['head_stage_mask']); editable = mask(inputs['edit_mask'])
        started = time.perf_counter()
        with torch.inference_mode():
            generated = pipe(prompt=request['prompt'], negative_prompt=request['negative_prompt'],
                image=Image.fromarray(source), mask_image=Image.open(inputs['whole_food_mask']).convert('RGB'),
                control_image=None, reference_images=None, control_context_scale=1., true_cfg_scale=1.,
                width=request['width'], height=request['height'], num_inference_steps=40,
                generator=torch.Generator('cuda').manual_seed(41), use_kv_cache=False).images[0]
        generated.save(folder/'generated.png')
        layer = np.asarray(Image.alpha_composite(Image.fromarray(source).convert('RGBA'),
            generated.convert('RGBA').resize((source.shape[1], source.shape[0]), Image.Resampling.LANCZOS)).convert('RGB'))
        Image.fromarray(layer).save(folder/'support_layer_native.png')
        full[source_stage] = source[source_stage]
        full[removed] = layer[removed]
        full[~editable] = source[~editable]
        assert np.array_equal(full[head_stage], base[head_stage])
        assert np.array_equal(full[source_stage & ~removed], source[source_stage & ~removed])
        assert np.array_equal(full[~editable], source[~editable])
        Image.fromarray(full).save(out/'unprojected.png')
        Image.fromarray(full).save(out/'candidate.png')
        shutil.copy2(folder/'request.json', out/'request.json')
        shutil.copy2(inputs['removed_mask'], out/'acceptance_mask.png')
        result = dict(case_id=case, seconds=time.perf_counter()-started, exact_removed_pixels=int(removed.sum()),
            changed_outside_frozen_edit_pixels=0, changed_retained_source_stage_pixels=0, changed_head_pixels=0,
            generated_sha256=sha(folder/'generated.png'), final_sha256=sha(out/'unprojected.png'),
            protocol_sha256=sha(protocol_path), request_sha256=sha(folder/'request.json'),
            runner_sha256=sha(__file__), vendor_sha256=sha(VENDOR/'videox_fun/pipeline/pipeline_qwenimage21_reference_control.py'),
            adapter=adapter, conditioning_audit=pipe.last_conditioning_audit,
            support_layer_realism_verified=False, true_background_inference_verified=False)
        write(out/'generation.json', result); completed.append(result)
        write(probe/'worker.json', dict(status='running', cases=completed))
        print(json.dumps(dict(case_id=case, seconds=result['seconds'], final_sha256=result['final_sha256'])), flush=True)
    write(probe/'worker.json', dict(status='complete', cases=completed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--probe-name', default='support_layer_probe'); p.add_argument('--variant', default=VARIANT)
    p.add_argument('--geometry', type=Path, required=True); p.add_argument('--guides', type=Path, required=True)
    p.add_argument('--captions', type=Path, required=True); p.add_argument('--indices', type=int, nargs='+', default=[2,13,14])
    p.add_argument('--resolution', type=int, default=1024); p.set_defaults(function=prepare)
    p = sub.add_parser('run'); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--probe-name', default='support_layer_probe'); p.set_defaults(function=run)
    args = parser.parse_args(); args.function(args)
