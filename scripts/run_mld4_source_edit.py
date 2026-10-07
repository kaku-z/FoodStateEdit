"""Frozen source-only Qwen Edit probe, accepting exactly the removed material footprint."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def rgb(path):
    return np.asarray(Image.open(path).convert('RGB')).copy()


def mask(path):
    return np.asarray(Image.open(path).convert('L')) > 127


def prepare(a):
    bundle = a.output / 'source_edit_probe' / ('frozen' if a.variant == 'source_edit_v1' else 'frozen_' + a.variant)
    bundle.mkdir(parents=True, exist_ok=False)
    captions = json.loads(a.captions.read_text(encoding='utf-8'))
    audit = json.loads(a.reference_audit.read_text(encoding='utf-8'))
    cases = []
    for index in a.indices:
        folder, = a.guides.glob(f'real_{index:02d}_*')
        case_id = folder.name
        dest = bundle / case_id
        dest.mkdir()
        sm = json.loads((folder / 'source' / 'stage_manifest.json').read_text(encoding='utf-8'))
        x0, y0, x1, y1 = sm['bbox_xyxy']
        source = rgb(folder / 'full_source.png')
        crop = source[y0:y1, x0:x1]
        removed = mask(folder / 'source' / 'source_removal_mask.png')
        source_stage = mask(folder / 'source_stage_full_mask.png')
        head_stage = mask(folder / 'head_stage_full_mask.png')
        full_removed = np.zeros(source.shape[:2], dtype=bool)
        full_removed[y0:y1, x0:x1] = removed
        assert not np.any(full_removed & ~source_stage)
        assert not np.any(full_removed & head_stage)
        annotated = crop.copy()
        annotated[removed] = np.rint(.25 * crop[removed] + .75 * np.array([255, 0, 255])).astype(np.uint8)
        for name, arr in [('source_crop.png', crop), ('selection_annotation.png', annotated),
                          ('acceptance_mask.png', removed.astype(np.uint8) * 255)]:
            Image.fromarray(arr).save(dest / name)
        input_order = ['source_crop.png', 'selection_annotation.png']
        if a.initialization == 'neutral_hole':
            incomplete = crop.copy()
            incomplete[removed] = 127
            Image.fromarray(incomplete).save(dest / 'missing_surface_input.png')
            input_order = ['missing_surface_input.png', 'source_crop.png']
        copies = {'full_source.png': folder / 'full_source.png',
                  'source_stage_full_mask.png': folder / 'source_stage_full_mask.png',
                  'head_stage_full_mask.png': folder / 'head_stage_full_mask.png',
                  'full_inpaint_mask.png': folder / 'full_inpaint_mask.png',
                  'base.png': a.output / 'real' / case_id / 'candidates' / a.base_variant / 'unprojected.png'}
        for name, path in copies.items():
            shutil.copy2(path, dest / name)
        caption = captions[case_id]
        surface = caption['source_surface']
        prompt = (
            'Edit IMAGE 1, the original close-up food photograph. IMAGE 2 is the same photograph with '
            'a translucent magenta selection marking the exact small portion to remove. '
            'Remove only this selected portion from IMAGE 1. Reveal the existing surface or remaining '
            'material immediately underneath it, continuously matching the adjacent visible scene. '
            'The selected portion is now absent. Preserve every surrounding unselected food item, '
            'its outline, texture, position, lighting and camera view. '
            + surface + ' Return a single natural photograph matching IMAGE 1 with this local removal. '
            'The magenta color belongs only to the selection annotation in IMAGE 2.')
        if a.initialization == 'neutral_hole':
            prompt = (
                'Refine IMAGE 1 into a natural food photograph. A small portion has already been removed; '
                'the neutral gray region marks the missing surface revealed by this removal. '
                'Fill only this missing region with the underlying surface, preserving its exact position and extent. '
                + surface + ' Keep the removed portion absent. IMAGE 2 is a reference only for the unchanged '
                'surrounding appearance and lighting; do not restore its intact food arrangement. '
                'Preserve all visible unselected food and the camera view from IMAGE 1.')
        size = [max(256, round(v * 1024 / max(x1-x0, y1-y0) / 32) * 32)
                for v in [x1-x0, y1-y0]]
        request = dict(case_id=case_id, bbox_xyxy=sm['bbox_xyxy'], canvas=size, prompt=prompt,
                       source_surface_caption=surface, semantic_status=caption.get('ground_truth_verified', False),
                       component_label=caption.get('component_label'), base_variant=a.base_variant,
                       exact_removed_pixels=int(removed.sum()), seam_native_pixels=0,
                       input_order=input_order, initialization=a.initialization, variant=a.variant,
                       inputs={p.name: sha(p) for p in dest.iterdir()},
                       guide_factorization_sha256=sha(folder / 'factorization.json'))
        write(dest / 'request.json', request)
        cases.append({'case_id': case_id, 'request_sha256': sha(dest / 'request.json')})
    config = dict(schema='mld4_source_edit_v1', variant=a.variant, initialization=a.initialization,
                  frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  runner_sha256=sha(__file__), captions_sha256=sha(a.captions),
                  reference_audit_sha256=sha(a.reference_audit), model_root=a.model,
                  expected_pipeline_sha256=audit['expected_pipeline_sha256'],
                  weight_files=audit['backend']['weight_files'], cases=cases,
                  inference=dict(steps=40, seed=41, true_cfg_scale=4., guidance_scale=None,
                                 negative_prompt=audit['inference']['negative_prompt']),
                  compositing='Reset the entire previous source stage to original RGB, then accept raw generated RGB only inside the exact source_removed mask. No feathering, dilation, seam pixels or material projection. Preserve the previous head and handle exactly.',
                  claim_limit='Unpaired diagnostic source-cavity experiment; selected material and hidden surface are not independently measured ground truth.')
    write(bundle / 'config.json', config)
    print(json.dumps({'bundle': str(bundle), 'config_sha256': sha(bundle / 'config.json'), 'cases': len(cases)}))


def run(a):
    bundle = a.output / 'source_edit_probe' / ('frozen' if a.variant == 'source_edit_v1' else 'frozen_' + a.variant)
    c = json.loads((bundle / 'config.json').read_text(encoding='utf-8'))
    assert c['runner_sha256'] == sha(__file__)
    state_path = a.output / 'source_edit_probe' / ('worker.json' if a.variant == 'source_edit_v1' else a.variant + '_worker.json')
    state = dict(status='verify', started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 config_sha256=sha(bundle / 'config.json'), physical_gpus=[6, 7], completed=[])
    write(state_path, state)
    model = Path(c['model_root'])
    weights = {}
    for rel, expected in c['weight_files'].items():
        path = model / rel
        weights[rel] = dict(size_matches=path.stat().st_size == expected['size_bytes'],
                            sha256=sha(path), expected_sha256=expected['sha256'])
        assert weights[rel]['size_matches'] and weights[rel]['sha256'] == expected['sha256']
    write(a.output / 'source_edit_probe' / (a.variant + '_weights_verified.json'), weights)
    os.environ.update(CUDA_VISIBLE_DEVICES='6,7', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false', TORCH_COMPILE_DISABLE='1', OMP_NUM_THREADS='4',
                      PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    import torch
    import diffusers
    from diffusers import QwenImageEditPlusPipeline
    torch.set_num_threads(4)
    pipeline_sha = sha(inspect.getfile(QwenImageEditPlusPipeline))
    assert pipeline_sha == c['expected_pipeline_sha256']
    state.update(status='loading', torch_version=torch.__version__, diffusers_version=diffusers.__version__, pipeline_sha256=pipeline_sha)
    write(state_path, state)
    pipe = QwenImageEditPlusPipeline.from_pretrained(str(model), torch_dtype=torch.bfloat16,
              local_files_only=True, device_map='balanced', max_memory={0:'44GiB',1:'44GiB'})
    state['device_map'] = pipe.hf_device_map
    def progress(pipe, step, timestep, kwargs):
        state.update(step=int(step), updated_unix=time.time())
        write(state_path, state)
        return kwargs
    inf = c['inference']
    for case in c['cases']:
        f = bundle / case['case_id']
        assert sha(f / 'request.json') == case['request_sha256']
        req = json.loads((f / 'request.json').read_text(encoding='utf-8'))
        for name, digest in req['inputs'].items():
            assert sha(f / name) == digest
        out = a.output / 'real' / case['case_id'] / 'candidates' / a.variant
        out.mkdir(parents=True, exist_ok=False)
        shutil.copy2(f / 'request.json', out / 'request.json')
        state.update(status='generating', current=case['case_id'], step=-1)
        write(state_path, state)
        start = time.perf_counter()
        with torch.inference_mode():
            result = pipe(image=[Image.open(f / n).convert('RGB') for n in req['input_order']],
                          prompt=req['prompt'], negative_prompt=inf['negative_prompt'],
                          true_cfg_scale=inf['true_cfg_scale'], guidance_scale=inf['guidance_scale'],
                          num_inference_steps=inf['steps'], width=req['canvas'][0], height=req['canvas'][1],
                          generator=torch.Generator(device=pipe._execution_device).manual_seed(inf['seed']),
                          callback_on_step_end=progress).images[0]
        result.save(out / 'raw.png')
        x0,y0,x1,y1 = req['bbox_xyxy']
        native = np.asarray(result.convert('RGB').resize((x1-x0,y1-y0), Image.Resampling.LANCZOS))
        Image.fromarray(native).save(out / 'raw_native.png')
        source, base = rgb(f / 'full_source.png'), rgb(f / 'base.png')
        full = base.copy()
        source_stage = mask(f / 'source_stage_full_mask.png')
        full[source_stage] = source[source_stage]
        removed = mask(f / 'acceptance_mask.png')
        full[y0:y1,x0:x1][removed] = native[removed]
        head = mask(f / 'head_stage_full_mask.png')
        editable = mask(f / 'full_inpaint_mask.png')
        kept_source = source_stage.copy()
        kept_source[y0:y1,x0:x1] &= ~removed
        assert np.array_equal(full[head], base[head])
        assert np.array_equal(full[kept_source], source[kept_source])
        assert np.array_equal(full[~editable], source[~editable])
        for name in ['candidate.png', 'unprojected.png']:
            Image.fromarray(full).save(out / name)
        row = dict(case_id=case['case_id'], seconds=time.perf_counter()-start,
                   raw_sha256=sha(out / 'raw.png'), final_sha256=sha(out / 'unprojected.png'),
                   accepted_generated_pixels=int(removed.sum()), seam_pixels=0,
                   head_changed_pixels=int(np.any(full!=base,axis=2)[head].sum()),
                   retained_source_changed_pixels=int(np.any(full!=source,axis=2)[kept_source].sum()),
                   exterior_changed_pixels=int(np.any(full!=source,axis=2)[~editable].sum()),
                   generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), reviewed=False)
        write(out / 'generation.json', row)
        state['completed'].append(row)
        write(state_path, state)
        print(json.dumps(row), flush=True)
    state.update(status='complete_unreviewed')
    write(state_path, state)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare','run'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--guides', type=Path)
    p.add_argument('--captions', type=Path)
    p.add_argument('--reference-audit', type=Path)
    p.add_argument('--indices', type=int, nargs='+', default=[2,13,14])
    p.add_argument('--base-variant', default='factorized_auto_depth')
    p.add_argument('--variant', default='source_edit_v1')
    p.add_argument('--initialization', choices=['annotation','neutral_hole'], default='annotation')
    p.add_argument('--model', default='/mnt/tmp/guo-z_first_bite_20260929_models/qwen-image-edit-2511')
    a = p.parse_args()
    (prepare if a.action == 'prepare' else run)(a)
