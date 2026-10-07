"""Complete an edge-connected utensil shaft on a finished first-bite photograph.

The geometry builder preserves the bowl and food. This pass accepts RGB only in
its shaft edit mask; the carried bite and source exposure remain byte-exact.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from run_mld4_reconstruction import mask, sha, write


PROMPT = (
    'A natural camera photograph of a small first bite of food on a stainless steel spoon. '
    'Complete only the narrow silver spoon handle indicated by the structural guide. '
    'The continuous handle joins the existing spoon neck and extends beyond the photograph boundary along the structural guide. '
    'Match the existing spoon metal, its thickness, perspective, soft reflections and lighting. '
    'Keep the food, spoon bowl, missing bite on the dish and entire scene unchanged. '
    'The hand holding this spoon is outside the photograph and is not visible. '
    'No person or hand appears in the photograph.')
NEGATIVE = 'hand, fingers, arm, person, face, duplicate utensil, disconnected handle, plastic, black rod, changed food'


def prepare_input(base, source, metal, old_shaft, new_shaft, edit):
    conditioning = base.copy()
    conditioning[old_shaft & edit] = source[old_shaft & edit]
    conditioning[new_shaft & edit] = metal[new_shaft & edit]
    return conditioning


def accept_shaft(base, candidate, edit):
    alpha = np.minimum(distance_transform_edt(edit) / 2., 1.)[..., None]
    result = np.rint(candidate * alpha + base * (1. - alpha)).clip(0, 255).astype(np.uint8)
    result[~edit] = base[~edit]
    return result


def rgb(path):
    return np.asarray(Image.open(path).convert('RGB'))


def run(args):
    import torch
    from probe_mld3_metal_conditioning import load_pipeline, VENDOR
    folders = sorted(args.geometry.glob('real_*'))
    if args.indices is not None:
        folders = [p for p in folders if int(p.name.split('_')[1]) in args.indices]
    args.output.mkdir(parents=True, exist_ok=True)
    pipe = None
    rows = []
    for folder in folders:
        out = args.output/'real'/folder.name
        out.mkdir(parents=True, exist_ok=False)
        basepath = args.production/'real'/folder.name/'final.png'
        base = rgb(basepath)
        source = rgb(folder/'source.png')
        old_shaft = mask(folder/'old_shaft_mask.png')
        new_shaft = mask(folder/'new_shaft_mask.png')
        edit = mask(folder/'acceptance_mask.png')
        geometry = json.loads((folder/'geometry.json').read_text(encoding='utf-8'))
        conditioning = prepare_input(base, source, rgb(folder/'handle_rgb.png'), old_shaft, new_shaft, edit)
        Image.fromarray(conditioning).save(out/'conditioning.png')
        Image.fromarray(edit.astype(np.uint8)*255).save(out/'edit_mask.png')
        start = time.perf_counter()
        if not edit.any():
            final = base.copy()
            audit = None
        else:
            if pipe is None:
                pipe, adapter = load_pipeline()
            h, w = base.shape[:2]
            scale = args.resolution/max(h, w)
            width, height = [max(256, round(v*scale/32)*32) for v in (w, h)]
            request = dict(case_id=folder.name, prompt=PROMPT, negative_prompt=NEGATIVE,
                seed=args.seed, steps=args.steps, resolution=[width, height],
                true_cfg_scale=args.cfg, control_context_scale=1., reference_images=0,
                base_sha256=sha(basepath), geometry_sha256=sha(folder/'geometry.json'),
                input_hashes={n: sha(folder/n) for n in ['source.png', 'handle_rgb.png',
                    'old_shaft_mask.png', 'new_shaft_mask.png', 'acceptance_mask.png', 'control.png']},
                conditioning_sha256=sha(out/'conditioning.png'), runner_sha256=sha(Path(__file__)),
                vendor_sha256=sha(VENDOR/'videox_fun/pipeline/pipeline_qwenimage21_reference_control.py'),
                adapter=adapter, scope='Only geometry-defined shaft edit region; no food/source regeneration')
            write(out/'request.json', request)
            with torch.inference_mode():
                generated = pipe(prompt=PROMPT, negative_prompt=NEGATIVE,
                    image=Image.fromarray(conditioning),
                    mask_image=Image.fromarray(edit.astype(np.uint8)*255).convert('RGB'),
                    reference_images=None, reference_resolution=512,
                    control_image=Image.open(folder/'control.png').convert('RGB'),
                    control_context_scale=1., true_cfg_scale=args.cfg,
                    height=height, width=width, num_inference_steps=args.steps,
                    generator=torch.Generator('cuda').manual_seed(args.seed), use_kv_cache=False).images[0]
            generated.save(out/'generated.png')
            candidate = np.asarray(Image.alpha_composite(Image.fromarray(conditioning).convert('RGBA'),
                generated.convert('RGBA').resize((w, h), Image.Resampling.LANCZOS)).convert('RGB'))
            Image.fromarray(candidate).save(out/'candidate.png')
            final = accept_shaft(base, candidate, edit)
            audit = pipe.last_conditioning_audit
        Image.fromarray(final).save(out/'final.png')
        row = dict(case_id=folder.name, seconds=time.perf_counter()-start,
            base_sha256=sha(basepath), final_sha256=sha(out/'final.png'),
            edit_pixels=int(edit.sum()), changed_outside_shaft_pixels=int(np.any(final != base, axis=2)[~edit].sum()),
            geometry=geometry, conditioning_audit=audit,
            interpretation='A projected shaft reaches the frame; an actual unseen hand or physical support is not observed.')
        write(out/'result.json', row)
        rows.append(row)
        write(args.output/(args.worker_name+'.json'), dict(status='running', cases=rows))
        print(json.dumps(dict(case_id=folder.name, seconds=row['seconds'], edit_pixels=row['edit_pixels'])), flush=True)
    write(args.output/(args.worker_name+'.json'), dict(status='complete', cases=rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--production', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--indices', type=int, nargs='+')
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--cfg', type=float, default=3.)
    parser.add_argument('--worker-name', default='handle_worker')
    run(parser.parse_args())
