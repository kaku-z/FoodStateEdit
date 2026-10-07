"""Geometry-conditioned image reconstruction with source-material projection.

This is an experimental renderer. Generated food pixels are explicitly recorded;
source correspondence and source-like texture are constraints, not a proof that
an unobserved surface or a semantic ingredient has been recovered correctly.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, record):
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def mask(path):
    return np.asarray(Image.open(path).convert('L')) > 127


def project_material(source, guide, candidate, editable, food, confidence, amount):
    """Project visible material detail/chroma while retaining inferred lighting.

    At each material sample, the quadratic objectives for high-pass luminance
    and low-pass chroma have a weighted-average minimizer. Unknown surfaces and
    the cavity have zero material confidence and retain the neural proposal.
    The exterior source is imposed exactly after the projection.
    """
    x = candidate.astype(np.float32) / 255
    if amount:
        t = np.asarray(Image.fromarray(guide).convert('YCbCr')).astype(np.float32) / 255
        g = np.asarray(Image.fromarray(candidate).convert('YCbCr')).astype(np.float32) / 255
        weight = np.clip(confidence, 0, 1) * np.clip(distance_transform_edt(food) / 3, 0, 1)
        t_low = gaussian_filter(t, sigma=(1.6, 1.6, 0))
        g_low = gaussian_filter(g, sigma=(1.6, 1.6, 0))
        g[..., 0] += amount * weight * ((t[..., 0] - t_low[..., 0]) - (g[..., 0] - g_low[..., 0]))
        g[..., 1:] += .6 * amount * weight[..., None] * (t_low[..., 1:] - g_low[..., 1:])
        x = np.asarray(Image.fromarray(np.rint(g*255).clip(0,255).astype(np.uint8), 'YCbCr').convert('RGB')).astype(np.float32) / 255
    alpha = np.minimum(distance_transform_edt(editable) / 3, 1)[..., None]
    final = np.rint(255 * x * alpha + source * (1 - alpha)).clip(0, 255).astype(np.uint8)
    final[~editable] = source[~editable]
    return final


def generate(args):
    import torch
    from probe_mld3_metal_conditioning import load_pipeline, VENDOR
    pipe, adapter = load_pipeline()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    worker_name = args.worker_name or args.variant
    folders = sorted(p for p in args.guides.glob('real_*') if p.is_dir())
    if args.indices is not None:
        folders = [p for p in folders if int(p.name.split('_')[1]) in args.indices]
    for folder in folders:
        source = Image.open(folder / 'source.png').convert('RGB')
        guide = Image.open(folder / 'coarse.png').convert('RGB')
        conditioning = Image.open(folder / 'conditioning_image.png').convert('RGB') if args.variant == 'owned_anchors' else guide
        control = Image.open(folder / 'edge.png').convert('RGB')
        editable = mask(folder / 'inpaint_mask.png')
        conditioning_edit = mask(folder / 'conditioning_mask.png') if args.variant == 'owned_anchors' else editable
        food = mask(folder / 'target_food_mask.png')
        source_np = np.asarray(source); guide_np = np.asarray(guide)
        manifest = json.loads((folder / 'guide_manifest.json').read_text(encoding='utf-8'))
        semantic = manifest.get('prompt', manifest.get('food_prompt', 'food'))
        prompt = (f'Photorealistic food photograph of the same {semantic} and the same dish. '
            'A real stainless steel spoon holds the first bite clearly lifted above and away from the plate. '
            'Preserve the lifted food position, amount, overall shape and contact with the spoon in the structural guide. '
            'The bite is made from the same food as the original meal: preserve its actual ingredients, colors and texture. '
            'Reconstruct natural three-dimensional food surfaces, moist highlights, soft self-shadowing and realistic silver reflections. '
            'At the small removal region on the plate, show the plausible remaining food or actual underlying dish after this bite was removed. '
            'Integrate that region seamlessly with the surrounding photograph; no flat painted patch and no black polygon outline. '
            'Keep the original scene, plate and food everywhere else. The utensil enters from the edge of the image. '
            'No new hand, person, body, mouth or extra food. The result is a single opaque natural camera photograph.')
        negative = ('CGI, 3D render, polygon, flat cutout, pasted patch, thick black outline, artificial gray cavity, '
            'plastic food, hand, fingers, face, person, arm, changed dish, duplicated bite, unsupported floating food, deep cup')
        reference = Image.open(folder / 'source_reference.png').convert('RGB')
        owned = args.variant.startswith('owned_')
        if owned:
            selected = mask(folder / 'source_reference_mask.png')
            reference_pixels = np.asarray(reference).copy()
            reference_pixels[~selected] = 128
            reference = Image.fromarray(reference_pixels)
            prompt += (' The reference contains ONLY the selected source material on a neutral gray mask background. '
                'Gray is not food. The mask boundary is artificial, not the target food outline. '
                'Copy only the selected food colors and texture; do not import colorful toppings or other parts '
                'visible elsewhere in the meal. Keep a clearly visible bite on the spoon. '
                'Show a definite small missing portion at the source edit region; do not repair the original food back to its intact state.')
        refs = [reference] if args.variant != 'geometry_only' else None
        if refs:
            prompt += ' The reference image is a close-up of the source food material, not the target layout.'
        factor = args.resolution / max(source.size)
        canvas = [max(256, round(v * factor / 32) * 32) for v in source.size]
        out = args.output / 'real' / folder.name / 'candidates' / args.variant
        out.mkdir(parents=True, exist_ok=True)
        if refs:
            reference.save(out / 'reference_used.png')
        conditioning.save(out / 'conditioning_used.png')
        Image.fromarray(conditioning_edit.astype(np.uint8)*255).save(out / 'conditioning_mask_used.png')
        request = dict(case_id=folder.name, variant=args.variant, prompt=prompt, negative_prompt=negative,
            steps=args.steps, seed=args.seed, resolution=canvas, control_context_scale=args.control_scale,
            true_cfg_scale=args.cfg, runner_sha256=sha(Path(__file__)),
            reference_mode='owned source material only, unselected pixels neutralgray' if owned else 'source crop including context' if refs else 'none',
            reference_sha256=sha(out/'reference_used.png') if refs else None, generated_food_permitted=True,
            conditioning_image_sha256=sha(out/'conditioning_used.png'),
            conditioning_mask_sha256=sha(out/'conditioning_mask_used.png'),
            kept_material_anchor_pixels=int((editable & ~conditioning_edit).sum()),
            input_hashes={n: sha(folder / n) for n in ['source.png','coarse.png','edge.png','inpaint_mask.png','source_reference.png']},
            model='Qwen-Image-2.1 + Fun ControlNet-Union', adapter=adapter,
            vendor_sha256=sha(VENDOR / 'videox_fun/pipeline/pipeline_qwenimage21_reference_control.py'))
        write(out / 'request.json', request)
        started = time.perf_counter()
        extra = {}; transport_audit = None
        if args.variant == 'owned_transport':
            from mld4_latent_transport import prepare_transport
            channels = np.load(folder / 'guide_channels.npz')
            texture = np.asarray(Image.open(folder/'material_texture.png').convert('RGB'))
            noise, callback, transport_audit = prepare_transport(pipe, guide, texture,
                channels['material_confidence'], food, canvas, args.seed)
            extra = dict(latents=noise, callback_on_step_end=callback)
        with torch.inference_mode():
            raw = pipe(prompt=prompt, negative_prompt=negative, image=conditioning,
                mask_image=Image.fromarray(conditioning_edit.astype(np.uint8)*255).convert('RGB'),
                reference_images=refs, reference_resolution=512, control_image=control,
                control_context_scale=args.control_scale, true_cfg_scale=args.cfg,
                height=canvas[1], width=canvas[0], num_inference_steps=args.steps,
                generator=torch.Generator('cuda').manual_seed(args.seed), use_kv_cache=False, **extra).images[0]
        if transport_audit is not None:
            write(out / 'transport_audit.json', transport_audit)
        raw.save(out / 'generated.png')
        rgba = raw.convert('RGBA').resize(source.size, Image.Resampling.LANCZOS)
        candidate = np.asarray(Image.alpha_composite(guide.convert('RGBA'), rgba).convert('RGB'))
        Image.fromarray(candidate).save(out / 'candidate.png')
        channels = np.load(folder / 'guide_channels.npz')
        confidence = channels['material_confidence'] if 'material_confidence' in channels else food.astype(np.float32)
        texture = np.asarray(Image.open(folder/'material_texture.png').convert('RGB')) if owned else guide_np
        projections = {}
        for name, amount in [('unprojected', 0.), ('projected', .5)]:
            final = project_material(source_np, texture, candidate, editable, food, confidence, amount)
            Image.fromarray(final).save(out / (name+'.png'))
            projections[name] = dict(material_projection=amount,
                material_texture_sha256=sha(folder/'material_texture.png') if owned else sha(folder/'coarse.png'),
                changed_outside_edit_pixels=int(np.any(final != source_np,axis=2)[~editable].sum()),
                changed_guide_food_pixels=int(np.any(final != guide_np,axis=2)[food].sum()),
                food_sha256=hashlib.sha256(final[food].tobytes()).hexdigest())
        row = dict(case_id=folder.name, variant=args.variant, seconds=time.perf_counter()-started,
            generated_sha256=sha(out / 'generated.png'),
            generated_alpha_mean_in_edit=float(np.asarray(rgba)[...,3][editable].mean()/255),
            projection=projections, generation_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        write(out / 'generation.json', row); rows.append(row)
        print(json.dumps(row), flush=True)
        write(args.output / (worker_name+'_worker.json'),dict(status='running',cases=rows))
    write(args.output / (worker_name+'_worker.json'),dict(status='complete',cases=rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--guides',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variant', choices=['geometry_only','material_ref','owned_material','owned_boundary','owned_anchors','owned_transport'],required=True)
    p.add_argument('--indices',type=int,nargs='+');p.add_argument('--steps',type=int,default=40)
    p.add_argument('--resolution',type=int,default=1024);p.add_argument('--seed',type=int,default=41)
    p.add_argument('--control-scale',type=float,default=.7)
    p.add_argument('--cfg',type=float,default=3.)
    p.add_argument('--worker-name')
    generate(p.parse_args())


if __name__ == '__main__':
    main()
