"""Factorized high-resolution bite/head and source-cavity reconstruction."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from run_mld4_reconstruction import mask, project_material, sha, write


HEAD_PROMPT = (
    'A close-up natural camera photograph of one small bite resting securely in a shallow stainless steel spoon. '
    'The reference shows exactly the selected material of this bite. Match that material, its colors and its texture. '
    'Preserve the bite outline and the separate strands if present in the structural guide. '
    'Keep its position, scale, spoon contact and camera view. The spoon is real thin silver metal, with soft reflections. '
    'The selected material in the reference is the only food to put on the spoon. '
    'The neutral gray surrounding the reference is a mask background, not an ingredient. '
    'Resolve natural depth, soft shading and moist surface detail. Keep the existing background unchanged. '
    'Do not infer additional ingredients from the meal name. Do not add ingredients absent from the selected material. '
    'No extra toppings, garnish, sauce, hand or person.')
SOURCE_PROMPT = (
    'A natural close-up food photograph after a small first bite has been removed at the indicated location. '
    'The edge guide shows the remaining food boundary. Preserve this open missing portion rather than restoring the intact food. '
    'Show the natural torn or cut food edge and the actual underlying dish or remaining food. '
    'Match the neighboring food surface, lighting and soft contact shadows. '
    'Keep all surrounding food, background and camera view unchanged. No added food or utensils. '
    'No flat painted patch, geometric cutout, black polygon, deep artificial crater, hand or person.')
NEGATIVE = 'CGI, polygon, plastic, flat pasted patch, invented toppings, hand, person, fingers, face, duplicate bite, floating food, deep cup'


def finish_candidate(source, texture, full, base, editable, active, food, confidence, amount):
    final = project_material(source, texture, full, editable, food, confidence, amount)
    # A local stage must not re-feather an already completed head or handle.
    final[~active] = base[~active]
    final[~editable] = source[~editable]
    return final


def run(args):
    import torch
    from probe_mld3_metal_conditioning import load_pipeline, VENDOR
    pipe, adapter = load_pipeline()
    binding = None
    if args.binding_adapter:
        from mld4_binding_adapter import load_material_binding_adapter
        binding = load_material_binding_adapter(pipe.transformer, args.binding_adapter)
    rows = []
    worker_name=args.worker_name or args.variant
    captions=json.loads(args.captions.read_text(encoding='utf-8')) if args.captions else {}
    folders = sorted(args.guides.glob('real_*'))
    if args.indices is not None:
        folders = [p for p in folders if int(p.name.split('_')[1]) in args.indices]
    for folder in folders:
        out = args.output/'real'/folder.name/'candidates'/args.variant
        out.mkdir(parents=True, exist_ok=True)
        basepath = args.output/'real'/folder.name/'candidates'/args.base_variant/'unprojected.png'
        full = np.asarray(Image.open(basepath).convert('RGB')).copy()
        base = full.copy()
        active = np.zeros(full.shape[:2], dtype=bool)
        source = np.asarray(Image.open(folder/'full_source.png').convert('RGB'))
        manifest = json.loads((folder/'factorization.json').read_text(encoding='utf-8'))
        stage_rows = []
        for stage in args.stages:
            f = folder/stage; dest = out/stage; dest.mkdir(exist_ok=True)
            sm = json.loads((f/'stage_manifest.json').read_text(encoding='utf-8'))
            x0,y0,x1,y1 = sm['bbox_xyxy']
            active[y0:y1,x0:x1] |= mask(f/'inpaint_mask.png')
            if binding is not None:
                # Offload hooks materialize inference tensors after the first call.
                # PEFT toggles gradients while enabling; keep that bookkeeping in inference mode.
                with torch.inference_mode():
                    if stage == 'head':
                        pipe.transformer.enable_adapters()
                    else:
                        pipe.transformer.disable_adapters()
                    pipe.transformer.requires_grad_(False)
            if stage == 'source' and args.source_mode == 'cavity':
                # Observed retained food is immutable; only the disoccluded area is synthesized.
                source_stage = mask(f/'inpaint_mask.png')
                local = full[y0:y1,x0:x1]
                local[source_stage] = source[y0:y1,x0:x1][source_stage]
            init = full[y0:y1,x0:x1].copy()
            anchors = mask(f/'anchor_mask.png')
            texture = np.asarray(Image.open(f/'material_texture.png').convert('RGB'))
            init[anchors] = texture[anchors]
            Image.fromarray(init).save(dest/'conditioning_used.png')
            edit = mask(f/'inpaint_mask.png')
            condmask = Image.open(f/'conditioning_mask.png').convert('RGB')
            control = Image.open(f/'edge.png').convert('RGB')
            if stage == 'head' and args.head_control == 'depth':
                control = Image.open(f/'depth_control.png').convert('RGB')
            reference = Image.open(f/'source_reference.png').convert('RGB')
            factor = args.resolution/max(x1-x0,y1-y0)
            width,height = [max(256,round(v*factor/32)*32) for v in [x1-x0,y1-y0]]
            prompt = HEAD_PROMPT if stage == 'head' else SOURCE_PROMPT
            if stage == 'source' and args.source_mode == 'seam':
                from refine_mld4_source_seam import PROMPT as SEAM_PROMPT
                channel = np.load(f/'guide_channels.npz')
                removed = channel['source_removed'].astype(bool)
                band = ((distance_transform_edt(removed) <= 4) &
                        (distance_transform_edt(~removed) <= 4))
                edit &= band
                condmask = Image.fromarray(edit.astype(np.uint8)*255).convert('RGB')
                control = None
                prompt = SEAM_PROMPT
            if stage == 'source' and args.source_mode == 'inpaint':
                control = None
            if stage == 'source' and args.source_mode == 'cavity':
                removed = mask(f/'source_removal_mask.png')
                edit &= distance_transform_edt(~removed) <= args.cavity_seam
                condmask = Image.fromarray(edit.astype(np.uint8)*255).convert('RGB')
                control = None
            if folder.name in captions:
                caption_key = 'source_surface' if stage == 'source' and args.source_mode == 'cavity' else stage
                prompt = captions[folder.name][caption_key]
            condmask.save(dest/'conditioning_mask_used.png')
            Image.fromarray(edit.astype(np.uint8)*255).save(dest/'acceptance_mask.png')
            request = dict(case_id=folder.name, variant=args.variant, stage=stage,
                bbox_xyxy=[x0,y0,x1,y1], canvas=[width,height], steps=args.steps, seed=args.seed,
                prompt=prompt, negative_prompt=NEGATIVE, control_context_scale=args.control_scale,
                true_cfg_scale=args.cfg,
                model='Qwen-Image-2.1 + Fun ControlNet-Union', adapter=adapter,
                base_variant=args.base_variant, base_sha256=sha(basepath),
                conditioning_sha256=sha(dest/'conditioning_used.png'),
                conditioning_mask_used_sha256=sha(dest/'conditioning_mask_used.png'),
                acceptance_mask_sha256=sha(dest/'acceptance_mask.png'),
                source_mode=args.source_mode, control_image_used=control is not None,
                cavity_seam_native_pixels=args.cavity_seam if args.source_mode=='cavity' else None,
                runner_sha256=sha(Path(__file__)),
                effective_control_context_scale=1. if control is None else args.control_scale,
                head_control=args.head_control,
                depth_control_sha256=sha(f/'depth_control.png') if stage=='head' and args.head_control=='depth' else None,
                captions_sha256=sha(args.captions) if args.captions else None,
                binding_adapter=binding, binding_enabled=binding is not None and stage=='head',
                inputs={n:sha(f/n) for n in ['conditioning_mask.png','inpaint_mask.png','edge.png','source_reference.png','anchor_mask.png','material_texture.png']},
                vendor_sha256=sha(VENDOR/'videox_fun/pipeline/pipeline_qwenimage21_reference_control.py'))
            write(dest/'request.json',request)
            started=time.perf_counter()
            with torch.inference_mode():
                result=pipe(prompt=prompt, negative_prompt=NEGATIVE, image=Image.fromarray(init),
                    mask_image=condmask, reference_images=[reference] if stage=='head' else None,
                    reference_resolution=512, control_image=control,
                    control_context_scale=1. if control is None else args.control_scale,
                    true_cfg_scale=args.cfg, height=height, width=width, num_inference_steps=args.steps,
                    generator=torch.Generator('cuda').manual_seed(args.seed), use_kv_cache=False).images[0]
            result.save(dest/'generated.png')
            raw=np.asarray(Image.alpha_composite(Image.fromarray(init).convert('RGBA'),
                result.convert('RGBA').resize((x1-x0,y1-y0),Image.Resampling.LANCZOS)).convert('RGB'))
            alpha=np.minimum(distance_transform_edt(edit)/2,1)[...,None]
            accepted=np.rint(raw*alpha+init*(1-alpha)).clip(0,255).astype(np.uint8)
            # Even preserved conditioning anchors remain soft generative constraints.
            accepted[~edit]=full[y0:y1,x0:x1][~edit]
            full[y0:y1,x0:x1]=accepted
            Image.fromarray(accepted).save(dest/'accepted.png')
            row=dict(stage=stage,seconds=time.perf_counter()-started,
                     generated_sha256=sha(dest/'generated.png'), editable_pixels=int(edit.sum()),
                     anchors=int(anchors.sum()), conditioning_audit=pipe.last_conditioning_audit)
            write(dest/'generation.json',row);stage_rows.append(row);print(json.dumps(row),flush=True)
        editable=mask(folder/'full_inpaint_mask.png')
        full[~editable]=source[~editable]
        Image.fromarray(full).save(out/'candidate.png')
        original=args.material_guides/folder.name
        channels=np.load(original/'guide_channels.npz')
        texture=np.asarray(Image.open(original/'material_texture.png').convert('RGB'))
        food=mask(original/'target_food_mask.png')
        for name,amount in [('unprojected',0.),('projected',.5)]:
            final=finish_candidate(source,texture,full,base,editable,active,food,channels['material_confidence'],amount)
            Image.fromarray(final).save(out/(name+'.png'))
        row=dict(case_id=folder.name,variant=args.variant,base_variant=args.base_variant,
            stages=stage_rows, source_sha256=sha(folder/'full_source.png'),
            frozen_edit_sha256=sha(folder/'full_inpaint_mask.png'),
            factorization_sha256=sha(folder/'factorization.json'),
            changed_outside_edit_pixels=int(np.any(final!=source,axis=2)[~editable].sum()),
            changed_inactive_stage_pixels=int(np.any(final!=base,axis=2)[~active & editable].sum()),
            generated_food_permitted=True, generation_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        write(out/'generation.json',row);rows.append(row)
        write(args.output/(worker_name+'_worker.json'),dict(status='running',cases=rows))
    write(args.output/(worker_name+'_worker.json'),dict(status='complete',cases=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--guides',type=Path,required=True)
    p.add_argument('--material-guides',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--indices',type=int,nargs='+');p.add_argument('--variant',default='factorized')
    p.add_argument('--base-variant',default='owned_anchors');p.add_argument('--resolution',type=int,default=1024)
    p.add_argument('--steps',type=int,default=40);p.add_argument('--seed',type=int,default=41)
    p.add_argument('--control-scale',type=float,default=.7);p.add_argument('--worker-name')
    p.add_argument('--source-mode',choices=['reconstruct','seam','inpaint','cavity'],default='reconstruct')
    p.add_argument('--cavity-seam',type=float,default=1.)
    p.add_argument('--captions',type=Path)
    p.add_argument('--binding-adapter',type=Path)
    p.add_argument('--cfg',type=float,default=3.)
    p.add_argument('--head-control',choices=['edge','depth'],default='edge')
    p.add_argument('--stages',nargs='+',choices=['head','source'],default=['head','source'])
    run(p.parse_args())
