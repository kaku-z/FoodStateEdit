"""Reconstruct the cut boundary while conditioning on the missing-region interior.

This separates the discrete removal layout from the appearance of its seam.
No contour image is supplied to the generative model in this experiment.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from run_mld4_reconstruction import mask, project_material, sha, write

PROMPT=('A natural food photograph. Retouch the narrow boundary between the remaining food and the region '
        'where a small bite was taken out. Preserve the existing missing portion and its underlying surface. '
        'Make the cut or torn food edge natural, with fine food texture and soft contact shadow matching the photograph. '
        'Preserve the surrounding food and plate and camera view. No drawn outline, black stroke, painted patch, '
        'extra food, utensil, hand or person.')


def run(args):
    import torch
    from probe_mld3_metal_conditioning import load_pipeline,VENDOR
    pipe,adapter=load_pipeline(); rows=[]
    folders=sorted(args.guides.glob('real_*'))
    if args.indices is not None:folders=[p for p in folders if int(p.name.split('_')[1]) in args.indices]
    for folder in folders:
        case=args.output/'real'/folder.name/'candidates'; out=case/args.variant;out.mkdir(parents=True,exist_ok=True)
        basepath=case/args.base_variant/'unprojected.png'
        full=np.asarray(Image.open(basepath).convert('RGB')).copy()
        source=np.asarray(Image.open(folder/'full_source.png').convert('RGB'))
        sm=json.loads((folder/'head/stage_manifest.json').read_text(encoding='utf-8'))
        x0,y0,x1,y1=sm['bbox_xyxy']; headpath=case/args.head_variant/'head/accepted.png'
        head=np.asarray(Image.open(headpath).convert('RGB')); hm=mask(folder/'head/inpaint_mask.png')
        crop=full[y0:y1,x0:x1];crop[hm]=head[hm]
        sm=json.loads((folder/'source/stage_manifest.json').read_text(encoding='utf-8'))
        x0,y0,x1,y1=sm['bbox_xyxy'];init=full[y0:y1,x0:x1].copy()
        z=np.load(folder/'source/guide_channels.npz');removed=z['source_removed'].astype(bool)
        band=(distance_transform_edt(removed)<=args.band)&(distance_transform_edt(~removed)<=args.band)
        band &= mask(folder/'source/inpaint_mask.png')
        Image.fromarray(init).save(out/'source_conditioning.png')
        Image.fromarray(band.astype(np.uint8)*255).save(out/'source_seam_mask.png')
        factor=args.resolution/max(x1-x0,y1-y0)
        width,height=[max(256,round(v*factor/32)*32) for v in [x1-x0,y1-y0]]
        request=dict(case_id=folder.name,variant=args.variant,base_variant=args.base_variant,
            head_variant=args.head_variant,base_sha256=sha(basepath),head_sha256=sha(headpath),
            bbox_xyxy=sm['bbox_xyxy'],band_native_pixels=args.band,band_pixels=int(band.sum()),
            mask_sha256=sha(out/'source_seam_mask.png'),source_conditioning_sha256=sha(out/'source_conditioning.png'),
            canvas=[width,height],steps=args.steps,seed=41,prompt=PROMPT,control_image=None,reference_images=None,
            model='Qwen-Image-2.1 + Fun ControlNet-Union pure inpaint',adapter=adapter,
            vendor_sha256=sha(VENDOR/'videox_fun/pipeline/pipeline_qwenimage21_reference_control.py'))
        write(out/'request.json',request);started=time.perf_counter()
        with torch.inference_mode():
            raw=pipe(prompt=PROMPT,negative_prompt='drawing, outline, black ink, polygon, artificial crater, restored intact food, hand, person',
                image=Image.fromarray(init),mask_image=Image.fromarray(band.astype(np.uint8)*255).convert('RGB'),
                control_image=None,reference_images=None,control_context_scale=1.,true_cfg_scale=3.,
                height=height,width=width,num_inference_steps=args.steps,
                generator=torch.Generator('cuda').manual_seed(41),use_kv_cache=False).images[0]
        raw.save(out/'source_generated.png')
        rgb=np.asarray(Image.alpha_composite(Image.fromarray(init).convert('RGBA'),raw.convert('RGBA').resize((x1-x0,y1-y0),Image.Resampling.LANCZOS)).convert('RGB'))
        alpha=np.minimum(distance_transform_edt(band)/2,1)[...,None]
        accepted=np.rint(rgb*alpha+init*(1-alpha)).clip(0,255).astype(np.uint8);accepted[~band]=init[~band]
        full[y0:y1,x0:x1]=accepted;editable=mask(folder/'full_inpaint_mask.png');full[~editable]=source[~editable]
        Image.fromarray(full).save(out/'candidate.png')
        mat=args.material_guides/folder.name;channels=np.load(mat/'guide_channels.npz')
        texture=np.asarray(Image.open(mat/'material_texture.png').convert('RGB'));food=mask(mat/'target_food_mask.png')
        for name,amount in [('unprojected',0.),('projected',.5)]:
            final=project_material(source,texture,full,editable,food,channels['material_confidence'],amount)
            Image.fromarray(final).save(out/(name+'.png'))
        row=dict(case_id=folder.name,variant=args.variant,seconds=time.perf_counter()-started,
            source_sha256=sha(folder/'full_source.png'),frozen_edit_sha256=sha(folder/'full_inpaint_mask.png'),
            changed_outside_edit_pixels=int(np.any(final!=source,axis=2)[~editable].sum()),
            generated_food_permitted=True,source_generated_sha256=sha(out/'source_generated.png'))
        write(out/'generation.json',row);rows.append(row);print(json.dumps(row),flush=True)
        write(args.output/(args.variant+'_worker.json'),dict(status='running',cases=rows))
    write(args.output/(args.variant+'_worker.json'),dict(status='complete',cases=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--guides',type=Path,required=True)
    p.add_argument('--material-guides',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--indices',type=int,nargs='+');p.add_argument('--variant',default='factorized_seam')
    p.add_argument('--base-variant',default='owned_anchors');p.add_argument('--head-variant',default='factorized')
    p.add_argument('--band',type=float,default=4);p.add_argument('--resolution',type=int,default=1024)
    p.add_argument('--steps',type=int,default=40);run(p.parse_args())
