"""Qwen2.1 appearance experiment with raw generations and explicit composites."""
import argparse
import os
from pathlib import Path
import sys
import json
import time
import hashlib
import numpy as np
from PIL import Image,ImageFilter
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt


def write(p,value):p.write_text(json.dumps(value,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1)
    ap.add_argument('--indices',type=int,nargs='+');ap.add_argument('--steps',type=int,default=32)
    ap.add_argument('--geometry-name',default='raw')
    ap.add_argument('--methods',nargs='+',default=['qwen_only','refined'])
    args=ap.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        VIDEOX_ATTENTION_TYPE='SDPA',OMP_NUM_THREADS='4')
    sys.path.insert(0,'/mnt/tmp/guo-z_first_bite_structure_20260930_runtime/vendor')
    import torch
    from safetensors.torch import load_file
    from diffusers import FlowMatchEulerDiscreteScheduler
    from videox_fun.models import AutoencoderKLQwenImage21,Qwen3VLForConditionalGeneration,Qwen3VLProcessor,QwenImage21ControlTransformer2DModel
    from videox_fun.pipeline.pipeline_qwenimage21_reference_control import QwenImage21ControlPipeline
    torch.set_num_threads(4)
    models=Path('/mnt/tmp/guo-z_first_bite_structure_20260930_models');model=models/'qwen-image-2.1';dtype=torch.bfloat16
    transformer=QwenImage21ControlTransformer2DModel.from_pretrained(str(model),subfolder='transformer',
        low_cpu_mem_usage=True,torch_dtype=dtype,
        transformer_additional_kwargs={'control_layers':list(range(0,32,2)),'control_in_dim':129})
    adapter=load_file(str(models/'controlnet-union/Qwen-Image-2.1-Fun-Controlnet-Union.safetensors'))
    missing,unexpected=transformer.load_state_dict(adapter,strict=False);del adapter
    vae=AutoencoderKLQwenImage21.from_pretrained(str(model),subfolder='vae',torch_dtype=dtype,local_files_only=True)
    processor=Qwen3VLProcessor.from_pretrained(str(model),subfolder='processor',local_files_only=True)
    text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(str(model),subfolder='text_encoder',
        torch_dtype=dtype,local_files_only=True,attn_implementation='sdpa')
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(str(model),subfolder='scheduler',local_files_only=True)
    pipe=QwenImage21ControlPipeline(vae=vae,text_encoder=text_encoder,processor=processor,transformer=transformer,scheduler=scheduler)
    pipe.enable_model_cpu_offload(gpu_id=0)
    cases=json.load(open('/host/space0/guo-z/tf-ufi/material_lineage_training_20261004/real_probes/manifest.json'))['cases']
    receipt=dict(status='running',seed=41,steps=args.steps,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        methods_requested=args.methods,
        same_prompt_and_seed=True,model='Qwen-Image-2.1 + Fun ControlNet-Union',adapter_unexpected=unexpected,
        source_only=True,model_trained_here=False,completed=[])
    rp=args.root/('appearance_worker_'+str(args.shard)+'.json');write(rp,receipt)
    for i,c in enumerate(cases):
        if i%args.shards!=args.shard or (args.indices is not None and i not in args.indices):continue
        folder=args.root/c['case_id'];src=Image.open(folder/'source.png').convert('RGB')
        geo=Image.open(folder/(args.geometry_name+'.png')).convert('RGB')
        mask=Image.open(folder/'edit_mask.png').convert('L')
        w,h=src.size;W=round(w/32)*32;H=round(h/32)*32
        W,H=max(256,W),max(256,H)
        food=c['metadata_caption_not_model_input'].split('of ')[-1].split(' on a plate')[0]
        plan=json.load(open(folder/('geometry_'+args.geometry_name+'.json')))
        center=plan['source_bite_center']
        prompt=(f'A realistic photograph of {food}. A single small first bite of this exact same food is lifted clearly above the dish '
            f'in the upper left of the image, resting inside a shiny concave stainless steel spoon extending to the right edge. '
            f'The food at position {round(center[0]/w*100)} percent across and {round(center[1]/h*100)} percent down '
            f'has a matching small missing bite at its outer edge. The remaining food and the lifted bite keep the original color, '
            f'ingredients and texture. Show a clear air gap, correct spoon contact, natural shadow, realistic metallic spoon. '
            f'Preserve the exact original dish, scene, lighting, background and camera view. No hand, person, face, mouth, extra food or text.')
        if (folder/'appearance_protocol.json').exists():
            prompt=json.load(open(folder/'appearance_protocol.json'))['prompt']
        negative='hand, fingers, person, face, arm, mouth, plastic, cartoon, smooth plastic food, floating unsupported food, duplicate food'
        edges=np.asarray(geo.filter(ImageFilter.FIND_EDGES).convert('L'))>45
        control=Image.fromarray(np.repeat((edges.astype(np.uint8)*255)[:,:,None],3,axis=-1))
        control.save(folder/'appearance_control.png')
        for method in args.methods:
            d=folder/method;d.mkdir(exist_ok=True);started=time.time()
            inputs=geo if method=='refined' else src
            job=dict(case_id=c['case_id'],method=method,prompt=prompt,negative_prompt=negative,seed=41,steps=args.steps,
                geometry_control=method=='refined',input_path=str(folder/(args.geometry_name+'.png' if method=='refined' else 'source.png')),
                width=W,height=H,context_composite=True,geometry_is_not_ground_truth=True)
            write(d/'request.json',job)
            with torch.inference_mode():
                raw=pipe(prompt=prompt,negative_prompt=negative,image=inputs,mask_image=mask.convert('RGB'),
                    reference_images=[src],reference_resolution=512,
                    control_image=control if method=='refined' else None,
                    control_context_scale=1.,true_cfg_scale=3.,height=H,width=W,num_inference_steps=args.steps,
                    generator=torch.Generator('cuda').manual_seed(41),use_kv_cache=False).images[0]
            raw.save(d/'generated_rgba.png')
            candidate=np.asarray(Image.alpha_composite(inputs.convert('RGBA'),raw.convert('RGBA').resize(src.size,Image.Resampling.LANCZOS)).convert('RGB')).astype(float)
            editable=np.asarray(mask)>127
            if method=='refined':
                moved=np.asarray(Image.open(folder/'observed_food_mask.png'))>0
                # Observation anchoring survives diffusion: only food boundary/cut/spoon appearance can change.
                immutable_food=binary_erosion(moved,iterations=3)
                editable &= ~immutable_food
            else:immutable_food=np.zeros(editable.shape,bool)
            alpha=np.clip(distance_transform_edt(editable)/3,0,1)[...,None]
            base=np.asarray(inputs).astype(float);final=np.clip(np.rint(base*(1-alpha)+candidate*alpha),0,255).astype(np.uint8)
            Image.fromarray(final).save(d/'composited.png')
            Image.fromarray(editable.astype(np.uint8)*255).save(d/'actual_edit_mask.png')
            row=dict(case_id=c['case_id'],method=method,seconds=time.time()-started,
                raw_mode=raw.mode,raw_wh=list(raw.size),food_anchor_pixels=int(immutable_food.sum()),
                appearance_change_pixels=int(np.any(final!=base,axis=-1).sum()),
                geometric_accuracy_guaranteed=False,files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir() if p.is_file()})
            write(d/'result.json',row);receipt['completed'].append(row);write(rp,receipt);print(json.dumps(row),flush=True)
    receipt['status']='complete';write(rp,receipt)


if __name__=='__main__':main()
