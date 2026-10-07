"""Experimental native layout and appearance references with target geometry control.

Raw RGBA and source-preserving composites are separate artifacts. Neither is
silently substituted for the other; no per-seed retry or output selection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
RUNTIME=Path('/mnt/tmp/guo-z_first_bite_structure_20260930_runtime')
MODELS=Path('/mnt/tmp/guo-z_first_bite_structure_20260930_models')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,c):
    p.write_text(json.dumps(c,indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1);a=ap.parse_args()
    cfg=json.loads(a.config.read_text());jobs=[j for i,j in enumerate(cfg['jobs']) if i%a.shards==a.shard]
    a.output.mkdir(parents=True,exist_ok=False);mp=a.output/'manifest.json'
    state={'status':'preflight','pid':os.getpid(),'gpu':a.gpu,'config_sha256':sha(a.config),'script_sha256':sha(__file__),
           'started_unix':time.time(),'expected_jobs':[j['id'] for j in jobs],'completed':[]}
    write(mp,state);(a.output/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    try:
        from run_qwen_image_edit_direct_baseline import gpu_snapshot,foreign_process_check
        snap=gpu_snapshot(a.gpu);assert not snap['compute_processes'] and snap['memory_free_mib']>47500
        assert snap['host_mem_available_mib']>50000
        receipt=json.loads((ROOT/'backend21/range_manifest_v3.json').read_text());assert receipt['status']=='complete_verified'
        for row in receipt['files']:
            assert Path(row['path']).stat().st_size==row['size']
        for j in jobs:
            for f in j['files'].values():assert sha(f['path'])==f['sha256']
        os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                          TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',VIDEOX_ATTENTION_TYPE='SDPA')
        sys.path.insert(0,str(RUNTIME/'vendor'))
        import importlib.metadata
        import numpy as np
        from PIL import Image
        import torch
        from scipy.ndimage import distance_transform_edt
        from safetensors.torch import load_file
        from diffusers import FlowMatchEulerDiscreteScheduler
        from videox_fun.models import AutoencoderKLQwenImage21,Qwen3VLForConditionalGeneration,Qwen3VLProcessor,QwenImage21ControlTransformer2DModel
        from videox_fun.pipeline.pipeline_qwenimage21_reference_control import QwenImage21ControlPipeline
        torch.set_num_threads(4)
        state.update(status='loading_model',packages={n:importlib.metadata.version(n) for n in ['torch','torchvision','diffusers','transformers','accelerate','peft']});write(mp,state)
        model=MODELS/'qwen-image-2.1';dtype=torch.bfloat16
        transformer=QwenImage21ControlTransformer2DModel.from_pretrained(str(model),subfolder='transformer',low_cpu_mem_usage=True,torch_dtype=dtype,
            transformer_additional_kwargs={'control_layers':list(range(0,32,2)),'control_in_dim':129}).to(dtype)
        adapter=load_file(str(MODELS/'controlnet-union/Qwen-Image-2.1-Fun-Controlnet-Union.safetensors'))
        expected_control={k for k in transformer.state_dict() if k.startswith(('control_img_in.','control_blocks.'))}
        assert set(adapter)==expected_control,{'missing_control':list(expected_control-set(adapter))[:10],'unexpected':list(set(adapter)-expected_control)[:10]}
        missing,unexpected=transformer.load_state_dict(adapter,strict=False)
        assert not unexpected and not any(k in expected_control for k in missing)
        write(a.output/'weight_load_audit.json',{'adapter_keys':len(adapter),'expected_control_keys':len(expected_control),'base_keys_missing_from_adapter':len(missing),
            'unexpected_keys':unexpected,'all_control_keys_loaded':True,'control_layers':list(range(0,32,2)),'control_in_dim':129})
        del adapter
        assert not any(p.is_meta for p in transformer.parameters())
        vae=AutoencoderKLQwenImage21.from_pretrained(str(model),subfolder='vae',torch_dtype=dtype,local_files_only=True)
        processor=Qwen3VLProcessor.from_pretrained(str(model),subfolder='processor',local_files_only=True)
        text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(str(model),subfolder='text_encoder',torch_dtype=dtype,local_files_only=True,attn_implementation='sdpa')
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(str(model),subfolder='scheduler',local_files_only=True)
        pipe=QwenImage21ControlPipeline(vae=vae,text_encoder=text_encoder,processor=processor,transformer=transformer,scheduler=scheduler)
        pipe.enable_model_cpu_offload(gpu_id=0)
        original_forward_control=transformer.forward_control
        for j in jobs:
            foreign_process_check(snap['uuid']);start=time.time();d=a.output/j['id'];d.mkdir(exist_ok=False)
            src=Image.open(j['files']['source']['path']).convert('RGB');mask=Image.open(j['files']['edit_mask']['path']).convert('RGB')
            control=Image.open(j['files']['control']['path']).convert('RGB') if 'control' in j['files'] else None
            W,H=cfg['inference']['width'],cfg['inference']['height']
            reference_keys=j.get('reference_keys', ['reference'] if 'reference' in j['files'] else [])
            references=[Image.open(j['files'][key]['path']).convert('RGB') for key in reference_keys]
            hint_audit=[]
            def controlled_hints(control_joint,joint_hidden_states,control_kwargs):
                hints=original_forward_control(control_joint,joint_hidden_states,control_kwargs)
                target_mask=control_kwargs['target_token_mask']
                assert target_mask is not None and int(target_mask.sum())==(W//16)*(H//16)
                if j['hint_scope']=='target':
                    weight=target_mask[None,:,None].to(device=hints[0].device,dtype=hints[0].dtype)
                    hints=[h*weight for h in hints]
                    assert torch.count_nonzero(hints[0][:,~target_mask])==0
                hint_audit.append({'scope':j['hint_scope'],'target_tokens':int(target_mask.sum()),'joint_prefix_tokens':int((~target_mask).sum()),'reference_hint_zero':j['hint_scope']=='target'})
                return hints
            transformer.forward_control=controlled_hints

            state.update(status='generating',current=j['id'],step=-1,updated_unix=time.time());write(mp,state);write(d/'request.json',j)
            def callback(pipeline,step,timestep,kwargs):
                state.update(step=int(step),updated_unix=time.time());write(mp,state)
                assert torch.isfinite(kwargs['latents']).all()
                return kwargs
            _allocator_probe=torch.empty(1,device="cuda");del _allocator_probe
            torch.cuda.reset_peak_memory_stats(0)
            with torch.inference_mode():
                raw=pipe(prompt=j['prompt'],image=src,mask_image=mask,control_image=control,
                    reference_images=references or None,reference_resolution=512,
                    control_context_scale=j.get('control_scale',1.),true_cfg_scale=cfg['inference']['true_cfg_scale'],
                    height=H,width=W,num_inference_steps=cfg['inference']['steps'],
                    generator=torch.Generator('cuda').manual_seed(j['seed']),use_kv_cache=False,callback_on_step_end=callback).images[0]
            raw.save(d/'raw_rgba.png');write(d/'reference_conditioning_audit.json',dict(pipe.last_conditioning_audit,hint_steps=hint_audit,reference_keys=reference_keys))
            generated=raw.convert('RGBA').resize(src.size,Image.Resampling.LANCZOS)
            # Native alpha is honored by compositing it over the real source.
            candidate=np.asarray(Image.alpha_composite(src.convert('RGBA'),generated).convert('RGB'),dtype=float)
            original=np.asarray(src,dtype=float);editable=np.asarray(mask.convert('L'))>127
            alpha=np.clip(distance_transform_edt(editable)/5,0,1)[...,None]
            final=np.uint8(np.clip(np.round(original*(1-alpha)+candidate*alpha),0,255))
            assert np.array_equal(final[~editable],original[~editable]);Image.fromarray(final).save(d/'composited.png')
            opacity=np.asarray(raw.convert('RGBA'))[:,:,3]
            row={'id':j['id'],'case_id':j['case_id'],'method':j['method'],'seed':j['seed'],'seconds':time.time()-start,
                 'raw_size':list(raw.size),'raw_mode':raw.mode,'alpha_min':int(opacity.min()),'nonopaque_fraction':float((opacity<254).mean()),
                 'peak_allocated_gib':torch.cuda.max_memory_allocated(0)/1024**3,'outside_source_exact':True,'raw_generator_is_composited':False,
                 'files':{p.name:sha(p) for p in d.iterdir()}}
            write(d/'result.json',row);state['completed'].append(row);write(mp,state)
        state.update(status='complete_unreviewed',finished_unix=time.time());write(mp,state)
    except BaseException as e:
        state.update(status='technical_failure',error=repr(e),updated_unix=time.time());write(mp,state)
        (a.output/'FAILED.txt').write_text(traceback.format_exc());raise

if __name__=='__main__':main()
