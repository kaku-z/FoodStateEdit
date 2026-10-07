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
        import torch.nn.functional as F
        from scipy.ndimage import distance_transform_edt
        from safetensors.torch import load_file
        from diffusers import FlowMatchEulerDiscreteScheduler
        from videox_fun.models import AutoencoderKLQwenImage21,Qwen3VLForConditionalGeneration,Qwen3VLProcessor,QwenImage21ControlTransformer2DModel
        from videox_fun.pipeline.pipeline_qwenimage21_reference_control import QwenImage21ControlPipeline
        from videox_fun.pipeline.pipeline_qwenimage21_reference_control import calculate_shift,retrieve_timesteps
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

            state.update(status='generating',current=j['id'],step=-1,updated_unix=time.time());write(mp,state)
            gen=torch.Generator('cuda').manual_seed(j['seed'])
            with torch.inference_mode():
                noise,_=pipe.prepare_latents(None,1,pipe.transformer.config.in_channels,H,W,dtype,pipe._execution_device,gen)
                x=pipe.image_processor.preprocess(src,height=H,width=W).unsqueeze(2)
                x=torch.cat([x,torch.ones_like(x[:,:1])],dim=1)
                grid=pipe._encode_vae_image(x.to(pipe._execution_device,dtype),gen)
                lh,lw=grid.shape[-2:];target=pipe._pack_latents(grid,1,grid.shape[1],lh,lw)
                editable=F.interpolate(torch.tensor(np.asarray(mask.convert('L')).copy()/255.,device=target.device)[None,None].float(),size=(lh,lw),mode='area')
                editable=F.max_pool2d(editable,3,stride=1,padding=1)
                keep=(editable<.001).to(dtype)[:,:,None].expand(1,grid.shape[1],1,lh,lw).contiguous()
                keep=pipe._pack_latents(keep,1,grid.shape[1],lh,lw)
                schedule=np.linspace(j['start_raw_sigma'],1/cfg['inference']['steps'],cfg['inference']['steps']).tolist()
                mu=calculate_shift(noise.shape[1],pipe.scheduler.config.get('base_image_seq_len',256),pipe.scheduler.config.get('max_image_seq_len',4096),pipe.scheduler.config.get('base_shift',.5),pipe.scheduler.config.get('max_shift',1.15))
                retrieve_timesteps(pipe.scheduler,cfg['inference']['steps'],pipe._execution_device,sigmas=schedule,mu=mu)
                s0=pipe.scheduler.sigmas[0].to(target.device,dtype)
                initial=(1-s0)*target+s0*noise
            request=dict(j,actual_start_sigma=float(s0),initial_latents_sha256=hashlib.sha256(initial.float().cpu().numpy().tobytes()).hexdigest(),context_lock='Noisy encoded source outside the dilated food latent mask at every step')
            write(d/'request.json',request);context_audit=[]
            def callback(pipeline,step,timestep,kwargs):
                state.update(step=int(step),updated_unix=time.time());write(mp,state)
                assert torch.isfinite(kwargs['latents']).all()
                sigma=pipeline.scheduler.sigmas[step+1].to(target.device,dtype)
                context=(1-sigma)*target+sigma*noise
                after=keep*context+(1-keep)*kwargs['latents']
                assert torch.equal(after[keep==1],context[keep==1])
                kwargs['latents']=after
                context_audit.append({'step':int(step),'sigma':float(sigma),'fixed_latent_fraction':float(keep.mean()),'context_equality':True})
                return kwargs
            _allocator_probe=torch.empty(1,device="cuda");del _allocator_probe
            torch.cuda.reset_peak_memory_stats(0)
            with torch.inference_mode():
                raw=pipe(prompt=j['prompt'],image=src,mask_image=mask,control_image=control,
                    reference_images=references or None,reference_resolution=512,
                    control_context_scale=j.get('control_scale',1.),true_cfg_scale=cfg['inference']['true_cfg_scale'],
                    height=H,width=W,num_inference_steps=cfg['inference']['steps'],
                    generator=gen,latents=initial,sigmas=schedule,use_kv_cache=False,callback_on_step_end=callback).images[0]
            raw.save(d/'raw_rgba.png');write(d/'reference_conditioning_audit.json',dict(pipe.last_conditioning_audit,hint_steps=hint_audit,reference_keys=reference_keys))
            write(d/'context_audit.json',{'steps':context_audit,'geometry_accuracy_guaranteed':False,'raw_generator_is_composited':False})
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

