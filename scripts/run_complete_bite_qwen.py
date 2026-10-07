"""Matched Qwen endpoint comparison with optional staged geometry projection."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
import time
import traceback


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(16*1024**2),b''):h.update(data)
    return h.hexdigest()


def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    c=json.loads(a.config.read_text());jobs=[j for i,j in enumerate(c['jobs']) if i%a.shards==a.shard]
    a.output.mkdir(parents=True,exist_ok=False);mp=a.output/'manifest.json';start=time.time()
    (a.output/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    state={'status':'preflight','host':socket.gethostname(),'pid':os.getpid(),'gpus':a.gpus,'started_unix':start,
        'config_sha256':sha(a.config),'script_sha256':sha(__file__),'expected_jobs':[j['id'] for j in jobs],'completed':[],'failed':[]}
    if c.get('shared_claim_root'):state['expected_jobs']=[]
    write(mp,state)
    try:
        from run_qwen_image_edit_direct_baseline import gpu_snapshot,foreign_process_check
        gpus=[int(x) for x in a.gpus.split(',')];assert len(gpus)==2 and len(set(gpus))==2
        snaps=[gpu_snapshot(g) for g in gpus]
        assert all(not s['compute_processes'] and s['memory_free_mib']>47500 for s in snaps),snaps
        assert snaps[0]['host_mem_available_mib']>50000
        state['resource_snapshots']=snaps
        audit=json.loads(Path(c['model_audit']).read_text());assert audit['all_verified'] and audit['expected_model_files']==c['backend']['weight_files']
        for name,info in c['backend']['weight_files'].items():assert (Path(c['backend']['model_root'])/name).stat().st_size==info['size_bytes']
        for job in jobs:
            for info in job['files'].values():assert sha(info['path'])==info['sha256']
        os.environ.update(CUDA_VISIBLE_DEVICES=a.gpus,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
            OMP_NUM_THREADS='4',TORCH_COMPILE_DISABLE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        import numpy as np
        from PIL import Image
        import torch
        import torch.nn.functional as F
        from diffusers import QwenImageEditPlusPipeline
        torch.set_num_threads(4)
        assert sha(inspect.getfile(QwenImageEditPlusPipeline))==c['expected_pipeline_sha256']
        state.update(status='loading_model',updated_unix=time.time());write(mp,state)
        pipe=QwenImageEditPlusPipeline.from_pretrained(c['backend']['model_root'],torch_dtype=torch.bfloat16,local_files_only=True,
            device_map='balanced',max_memory={0:'44GiB',1:'44GiB'})
        state.update(status='ready',device_map=pipe.hf_device_map,load_seconds=time.time()-start);write(mp,state)
        inf=c['inference'];dtype=torch.bfloat16;device=pipe._execution_device;channels=pipe.transformer.config.in_channels//4
        for job in jobs:
            for s in snaps:foreign_process_check(s['uuid'])
            if c.get('shared_claim_root'):
                claim=Path(c['shared_claim_root'])/job['id']
                try:claim.mkdir(parents=False,exist_ok=False)
                except FileExistsError:continue
                write(claim/'owner.json',{'pid':os.getpid(),'gpus':a.gpus,'output':str(a.output/job['id'])})
                state['expected_jobs'].append(job['id']);write(mp,state)
            folder=a.output/job['id'];folder.mkdir(exist_ok=False);begin=time.time();audit_steps=[]
            state.update(status='generating',current=job['id'],step=-1,updated_unix=time.time());write(mp,state)
            def img(key):return Image.open(job['files'][key]['path']).convert('RGB')
            source=img('source');W,H=source.size;images=[source]
            if job.get('control'):images.append(img(job['control']))
            if job.get('input_order'):images=[img(key) for key in job['input_order']]
            generator=torch.Generator(device=device).manual_seed(job['seed'])
            with torch.inference_mode():noise,_=pipe.prepare_latents(None,1,channels,H,W,dtype,device,generator)
            target=None;weights={};params=job.get('projection')
            if params:
                with torch.inference_mode():
                    vae_input=pipe.image_processor.preprocess(img('rgb_control'),height=H,width=W).unsqueeze(2).to(pipe.vae.device,dtype)
                    grid=pipe._encode_vae_image(vae_input,generator).to(device);lh,lw=grid.shape[-2:]
                    target=pipe._pack_latents(grid,1,channels,lh,lw)
                for key in ['rigid_mask','contact_mask','material_mask','hole_mask']:
                    mask=np.asarray(Image.open(job['files'][key]['path']).convert('L')).astype(np.float32)/255
                    weight=F.interpolate(torch.as_tensor(mask.copy(),device=device)[None,None],size=(lh,lw),mode='area').to(dtype)
                    wg=weight[:,:,None].expand(1,channels,1,lh,lw).contiguous();weights[key]=pipe._pack_latents(wg,1,channels,lh,lw)
                assert target.shape==noise.shape and all(w.shape==noise.shape for w in weights.values())
            def release(frac,hold,end):return max(0.,min(1.,(end-frac)/(end-hold)))
            def callback(pipeline,step,timestep,kwargs):
                state.update(step=int(step),updated_unix=time.time());write(mp,state)
                before=kwargs['latents'];result=before
                if target is not None:
                    sigma=pipeline.scheduler.sigmas[step+1].to(before.device,before.dtype)
                    clean=target.to(before.device);fixed_noise=noise.to(before.device);delta=(1-sigma)*clean+sigma*fixed_noise-before
                    frac=(step+1)/inf['steps']
                    rigid=weights['rigid_mask']*params['rigid_max']*release(frac,params['rigid_hold'],params['rigid_end'])
                    contact=weights['contact_mask']*params['contact_max']*release(frac,params['contact_hold'],params['contact_end'])
                    full=torch.maximum(rigid,contact).to(before.device)
                    coarse=torch.maximum(weights['material_mask'],weights['hole_mask'])*params['material_max']*release(frac,params['material_hold'],params['material_end'])
                    coarse=coarse.to(before.device)*(1-full)
                    grid_delta=pipeline._unpack_latents(delta,H,W,pipeline.vae_scale_factor)[:,:,0]
                    low=F.avg_pool2d(F.pad(grid_delta,(1,1,1,1),mode='reflect'),3,stride=1).unsqueeze(2)
                    low=pipeline._pack_latents(low,1,channels,lh,lw)
                    result=before+full*delta+coarse*low
                    assert torch.isfinite(result).all()
                    outside=(full+coarse)==0
                    max_out=float((result[outside]-before[outside]).abs().max()) if outside.any() else 0.
                    assert max_out==0
                    audit_steps.append({'step':int(step),'next_sigma':float(sigma),'full_weight_max':float(full.max()),
                        'coarse_weight_max':float(coarse.max()),'outside_support_update_max':max_out,
                        'update_rms':float((result.float()-before.float()).square().mean().sqrt())})
                return {'latents':result}
            request=dict(job,noise_sha256=hashlib.sha256(noise.float().cpu().numpy().tobytes()).hexdigest())
            write(folder/'request.json',request)
            for g in range(2):torch.cuda.reset_peak_memory_stats(g)
            with torch.inference_mode():
                result=pipe(image=images if len(images)>1 else source,prompt=job['prompt'],negative_prompt=inf['negative_prompt'],
                    true_cfg_scale=inf['true_cfg_scale'],guidance_scale=None,num_inference_steps=inf['steps'],width=W,height=H,
                    generator=generator,latents=noise.clone(),callback_on_step_end=callback).images[0]
            result.save(folder/'raw.png');write(folder/'projection_audit.json',{'enabled':target is not None,'steps':audit_steps,
                'no_pixel_compositor':True,'target':'RGB rendering of the same cut and lifted 3-D solid, not the unchanged source pose'})
            if target is not None:assert audit_steps[-1]['full_weight_max']==0 and audit_steps[-1]['coarse_weight_max']==0
            record={'id':job['id'],'case_id':job['case_id'],'method':job['method'],'seed':job['seed'],'seconds':time.time()-begin,
                'peak_allocated_gib':[torch.cuda.max_memory_allocated(g)/1024**3 for g in range(2)],
                'files':{f.name:sha(f) for f in folder.iterdir()}}
            write(folder/'result.json',record);state['completed'].append(record);write(mp,state)
            del target,weights,noise
        state.update(status='complete_unreviewed',elapsed_seconds=time.time()-start,updated_unix=time.time());write(mp,state)
        (a.output/'COMPLETE').write_text('complete_unreviewed\n')
    except BaseException as exc:
        state.update(status='technical_failure',error=repr(exc),updated_unix=time.time());write(mp,state)
        (a.output/'FAILED.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
