"""Soft food-interior ablation with no background lock or image-space compositor."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from foodstateedit.observed_edit.interior import InteriorProjection
from run_qwen_image_edit_direct_baseline import gpu_snapshot,gate_snapshot,foreign_process_check,sha256_file


def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--physical-gpu',type=int,required=True)
    p.add_argument('--condition',choices=['interior','detail'],required=True)
    args=p.parse_args()
    c=json.loads((args.bundle/'config.json').read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.time();manifest_path=args.output/'run_manifest.json'
    manifest={'host':socket.gethostname(),'status':'preflight','condition':args.condition,
              'started_unix':start,'config_sha256':sha256_file(args.bundle/'config.json'),
              'runner_sha256':sha256_file(Path(__file__)),
              'projection_sha256':sha256_file(Path(__file__).resolve().parents[1]/'foodstateedit/observed_edit/interior.py'),
              'completed':[],'physical_gpu':args.physical_gpu}
    write(manifest_path,manifest)
    try:
        for case in c['cases']:
            for f in case['files'].values():
                if sha256_file(args.bundle/f['path'])!=f['sha256']:raise ValueError('Input checksum mismatch')
        checks={};model=Path(c['backend']['model_root'])
        for rel,f in c['backend']['weight_files'].items():
            checks[rel]=(model/rel).stat().st_size==f['size_bytes'] and sha256_file(model/rel)==f['sha256']
            manifest['weight_files_checked']=len(checks);write(manifest_path,manifest)
        snapshot=gpu_snapshot(args.physical_gpu);resource=gate_snapshot(snapshot,c['resource_gate'])
        write(args.output/'preflight.json',{'weights':checks,'gpu':snapshot,'resource':resource})
        if not all(checks.values()) or not all(resource.values()):raise RuntimeError('Preflight failed')
        os.environ.update(CUDA_VISIBLE_DEVICES=str(args.physical_gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                          TOKENIZERS_PARALLELISM='false',TORCH_COMPILE_DISABLE='1',
                          PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        import inspect
        import torch
        import torch.nn.functional as F
        import diffusers
        from diffusers import QwenImageEditPlusPipeline
        pipeline_path=Path(inspect.getfile(QwenImageEditPlusPipeline))
        if sha256_file(pipeline_path)!=c['expected_pipeline_sha256']:
            raise ValueError('Pipeline version differs from inspected callback implementation')
        manifest.update(status='loading_model',torch_version=torch.__version__,diffusers_version=diffusers.__version__,
                        pipeline_sha256=sha256_file(pipeline_path))
        write(manifest_path,manifest)
        pipe=QwenImageEditPlusPipeline.from_pretrained(str(model),torch_dtype=torch.bfloat16,local_files_only=True)
        if pipe.scheduler.__class__.__name__!='FlowMatchEulerDiscreteScheduler':raise ValueError('Unsupported scheduler')
        pipe.enable_model_cpu_offload()
        inf=c['inference'];width,height=inf['width'],inf['height']
        channels=pipe.transformer.config.in_channels//4
        device=torch.device('cuda');dtype=torch.bfloat16
        for case in c['cases']:
            reference=Image.open(args.bundle/case['files']['reference.png']['path']).convert('RGB')
            state=np.load(args.bundle/case['files']['layers.npz']['path'],allow_pickle=False)
            for seed in c['seeds']:
                foreign_process_check(snapshot['uuid'])
                name=f"{case['case_id']}_{seed}_{args.condition}"
                manifest.update(status='generating',current=name,updated_unix=time.time());write(manifest_path,manifest)
                t0=time.time();generator=torch.Generator(device=device).manual_seed(seed)
                with torch.inference_mode():
                    # Explicit noise shared across ablations; VAE uses argmax, not random sampling.
                    noise,_=pipe.prepare_latents(None,1,channels,height,width,dtype,device,generator)
                    ref=pipe.image_processor.preprocess(reference,height,width).unsqueeze(2).to(device,dtype)
                    clean_grid=pipe._encode_vae_image(ref,generator)
                    lh,lw=clean_grid.shape[-2:]
                    clean=pipe._pack_latents(clean_grid,1,channels,lh,lw)
                    # Match the 2x2 latent packing spatially, not a flattened pixel mask.
                    weights=torch.from_numpy(state['core_weight'].astype(np.float32))[None,None].to(device)
                    weights=F.interpolate(weights,size=(lh,lw),mode='area').to(dtype)
                    weight_grid=weights[:, :, None].expand(1,channels,1,lh,lw).contiguous()
                    packed=pipe._pack_latents(weight_grid,1,channels,lh,lw)
                    projection=InteriorProjection(clean,noise,packed,height,width,inf['num_inference_steps'],args.condition,
                                                  maximum=c['projection']['maximum'],release_start=c['projection']['release_start'],
                                                  kernel_size=c['projection']['detail_kernel'])
                    callback=projection
                    result=pipe(image=reference,prompt=case['prompt'],negative_prompt=inf['negative_prompt'],
                                height=height,width=width,num_inference_steps=inf['num_inference_steps'],
                                true_cfg_scale=inf['true_cfg_scale'],guidance_scale=inf['guidance_scale'],
                                generator=generator,latents=noise.clone(),callback_on_step_end=callback).images[0]
                folder=args.output/name;folder.mkdir()
                result.save(folder/'raw.png')
                write(folder/'projection_audit.json',{'enabled':callback is not None,'steps':projection.audit,
                      'packed_shape':list(clean.shape),'unconstrained_fraction':float((packed==0).float().mean()),
                      'mean_core_weight':float(packed.float().mean()),'reference_grid':[lh,lw]})
                record={'name':name,'duration_seconds':time.time()-t0,'raw_size':list(result.size),
                        'files':{f.name:sha256_file(f) for f in folder.iterdir()}}
                manifest['completed'].append(record);write(manifest_path,manifest)
                del clean_grid,clean,ref,packed,weight_grid,weights,noise,projection
        manifest.update(status='complete_unreviewed',duration_seconds=time.time()-start)
        write(manifest_path,manifest);(args.output/'COMPLETE').write_text('complete_unreviewed\n')
    except BaseException as exc:
        manifest.update(status='technical_failure',error=repr(exc),duration_seconds=time.time()-start)
        write(manifest_path,manifest);(args.output/'FAILED').write_text(traceback.format_exc());raise
    print(json.dumps({'status':manifest['status'],'completed':len(manifest['completed'])}),flush=True)


if __name__=='__main__':main()
