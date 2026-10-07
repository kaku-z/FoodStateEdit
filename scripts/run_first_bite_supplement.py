"""Execute disjoint cells of a frozen first-bite matrix with no output compositor."""
import argparse,hashlib,inspect,json,os,socket,time,traceback
from pathlib import Path
from PIL import Image
from run_qwen_image_edit_direct_baseline import gpu_snapshot,gate_snapshot,foreign_process_check,sha256_file


def write(path,obj):
    p=path.with_suffix('.tmp');p.write_text(json.dumps(obj,indent=2),encoding='utf-8');p.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--physical-gpus',required=True);p.add_argument('--shard',type=int,required=True)
    p.add_argument('--shards',type=int,default=4)
    a=p.parse_args();c=json.loads((a.bundle/'config.json').read_text())
    gpus=[int(x) for x in a.physical_gpus.split(',')]
    if len(gpus)!=2 or len(set(gpus))!=2:raise ValueError('Two distinct GPUs required')
    by_id={case['case_id']:case for case in c['cases']}
    matrix=[(by_id[job['case_id']],job['method'],job['seed']) for job in c['jobs']]
    if len(matrix)!=c['expected_calls']:raise ValueError('Explicit matrix size mismatch')
    if len({(case['case_id'],method,seed) for case,method,seed in matrix})!=len(matrix):raise ValueError('Duplicate cell')
    jobs=[job for i,job in enumerate(matrix) if i%a.shards==a.shard]
    if not jobs:raise ValueError('Empty shard')
    a.output.mkdir(parents=True,exist_ok=False)
    manifest={'status':'preflight','host':socket.gethostname(),'started_unix':time.time(),'shard':a.shard,
              'shards':a.shards,'physical_gpus':gpus,'config_sha256':sha256_file(a.bundle/'config.json'),
              'runner_sha256':sha256_file(Path(__file__)),'expected_cells':[f"{case['case_id']}__{m}__{s}" for case,m,s in jobs],
              'completed':[],'failed':[],'model_calls':0}
    mp=a.output/'run_manifest.json';write(mp,manifest)
    try:
        for case in c['cases']:
            for info in case['files'].values():
                if sha256_file(a.bundle/info['path'])!=info['sha256']:raise ValueError('Input hash mismatch')
        model=Path(c['backend']['model_root']);checks={}
        for rel,info in c['backend']['weight_files'].items():
            path=model/rel;checks[rel]=path.stat().st_size==info['size_bytes'] and sha256_file(path)==info['sha256']
            manifest['weights_checked']=len(checks);write(mp,manifest)
        snapshots=[gpu_snapshot(g) for g in gpus]
        resources=[gate_snapshot(s,c['resource_gate']) for s in snapshots]
        write(a.output/'preflight.json',{'weights':checks,'snapshots':snapshots,'resources':resources})
        if not all(checks.values()) or not all(all(x.values()) for x in resources):raise RuntimeError('Preflight failed')
        os.environ.update(CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                          TOKENIZERS_PARALLELISM='false',TORCH_COMPILE_DISABLE='1',OMP_NUM_THREADS='4',
                          PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        import torch
        import diffusers
        from diffusers import QwenImageEditPlusPipeline
        torch.set_num_threads(4)
        pipeline_sha=sha256_file(Path(inspect.getfile(QwenImageEditPlusPipeline)))
        if pipeline_sha!=c['expected_pipeline_sha256']:raise RuntimeError('Pipeline changed')
        manifest.update(status='loading_model',torch_version=torch.__version__,diffusers_version=diffusers.__version__,pipeline_sha256=pipeline_sha)
        write(mp,manifest)
        pipe=QwenImageEditPlusPipeline.from_pretrained(str(model),torch_dtype=torch.bfloat16,local_files_only=True,
              device_map='balanced',max_memory={0:'44GiB',1:'44GiB'})
        manifest.update(device_map=pipe.hf_device_map,status='model_ready');write(mp,manifest)
        # A callback only logs progress and never replaces latents.
        def progress(pipe,step,timestep,kwargs):
            manifest.update(step=int(step),updated_unix=time.time());write(mp,manifest)
            return kwargs
        inf=c['inference']
        for case,method,seed in jobs:
            for snap in snapshots:foreign_process_check(snap['uuid'])
            name=f"{case['case_id']}__{method}__{seed}";d=a.output/name;d.mkdir()
            photo=Image.open(a.bundle/case['files']['source.png']['path']).convert('RGB')
            inputs=[photo]
            if method.startswith('C_'):inputs.append(Image.open(a.bundle/case['files']['planar.png']['path']).convert('RGB'))
            if method.startswith('D_'):inputs.append(Image.open(a.bundle/case['files']['geometry.png']['path']).convert('RGB'))
            W,H=case['output_size']
            manifest.update(status='generating',current=name,step=-1,updated_unix=time.time())
            manifest['model_calls']+=1;write(mp,manifest);start=time.time()
            call={'name':name,'case_id':case['case_id'],'method':method,'seed':seed,'input_count':len(inputs),
                  'width':W,'height':H,'prompt_sha256':hashlib.sha256(case['prompts'][method].encode()).hexdigest()}
            write(d/'request.json',call)
            try:
                with torch.inference_mode():
                    generator=torch.Generator(device=pipe._execution_device).manual_seed(seed)
                    result=pipe(image=inputs if len(inputs)>1 else inputs[0],prompt=case['prompts'][method],
                                negative_prompt=inf['negative_prompt'],true_cfg_scale=inf['true_cfg_scale'],
                                guidance_scale=inf['guidance_scale'],num_inference_steps=inf['num_inference_steps'],
                                width=W,height=H,generator=generator,callback_on_step_end=progress).images[0]
                result.save(d/'raw.png')
                call.update(duration_seconds=time.time()-start,raw_size=list(result.size),files={p.name:sha256_file(p) for p in d.iterdir()})
                manifest['completed'].append(call)
            except Exception as exc:
                (d/'FAILED.txt').write_text(traceback.format_exc())
                manifest['failed'].append(dict(call,error=repr(exc),duration_seconds=time.time()-start))
                write(mp,manifest)
                # Do not retry a failed cell. Fatal CUDA errors end this shard.
                raise
            manifest.update(updated_unix=time.time());write(mp,manifest)
        manifest.update(status='complete_unreviewed',duration_seconds=time.time()-manifest['started_unix']);write(mp,manifest)
        (a.output/'COMPLETE').write_text('complete_unreviewed\n')
    except BaseException as exc:
        manifest.update(status='technical_failure',error=repr(exc),duration_seconds=time.time()-manifest['started_unix']);write(mp,manifest)
        (a.output/'FAILED.txt').write_text(traceback.format_exc());raise
    print(json.dumps({'status':manifest['status'],'completed':len(manifest['completed'])}),flush=True)


if __name__=='__main__':main()
