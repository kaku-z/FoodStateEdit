"""Frozen still-frame endpoint experiments using the existing offline VACE runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(16*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    c=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    mp=a.output/'manifest.json';started=time.time()
    jobs=[j for i,j in enumerate(c['jobs']) if i%a.shards==a.shard]
    state={'status':'preflight','host':socket.gethostname(),'pid':os.getpid(),'gpu':a.gpu,'started_unix':started,
           'config_sha256':sha(a.config),'script_sha256':sha(__file__),'expected_jobs':[j['id'] for j in jobs],'completed':[],'failed':[]}
    if c.get('shared_claim_root'):state['expected_jobs']=[]
    write(mp,state)
    try:
        from run_qwen_image_edit_direct_baseline import gpu_snapshot,foreign_process_check
        snap=gpu_snapshot(a.gpu)
        assert not snap['compute_processes'] and snap['memory_free_mib']>47500,snap
        assert snap['host_mem_available_mib']>100000,snap
        state['resource_snapshot']=snap
        runtime=c['runtime'];model=Path(runtime['model_root'])
        # Verify bytes once in a launch-side audit shared by all resident shards.
        audit=json.loads(Path(c['model_audit']).read_text())
        assert audit['all_verified'] and audit['expected_model_files']==runtime['model_files']
        for name,info in runtime['model_files'].items():assert (model/name).stat().st_size==info['size_bytes']
        for name,h in runtime['geoedit_files'].items():assert sha(Path(runtime['geoedit_root'])/name)==h,name
        for job in jobs:
            for name,info in job['files'].items():assert sha(info['path'])==info['sha256'],name
        os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),DIFFSYNTH_SKIP_DOWNLOAD='true',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                          TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',HF_HOME=c['cache_root'],
                          DIFFSYNTH_MODEL_BASE_PATH=str(model.parent.parent),PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        sys.path.insert(0,runtime['geoedit_root'])
        import torch
        from PIL import Image
        from geoedit import inference
        torch.set_num_threads(4)
        state.update(status='loading_model',updated_unix=time.time());write(mp,state)
        pipe=inference.load_pipeline(45.0)
        state.update(status='ready',load_seconds=time.time()-started,torch_version=torch.__version__);write(mp,state)
        inf=c['inference'];n=inf['num_frames']
        def progress(iterable,**kwargs):
            for index,item in enumerate(iterable):
                state.update(step=index,updated_unix=time.time());write(mp,state)
                yield item
        for job in jobs:
            foreign_process_check(snap['uuid'])
            if c.get('shared_claim_root'):
                claim=Path(c['shared_claim_root'])/job['id']
                try:claim.mkdir(parents=False,exist_ok=False)
                except FileExistsError:continue
                write(claim/'owner.json',{'pid':os.getpid(),'gpu':a.gpu,'output':str(a.output/job['id'])})
                state['expected_jobs'].append(job['id']);write(mp,state)
            folder=a.output/job['id'];folder.mkdir(exist_ok=False);begin=time.time()
            state.update(status='generating',current=job['id'],step=-1,updated_unix=time.time());write(mp,state)
            def img(key,mode='RGB'):return Image.open(job['files'][key]['path']).convert(mode)
            source=img('source');control=img(job['control']);mask=img('edit_mask')
            W,H=source.size
            call=dict(prompt=job['prompt'],negative_prompt=inf['negative_prompt'],height=H,width=W,num_frames=n,
                num_inference_steps=inf['steps'],cfg_scale=inf['cfg_scale'],vace_scale=inf['vace_scale'],
                vace_video=[control.copy() for _ in range(n)],vace_video_mask=[mask.copy() for _ in range(n)],
                vace_reference_image=source,seed=job['seed'],tiled=True,progress_bar_cmd=progress)
            if job.get('warm_start_strength') is not None:
                call.update(input_video=[img('rgb_control') for _ in range(n)],denoising_strength=job['warm_start_strength'])
            if job.get('ttm'):
                spec=job['ttm'];call.update(enable_ttm=True,motion_signal_video=[img('rgb_control') for _ in range(n)],
                    motion_signal_mask=[img('rigid_mask') for _ in range(n)],ttm_contact_mask=[img('contact_mask') for _ in range(n)],
                    ttm_material_mask=[img('material_mask') for _ in range(n)],ttm_hole_mask=[img('hole_mask') for _ in range(n)],
                    ttm_replace_mode='mask_new',ttm_warm_start=False,tweak_index=0,tstrong_index=spec['rigid_end'],
                    contact_tstrong_index=spec['contact_end'],material_tstrong_index=spec['material_end'],hole_tstrong_index=spec['hole_end'])
            write(folder/'request.json',job)
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():frames=pipe(**call)
            assert len(frames)==n,(len(frames),n)
            for i,frame in enumerate(frames):frame.save(folder/f'frame_{i:03d}.png')
            frames[inf['selected_frame']].save(folder/'raw.png')
            inference.save_video(frames,str(folder/'all_frames.mp4'),fps=8,quality=7)
            result={'id':job['id'],'seed':job['seed'],'method':job['method'],'case_id':job['case_id'],
                    'seconds':time.time()-begin,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
                    'selected_frame':inf['selected_frame'],'selection':'fixed before generation; no output-dependent frame selection',
                    'files':{f.name:sha(f) for f in folder.iterdir() if f.is_file()}}
            write(folder/'result.json',result);state['completed'].append(result);write(mp,state)
        state.update(status='complete_unreviewed',elapsed_seconds=time.time()-started,updated_unix=time.time());write(mp,state)
        (a.output/'COMPLETE').write_text('complete_unreviewed\n')
    except BaseException as exc:
        state.update(status='technical_failure',error=repr(exc),updated_unix=time.time());write(mp,state)
        (a.output/'FAILED.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
