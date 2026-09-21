"""Consume frozen lossless controls; never download or train model weights."""
import argparse,contextlib,json,os,signal,sys,threading,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from PIL import Image
from run_high_lift_vace_pilot import resource_preflight,validate_file,sha256,run_text,write_json,project_outputs

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-config',type=Path,required=True);ap.add_argument('--controls',type=Path,required=True);ap.add_argument('--manifest-sha256',required=True);ap.add_argument('--job',required=True);ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    cfg=json.loads(a.base_config.read_text());manifest_path=a.controls/'manifest.json'
    assert sha256(manifest_path)==a.manifest_sha256
    manifest=json.loads(manifest_path.read_text());spec=manifest['config'];job=next(j for j in manifest['jobs'] if j['id']==a.job)
    data=a.controls/a.job
    for name,rec in job['files'].items():validate_file(data/name,rec)
    output=a.root/a.job;gate=resource_preflight(cfg,output,a.root/(a.job+'_preflight.json'),a.gpu)
    output.mkdir(exist_ok=False);start=time.monotonic();done=threading.Event()
    state=dict(schema='foodstateedit.spatial_stress_run.v1',status='initializing',job=a.job,pid=os.getpid(),gpu=a.gpu,started_utc=datetime.now(timezone.utc).isoformat(),pipeline_load_count=0,completed=[],manifest_sha256=a.manifest_sha256,base_config_sha256=sha256(a.base_config),runner_sha256=sha256(Path(__file__)),seed=spec['seed'],frames=21,steps=20,lora=False,ttm=False,hard_timeout_seconds=spec['hard_timeout_seconds_per_job'],claim_limit=spec['claim_limit'])
    write_json(output/'run_manifest.json',state)
    def abort(signum,frame):raise RuntimeError('Own worker stopped at safety/deadline boundary')
    signal.signal(signal.SIGTERM,abort);signal.signal(signal.SIGALRM,abort);signal.alarm(spec['hard_timeout_seconds_per_job'])
    def foreign():
        for line in run_text(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits']).splitlines():
            fields=[v.strip() for v in line.split(',')]
            if len(fields)>=4 and fields[0]==gate['gpu']['uuid'] and int(fields[1])!=os.getpid():raise RuntimeError('Foreign compute process '+line)
    def watch():
        while not done.wait(10):
            try:foreign()
            except Exception as exc:
                (output/'resource_stop.txt').write_text(repr(exc));os.kill(os.getpid(),signal.SIGTERM);return
    try:
        runtime=cfg['runtime'];validate_file(Path(runtime['model_hash_audit']['path']),runtime['model_hash_audit'])
        for name,expected in runtime['geoedit_files'].items():assert sha256(Path(runtime['geoedit_root'])/name)==expected
        for name,rec in runtime['model_files'].items():validate_file(Path(runtime['model_root'])/name,rec,hash_file=False)
        reference=Image.open(data/'reference.png').convert('RGB');alpha=Image.open(data/'shared_alpha.png').convert('RGB');w,h=reference.size
        arrays=np.load(data/'controls.npz',allow_pickle=False)
        assert sorted(arrays.files)==sorted(spec['arms'])
        for name in spec['arms']:assert arrays[name].shape==(21,h,w,3) and arrays[name].dtype==np.uint8
        os.environ.update(runtime['offline_environment']);os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),HF_HOME=str(output/'cache'),TRANSFORMERS_CACHE=str(output/'cache'),DIFFSYNTH_MODEL_BASE_PATH=str(Path(runtime['model_root']).parent.parent))
        sys.path.insert(0,runtime['geoedit_root']);foreign();threading.Thread(target=watch,daemon=True).start()
        state['status']='loading_pipeline';write_json(output/'run_manifest.json',state)
        from geoedit import inference
        with (output/'pipeline.log').open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):pipe=inference.load_pipeline(inference.resolve_vram_limit(None))
        state['pipeline_load_count']=1
        for name in spec['arms']:
            foreign();folder=output/name;folder.mkdir(exist_ok=False);begin=time.monotonic()
            state.update(status='running',active=name);write_json(output/'run_manifest.json',state)
            with (folder/'inference.log').open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                video=pipe(prompt=job['prompt'],negative_prompt=job['negative_prompt'],height=h,width=w,num_frames=21,num_inference_steps=20,vace_video=[Image.fromarray(f) for f in arrays[name]],vace_video_mask=[alpha]*21,vace_reference_image=reference,vace_scale=1.,enable_ttm=False,seed=spec['seed'],tiled=True)
                lossless=folder/'raw_frames';lossless.mkdir()
                for i,frame in enumerate(video):frame.save(lossless/f'{i:02d}.png')
                assert len(video)==21
                inference.save_video(video,str(folder/'raw.mp4'),fps=8,quality=5);del video
            project_outputs(folder/'raw.mp4',data/'reference.png',data/'shared_alpha.png',folder,21,8,[0,3,6,10,15,20])
            foreign();record=dict(condition=name,wall_seconds=time.monotonic()-begin,files={str(p.relative_to(folder)):dict(size_bytes=p.stat().st_size,sha256=sha256(p)) for p in folder.rglob('*') if p.is_file()})
            write_json(folder/'condition_manifest.json',record);state['completed'].append(record);write_json(output/'run_manifest.json',state);print('COMPLETED',a.job,name,flush=True)
        state['status']='complete_requires_visual_review'
    except BaseException as exc:
        state.update(status='technical_failure_preserved',error=repr(exc));(output/'failure.txt').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True)
    finally:
        done.set();signal.alarm(0);state.update(finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-start);write_json(output/'run_manifest.json',state)
    return 0 if state['status']=='complete_requires_visual_review' else 3
if __name__=='__main__':sys.exit(main())
