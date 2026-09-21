"""Bounded two-direction mechanism diagnostic. No training, no model download."""
import argparse, contextlib, copy, json, os, signal, sys, threading, time, traceback
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageFilter
from run_high_lift_vace_pilot import resource_preflight, validate_file, sha256, run_text, write_json, project_outputs
from build_multimaterial_pilot import render_solid_case, SPECS

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-config',type=Path,required=True);ap.add_argument('--case',choices=['soup','cake'],required=True);ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args();cfg=json.loads(args.base_config.read_text());case=args.case;output=args.root/case
    gate=resource_preflight(cfg,output,args.root/(case+'_preflight.json'),args.gpu)
    output.mkdir(exist_ok=False);started=time.monotonic()
    state=dict(schema='foodstateedit.teacher_counterfactual.v1',status='preparing',pid=os.getpid(),case=case,gpu=args.gpu,started_utc=datetime.now(timezone.utc).isoformat(),pipeline_load_count=0,completed=[],seed=2,frames=21,steps=20,scale=1,training=False,lora=False,ttm=False,hard_timeout_seconds=1500,
      claim_limit='Two development cases and one seed each. Counterfactual control-response diagnostic, not heldout efficacy or proof of 3D superiority.')
    write_json(output/'run_manifest.json',state)
    def abort(signum,frame):raise RuntimeError('Own worker stopped: hard deadline or foreign GPU process')
    signal.signal(signal.SIGALRM,abort);signal.signal(signal.SIGTERM,abort);signal.alarm(1500)
    done=threading.Event()
    def foreign():
        apps=run_text(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'])
        for line in apps.splitlines():
            parts=[x.strip() for x in line.split(',')]
            if len(parts)>=4 and parts[0]==gate['gpu']['uuid'] and int(parts[1])!=os.getpid():raise RuntimeError('Foreign process: '+line)
    def watch():
        while not done.wait(10):
            try:foreign()
            except Exception as e:
                (output/'resource_stop.txt').write_text(repr(e));os.kill(os.getpid(),signal.SIGTERM);return
    try:
        data=Path(cfg['dataset_root']);validate_file(data/'dataset_manifest.json',cfg['dataset_manifest']);dataset=json.loads((data/'dataset_manifest.json').read_text());meta=next(x for x in dataset['cases'] if x['case_id']==case)
        for rec in meta['files'].values():validate_file(data/rec['path'],rec)
        runtime=cfg['runtime'];validate_file(Path(runtime['model_hash_audit']['path']),runtime['model_hash_audit'])
        for name,expected in runtime['geoedit_files'].items():assert sha256(Path(runtime['geoedit_root'])/name)==expected
        for name,rec in runtime['model_files'].items():validate_file(Path(runtime['model_root'])/name,rec,hash_file=False)
        refpath=data/case/'reference.png';reference=Image.open(refpath).convert('RGB');source=np.array(reference);w,h=reference.size
        before=copy.deepcopy(SPECS[case]);controls={};geometries={}
        for name,arm in [('planar_original','planar'),('relative_original','relative3d'),('relative_counterfactual','relative3d')]:
            SPECS[case]=copy.deepcopy(before)
            if name=='relative_counterfactual':SPECS[case]['final'][0]=.60 if case=='soup' else .35
            controls[name],geometries[name]=render_solid_case(source,case,arm)
        SPECS[case]=before
        union=np.zeros((h,w),bool)
        for frames in controls.values():union|=np.any(np.stack(frames)!=source,axis=(0,3))
        alpha=np.array(Image.fromarray(union.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(33)))
        alphapath=output/'shared_alpha.png';Image.fromarray(alpha).save(alphapath)
        write_json(output/'geometry.json',geometries)
        for name,frames in controls.items():
            folder=output/name;folder.mkdir();Image.fromarray(frames[-1]).save(folder/'control_final.png')
        state.update(status='loading_pipeline',input_sha256=sha256(refpath),mask_sha256=sha256(alphapath),mask_policy='binary dilated union of all three control sequences, identical across arms',control_endpoints={k:v['trace'][-1]['anchor_uv'] for k,v in geometries.items()},prompt=meta['prompt'],base_config_sha256=sha256(args.base_config),runner_sha256=sha256(Path(__file__)))
        write_json(output/'run_manifest.json',state)
        os.environ.update(runtime['offline_environment']);os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),HF_HOME='/tmp/teacher_cf_cache_'+case,TRANSFORMERS_CACHE='/tmp/teacher_cf_cache_'+case,DIFFSYNTH_MODEL_BASE_PATH=str(Path(runtime['model_root']).parent.parent))
        sys.path.insert(0,runtime['geoedit_root']);foreign();threading.Thread(target=watch,daemon=True).start()
        from geoedit import inference
        with (output/'pipeline.log').open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):pipe=inference.load_pipeline(inference.resolve_vram_limit(None))
        state['pipeline_load_count']=1
        traces=[]
        def record_unit(unit):
            original=unit.process
            def wrapped(*a,**kw):
                result=original(*a,**kw)
                if isinstance(result,dict):
                    shapes={k:list(v.shape) for k,v in result.items() if hasattr(v,'shape')}
                    if shapes:traces.append(dict(unit=type(unit).__name__,shapes=shapes));write_json(output/'condition_shape_trace.json',traces)
                return result
            unit.process=wrapped
        for unit in pipe.units:
            if type(unit).__name__ in ['WanVideoUnit_VACE','WanVideoUnit_NoiseInitializer']:record_unit(unit)
        for name,frames in controls.items():
            foreign();begin=time.monotonic();state.update(status='running',active=name);write_json(output/'run_manifest.json',state)
            folder=output/name
            with (folder/'inference.log').open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                video=pipe(prompt=meta['prompt'],negative_prompt=meta['negative_prompt'],height=h,width=w,num_frames=21,num_inference_steps=20,
                    vace_video=[Image.fromarray(f) for f in frames],vace_video_mask=[Image.fromarray(alpha).convert('RGB')]*21,vace_reference_image=reference,vace_scale=1.,enable_ttm=False,seed=2,tiled=True)
                # Preserve lossless frame 20 before MP4 encoding as well as the full raw video.
                video[-1].save(folder/'raw_final_lossless.png')
                inference.save_video(video,str(folder/'raw.mp4'),fps=8,quality=5)
                del video
            project_outputs(folder/'raw.mp4',refpath,alphapath,folder,21,8,[0,3,6,10,15,20])
            foreign();rec=dict(condition=name,wall_seconds=time.monotonic()-begin,files={p.name:dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in folder.iterdir() if p.is_file()})
            state['completed'].append(rec);write_json(folder/'condition_manifest.json',rec);write_json(output/'run_manifest.json',state);print('COMPLETED',case,name,flush=True)
        state['status']='complete_requires_visual_review'
    except BaseException as e:
        state.update(status='technical_failure_preserved',error=repr(e));(output/'failure.txt').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True)
    finally:
        done.set();signal.alarm(0);state.update(finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-started);write_json(output/'run_manifest.json',state)
    return 0 if state['status'].startswith('complete') else 3
if __name__=='__main__':sys.exit(main())
