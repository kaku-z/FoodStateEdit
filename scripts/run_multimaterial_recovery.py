"""One admitted GPU job, one resident VACE pipeline, only the missing cake relative3d arm."""
import argparse
import contextlib
import json
import os
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from run_high_lift_vace_pilot import sha256, validate_file, resource_preflight, run_text, project_outputs, write_json


def foreign_process_check(gpu_uuid):
    raw=run_text(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'])
    foreign=[]
    for line in raw.splitlines():
        parts=[v.strip() for v in line.split(',')]
        if len(parts)>=4 and parts[0]==gpu_uuid and int(parts[1])!=os.getpid():
            foreign.append(line)
    if foreign:raise RuntimeError('Foreign compute process appeared; stop own batch without preemption: '+repr(foreign))
    return raw


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--gpu-index',type=int,required=True)
    parser.add_argument('--preflight-report',type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    if config['schema_version']!='foodstateedit.multimaterial_execution.v2' or not config['execution_allowed']:
        raise ValueError('Wrong or disabled freeze')
    if config['inference'] != dict(seed=1,num_frames=21,num_inference_steps=20,fps=8,vace_scale=1.0,enable_ttm=False,lora_enabled=False):
        raise ValueError('This runner supports only the exact frozen inference budget')
    if config['selected_conditions'] != ['cake__relative3d'] or config['expected_conditions'] != 1:
        raise ValueError('Recovery is restricted to the one missing non-noodle condition')
    for name,expected in config['implementation_files'].items():
        if sha256(Path(__file__).parent/name)!=expected:raise ValueError('Runtime script hash mismatch: '+name)
    data=Path(config['dataset_root']);output=Path(config['output_root'])
    validate_file(data/'dataset_manifest.json',config['dataset_manifest'])
    dataset=json.loads((data/'dataset_manifest.json').read_text())
    assert [x['case_id'] for x in dataset['cases']]==['soup','rice','cake','noodle']
    assert dataset['arms']==['planar','relative3d']
    for case in dataset['cases']:
        for rec in case['files'].values():validate_file(data/rec['path'],rec)
    runtime=config['runtime']
    validate_file(Path(runtime['model_hash_audit']['path']),runtime['model_hash_audit'])
    model_audit=Path(runtime['model_hash_audit']['path']).read_text()
    for name,rec in runtime['model_files'].items():
        validate_file(Path(runtime['model_root'])/name,rec,hash_file=False)
        assert rec['sha256'] in model_audit and name in model_audit
    for name,digest in runtime['geoedit_files'].items():
        if sha256(Path(runtime['geoedit_root'])/name)!=digest:raise ValueError('GeoEdit hash mismatch')
    preflight=resource_preflight(config,output,args.preflight_report,args.gpu_index)
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    manifest=dict(schema_version='foodstateedit.multimaterial_run.v1',status='loading_pipeline',
                  started_at=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),pid=os.getpid(),
                  physical_gpu=args.gpu_index,gpu_uuid=preflight['gpu']['uuid'],
                  config_sha256=sha256(args.config),dataset_manifest_sha256=sha256(data/'dataset_manifest.json'),
                  preflight_sha256=sha256(args.preflight_report),pipeline_load_count=0,expected_conditions=1,
                  completed_conditions=[],claim_limit=dataset['claim_limit'])
    write_json(output/'run_manifest.json',manifest)
    (output/'RUNNING').write_text(manifest['started_at'])
    os.environ.update(runtime['offline_environment'])
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu_index),HF_HOME=config['cache_root'],
                      TRANSFORMERS_CACHE=config['cache_root'],DIFFSYNTH_MODEL_BASE_PATH=str(Path(runtime['model_root']).parent.parent))
    sys.path.insert(0,runtime['geoedit_root'])
    try:
        from geoedit import inference
        with (output/'pipeline_stdout.log').open('x') as stdout,(output/'pipeline_stderr.log').open('x') as stderr,contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
            pipe=inference.load_pipeline(inference.resolve_vram_limit(None))
        manifest['pipeline_load_count']=1
        for case in dataset['cases']:
            for arm in dataset['arms']:
                condition=case['case_id']+'__'+arm
                if condition not in config['selected_conditions']:continue
                foreign_process_check(preflight['gpu']['uuid'])
                condition_root=output/condition;condition_root.mkdir(exist_ok=False)
                manifest.update(status='running',active_condition=condition)
                write_json(output/'run_manifest.json',manifest)
                begin=time.monotonic()
                ref=data/case['files']['reference.png']['path'];alpha=data/case['files']['edit_alpha.png']['path']
                control=data/case['files'][arm+'.mp4']['path']
                reference=Image.open(ref).convert('RGB');w,h=reference.size
                with (condition_root/'stdout.log').open('x') as stdout,(condition_root/'stderr.log').open('x') as stderr,contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                    control_frames=inference.load_video_or_image(control,h,w,21)
                    mask_frames=inference.load_video_or_image(alpha,h,w,21)
                    video=pipe(prompt=case['prompt'],negative_prompt=case['negative_prompt'],
                               height=h,width=w,num_frames=21,num_inference_steps=20,
                               vace_video=control_frames,vace_video_mask=mask_frames,
                               vace_reference_image=reference,vace_scale=1.0,enable_ttm=False,seed=1,tiled=True)
                    inference.save_video(video,str(condition_root/'raw.mp4'),fps=8,quality=5)
                    del video,control_frames,mask_frames
                projected=project_outputs(condition_root/'raw.mp4',ref,alpha,condition_root,21,8,[0,3,6,10,15,20])
                foreign_process_check(preflight['gpu']['uuid'])
                record=dict(condition=condition,case_id=case['case_id'],arm=arm,status='complete_requires_visual_review',
                            seed=1,frames=21,steps=20,vace_scale=1.0,lora_enabled=False,ttm_enabled=False,
                            wall_time_seconds=round(time.monotonic()-begin,3),
                            outside_support_max_difference_before_encoding=projected['outside_support_max_pixel_difference'],
                            files={p.name:dict(path=p.relative_to(output).as_posix(),size_bytes=p.stat().st_size,sha256=sha256(p))
                                   for p in condition_root.iterdir() if p.is_file()})
                write_json(condition_root/'condition_manifest.json',record)
                manifest['completed_conditions'].append(record)
                write_json(output/'run_manifest.json',manifest)
                print('COMPLETED '+condition,flush=True)
        assert len(manifest['completed_conditions'])==1
        manifest['status']='complete_requires_per_material_review'
        (output/'COMPLETE').write_text(datetime.now(timezone.utc).isoformat())
        code=0
    except Exception as exc:
        manifest.update(status='technical_failure_preserved',error=repr(exc))
        (output/'failure.txt').write_text(traceback.format_exc())
        (output/'FAILED').write_text(datetime.now(timezone.utc).isoformat())
        code=3
    manifest.update(finished_at=datetime.now(timezone.utc).isoformat(),wall_time_seconds=round(time.monotonic()-started,3))
    (output/'RUNNING').unlink(missing_ok=True)
    write_json(output/'run_manifest.json',manifest)
    print(json.dumps(manifest,indent=2),flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
