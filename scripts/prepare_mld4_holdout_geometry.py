"""Run unchanged frozen MLD3 perception/geometry, then MLD4 guides, on all holdouts.

Every frozen case receives a status, including failures. No image generator,
semantic VLM, new training, output selection, or per-case material override runs.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def utc():return datetime.now(timezone.utc).isoformat()


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def worker(args):
    import numpy as np
    from PIL import Image
    index=args.worker_index
    manifest=read(args.sources/'manifest.json')
    source_row=manifest['cases'][index]
    case_id=f"real_{16+index:02d}_{Path(source_row['source_file_name']).stem}"
    job=args.output/'jobs'/case_id;job.mkdir(parents=True,exist_ok=True)
    status=dict(case_id=case_id,holdout_case_id=source_row['case_id'],index=16+index,
        status='running',stage='initializing',started_utc=utc(),
        frozen_source_sha256=source_row['source_png_sha256'],food_prompt=source_row['food_prompt'],
        metadata_food_prompt_used=True,semantic_vlm_used=False,material_mode_policy='existing MLD3 auto routing from explicit metadata prompt',
        physical_cuda_gpu=4,egl_device_id=4,all_stage_artifacts_retained=True)
    start=time.time()
    def stage(name):
        status.update(stage=name,updated_utc=utc());write(job/'case_status.json',status)
        print(case_id,name,flush=True)
    try:
        core=args.output/'execution_snapshot'/'mld3'
        sys.path[:0]=[str(core/'scripts'),str(core)]
        mld3=load_module('run_mld3',core/'scripts/run_mld3.py')
        guide=load_module('holdout_guide_builder',args.output/'execution_snapshot/prepare_mld4_guides.py')
        import torch
        torch.set_num_threads(4);torch.manual_seed(41);np.random.seed(41)
        source=args.sources/source_row['image']
        if sha(source)!=source_row['source_png_sha256']:raise ValueError('Frozen source hash mismatch')
        work=args.output/'raw_work'/case_id
        work.mkdir(parents=True,exist_ok=True)
        raw=work/'observations/real_00_input'
        observations=args.output/'observations'/case_id
        raw_args=SimpleNamespace(image=source,food_prompt=source_row['food_prompt'],output=work,
            checkpoint=args.checkpoint,material_mode='auto')
        stage('fresh_SAM3_MoGe2_MLD2_observations')
        try:
            mld3.prepare_image(raw_args)
        finally:
            if raw.exists() and not observations.exists():
                observations.parent.mkdir(parents=True,exist_ok=True)
                raw.rename(observations)
        state_args=SimpleNamespace(output=args.output,checkpoint=args.checkpoint,indices=[16+index],
            material_mode='auto',iterations=args.iterations,compliance=None,lift_multiplier=1.,appearance=False)
        stage('existing_source_material_selection')
        mld3.prepare_selected_sources(args.output/'observations',state_args)
        source_equal=np.array_equal(np.asarray(Image.open(source).convert('RGB')),
            np.asarray(Image.open(observations/'source.png').convert('RGB')))
        if not source_equal:raise ValueError('Fresh pipeline changed frozen source RGB pixels')
        stage('MLD3_contact_aware_geometry')
        rendered=args.output/'real'/case_id
        state=mld3.run_case(observations,rendered,state_args,16+index)
        stage('MLD4_guides')
        guide_args=SimpleNamespace(source_margin=9,target_margin=5,source_shape_control=False)
        guides=guide.prepare_case(rendered,args.output,args.guides/case_id,guide_args)
        fields=read(observations/'geometry_raw.json')
        provenance=dict(frozen_source_case=source_row,holdout_manifest_sha256=sha(args.sources/'manifest.json'),
            holdout_freeze_lock=read(args.sources/'freeze_lock.json'),
            source_rgb_pixels_unchanged=source_equal,fresh_perception=True,
            SAM3_checkpoint='/host/space0/guo-z/Evol-SAM3/sam3/sam3.pt',
            MoGe2_checkpoint='/host/space0/guo-z/tf-ufi/food3d_repair_20260928/models/model.pt',
            MLD2_checkpoint=str(args.checkpoint),MLD2_checkpoint_sha256=sha(args.checkpoint),
            geometry_code_root=str(core),guide_builder_sha256=sha(args.output/'execution_snapshot/prepare_mld4_guides.py'),
            no_new_training=True,no_semantic_vlm=True,no_image_generator=True,
            all_frozen_cases_run_without_output_selection=True,
            material_mode=state['mode'],source_selection=state['source_selection'],
            original_bite_center=fields.get('source_bite_center'),
            guide_contract='Baseline hidden-surface-confidence guides; no source-shape-control or sparse-anchor ablation applied.')
        write(rendered/'holdout_provenance.json',provenance)
        write(args.guides/case_id/'holdout_provenance.json',provenance)
        status.update(status='complete',stage='complete',mode=state['mode'],
            source_observation=str(observations),geometry_case=str(rendered),guide_case=str(args.guides/case_id),
            source_pixels_unchanged=True,source_removed_pixels=guides['source_removed_pixels'],
            target_food_pixels=guides['target_food_pixels'],target_correspondence_coverage=guides['target_correspondence_coverage'],
            geometry_identifiers_preserved=guides['geometry_identifiers_preserved'])
    except Exception as exc:
        status.update(status='failed',exception_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        (job/'traceback.txt').write_text(status['traceback'],encoding='utf-8')
        print(status['traceback'],flush=True)
    status.update(completed_utc=utc(),seconds=time.time()-start)
    write(job/'case_status.json',status)
    print(json.dumps(status,ensure_ascii=False),flush=True)
    return status


def manager(args):
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Use a new geometry output directory; prior attempts remain immutable')
    if args.guides.exists() and any(args.guides.iterdir()):raise FileExistsError('Use a new guide output directory')
    sources=read(args.sources/'manifest.json');lock=read(args.sources/'freeze_lock.json')
    if sha(args.sources/'manifest.json')!=lock['manifest_sha256']:raise ValueError('Holdout freeze lock mismatch')
    if len(sources['cases'])!=8:raise ValueError('Expected all eight frozen sources')
    for row in sources['files']:
        if sha(args.sources/row['path'])!=row['sha256']:raise ValueError('Frozen input modified: '+row['path'])
    args.output.mkdir(parents=True);args.guides.mkdir(parents=True)
    snapshot=args.output/'execution_snapshot';snapshot.mkdir()
    shutil.copytree(args.mld3_code,snapshot/'mld3',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copyfile(args.guide_builder,snapshot/'prepare_mld4_guides.py')
    shutil.copyfile(__file__,snapshot/'prepare_mld4_holdout_geometry.py')
    snapshot_hashes={str(p.relative_to(snapshot)):sha(p) for p in sorted(snapshot.rglob('*')) if p.is_file()}
    write(snapshot/'executed_code_hashes.json',snapshot_hashes)
    recipe=dict(created_utc=utc(),holdout_manifest_sha256=lock['manifest_sha256'],
        frozen_source_selection_utc=lock['frozen_utc'],case_count=8,first_case_index=16,
        physical_gpu=4,egl_device_id=4,seed=41,iterations=args.iterations,
        checkpoint=str(args.checkpoint),checkpoint_sha256=sha(args.checkpoint),
        core_source=str(args.mld3_code),core_modified=False,guide_source=str(args.guide_builder),
        source_margin=9,target_margin=5,source_shape_control=False,
        material_mode='auto',food_prompt_policy='Explicit frozen metadata labels; no VLM labels',
        source_failures='Retain every failure and all stage artifacts; never replace a source or select by output appearance',
        generated_targets_used=False,new_training=False)
    write(args.output/'execution_recipe.json',recipe)
    rows=[dict(case_id=f"real_{16+i:02d}_{Path(r['source_file_name']).stem}",holdout_case_id=r['case_id'],status='pending') for i,r in enumerate(sources['cases'])]
    manifest=dict(status='running',cases=rows,source_only=True,fresh_perception=True,all_frozen_cases_included=True)
    logs=args.output/'logs';logs.mkdir()
    write(args.output/'run_manifest.json',manifest)
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='4',EGL_DEVICE_ID='4',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='4')
    for i,row in enumerate(rows):
        cmd=[sys.executable,str(snapshot/'prepare_mld4_holdout_geometry.py'),'--sources',str(args.sources),
            '--output',str(args.output),'--guides',str(args.guides),'--checkpoint',str(args.checkpoint),
            '--iterations',str(args.iterations),'--worker-index',str(i)]
        with (logs/(row['case_id']+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(cmd,env=environment,stdout=log,stderr=subprocess.STDOUT)
        status_path=args.output/'jobs'/row['case_id']/'case_status.json'
        rows[i]=read(status_path) if status_path.exists() else dict(row,status='failed',stage='worker_process',returncode=result.returncode)
        rows[i]['process_returncode']=result.returncode
        if result.returncode and rows[i]['status']!='failed':rows[i].update(status='failed',error='Worker exited unexpectedly')
        write(args.output/'run_manifest.json',manifest)
        print(json.dumps({'case_id':rows[i]['case_id'],'status':rows[i]['status'],'stage':rows[i].get('stage')}),flush=True)
    completed=[row for row in rows if row['status']=='complete']
    manifest.update(status='complete' if len(completed)==8 else 'complete_with_failures',completed_count=len(completed),
        failed_count=8-len(completed),completed_utc=utc(),image_generation_run=False,checkpoint_sha256=recipe['checkpoint_sha256'])
    write(args.output/'run_manifest.json',manifest)
    write(args.guides/'guides_manifest.json',dict(cases=[read(args.guides/r['case_id']/'guide_manifest.json') for r in completed],
        all_frozen_case_statuses=rows,expected_case_count=8,completed_count=len(completed),source_only=True,generated=False))
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--guides',type=Path,required=True)
    p.add_argument('--mld3-code',type=Path,default=Path('/host/space0/guo-z/tf-ufi/material_lineage_deformable_20261004/execution_snapshot_final'))
    p.add_argument('--guide-builder',type=Path,default=Path('/host/space0/guo-z/tf-ufi/material_lineage_reform_20261004/scripts/prepare_mld4_guides.py'))
    p.add_argument('--checkpoint',type=Path,default=Path('/host/space0/guo-z/tf-ufi/material_lineage_full_20261004/runs_continuous/shared_s41/inference_checkpoint.pt'))
    p.add_argument('--iterations',type=int,default=180);p.add_argument('--worker-index',type=int)
    args=p.parse_args()
    result=worker(args) if args.worker_index is not None else manager(args)
    print(json.dumps({'status':result['status'],'completed_count':result.get('completed_count')}),flush=True)


if __name__=='__main__':main()
