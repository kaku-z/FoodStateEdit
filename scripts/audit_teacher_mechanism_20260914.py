"""Reanalyse frozen outputs; never run inference or change the source evidence."""
import json, hashlib, sys, csv
from pathlib import Path
from datetime import datetime, timezone
import subprocess
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/teacher_mechanism_audit_20260914_v1'
RESULT=ROOT/'results/teacher_mechanism_audit_20260914_v1.json'
MODES=['native_scale_1p0','planar_scale_1p0','relative3d_scale_1p0']
CASES=['ramen','soup','rice','cake']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def decode(p):
    meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height','-of','json',str(p)]))['streams'][0]
    data=subprocess.check_output(['ffmpeg','-v','error','-i',str(p),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
    frames=np.frombuffer(data,dtype=np.uint8).reshape(-1,meta['height'],meta['width'],3)
    if len(frames)!=21:raise ValueError((str(p),len(frames)))
    return frames
def compare(a,b,mask=None):
    d=np.abs(a.astype(np.int16)-b.astype(np.int16))
    if mask is not None:d=d[mask]
    return dict(mae=float(d.mean()),max_difference=int(d.max()),exact_pixel_fraction=float(np.all(d==0,axis=-1).mean()))
def main():
    if OUT.exists() or RESULT.exists():raise FileExistsError('New evidence paths required')
    OUT.mkdir(parents=True)
    config=read(ROOT/'configs/day25_same_condition_ablation_v2_retry.json')
    dataset=ROOT/'artifacts/day19_multimaterial_dataset_v3'
    manifest=read(dataset/'dataset_manifest.json')
    assert sha(dataset/'dataset_manifest.json')==config['dataset_manifest']['sha256']
    for relative,expected in config['runtime']['geoedit_files'].items():
        local=ROOT/'vendor_overrides/GeoEdit'/relative
        if local.exists():assert sha(local)==expected,(str(local),'runtime differs')
    cells=read(ROOT/'results/day25_ablation_matrix_verified_v1.json')['cells']
    records=[]; pairs=[]; geometries=[]; allfinal={}; allraw={}
    for case in CASES:
        alias='noodle' if case=='ramen' else case
        meta=next(c for c in manifest['cases'] if c['case_id']==alias)
        for file in meta['files'].values():assert sha(dataset/file['path'])==file['sha256']
        reference=np.array(Image.open(dataset/alias/'reference.png').convert('RGB'))
        alpha=np.array(Image.open(dataset/alias/'edit_alpha.png').convert('L'))
        weight=alpha.astype(np.float32)[...,None]/255
        caseout=OUT/case;caseout.mkdir()
        Image.fromarray(reference).save(caseout/'input.png')
        controls={}
        for mode in ['native','planar','relative3d']:
            controls[mode]=np.repeat(reference[None],21,axis=0) if mode=='native' else decode(dataset/alias/f'{mode}.mp4')
            Image.fromarray(controls[mode][-1]).save(caseout/f'{mode}_control_f20.png')
        for seed in [1,2,3]:
            for condition in MODES:
                cell=next(c for c in cells if c['case_id']==case and c['seed']==seed and c['condition']==condition)
                finalpath=ROOT/cell['path'];assert sha(finalpath)==cell['sha256']
                rawpath=finalpath.parent/'raw.mp4'; raw=decode(rawpath)
                if raw.shape[1:3]!=reference.shape[:2]:raise ValueError('Size mismatch')
                final=np.array(Image.open(finalpath).convert('RGB'))
                reconstructed=np.clip(np.rint(raw[-1].astype(np.float32)*weight+reference.astype(np.float32)*(1-weight)),0,255).astype(np.uint8)
                assert np.array_equal(final,reconstructed),('Composite mismatch',case,seed,condition)
                mode=condition.split('_scale')[0]
                record=dict(case=case,seed=seed,condition=condition,frames=len(raw),input_sha256=sha(dataset/alias/'reference.png'),final_sha256=sha(finalpath),raw_sha256=sha(rawpath),source_final=str(finalpath.relative_to(ROOT)),source_raw=str(rawpath.relative_to(ROOT)),source_kind=meta['source_kind'],
                    raw_vs_input=compare(raw[-1],reference),raw_outside=compare(raw[-1],reference,alpha==0),raw_inside=compare(raw[-1],reference,alpha>0),
                    final_vs_input=compare(final,reference),final_outside=compare(final,reference,alpha==0),final_inside=compare(final,reference,alpha>0),
                    raw_vs_control_inside=compare(raw[-1],controls[mode][-1],alpha>0),composite_reproduced_exactly=True)
                if seed==2:
                    Image.fromarray(raw[-1]).save(caseout/f'{mode}_raw_seed2_f20.png')
                    Image.fromarray(final).save(caseout/f'{mode}_final_seed2_f20.png')
                records.append(record);allfinal[(case,seed,mode)]=final;allraw[(case,seed,mode)]=raw[-1]
        for seed in [1,2,3]:pairs.append(dict(case=case,seed=seed,raw_planar_vs_relative3d=compare(allraw[(case,seed,'planar')],allraw[(case,seed,'relative3d')],alpha>0),final_planar_vs_relative3d=compare(allfinal[(case,seed,'planar')],allfinal[(case,seed,'relative3d')],alpha>0)))
        if case!='ramen':
            geom=read(dataset/alias/'geometry.json');a=geom['planar'];b=geom['relative3d']
            uv=np.array(a['vertices_uv']);z=np.array(a['vertex_relative_depths']);K=np.array(a['camera_intrinsic']);xyz=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T*z[:,None]
            projected=xyz@K.T;roundtrip=projected[:,:2]/projected[:,2:]
            anchor_diff=max(float(np.max(np.abs(np.array(x['anchor_uv'])-np.array(y['anchor_uv'])))) for x,y in zip(a['trace'],b['trace']))
            geometries.append(dict(case=case,vertex_count=len(uv),K=K.tolist(),uv=uv.tolist(),relative_depth=z.tolist(),xyz=xyz.tolist(),backproject_project_max_error_px=float(np.abs(roundtrip-uv).max()),planar_relative3d_anchor_max_difference_px=anchor_diff,control_inside_difference=compare(controls['planar'][-1],controls['relative3d'][-1],alpha>0)))
    summaries=[]
    for mode in MODES:
        rs=[r for r in records if r['condition']==mode]
        summaries.append(dict(condition=mode,n_outputs=len(rs),n_independent_inputs=4,mean_raw_outside_mae=float(np.mean([r['raw_outside']['mae'] for r in rs])),mean_final_outside_mae=float(np.mean([r['final_outside']['mae'] for r in rs])),mean_final_inside_mae=float(np.mean([r['final_inside']['mae'] for r in rs])),mean_raw_control_inside_mae=float(np.mean([r['raw_vs_control_inside']['mae'] for r in rs]))))
    result=dict(schema='foodstateedit.teacher_mechanism_audit.v1',created_utc=datetime.now(timezone.utc).isoformat(),status='completed_local_reanalysis',new_model_inference=False,training_performed=False,source_experiment='Day25, frozen 4 cases x 3 seeds x 3 arms subset',records=records,summaries=summaries,pairs=pairs,geometry_checks=geometries,
        scope='Pixel-change diagnostic only, not action/photo quality or statistical superiority. MP4-decoded raw frames include codec effects. Final compositing uses identical alpha and reference in all three arms.',
        formulas={'MAE':'mean(abs(A-B)), RGB 0..255','outside':'alpha == 0','inside':'alpha > 0','composite':'round(raw*alpha/255 + input*(1-alpha/255))'},
        source_hashes={p:sha(ROOT/p) for p in ['configs/day25_same_condition_ablation_v2_retry.json','scripts/run_same_condition_ablation.py','scripts/build_multimaterial_pilot.py','vendor_overrides/GeoEdit/diffsynth/pipelines/wan_video.py']})
    RESULT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    print(json.dumps(dict(summary=summaries,geometry=geometries,output=str(RESULT)),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
