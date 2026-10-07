"""Matched G41 contexts with a pure-noise editable material initialization.

This isolates the appearance initialization rather than selecting favorable
cases. Source shape and garnish preservation remain compositor statements.
"""
import json, hashlib, subprocess, time, zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
PY='/home/yanai-lab/guo-z/.conda/envs/flux_pure_env/bin/python'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    for _ in range(10):
        try: return json.loads(p.read_text())
        except (OSError,json.JSONDecodeError):time.sleep(.3)
    raise RuntimeError(str(p))
def write(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def main():
    g=ROOT/'gate_v42';g.mkdir(exist_ok=False)
    (g/'executed_orchestrator.py').write_bytes(Path(__file__).read_bytes())
    cfg=read(ROOT/'gate_v41/config.json')
    cfg['stage']='development_source_plain_material_pure_noise';cfg['jobs']=[j for j in cfg['jobs'] if j['start_raw_sigma']==.6]
    for j in cfg['jobs']:
        j['method']='label_consistent_material_sigma100';j['id']=j['case_id']+'__'+j['method']+'__'+str(j['seed']);j['start_raw_sigma']=1.
    cfg['frozen_raw_cells']=24;cfg['selection_rule']='Matched G41 contexts, prompt, masks, seeds and compositor. Only initial editable-material sigma changes to pure noise.'
    write(g/'config.json',cfg)
    write(g/'frozen_plan.json',{'status':'frozen_before_generation','raw_cells':24,'seeds':[41,163,907],'requested_raw_sigma':1.,'no_case_or_seed_selection':True,'worker_sha256':sha(ROOT/'run_bite_material_sdedit_full_noise.py'),'config_sha256':sha(g/'config.json'),'scope':'Post-hoc development follow-up after partial G41 inspection. All eight source photographs remain development.'})
    write(g/'execution.json',{'status':'running','unix':time.time()})
    # The earlier two-worker ceiling was a resource precaution. Audit actual
    # idle pairs and available RAM before adding these separate balanced workers.
    from run_qwen_image_edit_direct_baseline import gpu_snapshot
    snapshots=[gpu_snapshot(i) for i in [1,6,5,7]]
    assert all(not s['compute_processes'] and s['memory_free_mib']>47500 for s in snapshots)
    assert min(s['host_mem_available_mib'] for s in snapshots)>180000
    write(g/'resource_preflight.json',{'snapshots':snapshots,'added_pairs':['1,6','5,7'],'existing_pairs':['0,2','3,4'],'host_available_floor_mib':180000,'foreign_jobs_untouched':True})
    procs=[]
    for i,gpus in enumerate(['1,6','5,7']):
        argv=[PY,'-u',str(ROOT/'run_bite_material_sdedit_full_noise.py'),'--config',str(g/'config.json'),'--output',str(g/('worker_'+str(i))),'--gpus',gpus,'--shard',str(i),'--shards','2']
        log=ROOT/'logs'/('gate_v42_worker_'+str(i)+'.log')
        with log.open('w') as h:p=subprocess.Popen(argv,stdout=h,stderr=subprocess.STDOUT,cwd=str(ROOT),start_new_session=True)
        write(g/('worker_'+str(i)+'_launch.json'),{'pid':p.pid,'argv':argv,'log':str(log)});procs.append(p)
    while any(p.poll() is None for p in procs):
        if any(p.poll() not in [None,0] for p in procs):raise RuntimeError('Retain completed cells; technical repair required')
        time.sleep(20)
    found={read(p)['id']:p.parent for p in g.glob('worker_*/*/result.json')}
    assert set(found)=={j['id'] for j in cfg['jobs']}
    cases={c['case_id']:c for c in read(ROOT/'inputs/manifest.json')['cases']};out=g/'collection_complete';out.mkdir();rows=[]
    for j in cfg['jobs']:
        tr=read(Path(j['transform']));base=np.asarray(Image.open(tr['parent']).convert('RGB'));assert sha(Path(tr['parent']))==tr['parent_sha256']
        x0,y0,x1,y1=tr['box'];mask=np.asarray(Image.open(Path(j['transform']).parent/'full_mask.png'))>0
        patch=np.asarray(Image.open(found[j['id']]/'raw.png').convert('RGB').resize((192,192),Image.Resampling.LANCZOS),float)
        target=base.astype(float).copy();target[y0:y1,x0:x1]=patch;alpha=np.clip(distance_transform_edt(mask)/3,0,1)[...,None]
        pre=np.uint8(np.clip(np.rint(base*(1-alpha)+target*alpha),0,255));assert np.array_equal(pre[~mask],base[~mask])
        c=cases[j['case_id']];l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h)
        geo=ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/j['case_id'];source=np.asarray(Image.open(geo/'source.png').convert('RGB'));edit=np.asarray(Image.open(geo/'edit_mask.png'))>0
        native=Image.fromarray(pre).crop(rect).resize(tuple(c['source_size']),Image.Resampling.LANCZOS).resize((w,h),Image.Resampling.LANCZOS);grid=source.copy();grid[t:t+h,l:l+w]=np.asarray(native)
        blend=np.clip(distance_transform_edt(edit)/5,0,1)[...,None];final=np.uint8(np.clip(np.rint(source*(1-blend)+grid*blend),0,255));assert np.array_equal(final[~edit],source[~edit])
        d=out/j['id'];d.mkdir();Image.fromarray(pre).save(d/'pre_sampling.png');Image.fromarray(final).save(d/'composited.png');Image.fromarray(final).crop(rect).save(d/'view.png')
        row={'id':j['id'],'case_id':j['case_id'],'method':j['method'],'seed':j['seed'],'raw':str(found[j['id']]/'raw.png'),'raw_sha256':sha(found[j['id']]/'raw.png'),'parent':tr['parent'],'parent_sha256':tr['parent_sha256'],'actual_start_sigma':read(found[j['id']]/'request.json')['actual_start_sigma'],'outside_editable_food_parent_exact_before_sampling':True,'outside_geometry_source_exact':True,'scope':'A pure-noise Qwen2511 material prior within an approximate source-chroma mask, followed by the disclosed geometry/garnish compositor. No final quality acceptance.'}
        write(d/'result.json',row);rows.append(row)
    for cid,c in cases.items():
        l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h);board=Image.new('RGB',(2048,780),'white');draw=ImageDraw.Draw(board)
        for ri,method in enumerate(['prior','label_consistent_material_sigma100']):
            paths=[('source',ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid/'source.png')]+[(method+' '+str(s),ROOT/'gate_v40/source_photo_cut'/(cid+'__'+str(s))/'composited.png' if method=='prior' else out/(cid+'__'+method+'__'+str(s))/'composited.png') for s in [41,163,907]]
            for col,(label,p) in enumerate(paths):
                im=Image.open(p).convert('RGB').crop(rect);im.thumbnail((504,354));board.paste(im,(col*512+(504-im.width)//2,ri*390+28));draw.text((col*512+5,ri*390+5),label,fill='black')
        board.save(g/(cid+'_review.jpg'),quality=95)
    write(g/'execution.json',{'status':'complete_unreviewed','raw_cells':24,'compositions':len(rows),'rows':rows,'unix':time.time()})
    with zipfile.ZipFile(ROOT/'gate_v42_complete.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in g.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(ROOT))
    print('FULL_NOISE_MATERIAL_COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
