"""Prepare matched local appearance calls for shallower cuts and a lit rounded fork."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
OLD=Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare_case(cid,d):
    d.mkdir(parents=True,exist_ok=False);gp=ROOT/'geometry_v3'/cid
    src=np.asarray(Image.open(ROOT/'inputs'/cid/'source.png').convert('RGB'))
    proxy=np.asarray(Image.open(gp/'rgb_control.png').convert('RGB'))
    def mask(n):return np.asarray(Image.open(gp/(n+'.png')))>0
    hole=mask('hole_mask');target=mask('material_mask')|mask('rigid_mask')
    remove=src.copy();remove[hole]=proxy[hole];lifted=src.copy();lifted[target]=proxy[target]
    arrays={'source':src,'proxy':proxy,'hole_edit':binary_dilation(hole,iterations=10),'target_edit':binary_dilation(target,iterations=10)}
    meta={}
    for kind,im in [('hole',remove),('target',lifted)]:
        m=arrays[kind+'_edit'];yy,xx=np.where(m)
        side=min(448,32*((max(224,int(max(np.ptp(xx)+1,np.ptp(yy)+1)*1.8))+31)//32))
        cx,cy=(xx.min()+xx.max())/2,(yy.min()+yy.max())/2
        x0=int(np.clip(round(cx-side/2),0,640-side));y0=int(np.clip(round(cy-side/2),0,480-side))
        box=(x0,y0,x0+side,y0+side);meta[kind]={'box_xyxy':box,'network_size':[512,512]}
        arrays[kind+'_crop']=np.asarray(Image.fromarray(im).crop(box).resize((512,512),Image.Resampling.LANCZOS))
        arrays[kind+'_crop_edit']=np.asarray(Image.fromarray(np.uint8(m)*255).crop(box).resize((512,512),Image.Resampling.NEAREST))
    arrays['target_composite_mask']=binary_dilation(arrays['target_edit'],iterations=14)&~binary_dilation(arrays['hole_edit'],iterations=4)
    files={}
    for name,arr in arrays.items():
        p=d/(name+'.png');Image.fromarray(np.uint8(arr)*255 if arr.dtype==bool else arr).save(p)
        files[name]={'path':str(p),'sha256':sha(p)}
    (d/'transforms.json').write_text(json.dumps(meta,indent=2)+'\n')
    return files

def main():
    geom=json.loads((ROOT/'geometry_v3/manifest.json').read_text());assert geom['status']=='complete'
    sources=json.loads((ROOT/'inputs/manifest.json').read_text());old=json.loads((OLD/'validation/frozen.json').read_text())
    cfg={k:old[k] for k in ['backend','expected_pipeline_sha256','model_audit','inference']};cfg.update(stage='development_rounded_fork',not_formal=True,jobs=[])
    out=ROOT/'gate_v2';out.mkdir(exist_ok=False)
    for cid in sources['development_ids']:
        if next(g for g in geom['cases'] if g['case_id']==cid)['status']!='geometry_ready':continue
        files=prepare_case(cid,out/'prepared'/cid)
        for kind in ['hole','target']:
            previous=next(j for j in old['jobs'] if j['case_id']==cid and j['method']==kind+'_locked')
            seed=907 if cid=='new_01_7442' else 41
            cfg['jobs'].append({'id':f'{cid}__{kind}_rounded__{seed}','case_id':cid,'method':kind+'_rounded','seed':seed,
                'input_order':[kind+'_crop'],'files':{kind+'_crop':files[kind+'_crop'],'edit_mask':files[kind+'_crop_edit']},
                'prompt':previous['prompt'],'lock_context':True})
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (out/'freeze.json').write_text(json.dumps({'created_unix':time.time(),'scope':'Previously seen sources, development only.',
        'config_sha256':sha(out/'config.json'),'raw_calls':len(cfg['jobs']),
        'hypothesis':'Source-only cut-depth bound and rounded illuminated utensil improve material recognition without internal latent RGB locking.'},indent=2)+'\n')
    print({'jobs':len(cfg['jobs'])})

if __name__=='__main__':main()
