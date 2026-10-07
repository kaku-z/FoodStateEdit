"""Compose explicitly labeled crop outputs; original context is pixel-exact."""
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    g=ROOT/'gate_v14';cfg=json.loads((g/'config.json').read_text())
    out=g/'compositions_v1';out.mkdir(exist_ok=True)
    cases={c['case_id']:c for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases']}
    rows=[]
    for cid in sorted({j['case_id'] for j in cfg['jobs']}):
        c=cases[cid]
        source=np.asarray(Image.open(ROOT/'inputs'/cid/'source.png').convert('RGB'))
        transforms={k:json.loads((g/cid/k/'transform.json').read_text()) for k in ['payload','recess']}
        baseline=np.asarray(Image.open(transforms['payload']['full_composite_baseline']).convert('RGB'),float)
        baseline_edit=np.asarray(Image.open(ROOT/'geometry_adaptive_v1'/cid/'edit_mask.png'))>0
        for method in ['no_reference','material_reference']:
            result=baseline.copy();union=baseline_edit.copy();parents=[]
            for component in ['payload','recess']:
                job=next(j for j in cfg['jobs'] if j['case_id']==cid and j['method']==component+'_'+method)
                paths=list(g.glob('worker*/'+job['id']+'/raw_rgba.png'))
                assert len(paths)==1,job['id']
                p=paths[0];transform=transforms[component];x0,y0,x1,y1=transform['box']
                raw=Image.open(p).convert('RGBA').resize((x1-x0,y1-y0),Image.Resampling.LANCZOS)
                under=Image.fromarray(source).crop((x0,y0,x1,y1)).convert('RGBA')
                patch=np.asarray(Image.alpha_composite(under,raw).convert('RGB'),float)
                mask=np.asarray(Image.open(g/cid/component/'full_mask.png'))>0
                # Both the semantic edit boundary and crop boundary are feathered.
                rect=np.zeros(mask.shape,bool);rect[y0:y1,x0:x1]=True
                alpha=np.minimum(np.clip(distance_transform_edt(mask)/4,0,1),np.clip(distance_transform_edt(rect)/4,0,1))[...,None]
                canvas=result.copy();canvas[y0:y1,x0:x1]=patch
                result=result*(1-alpha)+canvas*alpha;union|=mask
                parents.append(dict(raw=str(p),raw_sha256=sha(p),transform=transform))
            final=np.uint8(np.clip(np.rint(result),0,255));assert np.array_equal(final[~union],source[~union])
            d=out/(cid+'__'+method);d.mkdir(exist_ok=True);Image.fromarray(final).save(d/'composited.png')
            l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized']
            Image.fromarray(final).crop((l,t,l+w,t+h)).save(d/'view.png')
            row={'id':d.name,'raw_output':False,'baseline':transforms['payload']['full_composite_baseline'],
                 'parents':parents,'outside_union_source_exact':True,'fixed_original_content_crop':True}
            (d/'result.json').write_text(json.dumps(row,indent=2));rows.append(row)
    (out/'manifest.json').write_text(json.dumps({'status':'complete_unreviewed','rows':rows,'raw_count':len(cfg['jobs']),
        'composite_count':len(rows),'script_sha256':sha(Path(__file__))},indent=2))
    with zipfile.ZipFile(ROOT/'gate_v14_complete.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in g.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(ROOT))
    print('COMPLETE',len(rows))

if __name__=='__main__':main()
