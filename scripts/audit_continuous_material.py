"""Audit G46 spatial/color constraints, without a perceptual success claim."""
import json, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def rgb(p): return np.asarray(Image.open(p).convert('RGB'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def residual(a,b):
    gain=np.sum(a*b,1)/np.maximum(np.sum(a*a,1),1e-9)
    return float(np.max(np.abs(a*gain[:,None]-b))) if len(a) else None
def main():
    g=ROOT/'gate_v46'; ex=json.loads((g/'execution.json').read_text());assert len(ex['rows'])==72
    jobs={(j['case_id'],j['seed']):j for j in json.loads((ROOT/'gate_v42/config.json').read_text())['jobs']};rows=[]
    for row in ex['rows']:
        cid,s=row['case_id'],row['seed'];j=jobs[cid,s];tr=json.loads(Path(j['transform']).read_text());base=rgb(Path(tr['parent']));assert sha(Path(tr['parent']))==row['source_prior_sha256']
        f=np.load(ROOT/'source_fit_ellipsoid_food_fields_consumptive_v4'/cid/'field.npz');food=f['food'];observed=np.zeros(food.shape,bool);observed[f['yy'][f['valid']],f['xx'][f['valid']]]=True
        white=np.median(rgb(Path(j['transform']).parent/'reference.png').reshape(-1,3),0);white/=white.sum();chroma=base/np.maximum(base.sum(2)[...,None],1);distance=np.linalg.norm(chroma-white,axis=2);q=np.clip((.09-distance)/.06,0,1);confidence=q*q*(3-2*q);confidence[food&~observed]=1;confidence[~food]=0
        d=g/row['arm']/(cid+'__'+str(s));pre=rgb(d/'pre_sampling.png');final=rgb(d/'composited.png');assert sha(d/'composited.png')==row['composited_sha256'];assert np.array_equal(pre[confidence==0],base[confidence==0])
        check=observed&(confidence>0)&(pre.max(2)<254)&(base.min(2)>8);r_known=residual(base[check].astype(float),pre[check].astype(float));assert r_known is not None and r_known<1.01,(cid,s,r_known)
        unknown=food&~observed&(pre.max(2)<254);a=np.broadcast_to(white,(int(unknown.sum()),3));r_unknown=residual(a,pre[unknown].astype(float));assert r_unknown is None or r_unknown<1.01,(cid,s,r_unknown)
        geo=ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid;edit=np.asarray(Image.open(geo/'edit_mask.png'))>0;assert np.array_equal(final[~edit],rgb(geo/'source.png')[~edit]);assert not np.any((confidence>0)&~food)
        rows.append(dict(case_id=cid,seed=s,arm=row['arm'],observed_color_direction_residual=r_known,unknown_reference_color_direction_residual=r_unknown,zero_confidence_parent_exact=True,outside_geometry_source_exact=True))
    report=dict(status='verified',compositions_verified=len(rows),rows=rows,photographic_realism_verified=False,scope='Approximate color confidence, source appearance and inferred geometry only; unknown appearance is a reference-color hypothesis.',script_sha256=sha(Path(__file__)))
    (ROOT/'projected_material_audit_gate_v46.json').write_text(json.dumps(report,indent=2));print('G46_AUDIT_COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
