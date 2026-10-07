"""Audit G53 input receipts and actual protected/output pixels separately."""
from pathlib import Path
import numpy as np
from PIL import Image
from run_joint_action_full_noise import ROOT,read,write,sha,rgb

def main():
    g=ROOT/'gate_v53';ex=read(g/'execution.json');assert ex['status']=='complete_unreviewed';seen=set();rows=[]
    for r in ex['rows']:
        cid=r['case_id'];seed=r['seed'];arm=r['arm'];key=(cid,seed,arm);assert key not in seen;seen.add(key)
        geo=ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid;parent=ROOT/'gate_v52/geometry_light_only'/(cid+'__'+str(seed))/'pre_sampling.png';assert sha(parent)==r['parent_sha256']
        raw=next((ROOT/'gate_v50').glob('worker_*/'+cid+'__paired_patch_action__'+str(seed)+'/raw.png'));assert sha(raw)==r['raw_sha256']
        d=g/arm/(cid+'__'+str(seed));pre=rgb(d/'pre_sampling.png');final=rgb(d/'composited.png');base=rgb(parent)
        food=np.load(ROOT/'source_fit_ellipsoid_food_fields_consumptive_v4'/cid/'field.npz')['food'];fresh=np.load(ROOT/'gate_v37/coupled_source_cut_reconstruction_v3'/(cid+'_field')/'field.npz')['fresh'];mask=food|fresh
        protected=np.asarray(Image.open(ROOT/'gate_v50/contexts'/cid/str(seed)/'protected_source_color.png'))>0
        assert np.array_equal(pre[~mask],base[~mask]);assert np.array_equal(pre[protected],base[protected])
        edit=np.asarray(Image.open(geo/'edit_mask.png'))>0;assert np.array_equal(final[~edit],rgb(geo/'source.png')[~edit]);assert sha(d/'composited.png')==r['output_sha256'];assert r['poisson_max_absolute_residual']<1e-8
        assert np.isfinite(pre).all();rows.append(dict(case_id=cid,seed=seed,arm=arm,output_sha256=r['output_sha256']))
    assert len(rows)==72 and len({x[0] for x in seen})==8
    write(ROOT/'screened_surface_audit_gate_v53.json',dict(status='verified',compositions_verified=len(rows),rows=rows,new_neural_calls=0,photographic_realism_verified=False,physical_subsurface_scattering_verified=False,scope='Actual parent/raw receipts, protected material pixels and final source-exact area; linear-system residual is numerical consistency, not realism.',script_sha256=sha(Path(__file__))))
    print('SCREENED_SURFACE_AUDIT_COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
