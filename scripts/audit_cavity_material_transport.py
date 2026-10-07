"""Audit actual ray visibility and bounded neural material projection receipts."""
import time
from pathlib import Path
import numpy as np
from PIL import Image
from run_joint_action_full_noise import ROOT,read,write,sha,rgb

def main():
    reports={}
    for gn in [51,52]:
        g=ROOT/('gate_v'+str(gn))
        while read(g/'execution.json')['status']!='complete_unreviewed':time.sleep(20)
        ex=read(g/'execution.json');checked=[]
        for row in ex['rows']:
            cid=row['case_id'];seed=row['seed'];geo=ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid;cut=np.load(ROOT/'gate_v37/coupled_source_cut_reconstruction_v3'/(cid+'_field')/'field.npz');mask=cut['fresh'];parent=ROOT/('gate_v47/neural_source_grain' if gn==51 else 'gate_v46/source_detail')/(cid+'__'+str(seed))/'pre_sampling.png';assert sha(parent)==row['parent_sha256'];d=g/row['arm']/(cid+'__'+str(seed));pre=rgb(d/'pre_sampling.png');final=rgb(d/'composited.png')
            if gn==52:
                food=np.load(ROOT/'source_fit_ellipsoid_food_fields_consumptive_v4'/cid/'field.npz')['food'];mask=mask|food;protected=np.asarray(Image.open(ROOT/'gate_v50/contexts'/cid/str(seed)/'protected_source_color.png'))>0;assert np.array_equal(pre[protected],rgb(parent)[protected])
                raw=next((ROOT/'gate_v50').glob('worker_*/'+cid+'__paired_patch_action__'+str(seed)+'/raw.png'));assert sha(raw)==row['paired_raw_sha256']
                if row['arm']=='paired_neural_grain':
                    control=rgb(g/'geometry_light_only'/(cid+'__'+str(seed))/'pre_sampling.png');delta=np.abs(pre.astype(float)-control);bound=row['detail_luminance_bound'];assert delta.max()<=1+bound*2.5
                    yy,xx=np.where(mask);chroma=lambda x:x/np.maximum(x.sum(2)[...,None],1);assert np.max(np.linalg.norm(chroma(pre)[mask]-chroma(control)[mask],axis=1))<.025
            assert np.array_equal(pre[~mask],rgb(parent)[~mask]);edit=np.asarray(Image.open(geo/'edit_mask.png'))>0;assert np.array_equal(final[~edit],rgb(geo/'source.png')[~edit]);assert sha(d/'composited.png')==row['output_sha256'];checked.append(dict(case_id=cid,seed=seed,arm=row['arm'],outside_declared_material_parent_exact_before_sampling=True,outside_geometry_source_exact=True))
        rays=[]
        for p in (ROOT/'gate_v51/fields').glob('*/audit.json'):
            x=read(p);geo=ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/x['case_id'];assert sha(geo/'remaining.ply')==x['remaining_mesh_sha256'];assert sha(ROOT/'spoon_source_fit_ellipsoid_channels_v3'/x['case_id']/'geometry_channels.npz')==x['geometry_channels_sha256'];a=np.load(p.parent/'ambient_occlusion.npy');assert np.isfinite(a).all() and a.min()>=0 and a.max()<=1.00001;rays.append(x)
        report=dict(status='verified',compositions_verified=len(checked),rows=checked,ray_geometry_fields_verified=len(rays),ray_diagnostics=rays,new_neural_calls=0,photographic_realism_verified=False,real_geometry_lighting_verified=False,scope='Inferred meshes, retained ray diagnostics, parent/raw hashes and actual material region invariants. RGB appearance observations are not recovered albedo, calibrated light or real food physics.',script_sha256=sha(Path(__file__)))
        write(ROOT/('cavity_material_audit_gate_v'+str(gn)+'.json'),report);reports[gn]=report;print('CAVITY_MATERIAL_AUDIT_COMPLETE',gn,len(checked),flush=True)

if __name__=='__main__':main()
