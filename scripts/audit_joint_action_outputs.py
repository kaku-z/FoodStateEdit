"""Independently check retained G47/G48 pixels, input hashes and latent receipts."""
import json,time,hashlib,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rgb(p):return np.asarray(Image.open(p).convert('RGB'))
def read(p):
    for _ in range(10):
        try:return json.loads(p.read_text())
        except (OSError,json.JSONDecodeError):time.sleep(.3)
    raise RuntimeError(str(p))
def main():
    g=ROOT/'gate_v47';ex=read(g/'execution.json');assert len(ex['rows'])==72;rows=[]
    for row in ex['rows']:
        cid,s=row['case_id'],row['seed'];basepath=ROOT/'gate_v46/source_detail'/(cid+'__'+str(s))/'pre_sampling.png';assert sha(basepath)==row['base_sha256'];base=rgb(basepath);f=np.load(ROOT/'source_fit_ellipsoid_food_fields_consumptive_v4'/cid/'field.npz');cut=np.load(ROOT/'gate_v37/coupled_source_cut_reconstruction_v3'/(cid+'_field')/'field.npz');action=f['food']|cut['reconstruct'];d=g/row['arm']/(cid+'__'+str(s));pre=rgb(d/'pre_sampling.png');final=rgb(d/'composited.png');assert sha(d/'composited.png')==row['output_sha256'];assert np.array_equal(pre[~action],base[~action]);geo=ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid;edit=np.asarray(Image.open(geo/'edit_mask.png'))>0;assert np.array_equal(final[~edit],rgb(geo/'source.png')[~edit]);rows.append(dict(case_id=cid,seed=s,arm=row['arm'],outside_food_cut_parent_exact=True,outside_geometry_source_exact=True))
    write=dict(status='verified',compositions_verified=len(rows),rows=rows,photographic_realism_verified=False,scope='Joint scalar illumination is an uncalibrated prior. Predicted cavity material and phase grain are not real observations.');(ROOT/'joint_light_audit_gate_v47.json').write_text(json.dumps(write,indent=2));print('G47_AUDIT_COMPLETE',len(rows),flush=True)
    g=ROOT/'gate_v48'
    while read(g/'execution.json')['status']!='complete_unreviewed':time.sleep(20)
    cfg=read(g/'config.json');jobs={j['id']:j for j in cfg['jobs']};raw={read(p)['id']:p.parent for p in g.glob('worker_*/*/result.json')};assert set(raw)==set(jobs);receipts=[];paired={}
    for jid,j in jobs.items():
        d=raw[jid];request=read(d/'request.json');context=read(d/'context_audit.json');assert len(context['steps'])==24 and context['raw_output_no_pixel_compositor'];assert all(x['fixed_fraction']>0 for x in context['steps']);
        if j['method']=='joint_action_free':assert all(x['editable_latent_update_max']==0 for x in context['steps'])
        else:assert context['steps'][0]['structural_strength']==1 and abs(context['steps'][-1]['structural_strength']-.1)<1e-6
        for info in j['files'].values():assert sha(Path(info['path']))==info['sha256']
        result=read(d/'result.json')
        for name,value in result['files'].items():assert sha(d/name)==value
        paired.setdefault((j['case_id'],j['seed']),[]).append(request['noise_sha256']);receipts.append(dict(id=jid,actual_start_sigma=request['actual_start_sigma'],noise_sha256=request['noise_sha256'],context_steps=len(context['steps']),structural_anchoring=('structure' in j),raw_sha256=sha(d/'raw.png')))
    assert len(paired)==24 and all(len(x)==2 and x[0]==x[1] for x in paired.values());rows=[];ex=read(g/'execution.json');assert len(ex['rows'])==96
    for row in ex['rows']:
        j=jobs[row['id']];tr=read(Path(j['transform']));basepath=Path(tr['parent']);assert sha(basepath)==row['parent_sha256'];ctx=Path(j['transform']).parent;mask=np.asarray(Image.open(ctx/'edit_mask.png'))>0;d=g/row['variant']/row['id'];pre=rgb(d/'pre_sampling.png');final=rgb(d/'composited.png');assert sha(d/'composited.png')==row['output_sha256'];assert np.array_equal(pre[~mask],rgb(basepath)[~mask]);geo=Path(tr['original_source']).parent;edit=np.asarray(Image.open(geo/'edit_mask.png'))>0;assert np.array_equal(final[~edit],rgb(geo/'source.png')[~edit]);protected=np.asarray(Image.open(ctx/'protected_source_color.png'))>0;assert np.array_equal(pre[protected],rgb(basepath)[protected]);rows.append(dict(id=row['id'],variant=row['variant'],outside_bounded_mask_parent_exact_before_sampling=True,protected_approximate_source_color_parent_exact_before_sampling=True,outside_geometry_source_exact=True))
    report=dict(status='verified',raw_cells_verified=len(receipts),compositions_verified=len(rows),same_noise_pairs_verified=len(paired),raw_receipts=receipts,rows=rows,photographic_realism_verified=False,generated_silhouette_contact_volume_verified=False,scope='Coarse diffusion anchoring and masks are image constraints, not an exact geometry guarantee after boundary refinement.',script_sha256=sha(Path(__file__)));(ROOT/'joint_action_audit_gate_v48.json').write_text(json.dumps(report,indent=2));print('G48_AUDIT_COMPLETE',len(receipts),len(rows),flush=True)
if __name__=='__main__':main()
