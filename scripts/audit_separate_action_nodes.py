"""Independently audit G56 node receipts and restored photo boundaries."""
import time
from pathlib import Path
import numpy as np
from PIL import Image
from run_joint_action_full_noise import ROOT,read,write,sha,rgb

def main():
    g=ROOT/'gate_v56'
    while read(g/'execution.json')['status']!='complete_unreviewed':time.sleep(20)
    cfg=read(g/'config.json');jobs={j['id']:j for j in cfg['jobs']};found={read(p)['id']:p.parent for p in g.glob('worker_*/*/result.json')};assert set(found)==set(jobs) and len(found)==48
    receipts=[];rows=[];noise={}
    for jid,j in jobs.items():
        d=found[jid];q=read(d/'request.json');ca=read(d/'context_audit.json');assert q['actual_start_sigma']==1 and q['conditioning_input_order_executed']==['source','reference'];assert q['initial_latents_sha256']==q['noise_sha256'];assert len(ca['steps'])==24 and all(v['structural_strength']>0 and v['fixed_fraction']>0 for v in ca['steps']);assert rgb(d/'raw.png').shape==(384,384,3)
        for n,h in read(d/'result.json')['files'].items():assert sha(d/n)==h
        for v in j['files'].values():assert sha(Path(v['path']))==v['sha256']
        tr=read(Path(j['transform']));assert sha(Path(tr['parent']))==tr['parent_sha256'];old=ROOT/'gate_v50/contexts'/j['case_id']/str(j['seed']);assert sha(Path(j['files']['source']['path']))==sha(old/'source.png');assert sha(Path(j['files']['reference']['path']))==sha(old/'reference.png')
        key=(j['case_id'],j['seed']);noise.setdefault(key,[]).append(q['noise_sha256']);receipts.append(dict(id=jid,raw_sha256=sha(d/'raw.png'),noise_sha256=q['noise_sha256'],context_steps=24,actual_start_sigma=1.))
    assert all(len(v)==2 and v[0]==v[1] for v in noise.values())
    ex=read(g/'execution.json')
    for r in ex['rows']:
        cid=r['case_id'];seed=r['seed'];variant=r['variant'];old=ROOT/'gate_v50/contexts'/cid/str(seed);tr=read(old/'transform.json');parent=ROOT/'gate_v53/geometry_soft'/(cid+'__'+str(seed))/'pre_sampling.png';assert sha(parent)==r['parent_sha256'];mask=(np.asarray(Image.open(old/'mask_0.png'))>0)|(np.asarray(Image.open(old/'mask_1.png'))>0);protected=np.asarray(Image.open(old/'protected_source_color.png'))>0
        for i,node in enumerate(['food','cut']):assert sha(found[cid+'__node_'+node+'__'+str(seed)]/'raw.png')==r['raw_node_sha256'][i]
        d=g/variant/(cid+'__'+str(seed));pre=rgb(d/'pre_sampling.png');final=rgb(d/'composited.png');assert np.array_equal(pre[~mask],rgb(parent)[~mask]);assert np.array_equal(pre[protected],rgb(parent)[protected]);geo=ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid;edit=np.asarray(Image.open(geo/'edit_mask.png'))>0;assert np.array_equal(final[~edit],rgb(geo/'source.png')[~edit]);assert sha(d/'composited.png')==r['output_sha256'];rows.append(dict(case_id=cid,seed=seed,variant=variant))
    assert len(rows)==48
    write(ROOT/'separate_action_nodes_audit_gate_v56.json',dict(status='verified',raw_cells_verified=48,compositions_verified=48,raw_receipts=receipts,rows=rows,shared_noise_pairs=24,generated_silhouette_contact_volume_verified=False,photographic_realism_verified=False,scope='Actual 384px node receipts, separated real-source RGB conditioning, structural context, shared noise pairs, raw hashes and restored protected areas. Fixed masks do not prove output semantic identity or geometry.',script_sha256=sha(Path(__file__))))
    print('SEPARATE_NODES_AUDIT_COMPLETE',48,48,flush=True)

if __name__=='__main__':main()
