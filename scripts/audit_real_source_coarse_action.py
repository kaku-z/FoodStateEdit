"""Validate retained G55 receipts and disclosed crop restoration invariants."""
from pathlib import Path
import time
import numpy as np
from PIL import Image
from run_joint_action_full_noise import ROOT, read, write, sha, rgb

def main():
    g=ROOT/'gate_v55'
    while read(g/'execution.json')['status']!='complete_unreviewed':time.sleep(20)
    cfg=read(g/'config.json');jobs={j['id']:j for j in cfg['jobs']}
    raws={read(p)['id']:p.parent for p in g.glob('worker_*/*/result.json')};assert set(raws)==set(jobs) and len(raws)==24
    receipts=[];rows=[]
    for jid,j in jobs.items():
        d=raws[jid];req=read(d/'request.json');ctx=read(d/'context_audit.json')
        assert req['conditioning_input_order_executed']==['source','reference'] and req['context_layout_encoded_separately']
        assert req['actual_start_sigma']==1. and len(ctx['steps'])==24 and ctx['raw_output_no_pixel_compositor']
        assert all(s['structural_strength']>0 and s['fixed_fraction']>0 for s in ctx['steps'])
        old=read(ROOT/'gate_v54/config.json');oldjob=next(v for v in old['jobs'] if v['case_id']==j['case_id'] and v['seed']==j['seed']);oldreq=read(next((ROOT/'gate_v54').glob('worker_*/'+oldjob['id']+'/request.json')))
        assert j['structure']==dict(lowpass=True,kernel=3,begin_taper_fraction=.25,end_fraction=.65,tail_weight=.10)
        assert req['noise_sha256']==oldreq['noise_sha256'] and req['initial_latents_sha256']==oldreq['initial_latents_sha256']
        for key in ['layout','source','reference','edit_mask']:assert j['files'][key]['sha256']==oldjob['files'][key]['sha256']
        assert rgb(d/'raw.png').shape==(384,768,3)
        for name,value in read(d/'result.json')['files'].items():assert sha(d/name)==value
        for value in j['files'].values():assert sha(Path(value['path']))==value['sha256']
        tr=read(Path(j['transform']));p=Path(j['transform']).parent;base=rgb(Path(tr['parent']));assert sha(Path(tr['parent']))==tr['parent_sha256']
        masks=[np.asarray(Image.open(p/('mask_'+str(i)+'.png')))>0 for i in range(2)]
        assert not (masks[0]&masks[1]).any();mask=masks[0]|masks[1]
        for box,m in zip(tr['boxes'],masks):
            x,y,x1,y1=box;assert int(m.sum())==int(m[y:y1,x:x1].sum())
        receipts.append(dict(id=jid,actual_start_sigma=req['actual_start_sigma'],raw_sha256=sha(d/'raw.png'),noise_sha256=req['noise_sha256'],context_steps=24,paired_panel_layout_not_native_photo=True))
        for variant in ['direct_masked_raw','source_chroma_projected']:
            out=g/variant/jid;r=read(out/'result.json');pre=rgb(out/'pre_sampling.png');final=rgb(out/'composited.png')
            assert sha(out/'composited.png')==r['output_sha256'] and r['raw_sha256']==sha(d/'raw.png')
            assert np.array_equal(pre[~mask],base[~mask]);protected=np.asarray(Image.open(p/'protected_source_color.png'))>0;assert np.array_equal(pre[protected],base[protected])
            geo=Path(tr['original_source']).parent;edit=np.asarray(Image.open(geo/'edit_mask.png'))>0;assert np.array_equal(final[~edit],rgb(geo/'source.png')[~edit])
            rows.append(dict(id=jid,variant=variant,outside_crop_masks_parent_exact_before_sampling=True,protected_approximate_source_color_parent_exact_before_sampling=True,outside_geometry_source_exact=True))
    assert len(rows)==48
    write(ROOT/'real_source_coarse_action_audit_gate_v55.json',dict(status='verified',raw_cells_verified=24,compositions_verified=48,raw_receipts=receipts,rows=rows,photographic_realism_verified=False,generated_silhouette_contact_volume_verified=False,scope='Source hashes, actual latent context updates and coordinate restoration invariants. Fixed masks do not verify true food geometry or photographic appearance.',script_sha256=sha(Path(__file__))))
    print('G55_AUDIT_COMPLETE',24,48,flush=True)

if __name__=='__main__':main()
