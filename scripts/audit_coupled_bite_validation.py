"""Verify provenance, complete denominators and mechanical claims of the repair run."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
from PIL import Image


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--review',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();b=a.bundle
    def local(remote):return b/remote.split('/first_bite_coupled_20260929/',1)[1]
    freeze=read(b/'validation/freeze_decision.json');cfg=read(b/'validation/frozen.json');inv=read(b/'validation/inventory.json')
    assert inv['status']=='complete' and inv['expected_raw_calls']==inv['completed_raw_calls']==84
    assert len(inv['cells'])==72 and len({x['id'] for x in inv['cells']})==72
    assert all(x['status']=='generated' for x in inv['cells'])
    for f,k in [('inputs/manifest.json','source_annotation_manifest_sha256'),('geometry_v4/manifest.json','geometry_manifest_sha256'),
                ('validation/frozen.json','configuration_sha256'),('validation/expected_cells.json','expected_cells_sha256')]:assert sha(b/f)==freeze[k],f
    for f,h in freeze['code_sha256'].items():assert sha(b/f)==h,f
    sources=read(b/'inputs/manifest.json')['cases']
    geometry=[]
    for g in read(b/'geometry_v4/manifest.json')['cases']:
        assert g['status']=='geometry_ready';cs=g['cut_and_support']
        assert cs['relative_partition_volume_residual']<1e-9 and cs['fork_contact_gap_over_extent']<1e-8
        assert cs['com_inside_support_hull'] and cs['lifted_remaining_overlap_volume']<1e-10 and cs['fork_remaining_overlap_volume']<1e-10
        geometry.append({'case_id':g['case_id'],'source_silhouette_iou':g['fit']['source_silhouette_iou'],
                         'normal_agreement_degrees':g['fit']['normal_agreement_degrees'],
                         'cut_and_support':cs,'scope':'Mechanical checks within inferred cuboid; not recovered physical ground truth.'})
    for c in sources:
        for f,h in c['files'].items():assert sha(b/'inputs'/c['case_id']/f)==h
    jobs={j['id']:j for j in cfg['jobs']};assert len(jobs)==84
    for j in jobs.values():
        for f in j['files'].values():assert sha(local(f['path']))==f['sha256']
    noise=defaultdict(set);anchor_steps=0;anchored_jobs=0;raw_files=0
    for r in inv['raw_results']:
        d=local(r['raw_path']).parent
        for f,h in r['files'].items():assert sha(d/f)==h;raw_files+=1
        q=read(d/'request.json');j=jobs[r['id']]
        assert {k:q[k] for k in j}==j
        sz=Image.open(local(j['files'][j['input_order'][0]]['path'])).size
        noise[(r['seed'],sz)].add(q['noise_sha256'])
        ca=read(d/'context_audit.json');assert ca['enabled']==j['lock_context']
        if ca['enabled']:
            assert len(ca['steps'])==40 and all(s['editable_latent_update_max']==0 for s in ca['steps'])
            anchored_jobs+=1;anchor_steps+=len(ca['steps'])
    assert all(len(h)==1 for h in noise.values()) and anchored_jobs==36
    worker_summaries=[]
    for p in sorted((b/'validation').glob('worker_*/manifest.json')):
        m=read(p);assert m['status']=='complete_unreviewed' and len(m['completed'])==21
        assert m['config_sha256']==freeze['configuration_sha256'] and sha(p.parent/'executed_script.py')==freeze['code_sha256']['run_coupled_bite_qwen.py']
        worker_summaries.append({'worker':p.parent.name,'jobs':len(m['completed']),'elapsed_seconds':m['finished_unix']-m['started_unix']})
    notes=read(a.review/'../assistant_review_notes.json');ratings={r['id']:r for r in notes['rows']};assert len(ratings)==72
    composition_count=0
    for c in inv['cells']:
        p=local(c['output_path']);assert sha(p)==c['output_sha256']==ratings[c['id']]['sha256']
        assert len(ratings[c['id']]['ratings'])==6
        if c['composited']:
            co=read(p.parent/'composition.json');assert co['outside_source_exact'] and not co['physical_correspondence_guaranteed']
            prep=local(c['prepared']);source=np.asarray(Image.open(prep/'source.png'));out=np.asarray(Image.open(p));union=np.zeros(source.shape[:2],bool)
            for step in co['steps']:
                name='hole_edit' if step['region']=='hole' else 'target_composite_mask'
                mask=np.asarray(Image.open(prep/(name+'.png')))>0;x0,y0,x1,y1=step['crop']['box_xyxy'];support=np.zeros_like(mask);support[y0:y1,x0:x1]=True;union|=mask&support
            assert np.array_equal(source[~union],out[~union]);composition_count+=1
    assert composition_count==36
    completion=read(b/'compute_completion.json');assert completion['status']=='compute_complete_review_required'
    replay=read(b/'reproducibility/replay_audit.json')
    assert all(c['exact_file_match'] for c in replay['raw_checks']) and replay['exact_composited_file_match'] and replay['pixel_max_difference']==0
    obs=read(b/'observer/observations.json');assert obs['status']=='complete' and len(obs['images'])==76
    expected_hash={c['id']:c['output_sha256'] for c in inv['cells']}
    for c in sources:expected_hash[c['case_id']+'__source']=c['files']['source.png']
    assert set(expected_hash)=={r['id'] for r in obs['images']}
    for r in obs['images']:assert r['sha256']==expected_hash[r['id']]
    locality=read(a.review/'locality_metrics.json');observer=[];resource=[]
    for method in sorted({c['method'] for c in inv['cells']}):
        ids={c['id'] for c in inv['cells'] if c['method']==method};oo=[r for r in obs['images'] if r['id'] in ids];ll=[r for r in locality if r['id'] in ids]
        observer.append({'method':method,'images':len(oo),'fork_detected':sum(r['prompts']['fork']['instance_count']>0 for r in oo),
                         'hand_detected':sum(r['prompts']['hand']['instance_count']>0 for r in oo),
                         'locality_mae_mean_255':statistics.mean(r['outside_mae_255'] for r in ll),
                         'locality_mae_max_255':max(r['outside_mae_255'] for r in ll)})
    for method in sorted({r['method'] for r in inv['raw_results']}):
        rr=[r for r in inv['raw_results'] if r['method']==method]
        resource.append({'component':method,'count':len(rr),'mean_seconds':statistics.mean(r['seconds'] for r in rr),
                         'max_allocated_gpu_gib':max(max(r['peak_allocated_gib']) for r in rr)})
    supervisor=read(b/'validation/supervisor.json');assert supervisor['returncodes']==[0,0,0,0]
    result={'status':'verified','scope':'Recorded input/config/code integrity, completion, mechanical locality and one exact replay. NOT verified physical 3-D or human perceptual validity.',
        'frozen_configuration_sha256':freeze['configuration_sha256'],'source_images':4,'endpoints':72,'raw_calls':84,'rated_endpoints':72,
        'checked_raw_files':raw_files,'anchored_jobs':anchored_jobs,'checked_anchor_steps':anchor_steps,'composited_endpoints':composition_count,
        'same_noise_groups':[{'seed':k[0],'size':k[1],'hashes':list(v)} for k,v in noise.items()],
        'replay':replay,'observer_images':76,'observer_advisory_only':True,'advisory_observer_and_locality':observer,
        'workers':worker_summaries,'component_resources':resource,'formal_worker_wall_seconds':supervisor['elapsed_seconds'],
        'minimum_host_available_gib':min(s['mem_available_mib'] for s in supervisor['resource_samples'])/1024,
        'source_original_sha256':{c['file_name']:c['source_sha256'] for c in sources},
        'geometry_proxy_checks':geometry,
        'archive_case_mapping':'Linux validation/FROZEN.json is locally named freeze_decision.json; original archive preserves both case-distinct files.'}
    a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','source_images','endpoints','raw_calls','rated_endpoints','observer_images','formal_worker_wall_seconds']}))


if __name__=='__main__':main()
