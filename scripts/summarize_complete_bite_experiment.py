"""Descriptive paired summaries; repeated seeds are never independent photographs."""
import argparse
import csv
import json
from pathlib import Path

DIMS=['lift','utensil_support','matching_source_notch','scene_identity','no_person','photographic_realism']


def main():
    import numpy as np
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    inv=json.loads((a.bundle/'formal/inventory.json').read_text());assert inv['status']=='complete'
    with a.review.open(encoding='utf-8-sig',newline='') as f:reviews=list(csv.DictReader(f))
    keyed={(r['backend'],r['id']):r for r in reviews};assert len(keyed)==192
    cells=[]
    for x in inv['cells']:
        row=dict(x);r=keyed[x['backend'],x['id']]
        for dim in DIMS:
            if x['status']=='generated':assert r[dim] in ['pass','fail','uncertain'],(x['id'],dim,r[dim])
            row[dim]=r[dim] if x['status']=='generated' else 'no_output'
        row['success']=int(all(row[d]=='pass' for d in DIMS));row['review_notes']=r['notes'];cells.append(row)
    tables=[];photos=[];contrasts=[]
    for backend in ['qwen','vace']:
        for method in ['A_direct','B_planar','C_rgb3d','D_staged3d']:
            rs=[r for r in cells if r['backend']==backend and r['method']==method]
            byphoto=[]
            for cid in sorted(set(r['case_id'] for r in rs)):
                vals=[r for r in rs if r['case_id']==cid]
                assert len(vals)==3
                pr={'backend':backend,'method':method,'case_id':cid,'cluster_id':vals[0]['cluster_id'],'stratum':vals[0]['stratum'],'success_rate':sum(r['success'] for r in vals)/3}
                pr.update({d:sum(r[d]=='pass' for r in vals)/3 for d in DIMS});photos.append(pr);byphoto.append(pr)
            table={'backend':backend,'method':method,'intended':24,'generated':sum(r['status']=='generated' for r in rs),'complete_successes':sum(r['success'] for r in rs),'success_rate':sum(r['success'] for r in rs)/24,
                   'dimensions':{d:{v:sum(r[d]==v for r in rs) for v in ['pass','fail','uncertain','no_output']} for d in DIMS},
                   'strata':{s:{'n_photos':sum(p['stratum']==s for p in byphoto),'success_rate':float(np.mean([p['success_rate'] for p in byphoto if p['stratum']==s]))} for s in ['main','challenge']}}
            tables.append(table)
        for treatment,base in [('C_rgb3d','A_direct'),('C_rgb3d','B_planar'),('D_staged3d','C_rgb3d'),('D_staged3d','A_direct'),('D_staged3d','B_planar')]:
            diff=[]
            for cid in sorted(set(p['case_id'] for p in photos if p['backend']==backend)):
                d=next(p for p in photos if p['backend']==backend and p['case_id']==cid and p['method']==treatment)
                c=next(p for p in photos if p['backend']==backend and p['case_id']==cid and p['method']==base)
                diff.append({'case_id':cid,'cluster_id':d['cluster_id'],'difference':d['success_rate']-c['success_rate']})
            clusters=sorted(set(d['cluster_id'] for d in diff))
            values=np.array([np.mean([d['difference'] for d in diff if d['cluster_id']==cl]) for cl in clusters])
            contrasts.append({'backend':backend,'comparison':treatment+' minus '+base,'paired_photo_differences':diff,
                'mean_photo_difference':float(np.mean([d['difference'] for d in diff])),'mean_cluster_difference':float(values.mean()),
                'positive_clusters':int((values>0).sum()),'zero_clusters':int((values==0).sum()),'negative_clusters':int((values<0).sum()),
                'inference':'Descriptive only. Seven curated clusters and one nonindependent assistant reviewer do not support population-level significance claims.'})
    # Check that the Qwen seed comparison really reused the same initial packed noise.
    noise={}
    for x in inv['cells']:
        if x['backend']!='qwen' or x['status']!='generated':continue
        rel=x['raw_path'].split('/first_bite_complete_20260929/',1)[1]
        req=json.loads((a.bundle/rel).with_name('request.json').read_text())
        noise.setdefault((x['case_id'],x['seed']),set()).add(req['noise_sha256'])
    assert all(len(s)==1 for s in noise.values())
    audit={'reviewer':'Assistant diagnostic review; NOT independent blinded human ratings',
        'cells':cells,'summary':tables,'photo_averages':photos,'paired_contrasts':contrasts,
        'qwen_paired_noise_hash_check':{'groups':len(noise),'all_equal_within_photo_seed':True},
        'statistical_fallacy_scan':{'coverage':'11/11 considered','Simpson':'Main/challenge strata and photo-level differences reported; no pooled causal claim across backends.',
          'Ecological':'Inference stays at tested photos/conditions; no population or person-level claim.',
          'Berkson':'Curated cohesive-block source selection limits generalization; acknowledged.',
          'Collider':'No conditioning on successful reconstruction in the primary denominator.',
          'Base_rate':'All intended cells and all outcome categories reported; observer detections are not sensitivity/specificity.',
          'Regression_to_mean':'Fixed new source set and paired seeds; development excluded.',
          'Survivorship':'Preprocessing and technical failures included as zero complete success.',
          'Look_elsewhere':'All four methods, both backends, all seeds and all six dimensions reported; no best-seed claims.',
          'Forking_paths':'Development alternatives retained and final configuration hash frozen before formal generation; exploratory design, not external preregistration.',
          'Causal':'Ablations test this implementation on these fixed inputs; external physical and broad generalization claims unsupported.',
          'Reverse_causality':'Inputs and settings fixed before outputs; no output-driven parameter changes in formal matrix.'}}
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'summary.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(tables,ensure_ascii=False))


if __name__=='__main__':main()
