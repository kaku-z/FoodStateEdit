"""Descriptive outcome summaries with complete denominators and explicit review limits."""
import argparse
import csv
import json
from pathlib import Path
import statistics

DIMS=['lift','utensil_support','matching_source_notch','scene_identity','no_person','photographic_realism']
METHODS=['A_direct','C_prior_two_reference','E_full_single_reference','L0_no_cut_rgb','L1_free_hole','L2_context_locked']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--notes',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    inv=json.loads((a.bundle/'validation/inventory.json').read_text());assert inv['status']=='complete'
    notes=json.loads(a.notes.read_text(encoding='utf-8'));key={r['id']:r for r in notes['rows']};assert len(key)==len(notes['rows'])
    assert len(key)==sum(r['status']=='generated' for r in inv['cells'])
    rows=[]
    for c in inv['cells']:
        row=dict(c)
        if c['status']=='generated':
            r=key[c['id']];assert r['sha256']==c['output_sha256'] and len(r['ratings'])==6 and set(r['ratings'])<=set('PFU')
            row.update({d:v for d,v in zip(DIMS,r['ratings'])});row['review_notes']=r['notes']
        else:row.update({d:'N' for d in DIMS});row['review_notes']=c['status']
        row['success']=int(all(row[d]=='P' for d in DIMS));rows.append(row)
    tables=[];photos=[]
    for method in METHODS:
        rr=[r for r in rows if r['method']==method];assert len(rr)==12
        tables.append({'method':method,'intended':len(rr),'generated':sum(r['status']=='generated' for r in rr),
            'successes':sum(r['success'] for r in rr),'success_rate':statistics.mean(r['success'] for r in rr),
            'dimensions':{d:{v:sum(r[d]==v for r in rr) for v in 'PFUN'} for d in DIMS}})
        for cid in sorted({r['case_id'] for r in rr}):
            cc=[r for r in rr if r['case_id']==cid]
            photos.append({'case_id':cid,'method':method,'successes':sum(r['success'] for r in cc),'intended':3,'success_rate':statistics.mean(r['success'] for r in cc)})
    contrasts=[]
    for treatment,base in [('L1_free_hole','A_direct'),('L1_free_hole','C_prior_two_reference'),('L1_free_hole','E_full_single_reference'),
                           ('L1_free_hole','L0_no_cut_rgb'),('L2_context_locked','L1_free_hole')]:
        diffs=[]
        for cid in sorted({r['case_id'] for r in photos}):
            t=next(r for r in photos if r['case_id']==cid and r['method']==treatment);b=next(r for r in photos if r['case_id']==cid and r['method']==base)
            diffs.append({'case_id':cid,'difference':t['success_rate']-b['success_rate']})
        contrasts.append({'comparison':treatment+' minus '+base,'photo_differences':diffs,
            'mean_difference':statistics.mean(r['difference'] for r in diffs),
            'positive_photos':sum(r['difference']>0 for r in diffs),'negative_photos':sum(r['difference']<0 for r in diffs),
            'inference':'Descriptive only: four curated sources, input-aware geometry development, one unblinded assistant reviewer.'})
    scan={'coverage':'11/11 considered','Simpson':'Methods reported separately and per photo; no pooled backend claim.',
          'Ecological':'No inference beyond these photographs and conditions.','Berkson':'Source eligibility screening is deliberate selection.',
          'Collider':'All intended cells retained, including failed preparation.','Base_rate':'All pass/fail/uncertain/no-output counts available; no detector sensitivity claim.',
          'Regression_to_mean':'Development outputs excluded; new generation sources fixed before model outputs.',
          'Survivorship':'No replacement of failed cells or best-seed selection.','Look_elsewhere':'All six methods and six dimensions reported.',
          'Forking_paths':'Development gates and source-informed geometry revisions disclosed; formal outputs never retuned.',
          'Causal':'Ablations concern this implementation; L0 still shares geometry masks and target, local methods use extra calls.',
          'Reverse_causality':'Generation parameters and review display rule fixed before inspecting prospective outputs.'}
    result={'reviewer':'Assistant diagnostic, unblinded; no independent human ratings','summary':tables,'photo_averages':photos,
            'paired_contrasts':contrasts,'cells':rows,'statistical_fallacy_scan':scan,
            'denominator':'Four photographs times three seeds per method; seeds are repeated conditions, not independent photographs.'}
    a.output.mkdir(exist_ok=True,parents=True);(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    fields=['id','case_id','method','seed','status','composited','output_sha256']+DIMS+['success','review_notes']
    with (a.output/'assistant_review.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r.get(k,'') for k in fields} for r in rows)
    print(json.dumps([{'method':r['method'],'successes':r['successes'],'intended':r['intended']} for r in tables]))


if __name__=='__main__':main()
