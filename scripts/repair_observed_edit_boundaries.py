"""Post-hoc development fix of Poisson brightness drift; no new model samples."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw
from foodstateedit.observed_edit.constrained import poisson_merge
from review_observed_edit_v3 import mae,boundary_delta


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--previous-review',type=Path,required=True)
    p.add_argument('--real-pairs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();c=json.loads((a.bundle/'config.json').read_text())
    a.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for case in c['cases']:
        name=case['case_id'];d=a.bundle/'inputs'/name
        for f in case['files'].values():
            if sha(a.bundle/f['path'])!=f['sha256']:raise ValueError('Input mismatch')
        state=np.load(d/'layers.npz',allow_pickle=False)
        base=np.array(Image.open(d/'base.png').convert('RGB'))
        source=np.array(Image.open(d/'source.png').convert('RGB'))
        for condition in c['conditions']:
            run=a.runs/(condition+'_v1');manifest=json.loads((run/'run_manifest.json').read_text())
            if manifest['status']!='complete_unreviewed' or manifest['config_sha256']!=sha(a.bundle/'config.json'):
                raise ValueError('Incomplete or wrong generation run')
            run_name=f"{name}_{c['seeds'][0]}_{condition}"
            record=next(r for r in manifest['completed'] if r['name']==run_name)
            raw_path=run/run_name/'raw.png'
            if sha(raw_path)!=record['files']['raw.png']:raise ValueError('Raw image changed')
            raw=np.array(Image.open(raw_path).convert('RGB').resize((640,480),Image.Resampling.LANCZOS))
            # Fixed once, after v3 model results; no fitting to target pixels.
            result=poisson_merge(base,raw,state['free_mask'],screening=.1)
            Image.fromarray(result).save(a.output/f'{name}_{condition}_screened.png')
            target=np.array(Image.open(a.real_pairs/f"pair_{case['dataset_index']:03d}"/'target_rgb.png').convert('RGB'))
            rows.append({'case':name,'condition':condition,'source_hole_mae':mae(result,target,state['source_mask']),
                         'payload_copy_mae_engineering_only':mae(result,base,state['target_mask']),
                         'kept_food_mae':mae(result,source,state['protected_food']),
                         'boundary_edit_delta_engineering_only':boundary_delta(result,base,state['free_mask'])})
        # Full before/after; never hide the unconstrained or pure-Poisson output.
        sheet=Image.new('RGB',(1600,262),'white');draw=ImageDraw.Draw(sheet)
        paths=[d/'source.png',a.previous_review/f'{name}_projected_raw.png',
               a.previous_review/f'{name}_projected_poisson.png',
               a.output/f'{name}_projected_screened.png',a.output/f'{name}_scaffold_only_screened.png']
        labels=['source','projected raw','pure Poisson','screened projected','screened unprojected']
        for i,(path,label) in enumerate(zip(paths,labels)):
            sheet.paste(Image.open(path).convert('RGB').resize((320,240)),(320*i,0))
            draw.text((320*i+4,243),label,fill='black')
        sheet.save(a.output/f'{name}_boundary_comparison.png')
    with (a.output/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (a.output/'manifest.json').write_text(json.dumps({'stage':'posthoc_development_repair',
       'screening':.1,'model_calls_added':0,'target_used_to_fit_parameter':False,
       'selection_note':'Fixed .1 for both images/conditions to limit propagation to roughly 3 pixels; not a target-based parameter search.',
       'script_sha256':sha(Path(__file__)),'core_sha256':sha(Path(__file__).resolve().parents[1]/'foodstateedit/observed_edit/constrained.py'),
       'metrics':rows,'claim_limit':'Same two seen cases; author review only.'},indent=2))
    print(json.dumps(rows))


if __name__=='__main__':main()
