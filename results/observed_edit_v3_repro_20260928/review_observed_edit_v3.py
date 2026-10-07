"""Verify fixed repair runs and compare source-only postprocessing to v2."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw
from foodstateedit.observed_edit.constrained import composite_layers


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def mae(a,b,mask):
    return float(np.abs(a.astype(float)-b.astype(float))[mask].mean())


def boundary_delta(image,base,mask):
    # Diagnostic: changed color across a fixed edit-domain edge, not aesthetics.
    d=image.astype(float)-base.astype(float)
    values=[]
    for axis in [0,1]:
        edge=np.diff(mask.astype(np.int8),axis=axis)!=0
        values.append(np.abs(np.diff(d,axis=axis))[edge].reshape(-1))
    return float(np.concatenate(values).mean())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--real-pairs',type=Path,required=True)
    p.add_argument('--v2-review',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    config=json.loads((args.bundle/'config.json').read_text())
    for case in config['cases']:
        for f in case['files'].values():
            if sha(args.bundle/f['path'])!=f['sha256']:raise ValueError('Input changed after freeze')
    runs={};verified=0
    for condition in config['conditions']:
        folder=args.runs/(condition+'_v1')
        m=json.loads((folder/'run_manifest.json').read_text())
        expected={f"{case['case_id']}_{seed}_{condition}" for case in config['cases'] for seed in config['seeds']}
        if m['status']!='complete_unreviewed' or {r['name'] for r in m['completed']}!=expected:
            raise ValueError('Incomplete run')
        if m['config_sha256']!=sha(args.bundle/'config.json'):raise ValueError('Config mismatch')
        for r in m['completed']:
            for f,h in r['files'].items():
                if sha(folder/r['name']/f)!=h:raise ValueError('Server output mismatch')
                verified+=1
            audit=json.loads((folder/r['name']/'projection_audit.json').read_text())
            if condition=='projected':
                if len(audit['steps'])!=config['inference']['num_inference_steps']:raise ValueError('Missing projection steps')
                if audit['steps'][-1]['next_sigma']!=0:raise ValueError('Projection did not reach clean endpoint')
                if any(s['pinned_latent_max_error']!=0 for s in audit['steps']):raise ValueError('Broken latent projection')
        runs[condition]=(folder,m)
    args.output.mkdir(parents=True,exist_ok=False)
    columns=['source','v2_relit_locked','reference','scaffold_only_raw','scaffold_only_hard',
             'scaffold_only_poisson','projected_raw','projected_hard','projected_poisson']
    overview=Image.new('RGB',(320*len(columns),262*len(config['cases'])),'white')
    draw=ImageDraw.Draw(overview);rows=[];evidence=[]
    for ci,case in enumerate(config['cases']):
        name=case['case_id'];inputs=args.bundle/'inputs'/name
        load=lambda filename:np.array(Image.open(inputs/filename).convert('RGB'))
        source,base=load('source.png'),load('base.png')
        state=np.load(inputs/'layers.npz',allow_pickle=False)
        free=state['free_mask'];target=state['target_mask']
        protected=~(free|state['hole']|target)
        variants={'source':source,'reference':load('reference.png'),
                  'v2_relit_locked':np.array(Image.open(args.v2_review/(name+'_relit_locked.png')).convert('RGB'))}
        for condition,(run,manifest) in runs.items():
            raw=Image.open(run/f"{name}_{config['seeds'][0]}_{condition}"/'raw.png').convert('RGB')
            raw=np.array(raw.resize((source.shape[1],source.shape[0]),Image.Resampling.LANCZOS))
            start=time.perf_counter();hard,smooth=composite_layers(base,raw,free)
            elapsed=time.perf_counter()-start
            for suffix,img in [('raw',raw),('hard',hard),('poisson',smooth)]:
                label=condition+'_'+suffix;variants[label]=img
                Image.fromarray(img).save(args.output/(name+'_'+label+'.png'))
            evidence.append({'case':name,'condition':condition,'composition_seconds':elapsed})
        # Paired reference is first opened after both edited predictions exist.
        real=np.array(Image.open(args.real_pairs/f"pair_{case['dataset_index']:03d}"/'target_rgb.png').convert('RGB'))
        for label,img in variants.items():
            rows.append({'case':name,'variant':label,'source_hole_mae':mae(img,real,state['source_mask']),
                         'known_background_mae':mae(img,source,protected),
                         'kept_food_mae':mae(img,source,state['protected_food']),
                         'payload_copy_mae_engineering_only':mae(img,base,target),
                         'boundary_edit_delta_engineering_only':boundary_delta(img,base,free)})
        for j,label in enumerate(columns):
            overview.paste(Image.fromarray(variants[label]).resize((320,240)),(320*j,262*ci))
            draw.text((320*j+4,262*ci+243),name+' '+label,fill='black')
        selected=['source','v2_relit_locked','scaffold_only_raw','projected_raw','projected_poisson']
        comparison=Image.new('RGB',(320*len(selected),262),'white');d=ImageDraw.Draw(comparison)
        for j,label in enumerate(selected):
            comparison.paste(Image.fromarray(variants[label]).resize((320,240)),(320*j,0))
            d.text((320*j+4,243),label,fill='black')
        comparison.save(args.output/(name+'_repair_comparison.png'))
    overview.save(args.output/'all_variants.png')
    with (args.output/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result={'status':'verified_pending_semantic_review','server_files_verified':verified,
            'generation_count':sum(len(m['completed']) for _,m in runs.values()),
            'config_sha256':sha(args.bundle/'config.json'),'metrics':rows,'composition':evidence,
            'limits':['Two seen development images. No generalization or physical claim.',
                      'Copy, protected-region and latent zeros are construction invariants.',
                      'Boundary edit delta measures a specific numerical seam mechanism, not perceptual quality.',
                      'Real targets validate source-hole appearance only. No lift endpoint ground truth.',
                      'No independent human ballots.']}
    (args.output/'review_manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'verified':verified,'generated':result['generation_count']}))


if __name__=='__main__':main()
