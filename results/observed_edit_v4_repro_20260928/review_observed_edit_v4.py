"""Compare native generations only, including the old failure and no-lock control."""
import argparse,csv,hashlib,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def mae(a,b,m):return float(np.abs(a.astype(float)-b.astype(float))[m].mean())


def dilate(m,r):
    return cv2.dilate(m.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))).astype(bool)


def texture_ncc(a,b,m):
    # A descriptive source-texture correspondence proxy; not semantic identity.
    x=a[m].astype(float);y=b[m].astype(float)
    x-=x.mean(axis=0);y-=y.mean(axis=0)
    denominator=np.sqrt(np.sum(x*x)*np.sum(y*y))
    return float(np.sum(x*y)/denominator) if denominator>0 else None


def verified_images(run_root,condition,config,require_interior_audit=False):
    d=run_root/(condition+'_v1');m=json.loads((d/'run_manifest.json').read_text())
    expected={f"{c['case_id']}_{s}_{condition}" for c in config['cases'] for s in config['seeds']}
    if m['status']!='complete_unreviewed' or {r['name'] for r in m['completed']}!=expected:
        raise ValueError('Wrong or incomplete generation set')
    found={}
    for r in m['completed']:
        for f,h in r['files'].items():
            if sha(d/r['name']/f)!=h:raise ValueError('Output hash mismatch')
        if require_interior_audit:
            audit=json.loads((d/r['name']/'projection_audit.json').read_text())
            if len(audit['steps'])!=config['inference']['num_inference_steps']:raise ValueError('Incomplete callback')
            if audit['steps'][-1]['strength']!=0:raise ValueError('Endpoint not released')
            if any(s['outside_core_max_update']!=0 for s in audit['steps']):raise ValueError('Background was anchored')
        found[r['name'].split('_')[0]]=d/r['name']/'raw.png'
    return m,found


def main():
    p=argparse.ArgumentParser()
    for name in ['bundle','runs','v3_bundle','v3_runs','real_pairs','output']:
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a=p.parse_args();config=json.loads((a.bundle/'config.json').read_text())
    old=json.loads((a.v3_bundle/'config.json').read_text());paths={};runs=[]
    for case,previous in zip(config['cases'],old['cases']):
        if case['prompt']!=previous['prompt']:raise ValueError('Prompt changed')
        if case['files']['reference.png']['sha256']!=previous['files']['reference.png']['sha256']:
            raise ValueError('Reference changed')
        for f in case['files'].values():
            if sha(a.bundle/f['path'])!=f['sha256']:raise ValueError('Input hash mismatch')
    for condition in config['conditions']:
        manifest,found=verified_images(a.runs,condition,config,True)
        if manifest['config_sha256']!=sha(a.bundle/'config.json'):raise ValueError('Config changed')
        paths['v4_'+condition]=found;runs.append(manifest)
    for condition in ['scaffold_only','projected']:
        manifest,found=verified_images(a.v3_runs,condition,old)
        if manifest['config_sha256']!=sha(a.v3_bundle/'config.json'):raise ValueError('V3 config changed')
        paths['v3_'+condition]=found
    a.output.mkdir(parents=True,exist_ok=False)
    rows=[];mask_records=[]
    columns=['source','v3_scaffold_only','v3_projected','v4_interior','v4_detail']
    overview=Image.new('RGB',(320*len(columns),262*len(config['cases'])),'white');draw=ImageDraw.Draw(overview)
    for i,case in enumerate(config['cases']):
        name=case['case_id'];d=a.bundle/'inputs'/name
        state=np.load(d/'layers.npz',allow_pickle=False)
        old_state=np.load(a.v3_bundle/'inputs'/name/'layers.npz',allow_pickle=False)
        source=np.array(Image.open(d/'source.png').convert('RGB'))
        reference=np.array(Image.open(d/'reference.png').convert('RGB'))
        real=np.array(Image.open(a.real_pairs/f"pair_{case['dataset_index']:03d}"/'target_rgb.png').convert('RGB'))
        hole=state['source_mask'];target=state['target_mask'];kept=state['kept_food_mask']
        target_core=(state['core_weight']==1)&target
        tool=(old_state['free_mask'] & ~old_state['hole'])|target
        annulus=dilate(hole,48)&~hole&~dilate(kept,4)&~dilate(tool,12)
        far=~(dilate(hole,64)|dilate(tool,32)|dilate(kept,8))
        variants={'source':source}
        for label,by_case in paths.items():
            img=Image.open(by_case[name]).convert('RGB')
            variants[label]=np.array(img.resize((640,480),Image.Resampling.LANCZOS))
            Image.fromarray(variants[label]).save(a.output/(name+'_'+label+'.png'))
        for label,img in variants.items():
            rows.append({'case':name,'variant':label,'source_hole_mae':mae(img,real,hole),
                         'old_shadow_annulus_mae_new_diagnostic':mae(img,real,annulus),
                         'far_background_change_mae':mae(img,source,far),
                         'kept_food_change_mae':mae(img,source,kept),
                         'target_core_rgb_mae':mae(img,reference,target_core),
                         'target_core_texture_ncc_proxy':texture_ncc(img,reference,target_core)})
        masks=np.zeros_like(source);masks[annulus]=(0,160,255);masks[hole]=(255,160,0);masks[target_core]=(0,220,80)
        Image.fromarray(masks).save(a.output/(name+'_metric_regions.png'))
        mask_records.append({'case':name,'hole_pixels':int(hole.sum()),'annulus_pixels':int(annulus.sum()),
                             'target_core_pixels':int(target_core.sum()),'far_pixels':int(far.sum())})
        comparison=Image.new('RGB',(320*len(columns),262),'white');cd=ImageDraw.Draw(comparison)
        for j,label in enumerate(columns):
            thumb=Image.fromarray(variants[label]).resize((320,240))
            overview.paste(thumb,(320*j,262*i));comparison.paste(thumb,(320*j,0))
            draw.text((320*j+4,262*i+243),name+' '+label,fill='black')
            cd.text((320*j+4,243),label,fill='black')
        comparison.save(a.output/(name+'_comparison.png'))
    overview.save(a.output/'comparison.png')
    with (a.output/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={'status':'verified_pending_visual_review','new_model_calls':4,'new_server_files_verified':8,
            'previous_model_images_verified':4,'input_and_prompt_identical_to_v3':True,
            'runs':runs,'metrics':rows,'metric_regions':mask_records,
            'limits':['Two seen cases, one seed; no independent ratings or generalization claim.',
                      'All displayed results are model generations, without restored source pixels or blending.',
                      'Annulus is a new source-defined diagnostic, not an old preregistered endpoint.',
                      'Texture NCC measures interior correspondence, not complete semantic identity.',
                      'No measured lift endpoint, material albedo, or physical conservation claim.']}
    (a.output/'review_manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'new_calls':4,'verified_new_files':8,'previous_raws':4}))


if __name__=='__main__':main()
