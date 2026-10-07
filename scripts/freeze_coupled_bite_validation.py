"""Freeze all prospective endpoints before repaired-model outputs are observed."""
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929')
OLD=Path('/host/space0/guo-z/tf-ufi/first_bite_complete_20260929')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    geom=json.loads((ROOT/'geometry_v4/manifest.json').read_text());assert geom['status']=='complete'
    sources=json.loads((ROOT/'inputs/manifest.json').read_text())
    gate=json.loads((ROOT/'gate_v2.json').read_text())
    prior=json.loads((OLD/'formal/qwen_frozen.json').read_text())
    gate1=json.loads((ROOT/'gate_v1.json').read_text())
    formal=ROOT/'validation';formal.mkdir(exist_ok=False)
    inputs=formal/'prepared';inputs.mkdir()
    config={k:prior[k] for k in ['backend','expected_pipeline_sha256','model_audit','inference']}
    config.update(stage='prospective_generation_validation',not_formal=False)
    jobs=[];cells=[];transforms={}
    methods=['A_direct','C_prior_two_reference','E_full_single_reference','L0_no_cut_rgb','L1_free_hole','L2_context_locked']
    for case in sources['cases']:
        cid=case['case_id'];g=next(x for x in geom['cases'] if x['case_id']==cid)
        d=inputs/cid;d.mkdir()
        src=np.asarray(Image.open(ROOT/'inputs'/cid/'source.png').convert('RGB'))
        arrays={'source':src};meta={}
        if g['status']=='geometry_ready':
            gp=ROOT/'geometry_v4'/cid
            proxy=np.asarray(Image.open(gp/'rgb_control.png').convert('RGB'))
            hole=np.asarray(Image.open(gp/'hole_mask.png'))>0
            target=(np.asarray(Image.open(gp/'material_mask.png'))>0)|(np.asarray(Image.open(gp/'rigid_mask.png'))>0)
            remove=src.copy();remove[hole]=proxy[hole]
            lifted=src.copy();lifted[target]=proxy[target]
            arrays.update(proxy=proxy,remove_full=remove,lift_full=lifted,
                hole_edit=binary_dilation(hole,iterations=10),target_edit=binary_dilation(target,iterations=10))
            for kind,im,mask in [('hole',remove,arrays['hole_edit']),('target',lifted,arrays['target_edit'])]:
                yy,xx=np.where(mask)
                side=min(448,32*((max(224,int(max(np.ptp(xx)+1,np.ptp(yy)+1)*1.8))+31)//32))
                cx,cy=(xx.min()+xx.max())/2,(yy.min()+yy.max())/2
                x0=int(np.clip(round(cx-side/2),0,640-side));y0=int(np.clip(round(cy-side/2),0,480-side))
                box=(x0,y0,x0+side,y0+side);meta[kind]={'box_xyxy':box,'network_size':[512,512]}
                for label,array,resample in [(kind+'_crop',im,Image.Resampling.LANCZOS),
                    (kind+'_source_crop',src,Image.Resampling.LANCZOS),
                    (kind+'_crop_edit',np.uint8(mask)*255,Image.Resampling.NEAREST)]:
                    arrays[label]=np.asarray(Image.fromarray(array).crop(box).resize((512,512),resample))
            # Composition owns the source region even if a broad target feather
            # comes close to it; this prevents a target crop restoring the notch.
            arrays['target_composite_mask']=binary_dilation(arrays['target_edit'],iterations=14)&~binary_dilation(arrays['hole_edit'],iterations=4)
        files={}
        for name,arr in arrays.items():
            p=d/(name+'.png');Image.fromarray(np.uint8(arr)*255 if arr.dtype==bool else arr).save(p)
            files[name]={'path':str(p),'sha256':sha(p)}
        (d/'transforms.json').write_text(json.dumps(meta,indent=2)+'\n');transforms[cid]=meta
        def add(method,seed,order,prompt,lock=False,mask=None):
            jj={'id':cid+'__'+method+'__'+str(seed),'case_id':cid,'method':method,'seed':seed,
                'input_order':order,'files':{k:files[k] for k in order},'prompt':prompt,'lock_context':lock}
            if mask:jj['files']['edit_mask']=files[mask]
            jobs.append(jj);return jj['id']
        for seed in [41,163,907]:
            direct=next(j for j in prior['jobs'] if j['method']=='A_direct')
            previous=next(j for j in prior['jobs'] if j['method']=='C_rgb3d')
            # Replace the source-specific description but preserve the old prompt
            # protocol rather than selecting a stronger baseline after outputs.
            old_case=next(c for c in json.loads((OLD/'inputs/manifest.json').read_text())['cases'] if c['case_id']==direct['case_id'])
            direct_prompt=direct['prompt'].replace(old_case['description'],case['description'])
            ids={'A_direct':add('A_direct',seed,['source'],direct_prompt)}
            if g['status']=='geometry_ready':
                old_case=next(c for c in json.loads((OLD/'inputs/manifest.json').read_text())['cases'] if c['case_id']==previous['case_id'])
                pp=previous['prompt'].replace(old_case['description'],case['description'])
                ids['C_prior_two_reference']=add('C_prior_two_reference',seed,['proxy','source'],pp)
                prompt=next(j['prompt'] for j in gate1['jobs'] if j['method']=='E_joint_single_reference')
                prompt+=' The exposed cut is soft smooth silken tofu with very fine grain, not bread or sponge. The cavity has a solid pale tofu floor, not a tunnel through the bottom or a puddle of sauce.'
                ids['E_full_single_reference']=add('E_full_single_reference',seed,['proxy'],prompt)
                hole_prompt=next(j['prompt'] for j in gate['jobs'] if j['method']=='G_removal_crop')
                target_prompt=next(j['prompt'] for j in gate['jobs'] if j['method']=='J_target_crop_locked')
                # Linked layers share a cut plan, seed, source and target; their
                # synthesis is separate, so final physical conservation is not assumed.
                ids['hole_source']=add('hole_source',seed,['hole_source_crop'],hole_prompt.replace('has ALREADY been removed','must be removed').replace('shown in the image','at the nearest front corner'),True,'hole_crop_edit')
                ids['hole_free']=add('hole_free',seed,['hole_crop'],hole_prompt)
                ids['hole_locked']=add('hole_locked',seed,['hole_crop'],hole_prompt,True,'hole_crop_edit')
                ids['target_locked']=add('target_locked',seed,['target_crop'],target_prompt,True,'target_crop_edit')
            for m in methods:
                deps=([ids[m]] if m in ids else
                    [ids[{'L0_no_cut_rgb':'hole_source','L1_free_hole':'hole_free','L2_context_locked':'hole_locked'}[m]],ids['target_locked']]
                    if g['status']=='geometry_ready' else [])
                cells.append({'id':cid+'__'+m+'__'+str(seed),'case_id':cid,'method':m,'seed':seed,
                    'cluster_id':case['cluster_id'],'stratum':case['stratum'],'dependencies':deps,
                    'status':'scheduled' if deps else 'preprocessing_failed','geometry_status':g['status'],
                    'composited':m.startswith('L'),'prepared':str(d)})
    random.Random(292609).shuffle(jobs);config['jobs']=jobs
    config['composition']={'order':['target','hole'],'target_mask':'target_composite_mask','hole_mask':'hole_edit',
        'feather_pixels':5,'outside_original':True,'target_margin_expansion_pixels':14,'hole_exclusion_margin_pixels':4,
        'raw_and_composited_outputs_retained':True}
    (formal/'frozen.json').write_text(json.dumps(config,indent=2)+'\n')
    (formal/'expected_cells.json').write_text(json.dumps(cells,indent=2)+'\n')
    decision={'created_unix':time.time(),'development_sources':['7489.jpg'],'new_generated_validation_outputs_seen':0,
        'choice':'Fine tofu and pale-floor wording from development gate 2; compare free hole and frozen context. Use same fixed target synthesis across three layer methods.',
        'limits':['Development photo alone cannot establish efficacy.','Four source images and masks informed geometric framing fixes before generation; this is prospective generation validation, not untouched end-to-end held-out evaluation.',
                  'L0 removes only cut-RGB guidance; masks and target still use geometry.','Layer methods use two model calls per scored endpoint; target calls shared across L0/L1/L2.',
                  'Qwen only in this repair; no claim of a repaired VACE backend.'],
        'held_sources':[x['case_id'] for x in sources['cases']],
        'source_annotation_manifest_sha256':sha(ROOT/'inputs/manifest.json'),
        'geometry_manifest_sha256':sha(ROOT/'geometry_v4/manifest.json'),
        'configuration_sha256':sha(formal/'frozen.json'),'expected_cells_sha256':sha(formal/'expected_cells.json'),
        'code_sha256':{n:sha(ROOT/n) for n in ['run_coupled_bite_qwen.py','freeze_coupled_bite_validation.py','collect_coupled_bite_validation.py','coupled_bite_geometry_helper_v4.py','build_coupled_bite_geometry_v4.py']},
        'evaluation':'All six original criteria, PFU ratings, uncertain is not pass; failures kept in denominator. Assistant unblinded diagnostics, no independent human ratings.',
        'replay':'First source / L1 / seed 41: rerun both hole_free and target_locked in a fresh process, then recompute exact layer output.'}
    (formal/'FROZEN.json').write_text(json.dumps(decision,indent=2)+'\n')
    print(json.dumps({'intended_endpoints':len(cells),'raw_calls':len(jobs),'preprocessing_failures':sum(x['status']=='preprocessing_failed' for x in cells),'configuration_sha256':decision['configuration_sha256']}))


if __name__=='__main__':main()
