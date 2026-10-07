"""Freeze the real-photo endpoint matrix before generating any formal output."""
import hashlib
import json
from pathlib import Path
import random
import time


ROOT = Path('/host/space0/guo-z/tf-ufi/first_bite_complete_20260929')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False) + '\n')


def main():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
    assert not (ROOT/'formal/FROZEN.json').exists(), 'Refuse to overwrite frozen experiment'
    inputs = json.loads((ROOT/'inputs/manifest.json').read_text())
    geometry = json.loads((ROOT/'geometry/manifest.json').read_text())
    cases = {x['case_id']:x for x in geometry['cases']}
    decision=json.loads((ROOT/'pilot/development_decision.json').read_text())
    q = json.loads((ROOT/decision['qwen_config']).read_text())
    v = json.loads((ROOT/decision['vace_config']).read_text())
    cache=json.loads((ROOT/'model_cache_receipt.json').read_text())
    assert cache['status']=='complete'
    q['backend']['model_root']=cache['qwen_root']
    v['runtime']['model_root']=cache['vace_root']
    qp = next(j['projection'] for j in q['jobs'] if j.get('projection'))
    vt = next(j['ttm'] for j in v['jobs'] if j.get('ttm'))
    q['jobs']=[];v['jobs']=[]
    q.update(stage='formal_frozen',not_formal=False);v.update(stage='formal_frozen',not_formal=False)
    q.pop('reason', None); v.pop('selection_rule', None)
    for name,c in [('qwen',q),('vace',v)]:
        p=ROOT/'formal'/('claims_'+name);p.mkdir(exist_ok=False)
        c['shared_claim_root']=str(p)
    expected=[];source_audit=[]
    for info in inputs['cases']:
        cid=info['case_id'];g=cases[cid]
        original=ROOT/'inputs'/cid/'source.png'
        if g['status']=='geometry_ready':
            files={Path(k).stem:x for k,x in g['files'].items() if Path(k).suffix=='.png'}
            assert np.array_equal(np.asarray(Image.open(original).convert('RGB')),np.asarray(Image.open(files['source']['path']).convert('RGB')))
        else:
            folder=ROOT/'formal'/('fallback_'+cid);folder.mkdir()
            im=Image.open(original).convert('RGB');mask=Image.open(ROOT/'inputs'/cid/'food_mask.png').convert('L')
            # Source-only conservative edit region; no fabricated 3-D pose for a failed reconstruction.
            ImageDraw.Draw(mask).rectangle((0,0,240,240),fill=255)
            mask=mask.filter(ImageFilter.MaxFilter(25));mask.save(folder/'edit_mask.png')
            a=np.array(im);a[np.array(mask)>0]=127;Image.fromarray(a).save(folder/'masked_control.png')
            files={k:{'path':str(p),'sha256':sha(p)} for k,p in [('source',original),('edit_mask',folder/'edit_mask.png'),('masked_control',folder/'masked_control.png')]}
        source_audit.append({'case_id':cid,'source_file':files['source'],'geometry_status':g['status'],'stratum':info['stratum'],'cluster_id':info['cluster_id']})
        action=(f"Edit this exact food photograph: {info['description']} Exactly one small first bite of the same tofu is lifted on the four tines of a single stainless steel eating fork, above and to the upper left of the remaining tofu. There must be clear air below the lifted bite and the plate. The fork handle extends out of the left image edge. A matching small fresh notch is visible in the front corner of the remaining tofu where that bite was removed. Preserve the original camera, framing, scale, plate, garnish, lighting and background. The lifted bite is a small fraction of the main block. Natural moist tofu cut texture and reflective steel. No hand, arm, person, face or mouth. One realistic food photograph, no text, no collage.")
        refine=(f"Refine IMAGE 1 into a realistic photograph of this scene: {info['description']} IMAGE 1 already contains the required final arrangement: exactly one small tofu bite supported by a fork above and to the upper left of the remaining tofu, with a matching fresh notch at the source corner. Keep all object positions, sizes, camera viewpoint and composition as IMAGE 1. Preserve the lifted bite, source notch and fork support. Replace artificial-looking cut faces and the gray fork with moist natural white tofu texture, slightly irregular soft cut edges, reflective stainless steel and consistent subtle shadows. IMAGE 2 is the original scene and is only a reference for food appearance, plate, garnish and lighting; do not restore its uncut state. The four fork tines support the bite from below with clear air below it. Preserve the background. No hand, arm, person, face or mouth. Output one natural photograph, no labels, no diagram, no collage.")
        vp=action+' The camera and scene are completely stationary.'
        for seed in [41,163,907]:
            for method in ['A_direct','B_planar','C_rgb3d','D_staged3d']:
                for name,c in [('qwen',q),('vace',v)]:
                    jid=f'{cid}__{method}__{seed}'
                    cell={'backend':name,'id':jid,'case_id':cid,'method':method,'seed':seed,'stratum':info['stratum'],'cluster_id':info['cluster_id']}
                    if g['status']!='geometry_ready' and method!='A_direct':
                        cell.update(status='preprocessing_failed',reason=g.get('error','Geometry fitting failed'))
                        expected.append(cell);continue
                    cell['status']='scheduled';expected.append(cell)
                    control={'A_direct':'masked_control','B_planar':'planar_control','C_rgb3d':'rgb_control','D_staged3d':'rgb_control'}[method]
                    keys={'source','edit_mask',control}
                    if method=='D_staged3d': keys.update(['rigid_mask','contact_mask','material_mask','hole_mask','rgb_control'])
                    j={'id':jid,'case_id':cid,'method':method,'seed':seed,'files':{k:files[k] for k in sorted(keys)}}
                    if name=='qwen':
                        if method=='A_direct':j.update(prompt=action,control=None,projection=None)
                        else:j.update(prompt=refine,control=control,input_order=[control,'source'],projection=qp if method=='D_staged3d' else None)
                    else:j.update(prompt=vp,control=control,ttm=vt if method=='D_staged3d' else None)
                    c['jobs'].append(j)
    for name,c in [('qwen',q),('vace',v)]:
        random.Random(2026092901).shuffle(c['jobs'])
        write(ROOT/'formal'/f'{name}_frozen.json',c)
    write(ROOT/'formal/expected_cells.json',expected)
    write(ROOT/'formal/source_audit.json',source_audit)
    protocol={
        'frozen_unix':time.time(),'status':'FROZEN_BEFORE_FORMAL_GENERATION',
        'scope':'Feasibility on 8 curated real single-block tofu photographs; 5 main and 3 challenge cases, 7 conservative clusters. No paired first-bite photographs or metric 3-D truth.',
        'hypotheses':{'H1':'C: explicit 3-D food cut/lift/support plus appearance refinement improves complete first-bite success over A direct editing and B planar food rendering.',
                      'H2':'D: adding staged latent constraints to C improves complete success; the development failure mode is overconstrained synthetic-looking material, so harm is a plausible outcome.'},
        'matrix':{'backends':['Qwen-Image-Edit-2511','Wan2.2-VACE-Fun-A14B'],'cases':8,'methods':4,'seeds':[41,163,907],'intended_cells':192,'scheduled_calls':len(q['jobs'])+len(v['jobs']),'preprocessing_failures':sum(x['status']=='preprocessing_failed' for x in expected)},
        'methods':{'A_direct':'Original source and action prompt; VACE additionally gets the same editable spatial region with gray reactive pixels.',
                   'B_planar':'2-D source patch warp and hole inpainting using the shared projected source/target layout and fork. Ablates 3-D FOOD rendering; not a geometry-free planner.',
                   'C_rgb3d':'3-D cut/lift/support RGB proxy then learned appearance refinement; no latent projection.',
                   'D_staged3d':'Same C inputs and prompt plus staged latent constraints. Qwen uses this experiment\'s coarse material/full rigid projection; VACE uses the existing GeoEdit TTM implementation.'},
        'primary_diagnostic_endpoint':{'reviewer':'Assistant visual diagnostic review, not independent human evaluation',
            'dimensions':['lift','utensil_support','matching_source_notch','scene_identity','no_person','photographic_realism'],
            'values':['pass','fail','uncertain'],'complete_success':'All six dimensions pass; uncertain is not a pass. No output also counts as failure.',
            'no_truth_claim':'Image evidence alone cannot prove real 3-D support, conservation or actual removal from the same physical object.'},
        'automatic_metrics':'Outside-edit-mask source MAE and PSNR, advisory independent text segmentation with unedited-source negative controls. None is called physical accuracy or realism.',
        'aggregation':'All 192 intended cells retained. Average seeds within each photograph; summarize paired photo and conservative cluster effects descriptively. Seeds and video frames are not independent source samples. No confirmatory p-value claim.',
        'selection':'All formal raw outputs retained; VACE single frame 0, Qwen raw PNG. No best-seed/frame selection. Development uses only old 7489 and is reported separately.',
        'development_decision_sha256':sha(ROOT/'pilot/development_decision.json'),
        'local_model_cache_receipt_sha256':sha(ROOT/'model_cache_receipt.json'),
        'reproducibility_check':'After the main matrix, rerun test_01_7445 / C_rgb3d / seed 41 once per backend in a fresh process. Retain both and compare raw hashes/pixel error; replay is excluded from the 192-cell endpoint.',
        'source_selection_manifest_sha256':sha(ROOT/'inputs/manifest.json'),
        'geometry_manifest_sha256':sha(ROOT/'geometry/manifest.json'),
        'configuration_sha256':{n:sha(ROOT/'formal'/f'{n}_frozen.json') for n in ['qwen','vace']},
        'code_sha256':{n:sha(ROOT/n) for n in ['run_complete_bite_qwen.py','run_complete_bite_vace.py','freeze_complete_bite_experiment.py','build_complete_bite_geometry.py']},
        'limitations':['Curated small sample, narrow block-tofu prior, manual approximate source annotations.','Pretrained training-set overlap is unknown.','Missing or weak hidden geometry is not recovered ground truth.','Different native model interfaces: compare ablations within each backend; cross-backend rates are descriptive.','No automatic retries or post-hoc per-photo tuning. Technical reruns require explicit separate provenance and remain visible.','One depth-control image contains invalid depth normalization; depth controls are excluded from the formal matrix.']}
    write(ROOT/'formal/FROZEN.json',protocol)
    print(json.dumps(protocol['matrix']))


if __name__=='__main__':main()
