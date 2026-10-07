"""Fixed development comparison: no geometry signal, depth, and Canny."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def file(p):return {'path':str(p),'sha256':sha(p)}
def main():
    sources=json.loads((ROOT/'inputs/manifest.json').read_text());depth=json.loads((ROOT/'fracture_controls_v1/manifest.json').read_text());assert depth['status']=='complete'
    out=ROOT/'gate_v4';out.mkdir(exist_ok=False)
    cfg={'stage':'development_shared_irregular_cut','not_formal':True,'inference':{'width':1024,'height':768,'steps':40,'true_cfg_scale':1.,'use_kv_cache':False},'jobs':[]}
    cases=[next(c for c in sources['cases'] if c['case_id']==cid) for cid in ['new_03_7443','new_01_7442','new_02_7496','new_04_7459']]
    for case in cases:
        cid=case['case_id'];d=ROOT/'fracture_controls_v1'/cid
        prompt=(f"A natural, realistic food photograph. {case['description']} "
            "One small irregular rounded wedge of soft white tofu rests securely on the four tines of a single polished stainless steel eating fork entering from the left. "
            "The fork and its bite are suspended in the upper left of the photograph, clearly separated from the remaining tofu and plate by an air gap. "
            "A small shallow bite-sized notch is missing from the nearest front corner of the remaining tofu: the top surface ends at the open notch, "
            "and an open curved freshly torn inner surface leads down to pale tofu, with a delicate shadow inside the shallow recess. The recess is open from the top edge, never a closed tunnel or punched hole in the side. "
            "The lifted wedge is the same portion removed from the missing corner, slightly rotated to show its curved fresh torn surface. Some sauce from the original top is retained on its upper surface when present. Smooth moist silken tofu, fine grain, softly irregular cut edges, "
            "realistic metal reflections and contact beneath the bite. The original plate, toppings, lighting, perspective and tabletop remain the same. "
            "Only the existing food and tableware are visible; the fork handle continues toward the edge of the photograph, with no person or hand in view.")
        for method in ['depth','canny','grayscale','none']:
            files={'source':file(ROOT/'inputs'/cid/'source.png'),'edit_mask':file(d/'edit_mask.png')}
            if method!='none':files['control']=file(d/(method+'.png'))
            cfg['jobs'].append({'id':f'{cid}__{method}__41','case_id':cid,'method':method,'seed':41,'control_scale':0.7,'prompt':prompt,'files':files})
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    # Smoke is one predeclared cell; remaining cells have fixed settings already.
    first=dict(cfg,jobs=cfg['jobs'][:1]);rest=dict(cfg,jobs=cfg['jobs'][1:])
    (out/'smoke_config.json').write_text(json.dumps(first,indent=2)+'\n');(out/'remaining_config.json').write_text(json.dumps(rest,indent=2)+'\n')
    (out/'freeze.json').write_text(json.dumps({'created_unix':time.time(),'config_sha256':sha(out/'config.json'),
        'expected_raw_outputs':16,'scope':'Previously seen development sources only. No formal success rate or novelty claim.',
        'backend':'Qwen-Image-2.1 and Alibaba-PAI Qwen-Image-2.1-Fun-Controlnet-Union; both pretrained, not our model.',
        'hypothesis':'A shared irregular cut and rigid rotation reduce cuboid appearance while preserving a matching source notch; compare pretrained depth/edge/grayscale branches.',
        'control_ablation':'none retains the same source and inpainting mask with zero control-image channels; depth/canny change only the structural input.',
        'effective_cache':'Control pipeline disables prefix cache when control_context exists; explicitly false here.',
        'evaluation':'All cells retained; raw and composited evaluated separately. Assistant unblinded diagnostics, not human ratings.',
        'code_sha256':{n:sha(ROOT/n) for n in ['run_depth_bite_qwen21_v2.py','prepare_fracture_controls.py','freeze_fracture_gate.py','fracture_bite_geometry.py']}},indent=2)+'\n')
    print({'jobs':len(cfg['jobs']),'first':cfg['jobs'][0]['id']})

if __name__=='__main__':main()
