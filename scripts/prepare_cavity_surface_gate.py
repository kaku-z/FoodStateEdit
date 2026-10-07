"""Matched-noise cavity-only appearance experiment, eight development photos."""
import copy, json, hashlib, time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def entry(p):return {'path':str(p),'sha256':sha(p)}
def main():
    gate=ROOT/'gate_v33';gate.mkdir(exist_ok=False)
    cfg=json.loads((ROOT/'gate_v30/config.json').read_text());cfg['jobs']=[]
    cfg['stage']='development_cavity_only_matched_projection_ablation'
    cases=json.loads((ROOT/'inputs/manifest.json').read_text())['cases']
    for c in cases:
        cid=c['case_id'];d=gate/cid;d.mkdir()
        field=ROOT/'gate_v32/cavity_geometry_projection_v1'/(cid+'_field')
        cavity=np.asarray(Image.open(field/'cavity_mask.png'))>0
        geometry=ROOT/'geometry_spoon_box_cap_v1'/cid
        edit=np.asarray(Image.open(geometry/'edit_mask.png'))>0
        mask=binary_dilation(cavity,iterations=2)&edit
        yy,xx=np.where(mask);l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized']
        size=192;assert np.ptp(xx)<size-4 and np.ptp(yy)<size-4
        x0=int(np.clip(round((xx.min()+xx.max())/2-size/2),l,l+w-size))
        y0=int(np.clip(round((yy.min()+yy.max())/2-size/2),t,t+h-size));box=[x0,y0,x0+size,y0+size]
        assert mask[y0:y0+size,x0:x0+size].sum()==mask.sum()
        parent=ROOT/'gate_v32/cavity_geometry_projection_v1'/(cid+'__box_cap_food_intrinsic__41')/'projected.png'
        Image.open(parent).convert('RGB').crop(box).resize((640,640),Image.Resampling.LANCZOS).save(d/'layout.png')
        Image.fromarray(np.uint8(mask)*255).save(d/'full_mask.png')
        Image.fromarray(np.uint8(mask)*255).crop(box).resize((640,640),Image.Resampling.NEAREST).save(d/'edit_mask.png')
        Image.fromarray(np.uint8(cavity)*255).crop(box).resize((640,640),Image.Resampling.NEAREST).save(d/'structure_weight.png')
        transform={'box':box,'source_only_crop_selection':True,'full_composite_parent':str(parent),'parent_sha256':sha(parent),'mask_sha256':sha(d/'full_mask.png'),'parent_seed':41,'scope':'Only the shared cavity neighborhood is editable; already lifted food and utensil stay in the parent.'}
        (d/'transform.json').write_text(json.dumps(transform,indent=2))
        material=ROOT/'gate_v23'/cid/'material_reference.png'
        for anchored in [False,True]:
            method='cavity_low_frequency_projected' if anchored else 'cavity_unanchored'
            j={'id':cid+'__'+method+'__41','case_id':cid,'method':method,'seed':41,
               'start_raw_sigma':.8,'lock_context':True,'input_order':['layout','reference'],
               'transform':str(d/'transform.json'),
               'files':{'layout':entry(d/'layout.png'),'reference':entry(material),'edit_mask':entry(d/'edit_mask.png')},
               'prompt':'Image 1 is a close photograph of the original tofu block AFTER a first small mouthful has been removed from its front corner. The existing recessed cut-out is an empty concave opening in the remaining solid tofu: its back walls and gently scooped floor lie INSIDE the block. Image 2 shows the actual plain tofu material from this meal. Refine only the exposed fresh inner cut surfaces into dense moist cold silken soybean gel with subtle translucency, delicate irregular cut marks, soft natural light and continuous photographic texture. Preserve the exact empty cavity outline and depth arrangement, the remaining tofu, original sauce and garnish, plate and background. The missing mouthful is already elsewhere outside this crop. Add no piece inside this cavity, tray, cup, bowl, ledge, sauce, topping, leaf, colored spot, person or hand.'}
            if anchored:
                j['files']['structure_weight']=entry(d/'structure_weight.png')
                j['structure']={'begin_taper_fraction':.25,'end_fraction':.75,'tail_weight':.35,'lowpass':True,'kernel':3,'scope':'Low-frequency VAE initialization projection within the declared cavity; not a depth, shape or realism guarantee.'}
            cfg['jobs'].append(j)
    cfg['frozen_raw_cells']=16;cfg['selection_rule']='All eight development cases, paired matched-noise seed41 arms. Every raw output is retained; no per-case best selection.'
    cfg['inference']['negative_prompt']='additional food piece in the cavity, protruding ledge, filled missing corner, metal cavity, cup, tray, bowl, added sauce, colored spot, added garnish, hand, person, wax, plastic, CGI, drawing'
    cfg['limitations']='Monocular geometry and hidden material remain priors. The ambient initialization is approximate; no paired photos or blind human evaluation.'
    cfg['frozen_unix']=time.time();(gate/'config.json').write_text(json.dumps(cfg,indent=2));print('PREPARED',len(cfg['jobs']),flush=True)
if __name__=='__main__':main()
