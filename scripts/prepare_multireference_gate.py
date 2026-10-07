"""Freeze source-only appearance references and layout/reference ablations."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def entry(p):return dict(path=str(p),sha256=sha(p))

def main():
    g=ROOT/'gate_v13_run03';g.mkdir(exist_ok=False)
    (g/'executed_prepare_script.py').write_bytes(Path(__file__).read_bytes())
    cases=json.loads((ROOT/'inputs/manifest.json').read_text())['cases']
    previous=json.loads((ROOT/'gate_v12/config.json').read_text())
    prototypes={j['case_id']:j for j in previous['jobs'] if j['seed']==41}
    jobs=[]
    for c in cases:
        cid=c['case_id'];d=g/cid;d.mkdir()
        geometry=ROOT/'geometry_adaptive_v1'/cid
        source=np.asarray(Image.open(geometry/'source.png').convert('RGB'))
        food=np.asarray(Image.open(geometry/'food_mask.png'))>0
        anchor=np.array(c['anchors'][1],float)
        ax,ay=np.round(anchor).astype(int)
        seed=np.median(source[max(0,ay-2):ay+3,max(0,ax-2):ax+3].reshape(-1,3),axis=0)
        chroma=source.astype(float)/np.maximum(source.sum(2)[...,None].astype(float),1)
        seed_chroma=seed/seed.sum()
        color_distance=np.linalg.norm(chroma-seed_chroma,axis=2)
        bare=food&(color_distance<.085)&(source.mean(2)>max(70,.72*seed.mean()))
        distance=distance_transform_edt(bare)
        yy,xx=np.indices(bare.shape)
        for edge in [48,40,32,24,16,8]:
            candidates=distance>edge*.72
            if candidates.any():break
        assert candidates.any(),cid
        score=(xx-anchor[0])**2+(yy-anchor[1])**2
        score[~candidates]=np.inf
        y,x=np.unravel_index(np.argmin(score),score.shape)
        half=edge//2;box=(int(x-half),int(y-half),int(x+half),int(y+half))
        assert np.all(bare[y-half:y+half,x-half:x+half])
        patch=Image.fromarray(source).crop(box).resize((256,256),Image.Resampling.LANCZOS)
        patch.save(d/'material_reference.png')
        left,top=c['preprocessing']['pad_left_top'];width,height=c['preprocessing']['resized']
        for name,filename in [('layout','rgb_control.png'),('source','source.png')]:
            Image.open(geometry/filename).convert('RGB').crop((left,top,left+width,top+height)).save(d/(name+'_reference.png'))
        (d/'reference_selection.json').write_text(json.dumps({'source_only':True,'material_box_canvas':box,
            'material_box_edge':edge,'material_reference_low_observability':edge<16,'side_anchor_canvas':anchor.tolist(),
            'anchor_median_rgb':seed.tolist(),'bare_criterion':'Inside source food mask, normalized RGB distance below .085 from source-only side-anchor median, mean above max(70,.72*anchor mean); distance transform covers square',
            'not_measured_albedo':True,'scope':'Photographic texture includes original illumination; no paired first-bite truth'},indent=2))
        variants={'layout_only':['reference_layout'],'layout_source':['reference_layout','reference_source'],
                  'layout_material':['reference_layout','reference_material'],'material_only':['reference_material']}
        for method,keys in variants.items():
            j=copy.deepcopy(prototypes[cid]);j.update(id=f'{cid}__{method}__41',method=method,reference_keys=keys,hint_scope='target')
            for key in keys:j['files'][key]=entry(d/(key.replace('reference_','')+'_reference.png'))
            if method=='material_only':
                prefix='The reference is only a close-up of the original tofu material. It does not specify an object shape. '
            elif method=='layout_only':
                prefix='The reference is a rough geometry layout. Convert it to a natural food photograph while preserving the exact missing portion, lifted bite, fork and spatial arrangement. '
            else:
                prefix='Image 1 is the required rough geometry layout. Image 2 provides only the original food appearance. Convert image 1 to a natural food photograph: preserve its missing portion, lifted bite, fork and spatial arrangement, and use the colors and material of image 2. '
            j['prompt']=prefix+j['prompt'];jobs.append(j)
    cfg={'stage':'development_layout_and_source_appearance_references','not_formal':True,
         'inference':previous['inference'],'jobs':jobs,'data_status':'All eight sources are development; no unseen claims',
         'scope':'Architecture extension using pretrained image-reference prefix and target-only ControlNet hints; not trained for the combined interface'}
    (g/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (g/'freeze.json').write_text(json.dumps({'config_sha256':sha(g/'config.json'),
         'prepare_script_sha256':sha(Path(__file__)),'runner_sha256':sha(ROOT/'run_multireference_geometry_qwen21.py'),
         'expected_raw':len(jobs),'all_outputs_retained':True,'seeds':[41]},indent=2))
    print('FROZEN',len(jobs),flush=True)

if __name__=='__main__':main()
