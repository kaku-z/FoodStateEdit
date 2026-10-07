"""Source-geometry-selected crops, independently controlled recess and payload.

All crop boxes and masks are fixed before outputs. Full composites are separate
from crop generations. Existing fork-handle pixels come from the fixed baseline.
"""
import copy,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def entry(p):return dict(path=str(p),sha256=sha(p))

def main():
    g=ROOT/'gate_v14';g.mkdir(exist_ok=False)
    (g/'executed_prepare_script.py').write_bytes(Path(__file__).read_bytes())
    cases={c['case_id']:c for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases']}
    previous=json.loads((ROOT/'gate_v12/config.json').read_text())
    prototypes={j['case_id']:j for j in previous['jobs'] if j['seed']==41}
    ids=['new_02_7496','new_04_7459','prospective_02_7441','prospective_04_7498'];jobs=[]
    for cid in ids:
        c=cases[cid];geometry=ROOT/'geometry_adaptive_v1'/cid
        src=Image.open(geometry/'source.png').convert('RGB');control=Image.open(geometry/'canny_control.png').convert('RGB')
        left,top=c['preprocessing']['pad_left_top'];width,height=c['preprocessing']['resized']
        material=np.asarray(Image.open(geometry/'material_mask.png'))>0
        utensil=np.asarray(Image.open(geometry/'rigid_mask.png'))>0
        hole=np.asarray(Image.open(geometry/'hole_mask.png'))>0
        for component,center_mask,size,edit in [('payload',material,192,material|utensil),('recess',hole,224,binary_dilation(hole,iterations=8))]:
            d=g/cid/component;d.mkdir(parents=True)
            yy,xx=np.where(center_mask);cx=(xx.min()+xx.max())/2;cy=(yy.min()+yy.max())/2
            x0=int(np.clip(round(cx-size/2),left,left+width-size));y0=int(np.clip(round(cy-size/2),top,top+height-size))
            box=(x0,y0,x0+size,y0+size)
            full=np.zeros((480,640),bool);full[y0:y0+size,x0:x0+size]=edit[y0:y0+size,x0:x0+size]
            Image.fromarray(np.uint8(full)*255).save(d/'full_mask.png')
            transform={'source_only_crop':True,'component':component,'box':box,'input_resolution':[640,640],
                       'full_composite_baseline':str(next((ROOT/'gate_v12').glob(f'worker*/{cid}__adaptive_canny__41/composited.png'))),
                       'source_geometry_report_sha256':sha(geometry/'geometry_report.json')}
            (d/'transform.json').write_text(json.dumps(transform,indent=2))
            src.crop(box).resize((640,640),Image.Resampling.LANCZOS).save(d/'source.png')
            control.crop(box).resize((640,640),Image.Resampling.NEAREST).save(d/'control.png')
            Image.fromarray(np.uint8(full)*255).crop(box).resize((640,640),Image.Resampling.NEAREST).save(d/'edit_mask.png')
            if component=='payload':
                prompt='A real food photograph. One compact bite-sized wedge of smooth moist silken tofu is carried on the four tines of a polished stainless steel eating fork entering from the left edge. A solid soft tofu portion with a gently curved fresh scooped surface, fine dense texture and softly imperfect edges, with sauce on top only when present in the layout. The metal fork supports the underside of the food, with realistic reflected surroundings and a small contact shadow. The piece has the exact silhouette, position and orientation specified by the edge control. The existing photograph background remains. Only food and tableware, with no person or hand in view.'
            else:
                prompt='A real photograph of the existing smooth moist silken tofu after one small portion has been removed from its nearest front corner. The missing corner is an OPEN notch: it starts at the TOP boundary and remains open to the front and side. The top edge ends at the cut, without any food bridge across the opening. A softly irregular freshly scooped interior with the same color and fine dense texture as the existing tofu, realistic diffuse light and gentle cavity shading. This is a removed corner, never a tunnel, circular side hole, bread crust or machined cavity. Preserve the exact open cut geometry specified by the edge control, the remaining toppings, plate, perspective and original lighting.'
            for method in ['no_reference','material_reference']:
                j={'id':f'{cid}__{component}__{method}__41','case_id':cid,'method':f'{component}_{method}',
                   'component':component,'seed':41,'hint_scope':'target','control_scale':.7,'prompt':prompt,
                   'files':{'source':entry(d/'source.png'),'edit_mask':entry(d/'edit_mask.png'),'control':entry(d/'control.png')},
                   'reference_keys':[],'transform':str(d/'transform.json')}
                if method=='material_reference':
                    j['files']['reference_material']=entry(ROOT/'gate_v13_run03'/cid/'material_reference.png')
                    j['reference_keys']=['reference_material']
                    j['prompt']='The reference image specifies only the original food color and material, not an object shape. '+j['prompt']
                jobs.append(j)
    cfg={'stage':'development_component_geometry_control','not_formal':True,'jobs':jobs,
         'inference':dict(width=768,height=768,steps=40,true_cfg_scale=1.,use_kv_cache=False),
         'data_status':'All input sources are development; baseline composite is explicitly disclosed'}
    (g/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (g/'freeze.json').write_text(json.dumps({'config_sha256':sha(g/'config.json'),'script_sha256':sha(Path(__file__)),
              'expected_raw_crop_generations':len(jobs),'all_outputs_retained':True,'seeds':[41]},indent=2))
    print('FROZEN',len(jobs))

if __name__=='__main__':main()
