"""Freeze shallow spoon-scoop geometry and appearance-reference ablation."""
import copy,hashlib,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def entry(p):return dict(path=str(p),sha256=sha(p))

def main():
    geometry=ROOT/'geometry_spoon_v3'
    report=json.loads((geometry/'manifest.json').read_text())
    assert report['status']=='complete' and len(report['cases'])==8
    assert all(c['status']=='geometry_ready' for c in report['cases'])
    g=ROOT/'gate_v15';g.mkdir(exist_ok=False)
    (g/'executed_prepare_script.py').write_bytes(Path(__file__).read_bytes())
    previous=json.loads((ROOT/'gate_v12/config.json').read_text())
    prototypes={j['case_id']:j for j in previous['jobs'] if j['seed']==41}
    jobs=[]
    for c in report['cases']:
        cid=c['case_id'];folder=geometry/cid
        source=np.asarray(Image.open(folder/'rgb_control.png').convert('RGB'))
        Image.fromarray(cv2.Canny(source,80,160)).save(folder/'canny_control.png')
        original=prototypes[cid]['prompt']
        description=original.split('One small irregular rounded wedge')[0]
        prompt=description+'One compact rounded bite-sized scoop of smooth moist silken tofu rests inside and slightly above the bowl of a single polished stainless steel eating spoon entering from the left edge. The spoon is a normal shallow eating spoon, with a smooth reflective metal rim visible around the food. The spoon and its solid food portion are suspended above the plate in the upper left of the photograph, separated from the remaining tofu by a clear air gap. A small open rounded scoop is missing from the nearest front corner of the remaining tofu. The top boundary ends at the open scooped notch, and a gently curved fresh tofu surface continues down to the bottom of the small recess; there is no bridge over the opening. The lifted portion is the same piece removed from that corner, with original sauce on its upper surface when present. Fine dense moist tofu texture, gently imperfect fresh edges, natural original lighting and perspective, realistic stainless steel reflections and contact with the spoon bowl. Preserve the original plate, toppings and tabletop. Only food and tableware are visible; the spoon handle continues through the left edge, with no person or hand in view.'
        for method,seed in [('spoon_canny',41),('spoon_canny',163),('spoon_canny',907),('spoon_material_reference',41)]:
            j={'id':f'{cid}__{method}__{seed}','case_id':cid,'method':method,'seed':seed,
               'prompt':prompt,'control_scale':.7,'hint_scope':'target','reference_keys':[],
               'files':{'source':entry(folder/'source.png'),'edit_mask':entry(folder/'edit_mask.png'),
                        'control':entry(folder/'canny_control.png')}}
            if method=='spoon_material_reference':
                j['reference_keys']=['reference_material'];j['files']['reference_material']=entry(ROOT/'gate_v13_run03'/cid/'material_reference.png')
                j['prompt']='The reference image provides only the original tofu color and material, not the desired shape. '+j['prompt']
            jobs.append(j)
    cfg={'stage':'development_shared_scoop_and_supported_spoon','not_formal':True,'jobs':jobs,
         'inference':previous['inference'],'data_status':'All eight photos are development; no unseen claim',
         'geometry_scope':'Closed shared Boolean ellipsoid with shallow conforming spoon, static discretized friction prior .25, no measured dynamics'}
    (g/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (g/'freeze.json').write_text(json.dumps({'config_sha256':sha(g/'config.json'),
         'geometry_manifest_sha256':sha(geometry/'manifest.json'),'prepare_script_sha256':sha(Path(__file__)),
         'expected_raw':len(jobs),'seeds':[41,163,907],'all_outputs_retained':True},indent=2))
    print('FROZEN',len(jobs),flush=True)

if __name__=='__main__':main()
