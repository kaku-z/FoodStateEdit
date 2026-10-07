"""Freeze development tests for source-removal/target-appearance separation."""
from pathlib import Path
import hashlib
import json
import shutil
import time
import zipfile
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'outputs/first_bite_complete_20260929/server_results'
OUT = ROOT/'outputs/first_bite_coupled_20260929'
REMOTE = '/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out = OUT/'gate_v1_inputs'
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), 'Refuse to overwrite previously prepared inputs'
    old = OLD/'pilot/dev_7489_controls'
    src = np.asarray(Image.open(old/'source.png').convert('RGB'))
    proxy = np.asarray(Image.open(old/'rgb_control.png').convert('RGB'))
    hole = np.asarray(Image.open(old/'hole_mask.png')) > 0
    target = (np.asarray(Image.open(old/'material_mask.png')) > 0) | (np.asarray(Image.open(old/'rigid_mask.png')) > 0)
    remove = src.copy(); remove[hole] = proxy[hole]
    lifted = src.copy(); lifted[target] = proxy[target]
    arrays = {'source':src, 'proxy':proxy, 'remove_full':remove, 'lift_full':lifted,
              'hole_edit':binary_dilation(hole, iterations=10), 'target_edit':binary_dilation(target, iterations=10)}
    transforms = {}
    for kind, image, mask in [('hole', remove, arrays['hole_edit']), ('target', lifted, arrays['target_edit'])]:
        yy, xx = np.where(mask)
        side = max(224, int(max(np.ptp(xx)+1, np.ptp(yy)+1)*1.8))
        side = min(448, 32*((side+31)//32))
        cx, cy = (xx.min()+xx.max())/2, (yy.min()+yy.max())/2
        x0 = int(np.clip(round(cx-side/2), 0, 640-side))
        y0 = int(np.clip(round(cy-side/2), 0, 480-side))
        box = (x0, y0, x0+side, y0+side)
        transforms[kind] = {'box_xyxy':box, 'network_size':[512,512], 'source_size':[640,480],
                            'rule':'Source-control mask bounding box, fixed 1.8x extent, minimum 224, maximum 448; no output-dependent crop.'}
        arrays[kind+'_crop'] = np.asarray(Image.fromarray(image).crop(box).resize((512,512),Image.Resampling.LANCZOS))
        arrays[kind+'_source_crop'] = np.asarray(Image.fromarray(src).crop(box).resize((512,512),Image.Resampling.LANCZOS))
        arrays[kind+'_crop_edit'] = np.asarray(Image.fromarray(np.uint8(mask)*255).crop(box).resize((512,512),Image.Resampling.NEAREST))
    files = {}
    for name, arr in arrays.items():
        p = out/(name+'.png')
        Image.fromarray(np.uint8(arr)*255 if arr.dtype == bool else arr).save(p)
        files[name] = {'path':REMOTE+'/gate_v1_inputs/'+p.name, 'sha256':sha(p)}
    (out/'transforms.json').write_text(json.dumps(transforms,indent=2)+'\n')
    full_prompt = ('Photorealistic food photograph of the exact scene in IMAGE 1. The tofu has ALREADY had a small cubical bite removed from its nearest front corner. '
        'Keep the concave recessed missing corner, with visible fresh porous white interior walls and a floor below the sauce-coated top. The missing corner must remain missing. '
        'The removed small tofu bite is already raised in the upper left on a stainless steel eating fork. Maintain its position, dimensions and contact with the fork. '
        'Refine only synthetic flat surfaces into natural moist tofu texture and reflective metal. Keep the plate, viewpoint, lighting and original food details. '
        'One remaining tofu block, one lifted bite, one fork. No hand, arm or person. No labels or collage.')
    hole_prompt = ('A realistic close-up food photograph. A small cubical piece has ALREADY been removed from the nearest front corner of this tofu. '
        'Retain the clearly concave three-dimensional notch shown in the image: the sauce-coated top is interrupted, the two fresh white porous inner walls recede into the block, '
        'and the bottom of the recess is below the top surface. The corner is missing, not covered by a white patch and not filled by another cube. '
        'Give the exposed interior natural moist tofu texture, slightly soft irregular cut edges and subtle cavity shadows. '
        'Preserve the size and placement of the remaining tofu, existing brown sauce, plate pattern, lighting, camera and crop. '
        'There is no loose food piece, utensil, hand or person. Output the same composition as one natural photograph.')
    target_prompt = ('Refine this exact photograph. Preserve the small tofu bite already suspended above the plate on the fork. '
        'Replace the flat gray utensil with one realistic polished stainless steel eating fork with four tines supporting the bite from underneath. '
        'The bite is moist white tofu with the existing sauce and delicate porous cut texture, softly irregular edges and a small contact shadow on the fork. '
        'Keep the exact positions, sizes, camera crop, background and remaining food. Do not add another food piece or utensil. No hand, arm, face or person.')
    jobs = []
    def job(name, order, prompt, lock=False, mask=None):
        selected = {key:files[key] for key in order}
        if mask:
            selected['edit_mask'] = files[mask]
        jobs.append({'id':'dev_7489__'+name+'__20260929', 'case_id':'dev_7489', 'method':name, 'seed':20260929,
            'input_order':order, 'files':selected, 'prompt':prompt, 'lock_context':lock})
    job('C_previous_interface', ['proxy','source'], full_prompt+' IMAGE 2 is appearance reference only; do not restore its intact corner.')
    job('E_joint_single_reference', ['proxy'], full_prompt)
    job('F_removal_full', ['remove_full'], hole_prompt)
    job('G_removal_crop', ['hole_crop'], hole_prompt)
    job('H_removal_crop_locked', ['hole_crop'], hole_prompt, True, 'hole_crop_edit')
    job('I_removal_source_crop_locked', ['hole_source_crop'], hole_prompt.replace('has ALREADY been removed','must be removed').replace('shown in the image','at the nearest front corner'), True, 'hole_crop_edit')
    job('J_target_crop_locked', ['target_crop'], target_prompt, True, 'target_crop_edit')
    job('K_target_crop', ['target_crop'], target_prompt)
    frozen = json.loads((OLD/'formal/qwen_frozen.json').read_text())
    config = {k:frozen[k] for k in ['backend','expected_pipeline_sha256','model_audit','inference']}
    config.update(stage='development_only', not_formal=True, jobs=jobs, created_unix=time.time(),
        source='Old 7489 development photo only; previous eight formal photos are now regression evidence, never relabeled unseen.',
        hypotheses=['Removing the intact second reference may reduce restoration of the missing corner.',
                    'Local crop increases notch resolution without increasing the bite in scene coordinates.',
                    'Outside-region context anchoring should stabilize placement while allowing free material synthesis inside.'],
        selection='All eight outputs retained; inspect development before freezing a new-source evaluation.',
        algorithm_output='Any paired local result compositing will be explicitly labeled layered output, with raw local outputs separately retained.')
    (OUT/'gate_v1.json').write_text(json.dumps(config,indent=2)+'\n')
    with zipfile.ZipFile(OUT/'gate_v1_inputs.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            z.write(p, 'gate_v1_inputs/'+p.name)
    print(json.dumps({'jobs':len(jobs),'config_sha256':sha(OUT/'gate_v1.json'),'transforms':transforms}))


if __name__ == '__main__':
    main()
