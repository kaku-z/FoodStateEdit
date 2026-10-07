"""Keep v3 model inputs identical; change only the inference constraint."""
import argparse,hashlib,json,shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def soft_core(mask):
    distance=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)
    return np.clip((distance-12.)/12.,0,1).astype(np.float32)


def main():
    p=argparse.ArgumentParser();p.add_argument('--v3-bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    config=json.loads((a.v3_bundle/'config.json').read_text());cases=[]
    a.output.mkdir(parents=True,exist_ok=False)
    for case in config['cases']:
        for f in case['files'].values():
            if sha(a.v3_bundle/f['path'])!=f['sha256']:raise ValueError('Changed v3 input')
        d=a.output/'inputs'/case['case_id'];d.mkdir(parents=True)
        old=a.v3_bundle/'inputs'/case['case_id']
        for name in ['source.png','reference.png']:
            shutil.copyfile(old/name,d/name)
        rgb=np.array(Image.open(old/'source.png').convert('RGB'))
        state=np.load(old/'layers.npz',allow_pickle=False)
        # Segment retained food from the same source-only rectangle. Its
        # surrounding plate/shadow is not treated as immutable foreground.
        labels=np.zeros(rgb.shape[:2],np.uint8)
        cv2.setRNGSeed(281)
        cv2.grabCut(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),labels,tuple(case['protected_source_rect']),
                    np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),8,cv2.GC_INIT_WITH_RECT)
        kept=((labels==cv2.GC_FGD)|(labels==cv2.GC_PR_FGD)) & ~state['source_mask']
        core=np.maximum(soft_core(state['target_mask']),soft_core(kept))
        if np.count_nonzero(core==1)<100:raise ValueError('Insufficient reliable food interior')
        if np.any(core[state['source_mask']]):raise ValueError('Removed food may not be anchored')
        np.savez_compressed(d/'layers.npz',core_weight=core,kept_food_mask=kept,
                            source_mask=state['source_mask'],target_mask=state['target_mask'])
        Image.fromarray(np.rint(255*core).astype(np.uint8)).save(d/'core_weight.png')
        Image.fromarray(kept.astype(np.uint8)*255).save(d/'kept_food_mask.png')
        record={k:v for k,v in case.items() if k!='files'}
        record['files']={f.name:{'path':f.relative_to(a.output).as_posix(),'sha256':sha(f)} for f in d.iterdir()}
        cases.append(record)
    config.update(experiment_id='observed_edit_v4_interior_20260928',cases=cases,
                  conditions=['interior','detail'],expected_generation_count=4,
                  source_bundle_sha256=sha(a.v3_bundle/'config.json'),
                  projection={'maximum':.8,'release_start':.75,'core_zero_within_pixels':12,
                              'core_full_after_pixels':24,'detail_kernel':5},
                  compositor='none: native generated image only, no background/food pixel restoration',
                  change_scope='Model inputs, prompt, seed and resolution are unchanged from v3. Only constraint mechanism changes.')
    (a.output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(json.dumps({'config_sha256':sha(a.output/'config.json'),'cases':len(cases)}))


if __name__=='__main__':main()
