"""Freeze same-image repair experiment. Reads only v2 source assets."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from foodstateedit.observed_edit.constrained import build_layers


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--v2-bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    old=json.loads((args.v2_bundle/'config.json').read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    cases=[]
    for case in old['cases']:
        for f in case['files'].values():
            assert sha(args.v2_bundle/f['path'])==f['sha256']
        source=np.array(Image.open(args.v2_bundle/case['files']['source.png']['path']).convert('RGB'))
        state=np.load(args.v2_bundle/case['files']['state.npz']['path'],allow_pickle=False)
        # Source-only rectangles protect the other food and its immediate edge.
        rect={'pizza':(400,214,110,102),'corn':(155,225,185,182)}[case['case_id']]
        x,y,w,h=rect
        protected=np.zeros(source.shape[:2],bool);protected[y:y+h,x:x+w]=True
        layers=build_layers(source,state,case['handle_xy'],protected)
        folder=args.output/'inputs'/case['case_id'];folder.mkdir(parents=True)
        for name in ['background','reference','base']:
            Image.fromarray(layers[name]).save(folder/(name+'.png'))
        Image.fromarray(source).save(folder/'source.png')
        Image.fromarray(layers['free_mask'].astype(np.uint8)*255).save(folder/'free_mask.png')
        np.savez_compressed(folder/'layers.npz',free_mask=layers['free_mask'],hole=layers['hole'],
                            target_mask=layers['target_mask'],source_mask=state['source_mask'],protected_food=protected)
        (folder/'preparation.json').write_text(json.dumps(layers['metadata'],indent=2))
        prompt=('Make this rough composite a realistic photograph. The food has ALREADY been moved to its final '
                'location, and the old location is already empty. Keep every food piece exactly where it is, '
                'with the same orientation, silhouette, toppings, texture, size and color. Replace ONLY the flat gray '
                'spoon underneath the moved food with one realistic stainless-steel serving spoon, keeping its bowl '
                'and handle in exactly the indicated location and direction. Repair the plain filled patch at the '
                'old food location into smooth empty plate matching its surroundings; put no food or object there. '
                'Add subtle contact shadow immediately '
                'under the spoon and food. Preserve the plate, background, other food and lighting. '
                'Do not move or rotate any food. Do not add, copy or duplicate food. No hands, text or colored outlines.')
        cases.append({k:case[k] for k in ['case_id','dataset_index','target_xy','handle_xy','split']} |
                     {'prompt':prompt,'protected_source_rect':list(rect),'files':{f.name:{'path':f.relative_to(args.output).as_posix(),'sha256':sha(f)}
                                              for f in folder.iterdir() if f.is_file()}})
    cfg={k:old[k] for k in ['backend','resource_gate','claim_limit']}
    cfg.update(experiment_id='observed_edit_v3_repair_20260928',cases=cases,seeds=[281],
               conditions=['scaffold_only','projected'],expected_generation_count=4,
               inference={'num_inference_steps':40,'true_cfg_scale':4.,'guidance_scale':None,
                          'negative_prompt':' ','width':1184,'height':896},
               target_images_in_generation_bundle=False,
               source_bundle_sha256=sha(args.v2_bundle/'config.json'),
               compositor='fixed-observed-background-and-food; tool and removal domains; Dirichlet Poisson boundary',
               projection='flow next-sigma reference outside tool/removal domains at every denoising step')
    cfg['expected_pipeline_sha256']=sha(Path(__file__).resolve().parents[1]/'outputs/observed_edit_v2_20260928_runtime/pipeline_qwenimage_edit_plus_installed.py')
    (args.output/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    print(json.dumps({'config_sha256':sha(args.output/'config.json'),'cases':len(cases)}))


if __name__=='__main__':main()
