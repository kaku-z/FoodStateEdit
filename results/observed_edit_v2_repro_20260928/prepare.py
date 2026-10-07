"""Freeze source-only development inputs; never open the paired target images."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import cv2
from PIL import Image
from foodstateedit.observed_edit.core import build_state


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    definitions = [(5, 'pizza', 'the single pizza portion', [230, 220], [45, 390]),
                   (7, 'corn', 'the single round corn slice', [440, 175], [594, 385])]
    cases = []
    for index, name, noun, target, handle in definitions:
        old = root / 'outputs/nutrition5k_real_removal_20260928_validated' / f'pair_{index:03d}'
        src = np.asarray(Image.open(old / 'source.png').convert('RGB'))
        mask = np.asarray(Image.open(old / 'source_mask.png')) > 0
        # Prior removal masks were dilated 3px for inpainting. Undo that margin
        # before transporting visible food pixels; the margin is not food.
        mask = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        state = build_state(src, mask, target, handle)
        folder = args.output / 'inputs' / name
        folder.mkdir(parents=True)
        Image.fromarray(src).save(folder / 'source.png')
        Image.fromarray(mask.astype(np.uint8)*255).save(folder / 'mask.png')
        Image.fromarray(state.control).save(folder / 'control.png')
        Image.fromarray(state.scaffold).save(folder / 'scaffold.png')
        Image.fromarray(state.editable.astype(np.uint8)*255).save(folder / 'editable.png')
        np.savez_compressed(folder / 'state.npz', source_ids=state.source_ids,
                            transported_rgb=state.transported_rgb, editable=state.editable,
                            payload_core=state.payload_core, source_mask=state.source_mask,
                            target_mask=state.target_mask)
        common = (f'Edit the first real photograph: lift {noun} from its original position onto exactly one '
                  f'realistic stainless-steel serving spoon, with the food centered at pixel ({target[0]}, {target[1]}) '
                  f'in the original 640 by 480 coordinate system and the handle pointing toward pixel ({handle[0]}, {handle[1]}). '
                  'Show the same intact food piece supported by the spoon and remove it completely from its old position, '
                  'revealing the underlying plate. Preserve all other food, the plate, camera and photographic lighting. '
                  'No hands, duplicate payload, extra utensil, outlines or labels. Do not change the identity or size of the food. ')
        direct = common + ('The second image is a control diagram: yellow outlines the source food; cyan marks its destination '
                           'and spoon handle direction. Use the first image for photographic content, never render the control lines.')
        grounded = common + ('The second image is a rough state preview: the actual source food pixels have already been translated '
                             'to their requested destination, the old footprint is roughly filled, and a flat gray spoon shape marks '
                             'support. Turn that preview into a real photograph. Retain the exact food placement and appearance; '
                             'repair the revealed plate and render a realistic spoon underneath, with natural contact shadows.')
        cases.append({'case_id': name, 'dataset_index': index, 'split': 'development_seen',
                      'target_xy': target, 'handle_xy': handle, 'delta_xy': list(state.delta_xy),
                      'prompts': {'direct': direct, 'grounded': grounded},
                      'files': {f.name: {'path': f.relative_to(args.output).as_posix(), 'sha256': sha(f)}
                                for f in folder.iterdir() if f.is_file()}})
    old_config = json.loads((root/'configs/day27_qwen_image_edit_direct_baseline_v1.json').read_text())
    config = {'experiment_id': 'observed_edit_v2_dev_20260928',
              'status': 'frozen_before_new_generation', 'seeds': [281],
              'cases': cases, 'conditions': ['direct', 'grounded'],
              'inference': old_config['inference'], 'backend': old_config['backend'],
              'resource_gate': old_config['resource_gate'],
              'expected_generation_count': 4, 'target_images_in_generation_bundle': False,
              'claim_limit': 'Two previously seen development images; no confirmatory/generalization/physical-mass claim.'}
    (args.output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'cases': len(cases), 'config_sha256': sha(args.output/'config.json')}))


if __name__ == '__main__':
    main()
