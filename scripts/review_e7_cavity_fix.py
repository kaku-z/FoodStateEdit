"""Verify E7 bytes and make fixed-crop comparisons; no automatic success claim."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from run_high_lift_vace_pilot import sha256, validate_file, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hard', type=Path, required=True)
    p.add_argument('--soft', type=Path, required=True)
    p.add_argument('--controls', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    a = p.parse_args()
    runs = {
        'E5b / old donor / off': ROOT/'artifacts/e5b_bite_remain_gp40_20260917T063729Z',
        'E6 / old donor / hard': ROOT/'artifacts/e6_bite_remain_state_projection_gp40_20260917T141717Z',
        'E7 / pure donor / hard': a.hard,
        'E7 / pure donor / soft': a.soft,
    }
    with np.load(a.controls/'controls.npz', allow_pickle=False) as z:
        arrays = {k:z[k] for k in z.files}
    a.output_root.mkdir(parents=True, exist_ok=False)
    report = {'scope':'one synthetic cake; automated diagnostics plus separate visual review',
              'quality_success':None, 'runs':{},
              'metric_limit':'wall low-saturation is a donor-contamination diagnostic, not a photo-quality score'}
    crop = (280, 195, 465, 405)
    crops = Image.new('RGB', (370*4, 420+40), 'white')
    frames = [9, 12, 15, 20]
    timeline = Image.new('RGB', (370*4, (420+40)*4), 'white')
    reference = Image.open(a.controls/'reference.png').convert('RGB')
    donor_audit = json.loads((a.controls/'donor_audit.json').read_text())
    annotated = reference.copy()
    draw = ImageDraw.Draw(annotated)
    draw.rectangle((115,250,285,350), outline='red', width=3)
    draw.rectangle(donor_audit['donor_box_xyxy'], outline='lime', width=3)
    control_sheet = Image.new('RGB', (reference.width*3, reference.height+40), 'white')
    old_anchor = ROOT/'artifacts/e5b_bite_remain_controls_v1_20260917/cavity_anchor.png'
    for slot, (label, panel) in enumerate([
        ('Donor ROI: old red / corrected green', annotated),
        ('E6 reference BEFORE diffusion', Image.open(old_anchor).convert('RGB')),
        ('E7 reference BEFORE diffusion', Image.open(a.controls/'cavity_anchor.png').convert('RGB')),
    ]):
        ImageDraw.Draw(control_sheet).text((slot*reference.width+6,8), label, fill='black')
        control_sheet.paste(panel,(slot*reference.width,40))
    control_sheet.save(a.output_root/'donor_root_cause.png')
    for column, (name, root) in enumerate(runs.items()):
        records = json.loads((root/'files_manifest.json').read_text())
        for rel, rec in records.items():
            validate_file(root/rel, rec)
        run = json.loads((root/'run_manifest.json').read_text())
        if run['status'] != 'complete_requires_visual_review' or run['pipeline_load_count'] != 1:
            raise ValueError('incomplete or unexpected run: '+name)
        diagnostics = []
        for i in range(21):
            image = np.asarray(Image.open(root/'projected_frames'/f'{i:02d}.png').convert('RGB'))
            base, alpha, payload = (arrays[k][i] for k in ('baseline','projection_alpha','planned_payload'))
            diff = np.abs(image.astype(np.int16)-base.astype(np.int16))
            if diff[alpha==0].max(initial=0) or diff[payload>0].max(initial=0):
                raise AssertionError('protected pixel mismatch: '+name)
            wall = ((arrays['wall_back']>0) | (arrays['wall_left']>0)) & (alpha==255)
            wall = cv2.erode(wall.astype(np.uint8), np.ones((3,3), np.uint8))>0
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            diagnostics.append({'frame':i, 'wall_pixels':int(wall.sum()),
                'wall_low_saturation_fraction':float((hsv[...,1][wall]<42).mean()) if wall.any() else None})
        report['runs'][name] = {'path':str(root), 'verified_file_records':len(records),
            'files_manifest_sha256':sha256(root/'files_manifest.json'),
            'run_manifest':run, 'wall_diagnostics':diagnostics,
            'final_sha256':sha256(root/'projected_final_hold.png')}
        ImageDraw.Draw(crops).text((column*370+5,8), name, fill='black')
        image = Image.open(root/'projected_final_hold.png').convert('RGB')
        crops.paste(image.crop(crop).resize((370,420)),(column*370,40))
        for row, i in enumerate(frames):
            ImageDraw.Draw(timeline).text((column*370+5,row*460+8), f'{name} / frame {i}', fill='black')
            image = Image.open(root/'projected_frames'/f'{i:02d}.png').convert('RGB')
            timeline.paste(image.crop(crop).resize((370,420)),(column*370,row*460+40))
    crops.save(a.output_root/'final_cavity_comparison.png')
    timeline.save(a.output_root/'cavity_timeline.png')
    write_json(a.output_root/'verification.json', report)
    print(json.dumps({name: row['wall_diagnostics'][-1] for name,row in report['runs'].items()},indent=2))


if __name__ == '__main__':
    main()
