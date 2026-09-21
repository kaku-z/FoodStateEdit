"""Freeze corrected E7 assets, audit the E6 donor, and emit paired configs."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from bite_remain_consistency import dilate
from cavity_state_projection import render_material_pure_cavity
from run_high_lift_vace_pilot import sha256, validate_file, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', required=True, type=Path)
    a = p.parse_args()
    source = ROOT/'artifacts/e5b_bite_remain_controls_v1_20260917'
    expected = '14f007d57c42289c428600991b460add04a6c73552c3a774821a061a3ea024b2'
    if sha256(source/'manifest.json') != expected:
        raise ValueError('E5b source manifest changed')
    manifest = json.loads((source/'manifest.json').read_text())
    for rel, record in manifest['files'].items():
        validate_file(source/rel, record)
    with np.load(source/'controls.npz', allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    geometry_path = ROOT/'artifacts/e2_control_representation_controls_v1_20260917/cake/geometry.json'
    geometry = json.loads(geometry_path.read_text())
    vertices = np.asarray(geometry['vertices_uv'], float)
    reference = np.asarray(Image.open(source/'reference.png').convert('RGB'))
    yy, xx = np.indices(reference.shape[:2])
    hsv = cv2.cvtColor(reference, cv2.COLOR_RGB2HSV)
    old_donor = (xx >= 115) & (xx < 285) & (yy >= 250) & (yy < 350)
    # This is an explicit material rule for the cake diagnostic, not a learned
    # semantic segmenter. Restrict to the original crumb ROI and reject plate.
    material = old_donor & (hsv[..., 1] > 55) & (hsv[..., 2] > 65)
    removal = arrays['removal_projection'] > 0
    plate = ((xx > 265) & (xx < 510) & (yy > 300) & (yy < 440)
             & (hsv[..., 1] < 42) & ~dilate(removal, 18))
    fixed, faces, audit = render_material_pure_cavity(reference, vertices, material, plate)
    audit.update(
        source_controls_manifest_sha256=expected,
        geometry_sha256=sha256(geometry_path),
        old_donor_total_pixels=int(old_donor.sum()),
        old_donor_low_saturation_pixels=int((old_donor & (hsv[..., 1] < 42)).sum()),
        old_donor_outside_material_rule_pixels=int((old_donor & ~material).sum()),
        material_rule='within original crumb ROI, HSV saturation >55 and value >65',
        note='purity is with respect to the explicit mask, not ground-truth semantic annotation',
    )
    for i in range(len(arrays['condition'])):
        target = reference.copy(); target[removal] = fixed[removal]
        support = arrays['generation_mask'][i] > 0
        arrays['condition'][i] = arrays['baseline'][i].copy()
        arrays['condition'][i][support] = target[support]
    arrays['cavity_anchor'] = fixed
    a.output_root.mkdir(parents=True, exist_ok=False)
    controls = a.output_root/'controls'; controls.mkdir()
    np.savez_compressed(controls/'controls.npz', **arrays)
    Image.fromarray(reference).save(controls/'reference.png')
    Image.fromarray(fixed).save(controls/'cavity_anchor.png')
    Image.fromarray(material.astype(np.uint8)*255).save(controls/'material_donor_mask.png')
    box = audit['donor_box_xyxy']
    Image.fromarray(reference[box[1]:box[3], box[0]:box[2]]).save(controls/'material_donor.png')
    write_json(controls/'donor_audit.json', audit)
    cfg = dict(manifest['config'])
    cfg['claim_limit'] = 'E7 corrected material donor and smooth floor shading; synthetic cake diagnostic only.'
    new_manifest = dict(manifest, config=cfg, parent_manifest_sha256=expected,
        schema_version='foodstateedit.bite_remain_controls.v1',
        config_sha256=None, builder_sha256=sha256(Path(__file__)),
        render_helper_sha256=sha256(Path(__file__).with_name('cavity_state_projection.py')),
        files={f.name: {'sha256':sha256(f), 'size_bytes':f.stat().st_size}
               for f in controls.iterdir() if f.is_file()})
    write_json(controls/'manifest.json', new_manifest)
    e6 = json.loads((ROOT/'configs/e6_bite_remain_state_projection_cake_v1.json').read_text())
    for variant in ('hard', 'soft'):
        run_cfg = json.loads(json.dumps(e6))
        run_cfg.update(schema_version='foodstateedit.cavity_repair.v1', variant=variant,
            controls_manifest_sha256=sha256(controls/'manifest.json'),
            controls=str(controls),
            causal_baseline='E6 to E7 hard: material-pure donor + continuous shadow; E7 hard to soft: projection policy only',
            claim_limit='Single development case; control/rendering and projection diagnostic, no generalization claim.')
        run_cfg['soft_projection'] = {'max_strength': .35, 'feather_kernel': 3}
        write_json(a.output_root/f'e7_{variant}.json', run_cfg)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
