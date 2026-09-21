"""Restore original pixels outside per-frame motion and cavity support.

This postprocessor fixes inherited swept-background drift. It does not change
the diffusion result inside the operation core or constitute learned efficacy.
The motion support comes from the frozen proxy, so unplanned object motion
outside the margin can still be clipped and requires visual review.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from bite_remain_consistency import composite, dilate, inward_alpha
from run_high_lift_vace_pilot import sha256, validate_file, write_json


def frame_edit_alpha(reference, proxy, repair_alpha, payload, margin=12, feather=8):
    if proxy.shape != reference.shape or repair_alpha.shape != reference.shape[:2]:
        raise ValueError('reference/proxy/repair shape mismatch')
    if payload.shape != repair_alpha.shape or margin < feather or feather <= 0:
        raise ValueError('invalid payload mask or feather/margin')
    changed = np.any(proxy != reference, axis=-1)
    # At t=0 the planned payload still exists in the original photo. Do not
    # create an edit merely because its static segmentation is nonempty.
    if not changed.any() and not repair_alpha.any():
        return np.zeros_like(repair_alpha, dtype=np.uint8)
    core = changed | (repair_alpha > 0) | (payload > 0)
    support = dilate(core, margin)
    alpha = inward_alpha(support, feather)
    if np.any(alpha[core] != 255):
        raise AssertionError('operation core must be fully preserved')
    return alpha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--controls', type=Path, required=True)
    p.add_argument('--proxy-controls', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    a = p.parse_args()
    records = json.loads((a.run/'files_manifest.json').read_text())
    for rel, rec in records.items():
        validate_file(a.run/rel, rec)
    reference = np.asarray(Image.open(a.controls/'reference.png').convert('RGB'))
    with np.load(a.controls/'controls.npz', allow_pickle=False) as z:
        repair = z['projection_alpha']; payload = z['planned_payload']
    with np.load(a.proxy_controls/'controls.npz', allow_pickle=False) as z:
        proxy = z['appearance_laden_rgb']
    if len(proxy) != len(repair):
        raise ValueError('frame count mismatch')
    a.output_root.mkdir(parents=True, exist_ok=False)
    frames_dir=a.output_root/'frames'; frames_dir.mkdir()
    masks=[]; diagnostics=[]
    for i, frame in enumerate(proxy):
        candidate = np.asarray(Image.open(a.run/'projected_frames'/f'{i:02d}.png').convert('RGB'))
        alpha=frame_edit_alpha(reference,frame,repair[i],payload[i])
        final=composite(reference,candidate,alpha)
        outside=alpha==0
        error=int(np.abs(final.astype(np.int16)-reference.astype(np.int16))[outside].max(initial=0))
        core=alpha==255
        core_error=int(np.abs(final.astype(np.int16)-candidate.astype(np.int16))[core].max(initial=0))
        if error or core_error:
            raise AssertionError('compositor invariants failed')
        Image.fromarray(final).save(frames_dir/f'{i:02d}.png')
        masks.append(alpha);diagnostics.append({'frame':i,'support_pixels':int((alpha>0).sum()),
            'outside_original_max_difference':error,'operation_core_max_difference':core_error})
    Image.fromarray(final).save(a.output_root/'final.png')
    np.savez_compressed(a.output_root/'dynamic_alpha.npz',alpha=np.stack(masks))
    write_json(a.output_root/'manifest.json',{
        'schema_version':'foodstateedit.original_reference_compositor.v1',
        'source_run':str(a.run),'source_run_manifest_sha256':sha256(a.run/'files_manifest.json'),
        'proxy_controls_sha256':sha256(a.proxy_controls/'controls.npz'),
        'repair_controls_sha256':sha256(a.controls/'controls.npz'),
        'script_sha256':sha256(Path(__file__)),
        'parameters':{'margin_pixels':12,'inward_feather_pixels':8},
        'claim':'deterministic background restoration, not a new diffusion output',
        'diagnostics':diagnostics,
        'files':{f.relative_to(a.output_root).as_posix():{'sha256':sha256(f),'size_bytes':f.stat().st_size}
                 for f in a.output_root.rglob('*') if f.is_file()}})


if __name__ == '__main__':
    main()
