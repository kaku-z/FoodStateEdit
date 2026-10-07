"""Explicit observed-material path, with a bounded scalar illumination residual.

This diagnostic branch changes no geometry and makes no hidden-surface claim.
High-confidence interior pixels are transported source RGB times a smooth scalar
light field. Unknown surfaces and the transition band keep generative freedom.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter

from run_mld4_reconstruction import sha, write
from probe_mld4_locked_material import linear, srgb


def project_visible_material(texture, proposal, food, confidence):
    known = food & (confidence >= .75)
    # Boolean observed ownership, not fractional confidence, defines the core.
    # In v1 no udon pixel was fully bound because every confidence was below 1.
    alpha = np.clip((distance_transform_edt(known)-.5)/1.5, 0., 1.)
    core = alpha == 1.
    t, g = linear(texture), linear(proposal)
    luminance = np.array([.2126, .7152, .0722])
    ty, gy = t @ luminance, g @ luminance
    scale = max(food.shape)/640.
    sigma = max(1.5, 4.8*scale)
    norm = gaussian_filter(known.astype(np.float32), sigma)
    source_light = gaussian_filter(ty*known, sigma)/np.maximum(norm, 1e-6)
    target_light = gaussian_filter(gy*known, sigma)/np.maximum(norm, 1e-6)
    gain = np.clip((target_light+.01)/(source_light+.01), .72, 1.4)
    material = 255*srgb(t*gain[..., None]).clip(0., 1.)
    result = np.rint(material*alpha[..., None]+proposal*(1-alpha[..., None])).clip(0,255).astype(np.uint8)
    result[~known] = proposal[~known]
    return result, dict(known=known, core=core, alpha=alpha, gain=gain)


def rgb(path):
    return np.asarray(Image.open(path).convert('RGB'))


def run(args):
    protocol = json.loads((args.production/'protocol.json').read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in protocol['cases']:
        if args.indices is not None and case['index'] not in args.indices:
            continue
        guide = Path(case['source_guides']['boundary'])
        basepath = args.production/'real'/case['case_id']/'final.png'
        proposal = rgb(basepath)
        texture = rgb(guide/'material_texture.png')
        with np.load(guide/'guide_channels.npz') as z:
            confidence = z['material_confidence'].copy()
        food = np.asarray(Image.open(guide/'target_food_mask.png').convert('L')) > 127
        final, maps = project_visible_material(texture, proposal, food, confidence)
        folder = args.output/'real'/case['case_id']
        folder.mkdir(parents=True)
        Image.fromarray(final).save(folder/'final.png')
        Image.fromarray((maps['core']*255).astype(np.uint8)).save(folder/'bound_core_mask.png')
        np.savez_compressed(folder/'material_contract.npz', **maps)
        # A wrong UV control keeps color statistics and scalar lighting fixed.
        # Reflecting the observed atlas swaps correspondence, not ingredient text.
        wrong_texture = texture.copy()
        yy, xx = np.where(maps['known'])
        if len(xx):
            order = np.lexsort((xx, yy))
            wrong_texture[yy[order],xx[order]] = texture[yy[order[::-1]],xx[order[::-1]]]
        wrong_material = 255*srgb(linear(wrong_texture)*maps['gain'][...,None]).clip(0,1)
        wrong = np.rint(wrong_material*maps['alpha'][...,None]+proposal*(1-maps['alpha'][...,None])).clip(0,255).astype(np.uint8)
        wrong[~maps['known']] = proposal[~maps['known']]
        Image.fromarray(wrong).save(folder/'wrong_correspondence_control.png')
        expected = np.rint(255*srgb(linear(texture)*maps['gain'][...,None]).clip(0,1)).clip(0,255).astype(np.uint8)
        core = maps['core']
        row = dict(case_id=case['case_id'], base_sha256=sha(basepath),
            texture_sha256=sha(guide/'material_texture.png'), source_sha256=sha(guide/'source.png'),
            runner_sha256=sha(Path(__file__)), final_sha256=sha(folder/'final.png'),
            known_pixels=int(maps['known'].sum()), bound_core_pixels=int(core.sum()), food_pixels=int(food.sum()),
            core_fraction=float(core.sum()/max(1,food.sum())), scalar_gain_bounds=[.72,1.4],
            core_contract_max_rgb_error=int(np.abs(final.astype(int)-expected.astype(int))[core].max()) if core.any() else None,
            changed_outside_known_pixels=int(np.any(final!=proposal,axis=2)[~maps['known']].sum()),
            wrong_correspondence_core_mae=float(np.abs(wrong.astype(float)-final.astype(float))[core].mean()) if core.any() else None,
            generated_chromatic_content_in_bound_core=False,
            wrong_control='Reverse raster-order selected observed atlas samples; same gain and alpha. Constructed diagnostic, not a natural result.',
            limitation='Core identity follows by construction. Source RGB includes original lighting; natural appearance and unobserved surfaces remain unverified.')
        write(folder/'result.json', row)
        rows.append(row)
    write(args.output/'manifest.json', dict(status='complete', protocol_sha256=sha(args.production/'protocol.json'),
        method='Source-corresponded radiance texture with bounded achromatic low-frequency relighting',
        parameter_selection='Uniform threshold .75, core >=2 native pixels, gain .72..1.4; no per-image output selection',
        cases=rows, interpretation='Implementation contract, not a learned generalization or true first-bite metric'))
    print(json.dumps(dict(cases=len(rows), bound_core_pixels=sum(r['bound_core_pixels'] for r in rows))))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--production',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--indices',type=int,nargs='+')
    run(p.parse_args())
