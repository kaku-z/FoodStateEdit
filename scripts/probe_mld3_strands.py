"""Deterministic source-ridge/contact probes; pictures are diagnostics, not edits."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodstateedit.material_lineage.strands import reconstruct_strands, deform_strands, flatten_bundle, tube_geometry


def project(xyz, K):
    p = xyz @ K.T
    return p[:, :2]/p[:, 2:]


def draw_curves(source, bundle, K=None):
    im = Image.fromarray(source.copy());draw = ImageDraw.Draw(im)
    for i, curve in enumerate(bundle.curves):
        uv = curve['source_uv'] if K is None else project(curve['xyz'], K)
        color = (int(80+(i*71)%175), int(80+(i*53)%175), int(80+(i*113)%175))
        draw.line([tuple(p) for p in uv], fill=color, width=2)
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('outputs/material_lineage_full_20261004/exploratory_geometry_v2'))
    parser.add_argument('--output', type=Path, default=Path('outputs/material_lineage_deformable_20261004/strand_probe'))
    parser.add_argument('--indices', type=int, nargs='+', default=[1, 4, 9, 11, 14, 15])
    parser.add_argument('--actual-spoon', action='store_true', help='Use the exact V3 tilt, footprint, and bowl/neck/handle surface.')
    args = parser.parse_args();args.output.mkdir(parents=True, exist_ok=True)
    all_rows = [];previews = []
    folders = sorted(f for f in args.source.glob('real_*') if f.is_dir())
    for index in args.indices:
        folder = folders[index];out = args.output/folder.name;out.mkdir(exist_ok=True)
        source = np.asarray(Image.open(folder/'source.png').convert('RGB'))
        food = np.asarray(Image.open(folder/'food_mask.png')) > 0
        patch = np.asarray(Image.open(folder/'source_bite_mask.png')) > 0
        maps = dict(np.load(folder/'depth.npz'));points = maps['points'].copy()
        state = np.load(folder/'closure_raw.npz');points[patch] = state['source_camera_xyz'];axes = state['world_axes']
        meta = json.loads((folder/'geometry_raw.json').read_text());translation = np.asarray(meta['target_translation'])
        h, w = source.shape[:2];K = maps['intrinsics'].copy();K[0] *= w;K[1] *= h
        bundle = reconstruct_strands(source, food, points, patch, axes)
        flat = flatten_bundle(bundle)
        if not len(flat['xyz']):
            row = dict(index=index, case=folder.name, reconstruct=bundle.metrics, variants={})
            all_rows.append(row);continue
        center = (flat['xyz']+translation) @ axes
        ab = np.maximum(np.ptp(center[:, :2], axis=0)*.75, np.mean(flat['radii'])*5)
        curvature = .12*min(ab);cen = np.median(center, axis=0)
        radial = np.sum(((center[:, :2]-cen[:2])/ab)**2, axis=1)
        spoon = dict(axes=axes.tolist(), center_local=cen.tolist(), radius_ab=ab.tolist(),
            floor_local=float(np.min(center[:, 2]-flat['radii']-curvature*radial)), bowl_curvature_height=float(curvature))
        if args.actual_spoon:
            up=axes[:,2];view=np.array([0.,0.,-1.]);tangent=view-up*(up@view);tangent/=np.linalg.norm(tangent)
            angle=min(np.deg2rad(12),np.arccos(np.clip(up@view,-1,1)));normal=np.cos(angle)*up+np.sin(angle)*tangent
            right=np.array([1.,0.,0.]);right-=normal*(right@normal);right/=np.linalg.norm(right)
            spoon_axes=np.stack([right,np.cross(normal,right),normal],axis=1)
            lower=state['source_camera_xyz'].copy();local=lower@axes
            lower+=(state['floor_local']-local[:,2])[:,None]*up
            lower=(lower+translation)@spoon_axes;center=lower[:,:2].mean(axis=0)
            dimensions=np.ptp(lower[:,:2],axis=0);a=.64*max(dimensions);b=a/1.5
            radial=np.sum(((lower[:,:2]-center)/[a,b])**2,axis=1);inside=radial<.90
            curvature=.14*b;floor=float(np.min(lower[inside,2]-curvature*radial[inside]))
            spoon=dict(axes=spoon_axes.tolist(),spoon_axes_camera=spoon_axes.tolist(),center_local=center.tolist(),
                radius_ab=[float(a),float(b)],floor_local=floor,bowl_curvature_height=curvature,bowl_handle_blend=True)
        variants = {};deformed = None
        for name, options in [('full', {}), ('no_contact', {'contact':False}), ('no_gravity', {'gravity_strength':0.})]:
            moved, metrics = deform_strands(bundle, translation, spoon, axes, **options)
            transport = flatten_bundle(moved)
            transport['rigid_xyz'] = flat['xyz']+translation
            transport['support_signed_gap'] = np.concatenate([c['support_signed_gap'] for c in moved.curves])
            np.savez_compressed(out/(name+'_transport.npz'), **transport)
            variants[name] = metrics
            if name == 'full':
                deformed = moved
                mesh = tube_geometry(moved, source)
                np.savez_compressed(out/'textured_tubes.npz', **mesh)
        ridge = source.copy();sk = bundle.skeleton;ridge[sk] = [255, 40, 50]
        overlay = draw_curves(source, bundle)
        selected = source.copy();selected[bundle.removal_mask] = (.35*selected[bundle.removal_mask]+.65*np.array([40, 255, 80])).astype(np.uint8)
        lifted = draw_curves(source, deformed, K)
        panels = [Image.fromarray(ridge), overlay, Image.fromarray(selected), lifted]
        captioned = Image.new('RGB',(w*4,h+30),'white');draw = ImageDraw.Draw(captioned)
        for j, (panel, caption) in enumerate(zip(panels, ['All observed ridges', 'Selected source curves', 'Traced removal footprint', 'Deformed curve projection'])):
            captioned.paste(panel,(j*w,30));draw.text((j*w+5,8),caption,fill='black')
        captioned.save(out/'probe.jpg');previews.append(captioned)
        row = dict(index=index,case=folder.name,reconstruct=bundle.metrics,variants=variants,
            spoon_proxy=spoon,diagnostic_only=True,source=str(folder),
            surface_depth='Same locally stabilized v2 source rays; no target/ground-truth conditioning.')
        (out/'metrics.json').write_text(json.dumps(row,indent=2));all_rows.append(row)
        print(json.dumps(dict(index=index,case=folder.name,curves=len(bundle.curves),full=variants['full'])),flush=True)
    (args.output/'summary.json').write_text(json.dumps(dict(status='complete',cases=all_rows,
        semantics='Indices 09 omelet and 15 beef are negative food-type controls; do not render their detected texture ridges as noodles.',
        method='Deterministic multiscale RGB Hessian ridges, skeleton graph, source-ray sampling, cut paths, length/bending/contact PBD.'),indent=2))
    target_width = 1400;resized=[]
    for im in previews:
        resized.append(im.resize((target_width,int(im.height*target_width/im.width))))
    sheet = Image.new('RGB',(target_width,sum(im.height for im in resized)), 'white');y=0
    for im in resized:sheet.paste(im,(0,y));y+=im.height
    sheet.save(args.output/'six_real_probes.jpg')


if __name__ == '__main__':
    main()
