"""Extend only the distal spoon shaft to the image edge on resolved geometry.

This isolated branch changes no food state and runs no image generation model.
The smooth monotonic deformation fixes every vertex at local x <= bowl radius.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
LEGACY_RELEASE = Path('/host/space0/guo-z/tf-ufi/material_lineage_deformable_20261004/execution_snapshot_final')
sys.path.extend([str(LEGACY_RELEASE/'scripts'), str(LEGACY_RELEASE)])
sys.path.insert(0, '/host/space0/guo-z/tf-ufi/food3d_repair_20260928/compat_packages')
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, distance_transform_edt
import trimesh
import mld2_real_spoon_v3
from mld2_real_spoon_v3 import metal_pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def rgb(path):
    return np.asarray(Image.open(path).convert('RGB')).copy()


def mask(path):
    return np.asarray(Image.open(path).convert('L')) > 127


def save(path, array):
    Image.fromarray(array.astype(np.uint8)*255 if array.dtype == bool else array).save(path)


def dilate(array, radius):
    return distance_transform_edt(~array) <= radius if array.any() else array.copy()


def project(points, K):
    q = np.asarray(points) @ K.T
    return q[..., :2] / q[..., 2:]


def extend(mesh, info, K, width, target_xyz, margin):
    vertices = np.asarray(mesh.vertices).copy()
    axes = np.asarray(info['spoon_axes_camera'])
    a = info['radius_ab'][0]
    cx = info['center_local'][0]
    x = (vertices @ axes)[:, 0] - cx
    # All 24 source-bound target packets fit before this fixed bowl/shaft join.
    food_max_x = float((target_xyz @ axes)[:, 0].max() - cx)
    assert food_max_x < a, 'The prescribed fixed bowl does not contain food contact.'
    fixed = x <= a
    tip = x >= x.max() - 1e-5*a
    p = vertices[tip].mean(0)
    old_uv = project(p, K)
    already_outside = bool(old_uv[0] >= width)
    delta = 0.
    result = mesh.copy()
    if not already_outside:
        d = axes[:, 0]
        target_u = width + margin
        delta = float((target_u*(K[2]@p) - K[0]@p) / (K[0]@d - target_u*(K[2]@d)))
        assert np.isfinite(delta) and delta > 0
        t = np.clip((x-a)/(x.max()-a), 0., 1.)
        smooth = t*t*(3.-2.*t)
        moved = vertices + delta*smooth[:, None]*d
        moved[fixed] = vertices[fixed]
        assert (moved[:, 2] > 0).all()
        result.vertices = moved
    new_uv = project(result.vertices[tip].mean(0), K)
    assert np.array_equal(result.faces, mesh.faces)
    assert np.array_equal(result.vertices[fixed], vertices[fixed])
    if not already_outside:
        assert abs(new_uv[0] - (width+margin)) < 1e-7
    info_out = dict(noop=already_outside, reason='Original distal tip is already outside the right frame' if already_outside else 'Extend fixed distal handle to width + margin',
        old_tip_uv=old_uv.tolist(), new_tip_uv=new_uv.tolist(), image_width=width,
        margin_pixels=margin, local_delta=delta, bowl_radius_a=a, fixed_local_x_max=a,
        target_food_max_local_x=food_max_x, contact_axial_margin=a-food_max_x,
        vertex_count=len(vertices), face_count=len(mesh.faces), fixed_vertices=int(fixed.sum()),
        fixed_vertex_max_change=float(np.abs(result.vertices[fixed]-vertices[fixed]).max()),
        unchanged_face_indices=bool(np.array_equal(result.faces, mesh.faces)),
        old_watertight=bool(mesh.is_watertight), new_watertight=bool(result.is_watertight),
        minimum_camera_z=float(result.vertices[:, 2].min()),
        deformation='x_new=x+delta*smoothstep(clamp((x-a)/(old_tip_x-a),0,1)); local y,z unchanged. C1 join, positive axial Jacobian; bowl/contact x<=a fixed exactly.',
        collision_scope='Injective axial stretch preserves mesh self-overlap status. Carried food lies entirely in the fixed x<a slab. Unobserved scene/hand collisions are not verified.')
    return result, info_out


def shaft_from_depth(depth, K, info):
    yy, xx = np.indices(depth.shape)
    rays = np.stack((xx+.5, yy+.5, np.ones_like(xx)), -1) @ np.linalg.inv(K).T
    xyz = rays * depth[..., None]
    x = (xyz @ np.asarray(info['spoon_axes_camera']))[..., 0] - info['center_local'][0]
    return (depth > 0) & (x > info['radius_ab'][0])


def overlay(source, old, new, protected, acceptance):
    result = source.astype(float)
    result[acceptance] = .72*result[acceptance]+.28*np.array([255, 210, 20])
    for region, color in [(old, [255, 50, 50]), (new, [30, 230, 255]), (protected, [200, 70, 255])]:
        edge = region & ~binary_erosion(region)
        result[edge] = color
    return np.rint(result).clip(0, 255).astype(np.uint8)


def run_case(case, output, args):
    start = time.perf_counter()
    geometry = Path(case['resolved_geometry_case'])
    guide = Path(case['production_guides']['boundary'])
    out = output/case['case_id']
    out.mkdir()
    source = rgb(guide/'source.png')
    h, w = source.shape[:2]
    info = json.loads((geometry/'state.json').read_text())['spoon']
    with np.load(geometry/'transport.npz') as transport:
        K = transport['camera_intrinsics'].copy()
        target_xyz = transport['target_xyz'].copy()
    old = trimesh.load(geometry/'spoon.ply', process=False, force='mesh')
    new, audit = extend(old, info, K, w, target_xyz, args.margin)
    old_rgb, old_depth = metal_pass(old, info, source, K, w, h)
    new_rgb, new_depth = (old_rgb.copy(), old_depth.copy()) if audit['noop'] else metal_pass(new, info, source, K, w, h)
    old_shaft = shaft_from_depth(old_depth, K, info)
    new_shaft = shaft_from_depth(new_depth, K, info)
    bowl = (old_depth > 0) & ~old_shaft
    target = mask(guide/'target_tolerance_mask.png')
    source_region = mask(guide/'source_inpaint_mask.png')
    protected = target | source_region | dilate(bowl, 1.)
    edit = dilate(old_shaft | new_shaft, args.dilation) & ~protected
    if audit['noop']:
        edit[:] = False
    # Existing full-frame boundary guide remains exact outside the new edit area.
    control = rgb(guide/'edge.png')
    background_edges = cv2.Canny(source, 80, 160)
    replacement = np.repeat(background_edges[..., None], 3, axis=2)
    replacement[new_shaft] = 0
    replacement[dilate(new_shaft, 1.) & ~binary_erosion(new_shaft)] = 255
    control[edit] = replacement[edit]
    preview = rgb(guide/'coarse.png')
    preview[old_shaft & edit] = source[old_shaft & edit]
    preview[new_shaft & edit] = new_rgb[new_shaft & edit]
    shutil.copy2(guide/'source.png', out/'source.png')
    for name, array in dict(handle_rgb=new_rgb, old_shaft_mask=old_shaft, new_shaft_mask=new_shaft,
            acceptance_mask=edit, control=control, bowl_mask=bowl, protected_mask=protected,
            target_tolerance_mask=target, source_inpaint_mask=source_region,
            depth_valid=new_depth>0, old_depth_valid=old_depth>0,
            conditioning_preview=preview,
            mask_overlay=overlay(source, old_shaft, new_shaft, protected, edit)).items():
        save(out/(name+'.png'), array)
    np.save(out/'depth.npy', new_depth.astype(np.float32))
    new.export(out/'spoon.ply')
    reloaded = trimesh.load(out/'spoon.ply', process=False, force='mesh')
    fixed = (old.vertices @ np.asarray(info['spoon_axes_camera']))[:, 0]-info['center_local'][0] <= info['radius_ab'][0]
    audit['serialized_fixed_vertex_max_change'] = float(np.abs(reloaded.vertices[fixed]-old.vertices[fixed]).max())
    assert audit['serialized_fixed_vertex_max_change'] == 0.
    assert np.array_equal(reloaded.faces, old.faces)
    audit.update(case_id=case['case_id'], index=case['index'], input_geometry=str(geometry), input_boundary_guide=str(guide),
        input_hashes={str(p): sha(p) for p in [geometry/'spoon.ply', geometry/'state.json', geometry/'transport.npz', guide/'source.png', guide/'edge.png', guide/'coarse.png', guide/'target_tolerance_mask.png', guide/'source_inpaint_mask.png']},
        renderer='Existing mld2_real_spoon_v3.metal_pass; inferred neutral source palette and fixed PBR strip lights; 2x supersampling',
        renderer_path=str(Path(mld2_real_spoon_v3.__file__)), renderer_sha256=sha(Path(mld2_real_spoon_v3.__file__)),
        runner_sha256=sha(Path(__file__)), camera_intrinsics=K.tolist(),
        depth_semantics='Full new spoon camera Z; zero means no spoon. Food visibility is protected by target_tolerance, not baked into depth.',
        mask_semantics='Visible rasterized local x>a shaft; exact original bowl plus 1px Euclidean collar, target tolerance and full source inpaint mask are protected. Acceptance is union(old,new) dilated 3px minus protected. All no-op acceptance masks are empty.',
        original_food_protected=False, food_transport_changed=False, source_region_changed=False,
        old_shaft_pixels=int(old_shaft.sum()), new_shaft_pixels=int(new_shaft.sum()), acceptance_pixels=int(edit.sum()),
        acceptance_protected_overlap=int((edit & protected).sum()),
        new_shaft_source_inpaint_overlap=int((new_shaft & source_region).sum()),
        new_shaft_target_tolerance_overlap=int((new_shaft & target).sum()),
        original_shaft_right_edge=bool(old_shaft[:, -1].any()), new_shaft_right_edge=bool(new_shaft[:, -1].any()),
        new_shaft_top_edge=bool(new_shaft[0].any()), new_shaft_bottom_edge=bool(new_shaft[-1].any()),
        new_shaft_right_edge_accepted=bool((new_shaft & edit)[:, -1].any()),
        control_changed_outside_acceptance=int(np.any(control != rgb(guide/'edge.png'), axis=2)[~edit].sum()),
        preview_status='Geometry-only preview based on original coarse. Downstream must initialize from completed production RGB.',
        interface=['source.png','handle_rgb.png','old_shaft_mask.png','new_shaft_mask.png','acceptance_mask.png','control.png','depth.npy','geometry.json'],
        seconds=time.perf_counter()-start)
    assert audit['acceptance_protected_overlap'] == audit['control_changed_outside_acceptance'] == 0
    if not audit['noop']:
        assert audit['new_shaft_right_edge'] or audit['new_shaft_top_edge'] or audit['new_shaft_bottom_edge'], 'Extended shaft misses all allowed image boundaries.'
    audit['visible_exit'] = 'right' if audit['new_shaft_right_edge'] else ('top' if audit['new_shaft_top_edge'] else 'bottom')
    audit['right_edge_constraint_satisfied'] = audit['new_shaft_right_edge']
    audit['exit_limitation'] = None if audit['new_shaft_right_edge'] else 'Fixed local +x direction exits another boundary before reaching target u. No rotation or image-specific redirection was applied.'
    write(out/'geometry.json', audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--indices', type=int, nargs='+')
    parser.add_argument('--margin', type=float, default=10.)
    parser.add_argument('--dilation', type=float, default=3.)
    args = parser.parse_args()
    cases = json.loads(args.manifest.read_text())['cases']
    if args.indices is not None:
        cases = [c for c in cases if c['index'] in args.indices]
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'frozen.json', dict(manifest=str(args.manifest), manifest_sha256=sha(args.manifest),
        cases=[c['case_id'] for c in cases], runner_sha256=sha(Path(__file__)), margin=args.margin,
        dilation=args.dilation, already_outside_policy='No-op when tip u>=image width',
        scene_collision_policy='Preserve bowl/contact; positive-Jacobian axial deformation. Hidden scene collisions unobserved.',
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), egl_device_id=os.environ.get('EGL_DEVICE_ID')))
    shutil.copy2(__file__, args.output/Path(__file__).name)
    rows = []
    for case in cases:
        row = run_case(case, args.output, args)
        rows.append(row)
        write(args.output/'manifest.json', dict(status='running', cases=rows))
        print(json.dumps({k: row[k] for k in ['case_id','noop','old_tip_uv','new_tip_uv','acceptance_pixels','new_shaft_right_edge']}), flush=True)
    write(args.output/'manifest.json', dict(status='complete', cases=rows,
        extended=sum(not r['noop'] for r in rows), noop=sum(r['noop'] for r in rows),
        all_protected_unchanged=all(r['acceptance_protected_overlap']==0 for r in rows)))
    selected = [r for r in rows if r['index'] in [2,13,14]]
    sheet = Image.new('RGB', (900, 290*len(selected)), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(selected):
        for j, filename in enumerate(['mask_overlay.png','conditioning_preview.png','control.png']):
            im = Image.open(args.output/row['case_id']/filename).convert('RGB')
            im.thumbnail((294,260))
            sheet.paste(im, (j*300, i*290+25))
            draw.text((j*300+3,i*290+5), row['case_id']+' '+filename, fill='black')
    if selected:
        sheet.save(args.output/'three_case_overlays.png')


if __name__ == '__main__':
    main()
