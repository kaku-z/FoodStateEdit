"""Anchor only source-observed surfaces whose normal is unchanged by yaw.

This produces explicit composites, not raw generative outputs. CPU triangle
rasterization identifies visible target faces whose source texture was observed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt
import trimesh

from collect_structure_development import ROOT, read, sha


def observed_surface_mask(geometry, intrinsics):
    report = read(geometry / 'geometry_report.json')
    fit, motion = report['fit'], report['cut_and_support']
    rotation = np.array(fit['axes_camera_columns'])
    yaw = np.array(motion['rotation_food_frame'])
    destination = np.array(motion['destination_center'])
    origin = np.array(motion['source_center'])
    high = np.array(fit['high'])
    extent = np.max(high-np.array(fit['low']))
    mesh = trimesh.load(geometry / 'bite_lifted.ply', process=False)
    camera_vertices = np.asarray(mesh.vertices)
    source_vertices = (camera_vertices @ rotation-destination) @ yaw+origin
    source_mesh = trimesh.Trimesh(source_vertices, mesh.faces, process=False)
    inherited = np.zeros(len(mesh.faces), bool)
    for axis in [2]:
        inherited |= ((np.abs(source_mesh.triangles_center[:, axis]-high[axis]) < 1e-4*extent)
                      & (source_mesh.face_normals[:, axis] > .99))
    # Perspective-correct z buffer, with all faces competing before class labels.
    uvh = camera_vertices @ intrinsics.T
    uv = uvh[:, :2]/uvh[:, 2:]
    zbuffer = np.full((480, 640), np.inf)
    labels = np.zeros((480, 640), bool)
    for index, face in enumerate(mesh.faces):
        p = uv[face]
        z = camera_vertices[face, 2]
        if np.any(z <= 0):
            raise ValueError('Target geometry behind camera')
        low = np.maximum(np.floor(p.min(0)).astype(int), [0, 0])
        high_uv = np.minimum(np.ceil(p.max(0)).astype(int), [639, 479])
        if np.any(high_uv < low):
            continue
        x, y = np.meshgrid(np.arange(low[0], high_uv[0]+1)+.5,
                           np.arange(low[1], high_uv[1]+1)+.5)
        denom = (p[1,1]-p[2,1])*(p[0,0]-p[2,0])+(p[2,0]-p[1,0])*(p[0,1]-p[2,1])
        if abs(denom) < 1e-10:
            continue
        a = ((p[1,1]-p[2,1])*(x-p[2,0])+(p[2,0]-p[1,0])*(y-p[2,1]))/denom
        b = ((p[2,1]-p[0,1])*(x-p[2,0])+(p[0,0]-p[2,0])*(y-p[2,1]))/denom
        c = 1-a-b
        inside = (a >= -1e-7) & (b >= -1e-7) & (c >= -1e-7)
        inverse = a/z[0]+b/z[1]+c/z[2]
        depth = np.divide(1., inverse, out=np.full_like(inverse, np.inf), where=inverse > 0)
        patch = zbuffer[low[1]:high_uv[1]+1, low[0]:high_uv[0]+1]
        selected = inside & (depth < patch)
        patch[selected] = depth[selected]
        lab = labels[low[1]:high_uv[1]+1, low[0]:high_uv[0]+1]
        lab[selected] = inherited[index]
    rendered_bite = np.asarray(Image.open(geometry / 'bite_mask.png')) > 0
    rasterized = np.isfinite(zbuffer)
    iou = np.sum(rasterized & rendered_bite)/np.sum(rasterized | rendered_bite)
    assert iou > .94, f'CPU/GPU raster disagreement: {iou}'
    return labels & rendered_bite, dict(cpu_gpu_silhouette_iou=float(iou),
                                       inherited_source_faces=int(inherited.sum()),
                                       total_faces=len(inherited),
                                       scope='Only observed top faces: yaw preserves their surface normal. Rotated side-face baked illumination is not copied; distant-light and local-translation appearance approximation, not measured relighting')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', required=True, choices=['gate_v7', 'gate_v8', 'gate_v9'])
    args = parser.parse_args()
    config = read(ROOT / args.gate / 'config.json')
    source_manifest = read(ROOT / 'inputs/manifest.json')
    cases = {c['case_id']: c for c in source_manifest['cases']}
    out = ROOT / args.gate / 'compositions_v3'
    out.mkdir(exist_ok=True)
    records = []
    cached = {}
    for job in config['jobs']:
        cid = job['case_id']
        assert cid in source_manifest['development_ids']
        candidates = list((ROOT / args.gate).glob('worker_*/'+job['id']+'/raw.png'))
        if not candidates:
            continue
        assert len(candidates) == 1
        if cid not in cached:
            geometry = ROOT / 'geometry_fracture_v1' / cid
            maps = np.load(Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929/geometry') / cid / 'maps.npz')
            k = maps['intrinsics'].copy()
            k[0] *= 640
            k[1] *= 480
            observed, audit = observed_surface_mask(geometry, k)
            cached[cid] = observed, audit
        observed, raster_audit = cached[cid]
        geometry = ROOT / 'geometry_fracture_v1' / cid
        original = np.asarray(Image.open(ROOT / 'inputs' / cid / 'source.png').convert('RGB'), float)
        base_path = ROOT / f'gate_v4/worker_1/{cid}__canny__41/composited.png'
        base = np.asarray(Image.open(base_path).convert('RGB'), float)
        global_path = ROOT / f'gate_v5/worker_1/{cid}__controlled_refine__41/raw.png'
        global_raw = np.asarray(Image.open(global_path).convert('RGB'), float)
        hole = binary_dilation(np.asarray(Image.open(geometry/'hole_mask.png')) > 0, iterations=12)
        hole_alpha = np.clip(distance_transform_edt(hole)/5, 0, 1)[..., None]
        # Global refinement owns only the source recess, so its displaced target
        # food cannot leave a second silhouette around the separately refined bite.
        base = base*(1-hole_alpha)+global_raw*hole_alpha
        local = ROOT / 'gate_v7' / cid
        box = read(local/'transform.json')['box']
        x, y, x1, y1 = box
        raw = np.asarray(Image.open(candidates[0]).convert('RGB').resize((x1-x, y1-y), Image.Resampling.LANCZOS), float)
        mask = np.asarray(Image.open(local/'full_mask.png')) > 0
        alpha = np.clip(distance_transform_edt(mask)/4, 0, 1)[..., None]
        canvas = base.copy()
        canvas[y:y1, x:x1] = raw
        result = base*(1-alpha)+canvas*alpha
        proxy = np.asarray(Image.open(geometry/'rgb_control.png').convert('RGB'), float)
        observed_alpha = np.clip(distance_transform_edt(observed)/1.5, 0, 1)[..., None]
        for anchor in [False, True]:
            final = result if not anchor else result*(1-observed_alpha)+proxy*observed_alpha
            final = np.uint8(np.clip(np.rint(final), 0, 255))
            edit = np.asarray(Image.open(ROOT/'fracture_controls_v1'/cid/'edit_mask.png')) > 0
            assert np.array_equal(final[~(edit | mask | hole)], original[~(edit | mask | hole)])
            dest = out/(job['id']+('__observed_anchor' if anchor else '__separate_layers'))
            dest.mkdir(exist_ok=True)
            Image.fromarray(final).save(dest/'composited.png')
            Image.fromarray(np.uint8(observed)*255).save(dest/'observed_mask.png')
            prep = cases[cid]['preprocessing']
            left, top = prep['pad_left_top']
            width, height = prep['resized']
            Image.fromarray(final).crop((left, top, left+width, top+height)).save(dest/'view.png')
            record = dict(id=dest.name, raw_output=False, outside_action_exact=True,
                          local_raw_sha256=sha(candidates[0]), global_raw_sha256=sha(global_path),
                          base_sha256=sha(base_path), raw_crop=box, observed_anchor=anchor,
                          source_raster_audit=raster_audit,
                          geometry_sha256=sha(geometry/'geometry_report.json'), script_sha256=sha(Path(__file__)))
            (dest/'result.json').write_text(json.dumps(record, indent=2)+'\n')
            records.append(record)
    (out/'manifest.json').write_text(json.dumps(dict(expected=2*len(config['jobs']),
        status='complete' if len(records) == 2*len(config['jobs']) else 'partial', rows=records), indent=2)+'\n')
    print(args.gate, len(records), flush=True)


if __name__ == '__main__':
    main()
