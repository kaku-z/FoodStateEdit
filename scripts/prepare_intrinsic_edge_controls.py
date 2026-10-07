"""Geometry-only visible contours and creases, excluding proxy material edges.

Source photographic edges remain outside the action. Within new surfaces, an
RGB shader's tone boundaries must not masquerade as geometric creases.
"""
import json
from pathlib import Path
import hashlib
import time

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
import trimesh

from collect_structure_development import ROOT, read, sha


def rasterize(geometry, intrinsics):
    depth = np.full((480, 640), np.inf)
    normals = np.zeros((480, 640, 3), np.float32)
    labels = np.zeros((480, 640), np.uint8)
    for label, name in enumerate(['remaining', 'bite_lifted', 'fork'], 1):
        mesh = trimesh.load(geometry/(name+'.ply'), process=False)
        vertices = np.asarray(mesh.vertices)
        projected = vertices @ intrinsics.T
        uv = projected[:, :2]/projected[:, 2:]
        for index, face in enumerate(mesh.faces):
            p, z = uv[face], vertices[face, 2]
            if np.any(z <= 0):
                raise ValueError('Geometry crosses camera plane')
            low = np.maximum(np.floor(p.min(0)).astype(int), [0, 0])
            high = np.minimum(np.ceil(p.max(0)).astype(int), [639, 479])
            if np.any(high < low):
                continue
            x, y = np.meshgrid(np.arange(low[0], high[0]+1)+.5, np.arange(low[1], high[1]+1)+.5)
            denominator = (p[1,1]-p[2,1])*(p[0,0]-p[2,0])+(p[2,0]-p[1,0])*(p[0,1]-p[2,1])
            if abs(denominator) < 1e-10:
                continue
            a = ((p[1,1]-p[2,1])*(x-p[2,0])+(p[2,0]-p[1,0])*(y-p[2,1]))/denominator
            b = ((p[2,1]-p[0,1])*(x-p[2,0])+(p[0,0]-p[2,0])*(y-p[2,1]))/denominator
            c = 1-a-b
            inverse = a/z[0]+b/z[1]+c/z[2]
            znew = np.divide(1., inverse, out=np.full_like(inverse, np.inf), where=inverse > 0)
            region = np.s_[low[1]:high[1]+1, low[0]:high[0]+1]
            zold = depth[region]
            selected = (a >= -1e-7) & (b >= -1e-7) & (c >= -1e-7) & (znew < zold)
            zold[selected] = znew[selected]
            normals[region][selected] = mesh.face_normals[index]
            labels[region][selected] = label
    return depth, normals, labels


def main():
    out = ROOT/'intrinsic_edge_controls_v1'
    out.mkdir(exist_ok=False)
    sources = read(ROOT/'inputs/manifest.json')
    rows = []
    for case in sources['cases']:
        cid = case['case_id']
        # Development preserves the gate_v4 geometry and fork exactly. Future
        # sources use the already frozen frame-exiting handle geometry.
        g = ROOT/('geometry_fracture_v1' if cid in sources['development_ids'] else 'geometry_framed_v1')/cid
        mp = (Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929/geometry')/cid/'maps.npz'
              if cid in sources['development_ids'] else ROOT/'geometry_v3'/cid/'maps.npz')
        maps = np.load(mp)
        k = maps['intrinsics'].copy()
        k[0] *= 640
        k[1] *= 480
        depth, normal, labels = rasterize(g, k)
        report = read(g/'geometry_report.json')
        length = max(np.array(report['fit']['high'])-np.array(report['fit']['low']))
        geometric = np.zeros((480, 640), bool)
        crease = np.cos(np.deg2rad(55))
        for axis in [0, 1]:
            a = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
            b = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
            both = (labels[a] > 0) & (labels[b] > 0)
            boundary = labels[a] != labels[b]
            angle = np.sum(normal[a]*normal[b], axis=-1) < crease
            delta = np.zeros_like(depth[a])
            np.subtract(depth[a], depth[b], out=delta, where=both)
            discontinuity = np.abs(delta) > .02*length
            geometric[a] |= boundary | (both & (angle | discontinuity))
        source = np.asarray(Image.open(ROOT/'inputs'/cid/'source.png').convert('RGB'))
        proxy = np.asarray(Image.open(g/'rgb_control.png').convert('RGB'))
        original_edges = cv2.Canny(source, 80, 160)
        shader_edges = cv2.Canny(proxy, 80, 160)
        edit = np.zeros((480, 640), bool)
        for name in ['hole_mask', 'material_mask', 'rigid_mask']:
            edit |= np.asarray(Image.open(g/(name+'.png'))) > 0
        edit = binary_dilation(edit, iterations=12)
        control = original_edges.copy()
        control[edit] = np.uint8(geometric[edit])*255
        # Only geometric contours enter new surfaces in this controlled test.
        # Photo texture is supplied by the source, rather than by shader edges.
        folder = out/cid
        folder.mkdir()
        for name, array in [('intrinsic', control), ('rgb_proxy_canny', shader_edges),
                            ('source_canny', original_edges), ('geometric', np.uint8(geometric)*255),
                            ('edit_mask', np.uint8(edit)*255)]:
            Image.fromarray(array).save(folder/(name+'.png'))
        np.savez_compressed(folder/'raster_geometry.npz', depth=depth, normal=normal, labels=labels)
        bite = np.asarray(Image.open(g/'bite_mask.png')) > 0
        visible = labels == 2
        iou = np.sum(visible & bite)/np.sum(visible | bite)
        assert iou > .94, (cid, iou)
        assert np.array_equal(control[~edit], original_edges[~edit])
        row = dict(case_id=cid, cpu_gpu_bite_silhouette_iou=float(iou), crease_degrees=55,
                   depth_jump_over_extent=.02, source_edges_outside_action_exact=True,
                   changed_control_pixels_inside_action=int(np.sum((control != shader_edges) & edit)),
                   geometry_report_sha256=sha(g/'geometry_report.json'),
                   files={p.name:sha(p) for p in folder.iterdir()})
        (folder/'audit.json').write_text(json.dumps(row, indent=2)+'\n')
        rows.append(row)
        print(cid, 'READY', round(iou, 4), flush=True)
    (out/'manifest.json').write_text(json.dumps(dict(status='complete', rows=rows,
        created_unix=time.time(), script_sha256=sha(Path(__file__))), indent=2)+'\n')


if __name__ == '__main__':
    main()
