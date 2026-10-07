"""Source-preserving intrinsic geometry channels with explicit revealed-plane prior."""
import hashlib,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation,distance_transform_edt
import trimesh
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def rasterize(geometry, intrinsics, names=("remaining", "bite_lifted", "fork")):
    depth = np.full((480, 640), np.inf)
    normals = np.zeros((480, 640, 3), np.float32)
    labels = np.zeros((480, 640), np.uint8)
    for label, name in enumerate(names, 1):
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
    out=ROOT/'spoon_geometry_channels_v2';out.mkdir(exist_ok=False)
    (out/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    cases=json.loads((ROOT/'inputs/manifest.json').read_text())['cases'];rows=[]
    for c in cases:
        cid=c['case_id'];g=ROOT/'geometry_spoon_photo_v1'/cid;d=out/cid;d.mkdir()
        mp=Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929/geometry')/cid/'maps.npz' if cid.startswith('new_') else ROOT/'geometry_v3'/cid/'maps.npz'
        maps=np.load(mp);K=maps['intrinsics'].copy();K[0]*=640;K[1]*=480
        z,n,labels=rasterize(g,K);fullz,_,_=rasterize(g,K,['full'])
        remaining_z,_,_=rasterize(g,K,['remaining'])
        report=json.loads((g/'geometry_report.json').read_text())
        R=np.asarray(report['fit']['axes_camera_columns']);lo=np.asarray(report['fit']['low']);L=max(np.asarray(report['fit']['high'])-lo)
        geometric=np.zeros((480,640),bool)
        for axis in [0,1]:
            a=(slice(None,-1),slice(None)) if axis==0 else (slice(None),slice(None,-1))
            b=(slice(1,None),slice(None)) if axis==0 else (slice(None),slice(1,None))
            both=(labels[a]>0)&(labels[b]>0);edge=labels[a]!=labels[b]
            angle=np.sum(n[a]*n[b],axis=-1)<np.cos(np.deg2rad(55))
            delta=np.zeros_like(z[a]);np.subtract(z[a],z[b],out=delta,where=both)
            geometric[a]|=edge|(both&(angle|(np.abs(delta)>.02*L)))
        source=np.asarray(Image.open(g/'source.png').convert('RGB'))
        proxy=np.asarray(Image.open(g/'rgb_control.png').convert('RGB'))
        edit=np.asarray(Image.open(g/'edit_mask.png'))>0
        intrinsic=cv2.Canny(source,80,160);intrinsic[edit]=np.uint8(geometric[edit])*255
        raw_depth=np.asarray(maps['depth']);valid=np.isfinite(raw_depth)&(raw_depth>0)&maps['mask']
        nearest=distance_transform_edt(~valid,return_distances=False,return_indices=True)
        source_z=raw_depth[tuple(nearest)];edited=source_z.copy();touched=np.zeros((480,640),bool)
        cut=binary_dilation(np.asarray(Image.open(g/'source_bite_mask.png'))>0,iterations=2)
        remaining=np.isfinite(remaining_z)&cut;edited[remaining]=remaining_z[remaining];touched|=remaining
        # Deleted source foreground must reveal a support-plane hypothesis,
        # rather than silently retain the original food's depth.
        yy,xx=np.indices((480,640));rays=np.stack([xx,yy,np.ones_like(xx)],axis=-1)@np.linalg.inv(K).T
        denominator=rays@R[:,2];plate=np.divide(lo[2],denominator,out=np.full((480,640),np.nan),where=np.abs(denominator)>1e-9)
        revealed=np.isfinite(fullz)&~np.isfinite(remaining_z)&cut
        admissible=revealed&np.isfinite(plate)&(plate>0)
        assert not np.any(revealed&~admissible),(cid,'Invalid revealed support-plane depth')
        edited[admissible]=plate[admissible];touched|=admissible
        for label in [2,3]:
            shown=(labels==label)&(z<edited);edited[shown]=z[shown];touched|=shown
        assert np.isfinite(edited).all() and (edited>0).all()
        assert np.array_equal(edited[~touched],source_z[~touched])
        inv=1/edited;low,high=np.quantile(inv,[.005,.995])
        gray=np.uint8(np.clip((inv-low)/(high-low),0,1)*255)
        bite=np.asarray(Image.open(g/'bite_mask.png'))>0;visible=labels==2
        iou=float(np.sum(bite&visible)/np.sum(bite|visible));assert iou>.94,(cid,iou)
        for name,arr in [('intrinsic',intrinsic),('photo_canny',cv2.Canny(proxy,80,160)),
                         ('grayscale',cv2.cvtColor(proxy,cv2.COLOR_RGB2GRAY)),('depth',gray),
                         ('edit_mask',np.uint8(edit)*255),('revealed_support_plane',np.uint8(admissible)*255)]:
            Image.fromarray(arr).save(d/(name+'.png'))
        np.savez_compressed(d/'geometry_channels.npz',scene_depth=z,scene_normals=n,labels=labels,full_depth=fullz,remaining_depth=remaining_z,source_depth_filled=source_z,edited_depth=edited)
        row={'case_id':cid,'cpu_gpu_bite_silhouette_iou':iou,'outside_action_depth_exact':True,
             'revealed_plane_pixels':int(np.sum(admissible)),'revealed_support_plane':'Food-frame inferred base height with inferred upward axis; no measured plate plane',
             'finite_positive_depth':True,'source_depth_invalid_fraction':float(np.mean(~valid)),
             'source_geometry_report_sha256':sha(g/'geometry_report.json'),'normal_field_usage':'Diagnostic only; no normal-conditioned generation claimed',
             'files':{p.name:sha(p) for p in d.iterdir()}}
        (d/'audit.json').write_text(json.dumps(row,indent=2));rows.append(row);print(cid,'READY',iou,flush=True)
    (out/'manifest.json').write_text(json.dumps({'status':'complete','cases':rows,'script_sha256':sha(Path(__file__))},indent=2))

if __name__=='__main__':main()

