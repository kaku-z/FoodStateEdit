"""Render actual edited camera depth with finite-only scaling for control models."""
import hashlib
import json
import os
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='0',EGL_DEVICE_ID='0',PYOPENGL_PLATFORM='egl')
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation,distance_transform_edt
import trimesh
import structured_geometry_helper as geo

ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
OLD=Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=ROOT/'depth_controls_v2';out.mkdir(exist_ok=False);rows=[]
    sources=json.loads((ROOT/'inputs/manifest.json').read_text())
    for c in sources['cases']:
        cid=c['case_id'];g=ROOT/'geometry_v3'/cid;d=out/cid;d.mkdir()
        mp=(OLD/'geometry'/cid/'maps.npz') if cid.startswith('new_') else (g/'maps.npz')
        maps=np.load(mp);K=maps['intrinsics'].copy();K[0]*=640;K[1]*=480
        original=maps['depth'].copy();valid=np.isfinite(original)&(original>0)&maps['mask']
        nearest=distance_transform_edt(~valid,return_distances=False,return_indices=True)
        filled=original[tuple(nearest)];edited=filled.copy();depths={}
        renderer=geo.Renderer(K)
        for n in ['full','remaining','bite_lifted','fork']:
            mesh=trimesh.load(g/(n+'.ply'),process=False);renderer.add(mesh,[.7,.7,.7,1.]);_,depth=renderer.render();renderer.clear();depths[n]=depth.copy()
        renderer.close()
        # Remaining geometry owns the food depth; lifted pieces then use depth
        # testing against the scene. No RGB or shadow is used as geometric depth.
        cut_region=binary_dilation(np.asarray(Image.open(g/'source_bite_mask.png'))>0,iterations=2)
        remaining=(depths['remaining']>0)&cut_region;edited[remaining]=depths['remaining'][remaining]
        touched=remaining.copy()
        for n in ['bite_lifted','fork']:
            z=depths[n];m=(z>0)&(z<edited);edited[m]=z[m];touched|=m
        assert np.array_equal(edited[~touched],filled[~touched])
        assert np.isfinite(edited).all() and (edited>0).all()
        inverse=1/edited;lo,hi=np.quantile(inverse,[.005,.995]);gray=np.uint8(255*np.clip((inverse-lo)/(hi-lo),0,1))
        original_inv=1/filled;orig_gray=np.uint8(255*np.clip((original_inv-lo)/(hi-lo),0,1))
        proxy=np.asarray(Image.open(g/'rgb_control.png').convert('RGB'))
        canny=cv2.Canny(proxy,80,160)
        edit=np.zeros_like(gray,dtype=bool)
        for n in ['hole_mask','material_mask','rigid_mask']:edit|=np.asarray(Image.open(g/(n+'.png')))>0
        edit=binary_dilation(edit,iterations=12)
        for n,arr in [('depth',gray),('source_depth',orig_gray),('canny',canny),('edit_mask',np.uint8(edit)*255)]:Image.fromarray(arr).save(d/(n+'.png'))
        # Numerical depth retained independently of the viewable normalized map.
        np.savez_compressed(d/'depth_geometry.npz',source_filled=filled,edited=edited,valid_source=valid,**depths)
        bite=(np.asarray(Image.open(g/'bite_mask.png'))>0);cut=(np.asarray(Image.open(g/'source_bite_mask.png'))>0)
        changed=np.abs(edited-filled)>1e-5
        row={'case_id':cid,'source_invalid_fraction':float((~valid).mean()),'invalid_fill':'Nearest valid source depth, control input only; no measured geometry claimed',
             'normalization':'Global finite inverse depth, 0.5/99.5 percentiles; brighter is nearer',
             'scale_quantiles':[float(lo),float(hi)],'bite_changed_fraction':float(changed[bite].mean()),'cut_changed_fraction':float(changed[cut].mean()),
             'finite_edited_depth':True,'outside_action_depth_exact':True,'cut_depth_delta_median':float(np.median((edited-filled)[cut])),
             'source_geometry_report_sha256':sha(g/'geometry_report.json'),
             'files':{p.name:sha(p) for p in d.iterdir()}}
        (d/'audit.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
        print(cid,'CONTROL_READY',flush=True)
    (out/'manifest.json').write_text(json.dumps({'cases':rows,'script_sha256':sha(Path(__file__)),'status':'complete'},indent=2)+'\n')

if __name__=='__main__':main()
