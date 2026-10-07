"""Prepare a source-bound development gate; formal images are not rendered here."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_complete_20260929')
REPAIR=Path('/host/space0/guo-z/tf-ufi/food3d_repair_20260928')
sys.path[:0]=[str(REPAIR/'compat_packages'),str(REPAIR)]
os.environ['PYOPENGL_PLATFORM']='egl'
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
import trimesh
from repair_food3d_first_bite import Renderer


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(16*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    folder=ROOT/'pilot/dev_7489_controls';folder.mkdir(exist_ok=False)
    old=REPAIR/'results/repair_v5'
    source=np.asarray(Image.open(REPAIR/'inputs/source.jpg').convert('RGB'))
    proxy=np.asarray(Image.open(old/'source_texture_composite.png').convert('RGB'))
    maps=np.load(REPAIR/'results/depth_v1/maps.npz')
    K=maps['intrinsics'].copy();K[0]*=640;K[1]*=480
    renderer=Renderer(K);masks={};depths={}
    for name in ['bite_source','bite_lifted','fork','remaining']:
        renderer.add(trimesh.load(old/(name+'.ply'),force='mesh',process=False),[.7,.7,.7,1.0]);_,depth=renderer.render()
        masks[name]=depth>0;depths[name]=depth;renderer.clear()
    renderer.close()
    hole=binary_dilation(masks['bite_source'],iterations=7)
    material=binary_dilation(masks['bite_lifted'],iterations=4)
    rigid=binary_dilation(masks['fork'],iterations=4)
    contact=binary_dilation(masks['fork'],iterations=7)&binary_dilation(masks['bite_lifted'],iterations=7)
    edit=binary_dilation(hole|material|rigid,iterations=12)
    rgb=source.copy();rgb[edit]=proxy[edit]
    blank=source.copy();blank[edit]=127
    newdepth=maps['depth'].copy()
    for name in ['remaining','bite_lifted','fork']:
        d=depths[name];valid=d>0
        if name=='remaining':newdepth[valid]=d[valid]
        else:newdepth[valid]=np.minimum(newdepth[valid],d[valid])
    dlo,dhi=np.quantile(newdepth,[.01,.99]);gray=np.uint8(255*np.clip((dhi-newdepth)/(dhi-dlo),0,1));depthcontrol=source.copy();depthcontrol[edit]=np.repeat(gray[...,None],3,axis=-1)[edit]
    for name,array in [('source',source),('rgb_control',rgb),('masked_control',blank),('depth_control',depthcontrol),('edit_mask',edit),('hole_mask',hole),('material_mask',material),('rigid_mask',rigid),('contact_mask',contact)]:
        if array.dtype==bool:array=np.uint8(array)*255
        Image.fromarray(array).save(folder/(name+'.png'))
    files={p.stem:{'path':str(p),'sha256':sha(p)} for p in folder.glob('*.png')}
    runtime=json.loads((ROOT/'vace_runtime.json').read_text())
    start=time.time();checks={}
    for name,expected in runtime['model_files'].items():
        path=Path(runtime['model_root'])/name
        checks[name]=path.stat().st_size==expected['size_bytes'] and sha(path)==expected['sha256']
        print('AUDITED',name,checks[name],flush=True)
    assert all(checks.values())
    audit={'all_verified':True,'expected_model_files':runtime['model_files'],'checks':checks,'seconds':time.time()-start}
    (ROOT/'vace_model_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    prompt=('A stationary close-up photographic view of this exact block of cold tofu with brown soy sauce on the same blue and white patterned plate. '
        'Exactly one small bite of the same tofu is supported on the four tines of one stainless steel eating fork in the upper left of the image. '
        'The fork and bite are clearly suspended in air above the plate and above the original tofu top, with open space below. '
        'The fork handle extends out of the left frame edge. A matching small fresh cut notch is visible at the front corner of the original tofu. '
        'The same food, sauce texture, plate pattern, camera viewpoint and background remain recognizable. '
        'Natural moist tofu texture on the exposed interior, rounded slightly irregular cut edges, believable soft contact shadows and reflective metal. '
        'No person is visible. The pose stays completely still throughout all frames. Photorealistic food photography.')
    jobs=[]
    variants=[('A_reference','masked_control',None),('C_rgb3d','rgb_control',None),('C_depth3d','depth_control',None),
              ('D_staged3d','rgb_control',{'rigid_end':24,'contact_end':21,'material_end':12,'hole_end':12})]
    for method,control,ttm in variants:
        jobs.append({'id':'dev_7489__'+method+'__20260929','case_id':'dev_7489','method':method,'control':control,
                     'seed':20260929,'prompt':prompt,'files':files,'ttm':ttm})
    config={'stage':'development_backend_gate','not_formal':True,'jobs':jobs,'runtime':runtime,'model_audit':str(ROOT/'vace_model_audit.json'),
        'cache_root':str(ROOT/'vace_cache'),'inference':{'num_frames':9,'selected_frame':4,'steps':30,'cfg_scale':5.0,'vace_scale':1.0,
        'negative_prompt':'hand, arm, fingers, person, face, mouth, extra fork, extra utensil, floating unsupported food, missing fork, flat drawing, cartoon, plastic food, painted textures, text, watermark, blur, camera movement, motion'},
        'selection_rule':'All four gates retained and reviewed. Choose a rendering integration using only this old development image before formal protocol freeze.'}
    (ROOT/'pilot/vace_gate_v1.json').write_text(json.dumps(config,indent=2)+'\n')
    print('PILOT_CONTROLS_READY',flush=True)


if __name__=='__main__':main()
