"""Apply the frozen block-food geometry method to held-from-development sources."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import zipfile

ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
REPAIR=Path('/host/space0/guo-z/tf-ufi/food3d_repair_20260928')
os.environ.update(CUDA_VISIBLE_DEVICES='0',EGL_DEVICE_ID='0',PYOPENGL_PLATFORM='egl',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4')
sys.path[:0]=[str(REPAIR/'compat_packages'),str(REPAIR),str(REPAIR/'moge'),str(REPAIR/'utils3d_moge')]
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
import torch
import trimesh
import structured_geometry_helper as geo
from moge.model.v2 import MoGeModel


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def save_array(folder,name,array):
    if array.dtype==bool:array=np.uint8(array)*255
    Image.fromarray(array).save(folder/(name+'.png'))


def geometry(case,folder,maps):
    source_path=ROOT/'inputs'/case['case_id']/'source.png'
    source=np.asarray(Image.open(source_path).convert('RGB'));truth=np.asarray(Image.open(source_path.with_name('food_mask.png')))>0
    prep=case['preprocessing'];geo.VALID_IMAGE_RECT=tuple(prep['pad_left_top']+prep['resized']);geo.FOOD_MASK_FOR_PLANNING=truth
    geo.ANCHORS=[tuple(p) for p in case['anchors']]
    R,lo,hi,K,fit,pred=geo.fit_block(maps,truth)
    full,remain,bite,lifted,fork,translation,cut=geo.plan_bite(R,lo,hi,K)
    solids={'full':full,'remaining':remain,'bite_source':bite,'bite_lifted':lifted,'fork':fork}
    renderer=geo.Renderer(K);masks={};depths={};meshes={}
    for name,solid in solids.items():
        mesh=geo.camera_mesh(solid,R);mesh.export(folder/(name+'.ply'));meshes[name]=mesh
        renderer.add(mesh,[.7,.7,.7,1.]);_,depth=renderer.render();renderer.clear();masks[name]=depth>0;depths[name]=depth
    renderer.add(meshes['remaining'],[.75,.8,.83,1.]);renderer.add(meshes['bite_lifted'],[.97,.53,.19,1.]);renderer.add(meshes['fork'],[.36,.41,.47,1.]);clay,_=renderer.render();renderer.close()
    save_array(folder,'geometry',clay)
    tr=geo.texture_mesh(remain,R,lo,hi,K,source);tb=geo.texture_mesh(lifted,R,lo,hi,K,source,translation)
    renderer=geo.Renderer(K,neutral=False);renderer.add(tr);renderer.add(tb);texture,depth=renderer.render();renderer.close()
    # Separate metal illumination; the food still uses source-projected texture.
    renderer=geo.Renderer(K,neutral=True)
    renderer.scene.ambient_light=np.array([.12,.12,.12])
    import pyrender
    from scipy.spatial.transform import Rotation
    for angles,intensity in [([30,25,0],2.8),([-40,-60,10],1.8),([0,80,0],1.2)]:
        pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('xyz',angles,degrees=True).as_matrix()
        renderer.scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=intensity),pose=pose)
    metal=meshes['fork'].copy();metal.vertices=np.asarray(metal.vertices)*np.array([1,-1,-1])
    material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[.8,.82,.85,1],metallicFactor=.95,roughnessFactor=.20)
    renderer.scene.add(pyrender.Mesh.from_trimesh(metal,material=material,smooth=True))
    metal_rgb,metal_depth=renderer.render();renderer.close()
    texture=texture.copy();depth=depth.copy()
    visible=(metal_depth>0)&((depth==0)|(metal_depth<depth))
    texture[visible]=metal_rgb[visible];depth[visible]=metal_depth[visible]
    save_array(folder,'visible_fork_mask',visible)
    save_array(folder,'metal_render',metal_rgb)
    uncovered=truth&(depth==0);background=cv2.inpaint(source,np.uint8(uncovered)*255,3,cv2.INPAINT_TELEA)
    composite=background.copy();composite[depth>0]=texture[depth>0]
    hole=binary_dilation(masks['bite_source'],iterations=7)
    material=binary_dilation(masks['bite_lifted'],iterations=4)
    rigid=binary_dilation(masks['fork'],iterations=4)
    contact=binary_dilation(masks['fork'],iterations=7)&binary_dilation(masks['bite_lifted'],iterations=7)
    edit=binary_dilation(hole|material|rigid,iterations=12)
    rgb=source.copy();rgb[edit]=composite[edit]
    blank=source.copy();blank[edit]=127
    # The 2-D comparator receives exactly the same source footprint, target
    # footprint and utensil layout. It lacks volume-derived cut faces / views.
    # Therefore it is an ablation of 3-D food rendering, not a geometry-free
    # annotation baseline or an independently planned 2-D action.
    sy,sx=np.where(masks['bite_source']);ty,tx=np.where(masks['bite_lifted'])
    fx=(tx.max()-tx.min())/max(1,sx.max()-sx.min());fy=(ty.max()-ty.min())/max(1,sy.max()-sy.min())
    affine=np.array([[fx,0,tx.min()-fx*sx.min()],[0,fy,ty.min()-fy*sy.min()]],np.float32)
    patch=cv2.warpAffine(source,affine,(640,480));alpha=cv2.warpAffine(np.uint8(masks['bite_source'])*255,affine,(640,480))>127
    planar=cv2.inpaint(source,np.uint8(hole)*255,3,cv2.INPAINT_TELEA)
    planar[masks['fork']]=texture[masks['fork']];planar[alpha]=patch[alpha]
    newdepth=maps['depth'].copy()
    for name in ['remaining','bite_lifted','fork']:
        d=depths[name];valid=d>0;newdepth[valid]=np.minimum(newdepth[valid],d[valid]) if name!='remaining' else d[valid]
    dl,dh=np.quantile(newdepth,[.01,.99]);gray=np.uint8(255*np.clip((dh-newdepth)/(dh-dl),0,1))
    dc=source.copy();dc[edit]=np.repeat(gray[...,None],3,axis=-1)[edit]
    for name,array in [('source',source),('rgb_control',rgb),('planar_control',planar),('masked_control',blank),('depth_control',dc),
        ('edit_mask',edit),('hole_mask',hole),('material_mask',material),('rigid_mask',rigid),('contact_mask',contact),
        ('bite_mask',masks['bite_lifted']),('fork_mask',masks['fork']),('source_bite_mask',masks['bite_source']),('food_mask',truth),('uncovered_mask',uncovered)]:save_array(folder,name,array)
    scene=trimesh.Scene();scene.add_geometry(tr,node_name='remaining');scene.add_geometry(tb,node_name='bite');scene.add_geometry(meshes['fork'],node_name='fork');scene.export(folder/'scene.glb')
    record={'case_id':case['case_id'],'status':'geometry_ready','fit':fit,'cut_and_support':cut,
        'source_uv':geo.project(meshes['bite_source'].center_mass[None,:],K)[0].tolist(),
        'target_uv':geo.project(meshes['bite_lifted'].center_mass[None,:],K)[0].tolist(),
        'uncertainty_flag':max(fit['normal_agreement_degrees'])>20 or fit['source_silhouette_iou']<.8,
        'cut_material':'inferred directional tones; hidden surface not measured',
        'controls_scope':'Planar and 3-D controls share source/target annotations and projected fork. No output compositor is used in the primary generator score.',
        'files':{p.name:{'path':str(p),'sha256':sha(p)} for p in folder.iterdir() if p.is_file()}}
    (folder/'geometry_report.json').write_text(json.dumps(record,indent=2)+'\n');return record

