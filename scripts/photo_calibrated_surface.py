"""Source-calibrated continuous appearance for inferred fresh food surfaces.

This is an RGB appearance regression with an explicit smooth-light prior, not
measured albedo, calibrated illumination or recovered hidden texture. Observed
inherited surfaces retain source UVs; new surfaces use source texture statistics.
"""
import hashlib,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial import ConvexHull
import trimesh
import structured_geometry_helper as geo

STATE={}

def configure(case,maps,folder):
    STATE.clear();STATE.update(case=case,maps=maps,folder=folder)

def calibrate(source,R,lo,hi):
    cid=STATE['case']['case_id'];maps=STATE['maps']
    root=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
    selection=json.loads((root/'gate_v13_run03'/cid/'reference_selection.json').read_text())
    box=selection['material_box_canvas'];x0,y0,x1,y1=box
    patch=source[y0:y1,x0:x1].astype(float)/255.
    seed=np.median(patch.reshape(-1,3),axis=0);seed_chroma=seed/seed.sum()
    rgb=source.astype(float)/255.;chroma=rgb/np.maximum(rgb.sum(2)[...,None],1e-6)
    normals=np.asarray(maps['normal'],float)
    valid=binary_erosion(geo.FOOD_MASK_FOR_PLANNING,iterations=3)&maps['mask']
    valid&=np.isfinite(normals).all(2)&(np.linalg.norm(chroma-seed_chroma,axis=2)<.085)
    valid&=rgb.mean(2)>max(.28,.72*seed.mean())
    full=geo.mesh_of(geo.solid_box(lo,hi));planes=ConvexHull(full.vertices).equations
    plane_normals=np.unique(np.round(planes[:,:3],7),axis=0)@R.T
    center=(lo+hi)/2@R.T
    visible=plane_normals@(-center)>0
    plane_normals=plane_normals[visible]
    nn=[];color=[];counts=[]
    for normal in plane_normals:
        chosen=valid&(normals@normal>.945)
        if np.sum(chosen)<20:continue
        values=rgb[chosen]
        # Median rejects residual toppings and dark boundary pixels.
        nn.append(normal);color.append(np.median(values,axis=0));counts.append(int(np.sum(chosen)))
    if not nn:nn=[np.array([0.,0.,-1.])];color=[seed];counts=[0]
    X=np.c_[np.ones(len(nn)),np.asarray(nn)];Y=np.asarray(color)
    regularizer=np.diag([.015,.12,.12,.12]);prior=np.zeros((4,3));prior[0]=np.median(Y,axis=0)
    coefficient=np.linalg.solve(X.T@X+regularizer,X.T@Y+regularizer@prior)
    fitted=X@coefficient
    report={'source_only':True,'model':'RGB affine directional appearance with ridge toward ambient-only lighting',
        'source_material_box_canvas':box,'observed_plane_normals_camera':np.asarray(nn).tolist(),
        'plane_sample_counts':counts,'plane_median_rgb':(Y*255).tolist(),
        'coefficient_rgb':coefficient.tolist(),'fit_mean_absolute_rgb_error':float(np.mean(np.abs(fitted-Y))*255),
        'illumination_underconstrained':len(nn)<3,'material_reference_low_observability':selection['material_reference_low_observability'],
        'scope':'Visible source RGB fit and source high-frequency statistics. Hidden appearance and cavity occlusion remain priors; no measured relighting claim.'}
    (STATE['folder']/'appearance_calibration.json').write_text(json.dumps(report,indent=2))
    STATE.update(coefficient=coefficient,patch=patch,planes=planes)

def texture(solid,R,lo,hi,K,source,cutter,orientation=None,cavity=False):
    if 'coefficient' not in STATE:calibrate(source,R,lo,hi)
    orientation=np.eye(3) if orientation is None else orientation
    m=geo.mesh_of(solid)
    v,f=trimesh.remesh.subdivide_to_size(m.vertices,m.faces,max_edge=max(hi-lo)*.025,max_iter=8)
    m=trimesh.Trimesh(v,f,process=False)
    fc=m.triangles_center;fn=m.face_normals;verts=m.triangles.reshape(-1,3)
    planes=STATE['planes']
    inherited=np.any((np.abs(fc@planes[:,:3].T+planes[:,3])<max(hi-lo)*1e-5)&(fn@planes[:,:3].T>.999),axis=1)
    camera_visible=np.einsum('ij,ij->i',fn@R.T,-fc@R.T)>0
    seen=inherited&camera_visible
    fresh_normal=np.repeat(fn,3,axis=0)
    center=np.asarray(cutter['center']);radius=cutter['radius'];lower=cutter['lower'];upper=cutter['upper']
    q=fc-center;rz=np.where(q[:,2]>=0,upper,lower)
    implicit=(q[:,0]**2+q[:,1]**2)/radius**2+q[:,2]**2/rz**2
    curved=(~inherited)&(np.abs(implicit-1)<.035)
    for face in np.flatnonzero(curved):
        a=verts[3*face:3*face+3]-center
        zradius=np.where(a[:,2]>=0,upper,lower)
        gradient=np.c_[a[:,0]/radius**2,a[:,1]/radius**2,a[:,2]/zradius**2]
        gradient/=np.maximum(np.linalg.norm(gradient,axis=1)[:,None],1e-12)
        sign=1. if np.mean(gradient,axis=0)@fn[face]>=0 else -1.
        fresh_normal[3*face:3*face+3]=gradient*sign
    normal_camera=fresh_normal@orientation.T@R.T
    assert np.isfinite(normal_camera).all()
    texsize=512;H=max(512,source.shape[0]);W=source.shape[1]+texsize
    atlas=np.full((H,W,3),np.uint8(STATE['patch'].mean((0,1))*255),np.uint8)
    atlas[:source.shape[0],:source.shape[1]]=source
    # A normal-coordinate chart provides continuous directional appearance.
    # Its azimuth spans two turns so individual faces can unwrap at the seam.
    yy,xx=np.indices((texsize,texsize))
    phi=(xx/(texsize-1)*4*np.pi)-2*np.pi
    elevation=(yy/(texsize-1)*np.pi)-np.pi/2
    normals_chart=np.stack([np.cos(elevation)*np.cos(phi),np.cos(elevation)*np.sin(phi),np.sin(elevation)],axis=-1)
    predicted=np.concatenate([np.ones((texsize,texsize,1)),normals_chart],axis=2)@STATE['coefficient']
    patch=STATE['patch'].astype(np.float32)
    detail=patch-cv2.GaussianBlur(patch,(0,0),2.)
    detail=cv2.resize(detail,(texsize,texsize),interpolation=cv2.INTER_LINEAR)
    detail=np.clip(detail,-3/255.,3/255.)
    predicted=np.clip(predicted,.15,1.)+detail
    if cavity:predicted*=.94
    atlas[:texsize,source.shape[1]:]=np.uint8(np.clip(np.rint(predicted*255),0,255))
    uv=geo.project(verts@R.T,K)
    uv[:,0]/=W;uv[:,1]=1-uv[:,1]/H
    phi_vertices=np.arctan2(normal_camera[:,1],normal_camera[:,0]).reshape(-1,3)
    for i in range(len(phi_vertices)):
        mean_angle=np.arctan2(np.mean(normal_camera[3*i:3*i+3,1]),np.mean(normal_camera[3*i:3*i+3,0]))
        phi_vertices[i]=mean_angle+np.angle(np.exp(1j*(phi_vertices[i]-mean_angle)))
    px=(phi_vertices.reshape(-1)+2*np.pi)/(4*np.pi)*(texsize-1)
    py=(np.arctan2(normal_camera[:,2],np.linalg.norm(normal_camera[:,:2],axis=1))+np.pi/2)/np.pi*(texsize-1)
    hidden=~np.repeat(seen,3)
    uv[hidden,0]=(source.shape[1]+px[hidden])/W
    uv[hidden,1]=1-py[hidden]/H
    assert np.isfinite(uv).all()
    visual=trimesh.visual.texture.TextureVisuals(uv=uv,material=trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(atlas),baseColorFactor=[1.]*4,metallicFactor=0,roughnessFactor=1))
    return trimesh.Trimesh(verts@R.T,np.arange(len(verts)).reshape(-1,3),visual=visual,process=False)
