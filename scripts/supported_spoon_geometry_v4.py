"""Shared ellipsoidal scoop and conforming metal spoon, source-only planning.

Friction and hidden shape are explicit priors, not measured eating physics.
The same Boolean cutter defines the source recess and the carried food portion.
"""
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import manifold3d as manifold
from scipy.ndimage import distance_transform_edt,map_coordinates
from scipy.optimize import nnls
from scipy.spatial.transform import Rotation
import build_structured_geometry_v3 as base
import structured_geometry_helper as geo
import source_frustum_geometry as frustum
from structured_fork_framed import loft

ROOT=base.ROOT
OLD=Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929')
ORIGINAL_TEXTURE_BASE=geo.texture_mesh
ORIGINAL_TEXTURE=geo.texture_mesh
LAST={}

def equilibrium(bite,cutter,extent):
    """Nonnegative force balance on a discretized Coulomb contact cone."""
    m=geo.mesh_of(bite);centers=m.triangles_center;normal=m.face_normals
    cut=geo.mesh_of(cutter)
    # Cutter is convex. Inherited cutter faces share its supporting planes.
    from scipy.spatial import ConvexHull
    planes=ConvexHull(cut.vertices).equations
    distances=centers@planes[:,:3].T+planes[:,3]
    align=normal@planes[:,:3].T
    contact=np.any((np.abs(distances)<extent*1e-6)&(align>.999),axis=1)&(normal[:,2]<-.05)
    p=centers[contact];n=-normal[contact]
    p=p[::max(1,len(p)//160)];n=n[::max(1,len(n)//160)]
    assert len(p)>=8
    cm=m.center_mass
    t1=np.cross(n,np.tile([0.,0.,1.],(len(n),1)))
    degenerate=np.linalg.norm(t1,axis=1)<1e-5
    t1[degenerate]=np.cross(n[degenerate],np.array([1.,0.,0.]))
    t1/=np.linalg.norm(t1,axis=1)[:,None];t2=np.cross(n,t1)
    theta=np.linspace(0,2*np.pi,8,endpoint=False)
    b=np.array([0.,0.,1.,0.,0.,0.])
    for mu in [.25,.40,.60,.85]:
        forces=n[:,None,:]+mu*(t1[:,None,:]*np.cos(theta)[None,:,None]+t2[:,None,:]*np.sin(theta)[None,:,None])
        moments=np.cross((p-cm)[:,None,:],forces)/extent
        A=np.concatenate([forces,moments],axis=2).reshape(-1,6).T
        weights,residual=nnls(A,b,maxiter=20*A.shape[1])
        if residual<1e-5:
            return {'contact_face_samples':len(p),'coulomb_friction_coefficient_assumed':mu,
                    'force_and_torque_balance_relative_residual':float(residual),
                    'active_contact_cone_rays':int(np.sum(weights>1e-6)),
                    'scope':'Discretized static uniform-density rigid support with an assumed friction coefficient; no measured friction or deformation'}
    raise RuntimeError('No static support under the tested friction priors')

def plan(R,lo,hi,K):
    dims=hi-lo;L=float(max(dims));full=geo.solid_box(lo,hi)
    left,top,width,height=geo.VALID_IMAGE_RECT
    desired=np.array([left+.20*width,top+.25*height])
    us=np.unique(np.r_[np.linspace(left+18,left+width-18,48),desired[0]])
    vs=np.unique(np.r_[np.linspace(top+18,top+height-18,40),desired[1]])
    uu,vv=np.meshgrid(us,vs);pixels=np.c_[uu.ravel(),vv.ravel()]
    rays=np.c_[pixels,np.ones(len(pixels))]@np.linalg.inv(K).T
    distance=distance_transform_edt(~geo.FOOD_MASK_FOR_PLANNING)
    candidates=[]
    for fraction in [.25,.22,.18]:
        radius=fraction*min(dims[:2]);depth=float(min(.90*radius,.95*dims[2]))
        rz=depth*.35/.90;upper=depth*.85/.90
        center=np.r_[hi[:2]-.55*radius,hi[2]-depth*.55/.90]
        sphere=manifold.Manifold.sphere(1.,64)
        egg=geo.mesh_of(sphere);v=egg.vertices.copy();v[:,:2]*=radius
        v[:,2]*=np.where(v[:,2]>=0,upper,rz)
        cutter=manifold.Manifold(manifold.Mesh64(v,np.asarray(egg.faces,np.uint64))).translate(tuple(center))
        bite=full^cutter;remaining=full-cutter;bm=geo.mesh_of(bite);src=bm.center_mass
        assert bite.volume()>0
        support=equilibrium(bite,cutter,L)
        for yaw in [140.,110.,170.]:
            Q=Rotation.from_euler('z',yaw,degrees=True).as_matrix()
            relative=(bm.vertices-src)@Q.T
            for clearance in [.22,.34,.46]:
                zcenter=hi[2]+clearance*L+(src[2]-bm.bounds[0,2])
                ray_depth=zcenter/(rays@R[:,2]);centers_camera=rays*ray_depth[:,None];centers=centers_camera@R
                qs=(centers[:,None,:]+relative[None,:,:])@R.T
                projected=qs@K.T;projected=projected[:,:,:2]/projected[:,:,2:]
                low=projected.min(1);high=projected.max(1)
                good=(ray_depth>0)&np.all(qs[:,:,2]>.01,axis=1)&(low[:,0]>=left+14)&(high[:,0]<=left+width-14)&(low[:,1]>=top+18)&(high[:,1]<=top+height-18)
                for i in np.flatnonzero(good):
                    xx,yy=np.meshgrid(np.linspace(low[i,0],high[i,0],8),np.linspace(low[i,1],high[i,1],8))
                    if np.min(map_coordinates(distance,[yy.ravel(),xx.ravel()],order=1,mode='nearest'))<10:continue
                    score=np.linalg.norm((pixels[i]-desired)/[1,1.1])+40*(clearance-.22)+.12*abs(yaw-140)
                    candidates.append((score,centers[i],Q,yaw,radius,rz,center,bite,remaining,src,cutter,sphere,support,depth))
        if candidates:break
    if not candidates:raise RuntimeError('No in-frame supported scoop pose')
    _,dest,Q,yaw,radius,rz,center,bite,remaining,src,cutter,sphere,support,cutdepth=min(candidates,key=lambda x:x[0])
    transform=lambda solid:solid.translate(tuple(-src)).rotate((0,0,yaw)).translate(tuple(dest))
    lifted=transform(bite);inner=transform(cutter)
    bowl_center=(center-src)@Q.T+dest
    outer=sphere.scale((radius*1.10,radius*1.10,rz+.06*radius)).translate(tuple(bowl_center))
    bottom=np.array([bowl_center[0]-2*radius,bowl_center[1]-2*radius,bowl_center[2]-2*rz])
    topb=np.array([bowl_center[0]+2*radius,bowl_center[1]+2*radius,bowl_center[2]])
    bowl=(outer-inner)^manifold.Manifold.cube(tuple(topb-bottom)).translate(tuple(bottom))
    # A horizontal handle points toward the left edge in camera projection.
    direction=-R[0].copy();direction[2]=0;direction/=np.linalg.norm(direction)
    angle=float(np.rad2deg(np.arctan2(direction[1],direction[0])))
    origin=(bowl_center+np.array([0,0,-.04*radius]))@R.T;direction_camera=direction@R.T
    limit=left-12.;den=K[0,0]*direction_camera[0]-(limit-K[0,2])*direction_camera[2]
    travel=((limit-K[0,2])*origin[2]-K[0,0]*origin[0])/den
    length=max(3.5,travel/radius+.2)
    handle=loft([.95,1.08,1.25,1.6,max(2.,length*.65),length-.25,length],
                [.16,.16,.12,.105,.15,.20,.09],[-.015,-.025,-.04,-.055,-.06,-.05,-.05],
                .075,np.array([0.,0.,0.])).scale((radius,radius,radius)).rotate((0,0,angle)).translate(tuple(bowl_center))
    spoon=(bowl+handle)-inner
    spoon_mesh=geo.mesh_of(spoon)
    assert spoon_mesh.is_watertight and len(spoon_mesh.split(only_watertight=False,repair=False))==1
    end=(bowl_center+length*radius*direction)@R.T;end_uv=geo.project(end[None,:],K)[0]
    assert end_uv[0]<left-8 and end[2]>.01
    residual=abs(full.volume()-bite.volume()-remaining.volume())/full.volume()
    overlaps={'partition':float((bite^remaining).volume()),'spoon_food':float((lifted^spoon).volume()),
              'lifted_remaining':float((lifted^remaining).volume()),'spoon_remaining':float((spoon^remaining).volume())}
    assert residual<1e-8 and all(abs(x)<1e-9 for x in overlaps.values()), dict(residual=residual,overlaps=overlaps)
    for solid in [full,bite,remaining,lifted,spoon]:assert geo.mesh_of(solid).is_watertight
    report={'type':'Shared asymmetric smooth scoop; shallow lower ellipsoid and plump upper cap, carried portion is the exact rigidly transformed Boolean intersection',
        'utensil':'Shallow conforming metal eating spoon; payload extends above bowl rim','legacy_artifact_name':'fork.ply and fork masks name the generic utensil channel only',
        'source_cutter_center':center.tolist(),'cutter_radii':[radius,radius,rz],'bowl_depth_over_radius':rz/radius,'payload_top_above_bowl_rim_over_radius':float((hi[2]-center[2])/radius),'upper_vertical_radius_over_bowl_radius':float((cutdepth*.85/.90)/radius),
        'cut_depth_over_food_height':cutdepth/dims[2],'radius_fraction':fraction,'yaw_degrees':yaw,
        'rotation_food_frame':Q.tolist(),'source_center':src.tolist(),'destination_center':dest.tolist(),
        'translation_food_frame':(dest-src).tolist(),'relative_partition_volume_residual':residual,'overlap_volumes':overlaps,
        'bite_fraction':bite.volume()/full.volume(),'static_contact':support,
        'handle_end_uv':end_uv.tolist(),'handle_exits_original_frame':True,
        'bottom_clearance_above_food_top_over_extent':(geo.mesh_of(lifted).bounds[0,2]-hi[2])/L,
        'all_meshes_watertight':True,'scope':'Monocular source shape plus shared scoop prior. Curvature and friction are hypotheses, not real 3D measurements.'}
    LAST.update(bite=bite,src=src.copy(),dest=dest.copy(),Q=Q.copy())
    return full,remaining,bite,lifted,spoon,dest-src,report

def texture(solid,R,lo,hi,K,source,translation=None):
    if translation is None:return ORIGINAL_TEXTURE(solid,R,lo,hi,K,source)
    mesh=ORIGINAL_TEXTURE(LAST['bite'],R,lo,hi,K,source)
    world=np.asarray(mesh.vertices)@R
    mesh.vertices=((world-LAST['src'])@LAST['Q'].T+LAST['dest'])@R.T
    return mesh

def main():
    os.environ.update(CUDA_VISIBLE_DEVICES='0',EGL_DEVICE_ID='0')
    geo.plan_bite=plan;geo.texture_mesh=texture
    original_fit=geo.fit_block;original_solid=geo.solid_box
    out=ROOT/'geometry_spoon_v4';out.mkdir(exist_ok=False);rows=[]
    (out/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases']:
        global ORIGINAL_TEXTURE
        if c['case_id']=='prospective_02_7441':geo.fit_block=frustum.fit;geo.solid_box=frustum.solid;ORIGINAL_TEXTURE=frustum.texture
        else:geo.fit_block=original_fit;geo.solid_box=original_solid;ORIGINAL_TEXTURE=ORIGINAL_TEXTURE_BASE
        d=out/c['case_id'];d.mkdir();start=time.time()
        try:
            mp=(OLD/'geometry'/c['case_id']/'maps.npz') if c['case_id'].startswith('new_') else (ROOT/'geometry_v3'/c['case_id']/'maps.npz')
            maps=dict(np.load(mp));row=base.geometry(c,d,maps);row['seconds']=time.time()-start
        except Exception as e:
            row={'case_id':c['case_id'],'status':'geometry_failed','error':repr(e)};(d/'FAILED.txt').write_text(traceback.format_exc())
        rows.append(row);(out/'manifest.json').write_text(json.dumps({'status':'building','cases':rows},indent=2))
        print(c['case_id'],row['status'],row.get('error'),flush=True)
    (out/'manifest.json').write_text(json.dumps({'status':'complete','cases':rows,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))

if __name__=='__main__':main()
