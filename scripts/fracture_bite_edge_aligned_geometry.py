"""Shared irregular cut surface, rigid yaw and explicit fork support.

The cut is a procedural fracture prior, not a learned or measured fracture law.
One Boolean partition defines both pieces; the target rotates that same piece.
"""
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt,map_coordinates
from scipy.spatial import ConvexHull
import manifold3d as manifold
import build_structured_geometry_v3 as base
import structured_geometry_helper as geo
from structured_fork import make_fork

ROOT=base.ROOT
OLD=Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929')
ORIGINAL_TEXTURE=geo.texture_mesh
LAST={}

def plan(R,lo,hi,K):
    dims=hi-lo;L=float(max(dims));full=geo.solid_box(lo,hi)
    left,top,width,height=geo.VALID_IMAGE_RECT;desired=np.array([left+.18*width,top+.25*height])
    us=np.unique(np.r_[np.linspace(left+12,left+width-12,48),desired[0]])
    vs=np.unique(np.r_[np.linspace(top+12,top+height-12,40),desired[1]])
    uu,vv=np.meshgrid(us,vs);pixels=np.c_[uu.ravel(),vv.ravel()]
    rays=np.c_[pixels,np.ones(len(pixels))]@np.linalg.inv(K).T
    distance=distance_transform_edt(~geo.FOOD_MASK_FOR_PLANNING)
    candidates=[]
    for fraction in [.30,.26,.22]:
        radius=fraction*min(dims[:2]);depth=min(.26*min(dims[:2]),.40*dims[2]);z0=hi[2]-depth
        theta=np.linspace(0,2*np.pi,72,endpoint=False)
        rad=radius*(1+.055*np.sin(3*theta+.7)+.025*np.sin(7*theta+1.8))
        center=hi[:2]-.12*radius
        polygon=center+np.c_[np.cos(theta)*rad,np.sin(theta)*rad]
        cutter=manifold.CrossSection([polygon]).extrude(depth+.005*L).translate((0,0,z0))
        bite=full^cutter;remaining=full-cutter;bm=geo.mesh_of(bite);src=bm.center_mass
        for yaw in [140.,110.,170.]:
            angle=np.deg2rad(yaw);Q=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
            relative=(bm.vertices-src)@Q.T
            for clearance in [.18,.28,.40]:
                zcenter=hi[2]+clearance*L+(src[2]-z0)
                ray_depth=zcenter/(rays@R[:,2]);centers_camera=rays*ray_depth[:,None];centers=centers_camera@R
                qs=(centers[:,None,:]+relative[None,:,:])@R.T
                projected=qs@K.T;projected=projected[:,:,:2]/projected[:,:,2:]
                low=projected.min(1);high=projected.max(1)
                good=(ray_depth>0)&np.all(qs[:,:,2]>.01,axis=1)&(low[:,0]>=left+10)&(high[:,0]<=left+width-10)&(low[:,1]>=top+12)&(high[:,1]<=top+height-12)
                for i in np.flatnonzero(good):
                    xx,yy=np.meshgrid(np.linspace(low[i,0],high[i,0],8),np.linspace(low[i,1],high[i,1],8))
                    if np.min(map_coordinates(distance,[yy.ravel(),xx.ravel()],order=1,mode='nearest'))<8:continue
                    score=np.linalg.norm((pixels[i]-desired)/[1,1.1])+40*(clearance-.18)+.12*abs(yaw-140)
                    candidates.append((score,centers[i],Q,yaw,radius,bite,remaining,src,z0,polygon,depth,fraction))
        if candidates:break
    if not candidates:raise RuntimeError('No in-frame irregular bite pose')
    _,dest,Q,yaw,radius,bite,remaining,src,z0,polygon,cutdepth,fraction=min(candidates,key=lambda x:x[0])
    lifted=bite.translate(tuple(-src)).rotate((0,0,yaw)).translate(tuple(dest))
    contact=dest.copy();contact[2]=z0+(dest-src)[2]
    fork,patches=make_fork(contact,radius)
    footprint=lifted.slice(contact[2]+L*1e-7);contact_vertices=[];nonempty=0
    for low,high in patches:
        rectangle=np.array([[low[0],low[1]],[high[0],low[1]],[high[0],high[1]],[low[0],high[1]]])
        polygons=(footprint^manifold.CrossSection([rectangle])).to_polygons()
        if polygons:nonempty+=1
        for p in polygons:contact_vertices.extend(p)
    hull=ConvexHull(np.asarray(contact_vertices));cm=geo.mesh_of(lifted).center_mass
    margin=float(np.min(-(hull.equations[:,:2]@cm[:2]+hull.equations[:,2])))
    residual=abs(full.volume()-bite.volume()-remaining.volume())/full.volume()
    overlaps={'partition':float((bite^remaining).volume()),'fork_food':float((lifted^fork).volume()),'lifted_remaining':float((lifted^remaining).volume()),'fork_remaining':float((fork^remaining).volume())}
    assert residual<1e-9 and all(abs(x)<1e-9 for x in overlaps.values()) and margin>0 and nonempty>=2
    for solid in [full,bite,remaining,lifted,fork]:assert geo.mesh_of(solid).is_watertight
    gap=float(geo.mesh_of(lifted).bounds[0,2]-geo.mesh_of(fork).bounds[1,2]);assert abs(gap/L)<1e-8
    report={'type':'Shared smooth irregular vertical fracture prior; same partitioned bite undergoes rigid yaw and lift',
        'source_cut_polygon_xy':polygon.tolist(),'cut_depth_over_food_height':cutdepth/dims[2],'radius_fraction':fraction,
        'yaw_degrees':yaw,'rotation_food_frame':Q.tolist(),'source_center':src.tolist(),'destination_center':dest.tolist(),
        'translation_food_frame':(dest-src).tolist(),'relative_partition_volume_residual':residual,'overlap_volumes':overlaps,
        'bite_fraction':bite.volume()/full.volume(),'com_inside_support_hull':True,'support_margin_over_bite_radius':margin/radius,
        'support_contact_patches':nonempty,'fork_contact_gap_over_extent':gap/L,
        'bottom_clearance_above_food_top_over_extent':(contact[2]-hi[2])/L,'all_meshes_watertight':True,
        'scope':'Procedural hidden fracture and rigid, uniform-density static support. No measured fracture, friction or dynamic eating physics.'}
    LAST.update(bite=bite,src=src.copy(),dest=dest.copy(),Q=Q.copy())
    return full,remaining,bite,lifted,fork,dest-src,report

def texture(solid,R,lo,hi,K,source,translation=None):
    if translation is None:return ORIGINAL_TEXTURE(solid,R,lo,hi,K,source)
    mesh=ORIGINAL_TEXTURE(LAST['bite'],R,lo,hi,K,source)
    world=np.asarray(mesh.vertices)@R
    mesh.vertices=((world-LAST['src'])@LAST['Q'].T+LAST['dest'])@R.T
    return mesh

def main():
    # GPU 6 is independent of the ongoing image-generation workers on 0..3.
    os.environ.update(CUDA_VISIBLE_DEVICES='3',EGL_DEVICE_ID='3')
    import edge_aligned_geometry_helper as aligned
    def fit(maps,mask):
        aligned.ANCHORS=geo.ANCHORS
        return aligned.fit_block(maps,mask)
    geo.fit_block=fit
    geo.plan_bite=plan;geo.texture_mesh=texture
    out=ROOT/'geometry_fracture_edges_v1';out.mkdir(exist_ok=False);rows=[]
    c=json.loads((ROOT/'inputs/manifest.json').read_text())
    for case in c['cases']:
        d=out/case['case_id'];d.mkdir();start=time.time()
        try:
            mp=(OLD/'geometry'/case['case_id']/'maps.npz') if case['case_id'].startswith('new_') else (ROOT/'geometry_v3'/case['case_id']/'maps.npz')
            maps=dict(np.load(mp))
            row=base.geometry(case,d,maps);row['seconds']=time.time()-start
        except Exception as e:
            row={'case_id':case['case_id'],'status':'geometry_failed','error':repr(e)};(d/'FAILED.txt').write_text(traceback.format_exc())
        rows.append(row);(out/'manifest.json').write_text(json.dumps({'status':'building','cases':rows},indent=2)+'\n')
        print(case['case_id'],row['status'],row.get('error'),flush=True)
    (out/'manifest.json').write_text(json.dumps({'status':'complete','cases':rows,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')

if __name__=='__main__':main()
