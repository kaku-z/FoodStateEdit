"""Source-only tapered tofu prior for an almost frontal, two-face photograph.

Manual landmarks are approximate image observations, not measured 3D truth.
Rear landmarks are occluded and receive less weight. No generated image is fit.
"""
import numpy as np
import cv2
import manifold3d as manifold
import trimesh
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import structured_geometry_helper as geo

TAPER = None
PLANES = None
LANDMARKS_ORIGINAL = np.array([[104,120],[204,122],[182,82],[119,75],[116,160],[187,165]],float)

def vertices(lo, hi, taper):
    center=(lo+hi)/2; ex=(hi-lo)/2
    top=np.array([[hi[0],lo[1],hi[2]],[hi[0],hi[1],hi[2]],
                  [lo[0],hi[1],hi[2]],[lo[0],lo[1],hi[2]]])
    bottom=top.copy();bottom[:,:2]=center[:2]+(bottom[:,:2]-center[:2])*taper
    bottom[:,2]=lo[2]
    return np.r_[top,bottom]

def fit(maps, original_mask):
    global TAPER, PLANES
    h,w=original_mask.shape
    K=maps['intrinsics'].copy();K[0]*=w;K[1]*=h
    uv=LANDMARKS_ORIGINAL*np.array([640/280,480/210])
    valid=original_mask & maps['mask'] & np.isfinite(maps['depth']) & (maps['depth']>0)
    front=np.array(maps['normal'][338,345],float);front/=np.linalg.norm(front)
    if front[2]>0:front=-front
    up=np.array([0.,-1.,0.]);up-=np.dot(up,front)*front;up/=np.linalg.norm(up)
    right=np.cross(up,front);right/=np.linalg.norm(right)
    R0=np.stack([front,right,up],axis=1);assert np.linalg.det(R0)>.999
    depth=float(np.median(maps['depth'][valid]))
    pc=np.r_[uv[:4].mean(0),1.]@np.linalg.inv(K).T*depth
    ey=(uv[1,0]-uv[0,0])*depth/K[0,0]
    extent=np.array([.65*ey,ey,.48*ey]);c0=pc@R0;scale=ey
    p0=np.r_[Rotation.from_matrix(R0).as_rotvec(),c0,np.log(extent),np.log([.78,.75])]
    weights=np.array([1.,1.,.35,.35,1.,1.])
    def decode(p):
        R=Rotation.from_rotvec(p[:3]).as_matrix();e=np.exp(p[6:9]);c=p[3:6]
        return R,c-e/2,c+e/2,np.exp(p[9:11])
    def residual(p):
        R,lo,hi,t=decode(p);cam=vertices(lo,hi,t)@R.T
        observed=geo.project(cam,K)[[0,1,2,3,4,5]]
        reproj=((observed-uv)*weights[:,None]/max(h,w)).ravel()
        normal=.045*(R[:,0]-front);gravity=.02*(R[:,2]-up)
        depth_prior=np.array([.01*(np.mean(cam[:,2])-depth)/depth])
        return np.r_[reproj,normal,gravity,depth_prior]
    lower=np.r_[p0[:3]-.55,c0-.45*scale,np.log(extent*.3),np.log([.45,.45])]
    upper=np.r_[p0[:3]+.55,c0+.45*scale,np.log(extent*2.8),np.log([1.05,1.05])]
    solved=least_squares(residual,p0,bounds=(lower,upper),max_nfev=800,
                         ftol=1e-11,xtol=1e-11,gtol=1e-11)
    R,lo,hi,TAPER=decode(solved.x)
    vv=vertices(lo,hi,TAPER);PLANES=ConvexHull(vv).equations
    projected=geo.project(vv@R.T,K)
    pred=np.zeros_like(original_mask,np.uint8)
    cv2.fillConvexPoly(pred,np.round(projected[ConvexHull(projected).vertices]).astype(np.int32),1)
    body=np.zeros_like(original_mask,np.uint8)
    cv2.fillConvexPoly(body,np.round(uv[ConvexHull(uv).vertices]).astype(np.int32),1)
    def iou(a,b):return float(np.sum(a&b)/np.sum(a|b))
    report={'prior':'Closed tapered rectangular frustum; two-face source landmark fit',
        'landmarks_original_pixels':LANDMARKS_ORIGINAL.tolist(),'landmarks_canvas_pixels':uv.tolist(),
        'landmark_weights':weights.tolist(),'rear_landmarks_occluded_inferred':True,
        'projected_landmarks':projected[[0,1,2,3,4,5]].tolist(),
        'landmark_reprojection_pixels':np.linalg.norm(projected[[0,1,2,3,4,5]]-uv,axis=1).tolist(),
        'original_mask_includes_garnish':True,'source_silhouette_iou':iou(pred>0,original_mask),
        'source_body_annotation_iou':iou(pred>0,body>0),'bottom_taper_xy':TAPER.tolist(),
        'axes_camera_columns':R.tolist(),'low':lo.tolist(),'high':hi.tolist(),
        'normal_agreement_degrees':[float(np.rad2deg(np.arccos(np.clip(R[:,0]@front,-1,1))))],
        'solver_success':bool(solved.success),'solver_message':solved.message,'evaluations':solved.nfev,
        'scale':'Arbitrary monocular scale; no centimetres or independently measured dimensions',
        'evaluation':'In-sample manually annotated source fit; no independent 3D accuracy claim'}
    return R,lo,hi,K,report,pred

def solid(lo,hi):
    m=trimesh.convex.convex_hull(vertices(lo,hi,TAPER));assert m.is_watertight
    return manifold.Manifold(manifold.Mesh(np.asarray(m.vertices,np.float32),np.asarray(m.faces,np.uint32)))

def texture(solid,R,lo,hi,K,source,translation=None):
    assert translation is None
    m=geo.mesh_of(solid)
    v,f=trimesh.remesh.subdivide_to_size(m.vertices,m.faces,max_edge=max(hi-lo)*.025,max_iter=8)
    m=trimesh.Trimesh(v,f,process=False)
    fc=m.triangles_center;fn=m.face_normals
    distance=fc@PLANES[:,:3].T+PLANES[:,3]
    alignment=fn@PLANES[:,:3].T
    inherited=np.any((np.abs(distance)<max(hi-lo)*1e-5)&(alignment>.999),axis=1)
    camera_visible=np.einsum('ij,ij->i',fn@R.T,-fc@R.T)>0
    seen=inherited&camera_visible
    cam=m.triangles.reshape(-1,3)@R.T;uv=geo.project(cam,K)
    h,w=source.shape[:2];atlas=np.full((h,w+64,3),[225,221,206],np.uint8);atlas[:,:w]=source
    for i,tone in enumerate([[180,179,166],[207,205,190],[234,231,217]]):atlas[i*(h//3):(i+1)*(h//3),w:]=tone
    uv[:,0]/=w+64;uv[:,1]=1-uv[:,1]/h
    hidden=~np.repeat(seen,3);direction=np.repeat(np.argmax(np.abs(fn),axis=1),3)
    uv[hidden,0]=(w+32)/(w+64);uv[hidden,1]=1-(direction[hidden]+.5)/3
    visual=trimesh.visual.texture.TextureVisuals(uv=uv,material=trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(atlas),baseColorFactor=[1.]*4,metallicFactor=0,roughnessFactor=1))
    return trimesh.Trimesh(cam,np.arange(len(cam)).reshape(-1,3),visual=visual,process=False)
