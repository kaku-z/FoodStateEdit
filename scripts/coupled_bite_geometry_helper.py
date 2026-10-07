"""Observation-constrained block-food geometry and supported first-bite diagnostic.

This deliberately uses a cuboid prior for the one tofu development image. It is
not a general reconstruction model, measured 3-D truth, or a generated photograph.
Run after run_food3d_depth.py. Raw Hunyuan baseline meshes are never modified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import importlib.metadata

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import cv2
import manifold3d as manifold
import numpy as np
from PIL import Image, ImageDraw
import pyrender
from scipy.ndimage import binary_erosion, distance_transform_edt, map_coordinates
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh

# Fixed source-only semantic anchors: top, left-facing side, right-facing side.
ANCHORS = [(360, 170), (260, 310), (450, 330)]


def project(points, K):
    p = np.asarray(points) @ K.T
    return p[:, :2] / p[:, 2:]


def corners(lo, hi):
    return np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])


def fit_block(maps, mask):
    points, normals = maps['points'], maps['normal']
    valid = binary_erosion(mask & maps['mask'], iterations=5)
    rough_extent = float(np.max(np.quantile(points[valid],.98,axis=0)-np.quantile(points[valid],.02,axis=0)))
    seeds = np.array([normals[y,x] for x,y in ANCHORS])
    for _ in range(12):
        labels = np.argmax(normals @ seeds.T, axis=-1)
        for k in range(3):
            selected = valid & (labels == k) & (normals @ seeds[k] > .97)
            seeds[k] = np.median(normals[selected], axis=0)
            seeds[k] /= np.linalg.norm(seeds[k])
    clouds = []
    for k in range(3):
        q = points[valid & (labels == k) & (normals @ seeds[k] > .99)]
        signed = q @ seeds[k]
        # Robust rejection is necessary: a few edge pixels have background depth.
        q = q[np.abs(signed - np.median(signed)) < .045*rough_extent]
        clouds.append(q[::max(1, len(q)//1800)])
    # x: left-facing side; y: right-facing side; z: upward top face.
    initial_axes = np.array([seeds[1], seeds[2], seeds[0]]).T
    u,_,vt = np.linalg.svd(initial_axes)
    R = u @ vt
    side_axes_swapped = bool(np.linalg.det(R) < 0)
    if side_axes_swapped:
        # Exchange the two side coordinate labels, never reflect the solid or flip up.
        seeds = seeds[[0,2,1]]
        clouds = [clouds[0], clouds[2], clouds[1]]
        u,_,vt = np.linalg.svd(np.array([seeds[1], seeds[2], seeds[0]]).T)
        R = u @ vt
    assert np.linalg.det(R) > .999
    all_points = np.concatenate(clouds)
    lo = np.quantile(points[valid] @ R, .008, axis=0)
    hi = np.array([np.median(clouds[k] @ R[:,i]) for k,i in [(1,0),(2,1),(0,2)]])
    scale = float(np.max(hi-lo))
    center, extent = (lo+hi)/2, hi-lo
    h,w = mask.shape
    K = maps['intrinsics'].copy(); K[0] *= w; K[1] *= h
    contour = max(cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=len)[:,0,:].astype(float)
    contour = contour[::max(1,len(contour)//250)]
    sdf = distance_transform_edt(~mask)-distance_transform_edt(mask)
    x0 = np.r_[Rotation.from_matrix(R).as_rotvec(), center, np.log(extent)]

    def decode(p):
        rot=Rotation.from_rotvec(p[:3]).as_matrix(); e=np.exp(p[6:9]); c=p[3:6]
        return rot,c-e/2,c+e/2

    def silhouette_samples(rot, low, high):
        pixels=project(corners(low,high) @ rot.T,K)
        poly=pixels[ConvexHull(pixels).vertices]
        t=np.linspace(0,1,24,endpoint=False)
        samples=(poly[:,None,:]*(1-t[None,:,None])+np.roll(poly,-1,axis=0)[:,None,:]*t[None,:,None]).reshape(-1,2)
        return poly,samples

    def residual(p):
        rot, low, high=decode(p)
        plane=np.concatenate([(clouds[k]@rot[:,i]-high[i])/scale for k,i in [(1,0),(2,1),(0,2)]])
        # Every term is normalized by its sample count: point density cannot
        # silently overwhelm silhouette or normal constraints.
        plane=.8*plane/np.sqrt(len(plane))
        nr=(rot.T-np.array([seeds[1],seeds[2],seeds[0]])).ravel()*.08
        poly,samples=silhouette_samples(rot,low,high)
        sd=map_coordinates(sdf,[samples[:,1],samples[:,0]],order=1,mode='nearest')/max(h,w)
        a=poly; ab=np.roll(poly,-1,axis=0)-poly
        ap=contour[:,None,:]-a[None,:,:]
        t=np.clip(np.sum(ap*ab[None,:,:],axis=-1)/np.sum(ab*ab,axis=-1)[None,:],0,1)
        d=np.sqrt(np.min(np.sum((ap-t[:,:,None]*ab[None,:,:])**2,axis=-1),axis=1))/max(h,w)
        return np.r_[plane,nr,1.8*sd/np.sqrt(len(sd)),1.8*d/np.sqrt(len(d))]

    lower=x0-np.r_[[.2]*3,[.15*scale]*3,[.3]*3]
    upper=x0+np.r_[[.2]*3,[.15*scale]*3,[.3]*3]
    fit=least_squares(residual,x0,bounds=(lower,upper),max_nfev=160,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    R,lo,hi=decode(fit.x)
    pred=np.zeros_like(mask,dtype=np.uint8)
    uv=project(corners(lo,hi)@R.T,K)
    cv2.fillConvexPoly(pred,np.round(uv[ConvexHull(uv).vertices]).astype('int32'),1)
    report={'prior':'Closed orthogonal cuboid for cohesive block-shaped tofu only; hidden faces inferred.',
        'semantic_anchor_pixels_top_left_right':ANCHORS,'side_axes_swapped_for_right_handed_coordinates':side_axes_swapped,'normals_cluster_centers':seeds.tolist(),
        'axes_camera_columns':R.tolist(),'low':lo.tolist(),'high':hi.tolist(),
        'dimensions_relative':((hi-lo)/np.max(hi-lo)).tolist(),
        'source_silhouette_iou':float(np.sum((pred>0)&mask)/np.sum((pred>0)|mask)),
        'solver_success':bool(fit.success),'solver_message':fit.message,'evaluations':fit.nfev,
        'plane_depth_residual_median_over_food_extent':[float(np.median(np.abs(clouds[k]@R[:,i]-hi[i]))/np.max(hi-lo)) for k,i in [(0,2),(1,0),(2,1)]],
        'normal_agreement_degrees':[float(np.rad2deg(np.arccos(np.clip(seeds[k]@R[:,i],-1,1)))) for k,i in [(0,2),(1,0),(2,1)]],
        'scale':'Arbitrary scale inherited from monocular prediction; no centimetre claims.',
        'evaluation':'In-sample observation fit, not independent 3-D accuracy.'}
    return R,lo,hi,K,report,pred


def solid_box(low,high):
    return manifold.Manifold.cube(tuple(high-low)).translate(tuple(low))


def mesh_of(solid):
    m=solid.to_mesh64()
    return trimesh.Trimesh(np.asarray(m.vert_properties)[:,:3],np.asarray(m.tri_verts),process=False)


def make_fork(center, edge):
    # Four rectangular tines and a joined neck/handle. Their upper face is the
    # underside contact plane. This is an explicit geometric utensil proxy.
    x,y,z=center
    pieces=[]; contacts=[]
    for offset in [-.36,-.12,.12,.36]:
        low=np.array([x-.67*edge,y+(offset-.055)*edge,z-.065*edge])
        high=np.array([x+.58*edge,y+(offset+.055)*edge,z])
        pieces.append(solid_box(low,high))
        contacts.append((low[:2],high[:2]))
    pieces.append(solid_box(np.array([x+.43*edge,y-.44*edge,z-.085*edge]),np.array([x+.79*edge,y+.44*edge,z])))
    pieces.append(solid_box(np.array([x+.70*edge,y-.13*edge,z-.085*edge]),np.array([x+2.8*edge,y+.13*edge,z-.015*edge])))
    fork=pieces[0]
    for p in pieces[1:]:fork=fork+p
    return fork,contacts


def plan_bite(R,lo,hi,K):
    dims=hi-lo; L=float(max(dims))
    # Compact cube removes the top, front corner from this same closed solid.
    edge=.32*min(dims[:2]); blo=hi-edge; bhi=hi.copy()
    full=solid_box(lo,hi); cutter=solid_box(blo,bhi+.001*L)
    bite=full^cutter; remain=full-cutter
    # An explicit relative-height target, followed by image-framing selection.
    # Final bottom is 0.18 L above original food top, not merely above the plate.
    dz=hi[2]+.18*L-blo[2]
    src_center=(blo+bhi)/2
    # Search horizontal displacement only; the inferred surface normal fixes up.
    candidates=[]
    for dx in np.linspace(.05*L,.95*L,55):
        for dy in np.linspace(-1.1*L,.15*L,65):
            trans=np.array([dx,dy,dz]); dest=src_center+trans
            q=project(corners(blo+trans,bhi+trans)@R.T,K)
            if np.any(q[:,0]<12) or np.any(q[:,0]>628) or np.any(q[:,1]<20) or np.any(q[:,1]>455):continue
            center=q.mean(0)
            # Desired empty left region of this development image. No moving
            # the camera or shrinking the bite to manufacture a visible lift.
            score=np.linalg.norm((center-np.array([87,163]))/np.array([1,1.1]))
            candidates.append((score,trans))
    if not candidates:raise RuntimeError('No in-frame lifted bite at required height')
    _,translation=min(candidates,key=lambda item:item[0])
    lifted=bite.translate(tuple(translation))
    contact_center=src_center+translation;contact_center[2]=blo[2]+translation[2]
    fork,contacts=make_fork(contact_center,edge)
    bm=mesh_of(lifted); remaining=mesh_of(remain)
    # Actual intersection rectangles of tine tops and the bite bottom.
    footlo=blo[:2]+translation[:2]; foothi=bhi[:2]+translation[:2]
    patches=[]
    for a,b in contacts:
        a=np.maximum(a,footlo);b=np.minimum(b,foothi)
        if np.all(b>a):patches.extend([[a[0],a[1]],[a[0],b[1]],[b[0],a[1]],[b[0],b[1]]])
    support=ConvexHull(np.array(patches));com=bm.center_mass
    distances=-(support.equations[:,:2]@com[:2]+support.equations[:,2])
    support_margin=float(min(distances))
    overlap=float((lifted^fork).volume());interpenetration=float((lifted^remain).volume())
    fork_remaining_overlap=float((fork^remain).volume())
    residual=abs(full.volume()-bite.volume()-remain.volume())/full.volume()
    contact_gap=abs(float(bm.bounds[0,2]-mesh_of(fork).bounds[1,2]))
    report={'bite_fraction':bite.volume()/full.volume(),'bite_dimensions_over_food_extent':[edge/L]*3,
        'bite_aspect_ratio':float(np.max(mesh_of(bite).extents)/np.min(mesh_of(bite).extents)),
        'relative_partition_volume_residual':residual,'partition_overlap_volume':float((bite^remain).volume()),
        'translation_food_frame':translation.tolist(),'lift_height_over_food_extent':translation[2]/L,
        'bottom_clearance_above_food_top_over_extent':(bm.bounds[0,2]-hi[2])/L,
        'bottom_clearance_above_assumed_food_base_over_extent':(bm.bounds[0,2]-lo[2])/L,
        'fork_contact_gap_over_extent':contact_gap/L,'fork_food_overlap_volume':overlap,
        'fork_remaining_overlap_volume':fork_remaining_overlap,
        'lifted_remaining_overlap_volume':interpenetration,'com_inside_support_hull':support_margin>0,
        'support_margin_over_bite_width':support_margin/edge,'support_contact_patches':len(patches)//4,
        'original_watertight':mesh_of(full).is_watertight,'remaining_watertight':remaining.is_watertight,
        'bite_watertight':bm.is_watertight,'fork_watertight':mesh_of(fork).is_watertight,
        'bite_components':len(bm.split(only_watertight=False,repair=False)),
        'fork_components':len(mesh_of(fork).split(only_watertight=False,repair=False)),
        'scope':'Static rigid geometric contact under uniform density. No tofu fracture, sag, friction or dynamic insertion model.'}
    assert residual<1e-9 and overlap<1e-10 and interpenetration<1e-10 and fork_remaining_overlap<1e-10
    assert report['bite_watertight'] and report['remaining_watertight'] and report['fork_watertight']
    assert report['bite_components']==1 and report['fork_components']==1 and support_margin>0 and contact_gap/L<1e-8
    return full,remain,bite,lifted,fork,translation,report


class Renderer:
    def __init__(self,K,w=640,h=480,neutral=True):
        self.r=pyrender.OffscreenRenderer(w,h)
        self.scene=pyrender.Scene(bg_color=[.965,.975,.99,1],ambient_light=[.45]*3 if neutral else [1.0]*3)
        self.scene.add(pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2],K[1,2],znear=.01,zfar=30))
        if neutral:
            self.scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2.2))
        self.nodes=[]

    def add(self,mesh,color=None):
        m=mesh.copy();m.vertices=np.asarray(m.vertices)*np.array([1,-1,-1])
        material=None if color is None else pyrender.MetallicRoughnessMaterial(baseColorFactor=color,metallicFactor=.7 if color[0]<.5 else 0,roughnessFactor=.45)
        self.nodes.append(self.scene.add(pyrender.Mesh.from_trimesh(m,material=material,smooth=False)))

    def render(self):return self.r.render(self.scene)
    def clear(self):
        for node in self.nodes:self.scene.remove_node(node)
        self.nodes=[]
    def close(self):self.r.delete()


def camera_mesh(solid,R):
    m=mesh_of(solid);m.vertices=m.vertices@R.T
    return m


def texture_mesh(solid,R,lo,hi,K,source,translation=None):
    m=mesh_of(solid)
    if translation is not None:m.vertices=np.asarray(m.vertices)-translation
    v,f=trimesh.remesh.subdivide_to_size(m.vertices,m.faces,max_edge=max(hi-lo)*.025,max_iter=8)
    m=trimesh.Trimesh(v,f,process=False)
    # Duplicate vertices per face so cut interiors never inherit a visible-face UV.
    verts=m.triangles.reshape(-1,3);face_centers=m.triangles_center;normals=m.face_normals
    seen=np.zeros(len(m.faces),bool)
    for axis in range(3):
        seen|=(np.abs(face_centers[:,axis]-hi[axis])<1e-6)&(normals[:,axis]>.9)
    cam=verts@R.T;uv=project(cam,K)
    h,w=source.shape[:2]
    atlas=np.full((h,w+64,3),[225,221,206],dtype=np.uint8);atlas[:,:w]=source
    # Separate directional tones reveal the three inferred cut faces. These
    # deliberately simple material priors are never sampled from hidden food.
    for i,tone in enumerate([[180,179,166],[207,205,190],[234,231,217]]):
        atlas[i*(h//3):(i+1)*(h//3),w:]=tone
    uv[:,0]/=(w+64);uv[:,1]=1-uv[:,1]/h
    hidden=~np.repeat(seen,3);direction=np.repeat(np.argmax(np.abs(normals),axis=1),3)
    uv[hidden,0]=float(w+32)/(w+64)
    uv[hidden,1]=1-(direction[hidden]+.5)/3
    if translation is not None:cam+=translation@R.T
    visual=trimesh.visual.texture.TextureVisuals(uv=uv,material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(atlas),baseColorFactor=[1.0]*4,metallicFactor=0,roughnessFactor=1))
    return trimesh.Trimesh(cam,np.arange(len(cam)).reshape(-1,3),visual=visual,process=False)


def panel(array,title):
    img=Image.new('RGB',(640,520),'white');img.paste(Image.fromarray(array[:,:,:3]),(0,40))
    ImageDraw.Draw(img).text((12,12),title,fill=(20,30,45));return img


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output-name',default='repair_v1');args=p.parse_args()
    root=args.root;out=root/'results'/args.output_name;out.mkdir(exist_ok=False)
    (out/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    source_path=root/'inputs/source.jpg'
    expected='cd92b2477620cba4688a4cd73bd32d155a357ca77e32effe138bb1b8a665a7e2'
    assert hashlib.sha256(source_path.read_bytes()).hexdigest()==expected
    source=np.asarray(Image.open(source_path).convert('RGB'));mask=np.asarray(Image.open(root/'inputs/food_mask.png'))>0
    maps=np.load(root/'results/depth_v1/maps.npz')
    R,lo,hi,K,fit,pred=fit_block(maps,mask)
    full,remain,bite,lifted,fork,translation,cut=plan_bite(R,lo,hi,K)
    report={'source_sha256':expected,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'runtime_packages':{name:importlib.metadata.version(name) for name in ['numpy','scipy','trimesh','manifold3d','pyrender','PyOpenGL']},
        'method':'MoGe visible-surface observations + tofu block prior + local solid partition + static utensil support',
        'fit':fit,'cut_and_support':cut,'model_provenance':json.loads((root/'results/depth_v1/manifest.json').read_text()),
        'limits':['One manually segmented development image, no held-out evaluation.','Hunyuan is retained as the failed baseline, not secretly credited for this proxy geometry.',
                  'Hidden geometry, cut material and absolute scale are inferred.','Renderings and projection composites are diagnostic, not final photorealistic VACE outputs.']}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    renderer=Renderer(K)
    full_cam=camera_mesh(full,R);renderer.add(full_cam,[.75,.80,.83,1]);fitted,depth=renderer.render();renderer.clear()
    renderer.add(camera_mesh(remain,R),[.75,.80,.83,1]);renderer.add(camera_mesh(lifted,R),[.97,.53,.19,1]);renderer.add(camera_mesh(fork,R),[.36,.41,.47,1]);edited,edepth=renderer.render();renderer.close()
    Image.fromarray(fitted).save(out/'fitted_block.png');Image.fromarray(edited).save(out/'supported_bite_geometry.png')
    overlay=source.copy();overlay[pred>0]=(.55*overlay[pred>0]+.45*np.array([20,160,220])).astype('uint8');Image.fromarray(overlay).save(out/'projection_overlay.png')
    scene=trimesh.Scene()
    for name,solid,color in [('full',full,[190,203,211,255]),('remaining',remain,[190,203,211,255]),('bite_source',bite,[248,135,45,255]),('bite_lifted',lifted,[248,135,45,255]),('fork',fork,[100,110,125,255])]:
        m=camera_mesh(solid,R);m.visual.face_colors=color;m.export(out/(name+'.ply'))
        if name in ('remaining','bite_lifted','fork'):scene.add_geometry(m,node_name=name)
    scene.export(out/'supported_bite_scene.glb')
    # Source-projected appearance, without a diffusion model. Outside the proxy
    # footprint the original photograph is retained. Only newly uncovered pixels
    # receive clearly documented small-region Telea inpainting.
    textured_remain=texture_mesh(remain,R,lo,hi,K,source)
    textured_bite=texture_mesh(lifted,R,lo,hi,K,source,translation)
    renderer=Renderer(K,neutral=False)
    renderer.add(texture_mesh(full,R,lo,hi,K,source));source_check,check_depth=renderer.render();renderer.clear()
    check_mask=binary_erosion(mask,iterations=5)&(check_depth>0)
    color_mae=float(np.mean(np.abs(source_check[check_mask].astype(float)-source[check_mask].astype(float))))
    assert color_mae<12,'Source texture is not reproduced; inspect color factors / renderer lighting'
    Image.fromarray(source_check).save(out/'source_projection_color_check.png')
    renderer.add(textured_remain);renderer.add(textured_bite);renderer.add(camera_mesh(fork,R),[.42,.46,.5,1]);texture,td=renderer.render();renderer.close()
    removal=(mask&(td==0)).astype('uint8')*255
    background=cv2.inpaint(source,removal,3,cv2.INPAINT_TELEA)
    composite=background.copy();composite[td>0]=texture[td>0]
    Image.fromarray(removal).save(out/'uncovered_inpaint_mask.png')
    Image.fromarray(composite).save(out/'source_texture_composite.png')
    report['appearance']={'type':'Source-projected texture diagnostic with uniform inferred interiors and Telea background fill.',
        'uncovered_filled_pixels':int(np.sum(removal>0)),'source_texture_preserved':True,'generated_photograph':False,
        'unchanged_pose_color_mae_0_to_255':color_mae,'color_check_scope':'Texture mapping correctness by same-view round trip; not geometry accuracy.',
        'known_limits':'No physically rendered cast shadows, tofu rough fracture, utensil reflections or contact deformation.'}
    scene=trimesh.Scene();scene.add_geometry(textured_remain,node_name='remaining');scene.add_geometry(textured_bite,node_name='bite_lifted');scene.add_geometry(camera_mesh(fork,R),node_name='fork');scene.export(out/'textured_supported_scene.glb')
    board=Image.new('RGB',(1280,1040),'white')
    arrays=[(source,'REAL INPUT | unchanged UECFOOD photo'),(fitted,'REPAIR | depth / normal constrained tofu block prior'),(edited,'REPAIR | compact bite + 3D fork contact + visible lift'),(composite,'TEXTURE DIAGNOSTIC | projected photo, inferred cut faces')]
    for i,(array,title) in enumerate(arrays):board.paste(panel(array,title),((i%2)*640,(i//2)*520))
    board.save(out/'repair_overview.jpg',quality=95)
    report['files']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file() and f.name!='report.json'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
