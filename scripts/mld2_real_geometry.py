"""Observation-constrained local closed volume and source-anchored material transport.

The inferred underside and local plane are monocular hypotheses, not scans.
"""
import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
from pathlib import Path
import sys
import json
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, distance_transform_edt
from scipy.spatial.transform import Rotation
import trimesh
sys.path.insert(0, '/host/space0/guo-z/tf-ufi/food3d_repair_20260928/compat_packages')
import pyrender
from foodstateedit.material_lineage.observations import smooth_observed_points


def project(p, K):
    q = np.asarray(p) @ K.T
    return q[..., :2] / q[..., 2:]


def axis_frame(points, mask, valid, other_food=None):
    ring = binary_dilation(mask, iterations=25) & ~mask & valid
    if other_food is not None:
        observed_plate=ring & ~binary_dilation(other_food,iterations=2)
        if observed_plate.sum()>100:ring=observed_plate
    q = points[ring]
    q = q[::max(1, len(q)//2000)]
    center = np.median(q, axis=0)
    # Robust local plate/table tangent estimated from source pixels only.
    for _ in range(4):
        _, _, vt = np.linalg.svd(q-center, full_matrices=False)
        n = vt[-1]
        d = (q-center) @ n
        q = q[np.abs(d) <= np.quantile(np.abs(d), .80)]
        center = np.median(q, axis=0)
    if n[1] > 0: n = -n
    n = n/np.linalg.norm(n)
    right = np.array([1.,0.,0.]); right -= n*(right@n); right /= np.linalg.norm(right)
    back = np.cross(n, right); back /= np.linalg.norm(back)
    return np.stack([right,back,n], axis=1), center


def choose_patch(mask, image):
    h,w = mask.shape
    dist = distance_transform_edt(mask)
    yy,xx = np.indices(mask.shape)
    ys,xs = np.where(mask)
    scale = min(xs.max()-xs.min(),ys.max()-ys.min())
    radius=max(10., .055*min(w,h), .11*scale)
    desired = np.array([np.quantile(xs,.85),np.quantile(ys,.40)])
    # Bite a visible outer edge, producing an open notch instead of a drilled hole.
    score = -np.abs(dist-.42*radius) - .35*np.hypot(xx-desired[0],yy-desired[1])
    score[~mask]=-1e6
    y,x = np.unravel_index(np.argmax(score), score.shape)
    angle=np.arctan2(yy-y,xx-x)
    uneven=1+.10*np.sin(3*angle)+.035*np.cos(7*angle)
    patch = ((xx-x)**2+(1.1*(yy-y))**2 < (radius*uneven)**2) & mask
    return patch, (x,y), radius


def texture_top(vertices, faces, pixels, image):
    uv = np.c_[pixels[:,0]/image.shape[1], 1-pixels[:,1]/image.shape[0]]
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(image),
        baseColorFactor=[1.,1.,1.,1.], metallicFactor=0.,roughnessFactor=1.,doubleSided=True)
    mesh=trimesh.Trimesh(vertices, faces, process=False)
    mesh.visual=trimesh.visual.texture.TextureVisuals(uv=uv,material=material)
    return mesh


def color_mesh(vertices, faces, rgb):
    mesh=trimesh.Trimesh(vertices,faces,process=False)
    mesh.visual.vertex_colors=np.c_[np.clip(rgb,0,255).astype(np.uint8),np.full(len(vertices),255,dtype=np.uint8)]
    return mesh


def closed_patch(points, patch, floor, R, interior_rgb, source, bottom_rgb=None):
    # One persistent pixel identity per visible vertex; underside shares coordinates.
    h,w=patch.shape
    yy,xx=np.where(patch)
    lookup=np.full((h,w),-1,dtype=np.int32); lookup[yy,xx]=np.arange(len(xx))
    top=points[yy,xx].copy(); low=top.copy()
    coords=top@R; low += (floor-coords[:,2])[:,None]*R[:,2]
    faces=[];edges=[]
    for y,x in zip(yy,xx):
        a=lookup[y,x]
        if x+1<w and y+1<h:
            b,c,d=lookup[y,x+1],lookup[y+1,x],lookup[y+1,x+1]
            if b>=0 and c>=0:faces.append([a,c,b])
            if b>=0 and c>=0 and d>=0:faces.append([b,c,d])
    f=np.asarray(faces,dtype=int)
    # Triangulated top boundary produces exactly corresponding cavity/bite walls.
    unique,counts=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0,return_counts=True)
    edges=unique[counts==1]
    N=len(top); sides=np.array([[a,b,b+N] for a,b in edges]+[[a,b+N,a+N] for a,b in edges])
    if bottom_rgb is None:bottom_rgb=interior_rgb
    allv=np.r_[top,low]; colors=np.r_[interior_rgb,bottom_rgb*.72]
    side=color_mesh(allv,sides,colors)
    bottom=color_mesh(low,f[:,::-1],bottom_rgb*.62)
    topmesh=texture_top(top,f,np.c_[xx+.5,yy+.5],source)
    cavityfloor=color_mesh(low,f,bottom_rgb*.82)
    cavityside=color_mesh(allv,sides[:,::-1],np.r_[interior_rgb*.85,bottom_rgb*.60])
    complete=trimesh.Trimesh(allv,np.r_[f,f[:,::-1]+N,sides],process=False)
    complete.fix_normals(multibody=True)
    return dict(top=topmesh,side=side,bottom=bottom,cavityfloor=cavityfloor,cavityside=cavityside,
        complete=complete,points=top,low=low,faces=f,pixels=np.c_[xx,yy],boundary_edges=edges)


def spoon(bite, R, target, source):
    # Tilt the spoon bowl toward the viewer while retaining one exact support contact.
    bowl_up=.35*R[:,2]+.65*np.array([0.,0.,-1.]);bowl_up/=np.linalg.norm(bowl_up)
    bowl_right=np.array([1.,0.,0.]);bowl_right-=bowl_up*(bowl_right@bowl_up);bowl_right/=np.linalg.norm(bowl_right)
    bowl_back=np.cross(bowl_up,bowl_right);R=np.stack([bowl_right,bowl_back,bowl_up],axis=1)
    pts=(bite['low']+target)@R; cen=pts.mean(axis=0)
    dims=np.ptp(pts[:,:2],axis=0)
    a,b=max(dims[0]*.74,dims[1]*.7),max(dims[1]*.74,dims[0]*.45)
    contact_r2=((pts[:,0]-cen[0])/a)**2+((pts[:,1]-cen[1])/b)**2
    curvature=.10*b
    floor=np.min(pts[:,2]-curvature*contact_r2)
    vs=[[cen[0],cen[1],floor]]; fs=[]
    rings=14;segments=80
    for i in range(1,rings+1):
        r=i/rings
        for j in range(segments):
            theta=j*2*np.pi/segments
            vs.append([cen[0]+a*r*np.cos(theta),cen[1]+b*r*np.sin(theta),floor+curvature*r*r])
    for j in range(segments):fs.append([0,1+j,1+(j+1)%segments])
    for i in range(rings-1):
        s=1+i*segments;d=s+segments
        for j in range(segments):
            k=(j+1)%segments;fs.extend([[s+j,d+j,s+k],[s+k,d+j,d+k]])
    # Joined spoon handle extends away from the bowl along source camera right.
    length=7*a;handle=[];nf=36
    for i in range(nf+1):
        u=i/nf; x=cen[0]+a*(.8+7*u); half=.14*b*(1-.15*u)
        z=floor+.16*b+.025*b*np.sin(np.pi*u)
        handle.extend([[x,cen[1]-half,z],[x,cen[1]+half,z]])
    base=len(vs);vs.extend(handle)
    for i in range(nf):fs.extend([[base+2*i,base+2*i+1,base+2*i+2],[base+2*i+1,base+2*i+3,base+2*i+2]])
    local=np.asarray(vs); vert=local@R.T
    bowlrad=np.sqrt(((local[:,0]-cen[0])/a)**2+((local[:,1]-cen[1])/b)**2)
    bands=.20+.78*np.cos((local[:,1]-cen[1])/b*3.7)**10+.25*np.cos((local[:,1]-cen[1])/b*1.8)
    env=np.mean(source.reshape(-1,3),axis=0)
    rgb=np.clip(205*bands[:,None]+.10*env,0,255)
    mesh=color_mesh(vert,fs,rgb)
    contact_gaps=pts[:,2]-(floor+curvature*contact_r2)
    return mesh,dict(center_local=cen.tolist(),radius_ab=[float(a),float(b)],floor_local=float(floor),
        inferred_contact_min_gap=float(contact_gaps.min()),inferred_penetration_depth=float(max(0,-contact_gaps.min())),
        spoon_axes_camera=R.tolist(),bowl_curvature_height=float(curvature),
        proxy='concave metal spoon; center plane contact, no hand')


class Render:
    def __init__(self,K,w,h):
        self.r=pyrender.OffscreenRenderer(w,h)
        self.scene=pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[1,1,1])
        pose=np.diag([1.,-1.,-1.,1.])
        self.scene.add(pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2],K[1,2],znear=.001,zfar=100),pose=pose)
        self.nodes=[]
    def add(self,mesh):
        self.nodes.append(self.scene.add(pyrender.Mesh.from_trimesh(mesh,smooth=False)))
    def image(self):
        rgb,depth=self.r.render(self.scene,flags=pyrender.RenderFlags.FLAT|pyrender.RenderFlags.RGBA|pyrender.RenderFlags.SKIP_CULL_FACES)
        for n in self.nodes:self.scene.remove_node(n)
        self.nodes=[]
        return rgb[:,:,:3],depth
    def close(self):self.r.delete()


def field_context(model, source, mask, torch, max_edge=40):
    yy,xx=np.where(mask);margin=int(.1*max(np.ptp(xx),np.ptp(yy)))+4
    x0,x1=max(0,xx.min()-margin),min(source.shape[1],xx.max()+margin+1)
    y0,y1=max(0,yy.min()-margin),min(source.shape[0],yy.max()+margin+1)
    crop=source[y0:y1,x0:x1].copy(); cm=mask[y0:y1,x0:x1]
    crop[~binary_dilation(cm,iterations=2)]=255
    im=Image.fromarray(crop);im.thumbnail((max_edge,max_edge),Image.Resampling.LANCZOS)
    canvas=Image.new('RGB',(64,64),'white'); off=((64-im.width)//2,(64-im.height)//2);canvas.paste(im,off)
    t=torch.as_tensor(np.asarray(canvas).copy(),device='cuda').float().permute(2,0,1)[None]/255
    cache=model.encode_image(t)
    return cache,dict(box=[int(x0),int(y0),int(x1),int(y1)], resized=[im.width,im.height],offset=list(off),
        canonical_camera_input_max_edge=max_edge),canvas


def prior_queries(model, cache, coords, pixels, roi, torch):
    # Explicit real-camera UV is independent of unknown hidden GT.
    xyz=np.asarray(coords,dtype=np.float32)
    uv=roi_uv(pixels,roi)
    outputs=[]
    for a in range(0,len(xyz),2048):
        p=model.query(cache,torch.as_tensor(xyz[a:a+2048],device='cuda')[None],
            torch.as_tensor(uv[a:a+2048],device='cuda')[None])
        outputs.append({k:v[0].cpu().numpy() for k,v in p.items() if k in ['sdf','rgb']})
    return {k:np.concatenate([p[k] for p in outputs]) for k in ['sdf','rgb']}


def roi_uv(pixels,roi):
    x0,y0,x1,y1=roi['box'];rw,rh=roi['resized'];ox,oy=roi['offset']
    return np.c_[((pixels[:,0]-x0)/(x1-x0)*rw+ox)/32-1,
                 ((pixels[:,1]-y0)/(y1-y0)*rh+oy)/32-1].astype('float32')


def canonical_observation(points,pixels,roi,extent,reference_depth,predicted_toward_depth=None):
    uv=roi_uv(pixels,roi)
    yaw,elev=np.deg2rad([30,35])
    u=np.array([-np.sin(yaw),np.cos(yaw),0.])
    v=np.array([np.sin(elev)*np.cos(yaw),np.sin(elev)*np.sin(yaw),-np.cos(elev)])
    view=np.cross(u,v)
    depth=(points[:,2]-reference_depth)/extent if predicted_toward_depth is None else -predicted_toward_depth
    xyz=1.45*(uv[:,0,None]*u+uv[:,1,None]*v)+depth[:,None]*view
    return xyz,view


def lift_plan(selected,R,extent,K,w,h):
    target_uv=np.array([.23*w,.20*h]);src_center=selected.mean(axis=0)
    up=R[:,2];lift=.32*extent
    ray=np.r_[target_uv,1.]@np.linalg.inv(K).T
    target_depth=(src_center@up+lift)/(ray@up)
    return ray*target_depth-src_center


def build_case(folder, model, torch, baseline=False):
    source=np.asarray(Image.open(folder/'source.png').convert('RGB'));h,w=source.shape[:2]
    maps=dict(np.load(folder/'depth.npz'));mask=np.asarray(Image.open(folder/'food_mask.png'))>0
    valid=maps['mask']&np.isfinite(maps['points']).all(axis=-1)
    mask &= valid
    seg=np.load(folder/'segmentation.npz') if (folder/'segmentation.npz').exists() else None
    other_food=None
    if seg is not None:
        selected=(seg['scores']>.5)&(seg['prompts']=='food')
        if selected.any(): other_food=seg['masks'][selected].any(axis=0)
    R,plate=axis_frame(maps['points'],mask,valid,other_food)
    patch,center,radius=choose_patch(mask,source)
    # Missing observations supply neither coordinates nor smoothing weight.
    points=smooth_observed_points(maps['points'],valid,.7)
    food=points[mask]@R;mean=food.mean(axis=0)
    extent=np.max(np.quantile(food,.98,axis=0)-np.quantile(food,.02,axis=0));extent=max(extent,1e-5)
    K=maps['intrinsics'].copy();K[0]*=w;K[1]*=h
    # Increase the actual source portion when perspective would make the moved bite unreadable.
    # The carried mesh is never resized independently of the removed source volume.
    for _ in range(2):
        selected=points[patch];target=lift_plan(selected,R,extent,K,w,h)
        major=float(np.max(np.ptp(project(selected+target,K),axis=0)))
        need=max(35.,.08*w)
        if major>=need:break
        radius*=min(1.65,need/max(major,1))
        gy,gx=np.indices(mask.shape);cx,cy=center;angle=np.arctan2(gy-cy,gx-cx)
        uneven=1+.10*np.sin(3*angle)+.035*np.cos(7*angle)
        patch=((gx-cx)**2+(1.1*(gy-cy))**2<(radius*uneven)**2)&mask
    selected=points[patch];local=selected@R
    coord=(local-mean)/extent*1.7
    yy,xx=np.where(patch);pix=np.c_[xx,yy]
    prior_stats={}
    if model is not None:
        cache,roi,crop=field_context(model,source,mask,torch);crop.save(folder/'mld2_roi.png')
        uv=roi_uv(pix,roi)
        predicted_depth=torch.nn.functional.grid_sample(cache['depth'],torch.as_tensor(uv,device='cuda')[None,:,None],
            align_corners=False,padding_mode='border')[0,0,:,0].cpu().numpy()
        coord,canonical_view=canonical_observation(selected,pix,roi,extent,np.median(points[mask][:,2]),predicted_depth)
        p=prior_queries(model,cache,coord,pix,roi,torch)
        # The learned hidden volume controls local depth, then source observations cap it.
        zsteps=np.linspace(-.85,.85,65)
        c=np.repeat(coord,65,axis=0);c+=np.tile(zsteps,len(coord))[:,None]*canonical_view
        pp=prior_queries(model,cache,c,np.repeat(pix,65,axis=0),roi,torch)
        ray_supported=np.all(np.abs(c)<=1,axis=-1).reshape(-1,65)
        occupancy=(pp['sdf'].reshape(-1,65)<0)&ray_supported
        # A continuous occupied interval supplies local underside depth, rather than summing disconnected blobs.
        inside_lengths=[];anchor_distances=[]
        for row in occupancy:
            changes=np.diff(np.r_[False,row,False].astype(int));starts=np.where(changes==1)[0];ends=np.where(changes==-1)[0]
            if len(starts):
                low=zsteps[starts];high=zsteps[np.minimum(ends,64)]
                distance=np.maximum(low,0)+np.maximum(-high,0)
                k=int(np.argmin(distance));inside_lengths.append(float(ends[k]-starts[k])*(zsteps[1]-zsteps[0]));anchor_distances.append(float(distance[k]))
            else:inside_lengths.append(0.);anchor_distances.append(1.7)
        thickness=np.asarray(inside_lengths)/1.7*extent*.25
        posterior_rgb=np.clip((p['rgb']+1)/2,0,1)
        posterior_rgb=np.where(posterior_rgb<=.0031308,posterior_rgb*12.92,1.055*posterior_rgb**(1/2.4)-.055)*255
        obs=source[yy,xx].astype(float)
        # Preserve observed food color; learned residual is used only on newly exposed sides.
        posterior_rgb=obs.mean(axis=0)+.25*(posterior_rgb-posterior_rgb.mean(axis=0))
        interior=.70*obs+.30*posterior_rgb
        prior_stats=dict(source_anchor_actual_sdf_abs_mean=float(np.mean(np.abs(p['sdf']))*.5),
            source_anchor_actual_sdf_abs_p90=float(np.quantile(np.abs(p['sdf']),.9)*.5),
            predicted_canonical_toward_depth_quantiles=np.quantile(predicted_depth,[0,.25,.5,.75,1]).tolist(),
            source_ray_nonempty_fraction=float(occupancy.any(-1).mean()),
            source_interval_anchor_distance_mean=float(np.mean(anchor_distances)),
            source_interval_anchor_distance_p90=float(np.quantile(anchor_distances,.9)),
            source_prior_anchor_gap_over_point1_fraction=float((np.asarray(anchor_distances)>.1).mean()),
            posterior_thickness_raw_quantiles_over_extent=np.quantile(thickness/extent,[0,.25,.5,.75,1]).tolist(),
            posterior_thickness_lower_clip_fraction=float((thickness<=.03*extent).mean()),
            posterior_thickness_upper_clip_fraction=float((thickness>=.085*extent).mean()),
            posterior_centered_color_std=float(np.std(posterior_rgb-posterior_rgb.mean(0))),
            interior_color_mae_vs_observed=float(np.mean(np.abs(interior-obs))),
            learned_material_only_mae=float(np.mean(np.abs(interior-(.70*obs+.30*obs.mean(0))))),
            visible_field_queries_outside_canonical_fraction=float(np.any(np.abs(coord)>1,axis=-1).mean()))
        prior_stats['hidden_ray_queries_outside_canonical_fraction']=float((~ray_supported).mean())
        np.savez_compressed(folder/'posterior_field.npz', xyz=coord,uv_pixels=pix,sdf=p['sdf'],rgb=p['rgb'],
            depth_sdf=pp['sdf'].reshape(-1,65),depth_candidates=zsteps,thickness=thickness,
            predicted_toward_camera_depth=predicted_depth,chosen_interval_anchor_distance=anchor_distances)
    else:
        roi=None;thickness=np.full(len(coord),.12*extent);interior=source[yy,xx].astype(float)
    observed_height=np.median(local[:,2]-(plate@R)[2])
    if model is not None and not baseline:
        depth=np.clip(thickness,.030*extent,.085*extent)
        canvas=np.zeros(mask.shape);weight=np.zeros(mask.shape)
        canvas[yy,xx]=depth;weight[yy,xx]=1
        smooth=gaussian_filter(canvas,.8)/np.maximum(gaussian_filter(weight,.8),1e-8)
        depth=smooth[yy,xx]
    else:depth=np.full(len(coord),.055*extent)
    # Source height/plane constrain unseen closure, irrespective of the learned prior.
    height_cap=max(.05*extent,observed_height*.85)
    prior_stats['source_height_cap_active_fraction']=float((depth>height_cap).mean())
    depth=np.minimum(depth,height_cap)
    provisional_floor=local[:,2]-depth
    floor=np.maximum(provisional_floor, (plate@R)[2]+.015*extent)
    prior_stats['source_plate_floor_active_fraction']=float((floor>provisional_floor).mean())
    prior_stats['minimum_thickness_active_fraction']=float((floor>local[:,2]-.03*extent).mean())
    floor=np.minimum(floor,local[:,2]-.03*extent)
    prior_stats['final_closure_thickness_quantiles_over_extent']=np.quantile((local[:,2]-floor)/extent,[0,.25,.5,.75,1]).tolist()
    uniform_depth=np.full(len(local),min(.055*extent,height_cap))
    uniform_floor=np.minimum(np.maximum(local[:,2]-uniform_depth,(plate@R)[2]+.015*extent),local[:,2]-.03*extent)
    prior_stats['learned_vs_uniform_floor_mean_absolute_delta_over_extent']=float(np.mean(np.abs(floor-uniform_floor))/extent)
    prior_stats['learned_vs_uniform_floor_p90_absolute_delta_over_extent']=float(np.quantile(np.abs(floor-uniform_floor),.9)/extent)
    bottom_rgb=interior
    if model is not None and not baseline:
        # Query the same frozen canonical material field at the newly exposed interior.
        canonical_floor=coord+((local[:,2]-floor)/extent*1.7*abs(R[2,2]))[:,None]*canonical_view
        hidden=prior_queries(model,cache,canonical_floor,pix,roi,torch)
        linear=np.clip((hidden['rgb']+1)/2,0,1)
        srgb=np.where(linear<=.0031308,linear*12.92,1.055*linear**(1/2.4)-.055)*255
        calibrated=obs.mean(0)+.25*(srgb-srgb.mean(0))
        bottom_rgb=.70*obs+.30*calibrated
        prior_stats['hidden_material_query_delta_vs_surface_rgb_mae']=float(np.mean(np.abs(bottom_rgb-interior)))
        prior_stats['hidden_learned_material_only_mae']=float(np.mean(np.abs(bottom_rgb-(.70*obs+.30*obs.mean(0)))))
        prior_stats['hidden_material_queries_outside_canonical_fraction']=float(np.any(np.abs(canonical_floor)>1,axis=-1).mean())
        prior_stats['new_cut_material_field_queries']=len(canonical_floor)
        np.savez_compressed(folder/'hidden_material_field.npz',xyz=canonical_floor,uv_pixels=pix,
            rgb=hidden['rgb'],sdf=hidden['sdf'],render_rgb=bottom_rgb)
    np.savez_compressed(folder/('closure_'+('baseline' if baseline else 'raw')+'.npz'),
        source_local_xyz=local,world_axes=R,source_camera_xyz=selected,floor_local=floor,
        uniform_floor_local=uniform_floor,extent=extent,interior_rgb=interior,bottom_rgb=bottom_rgb,source_uv_pixels=pix)
    bite=closed_patch(points,patch,floor,R,interior,source,bottom_rgb)
    # A fixed prominent upper-left placement makes the first bite visibly off the dish.
    up=R[:,2];target=lift_plan(selected,R,extent,K,w,h)
    if not np.isfinite(target).all():
        raise ValueError('Observed source camera/support cannot define a finite lift translation')
    spoonmesh,spoonmeta=spoon(bite,R,target,source)
    if baseline and (folder/'geometry_raw.json').exists():
        frozen=json.loads((folder/'geometry_raw.json').read_text())
        target=np.asarray(frozen['target_translation'])
        spoonmesh=trimesh.load(folder/'spoon_raw.ply',process=False)
        spoonmeta=frozen['spoon']
        spoonmeta=dict(spoonmeta,shared_with_mld2_core=True)
        axes=np.asarray(spoonmeta['spoon_axes_camera']);bl=(bite['low']+target)@axes
        bc=np.asarray(spoonmeta['center_local']);a,b=spoonmeta['radius_ab']
        rr=((bl[:,0]-bc[0])/a)**2+((bl[:,1]-bc[1])/b)**2
        gaps=bl[:,2]-(spoonmeta['floor_local']+spoonmeta['bowl_curvature_height']*rr)
        spoonmeta['inferred_contact_min_gap']=float(gaps.min())
        spoonmeta['inferred_penetration_depth']=float(max(0,-gaps.min()))
    renderer=Render(K,w,h)
    renderer.add(bite['top']);source_check,source_depth=renderer.image()
    source_mae=float(np.mean(np.abs(source_check[patch].astype(float)-source[patch].astype(float))))
    renderer.add(bite['cavityfloor']);renderer.add(bite['cavityside']);cavity,cd=renderer.image()
    observed_top=bite['top'].copy();observed_top.apply_translation(target)
    renderer.add(observed_top);_,observed_depth=renderer.image()
    for name in ['top','side','bottom']:
        mesh=bite[name].copy();mesh.apply_translation(target);renderer.add(mesh)
    moved,md=renderer.image()
    renderer.add(spoonmesh);spoon_rgb,sd=renderer.image()
    renderer.close()
    result=source.copy();result[patch]=cavity[patch]
    # Only a source-local cavity is replaced; retained observations keep exact pixels.
    sourcewall=(cd>0)&binary_dilation(patch,iterations=2);result[sourcewall]=cavity[sourcewall]
    alpha=np.maximum((md>0).astype(float),(sd>0).astype(float))
    # Broad displaced shadow on the source is a rendering cue, not a measured lighting estimate.
    shadow=gaussian_filter(np.roll(alpha,int(.11*h),axis=0),max(2,radius*.30))
    shadow *= (mask|binary_dilation(mask,iterations=8))*.22
    result=np.clip(result*(1-shadow[...,None]),0,255).astype(np.uint8)
    spoonvis=(sd>0)&((md==0)|(sd<md));bitevis=(md>0)&((sd==0)|(md<=sd))
    result[spoonvis]=spoon_rgb[spoonvis];result[bitevis]=moved[bitevis]
    edit=binary_dilation(patch|spoonvis|bitevis|(shadow>.005),iterations=9)
    frozen_mask=folder/'frozen_edit_mask.png'
    if frozen_mask.exists():edit=np.asarray(Image.open(frozen_mask))>127
    pre_projection_outside=int(np.any(result!=source,axis=-1)[~edit].sum())
    geometry_outside=int(((spoonvis|bitevis)&~edit).sum())
    if frozen_mask.exists():result[~edit]=source[~edit]
    suffix='baseline' if baseline else 'raw'
    for name,array in [(suffix,result),('source_bite_mask',patch),('bite_mask',bitevis),('spoon_mask',spoonvis),
        ('edit_mask',edit),('observed_food_mask',(observed_depth>0)&bitevis),
        ('cavity',cavity),('moved_food',moved),('source_render_check',source_check)]:
        if array.dtype==bool:array=array.astype(np.uint8)*255
        if baseline and (folder/'geometry_raw.json').exists() and name!='baseline':name='baseline_'+name
        Image.fromarray(array).save(folder/(name+'.png'))
    moved_complete=bite['complete'].copy();moved_complete.apply_translation(target)
    bite['complete'].export(folder/('bite_source_'+suffix+'.ply'));moved_complete.export(folder/('bite_lifted_'+suffix+'.ply'))
    spoonmesh.export(folder/('spoon_'+suffix+'.ply'))
    scene=trimesh.Scene()
    for key in ['top','side','bottom']:
        part=bite[key].copy();part.apply_translation(target);scene.add_geometry(part,node_name='food_'+key)
    scene.add_geometry(spoonmesh,node_name='spoon');scene.export(folder/('scene_'+suffix+'.glb'))
    ids=bite['pixels'][:,1]*w+bite['pixels'][:,0]
    np.savez_compressed(folder/('transport_'+suffix+'.npz'),source_points=bite['complete'].vertices,
        carried_points=moved_complete.vertices,translation=target,faces=bite['complete'].faces,
        visible_material_ids=ids,source_rgb=source[yy,xx],source_top_uv_pixels=bite['pixels'])
    footprint=np.c_[np.where(bitevis)[1],np.where(bitevis)[0]]
    report=dict(method='SAM3 + MoGe2 + '+('observation anchored MLD2' if model is not None and not baseline else 'uniform unseen thickness'),
        source_only=True, source_pixel_anchor_mae=source_mae, local_food_plane=R.tolist(),plane_center=plate.tolist(),
        source_bite_center=[int(v) for v in center],source_bite_radius=float(radius),source_pixels=int(patch.sum()),
        target_translation=target.tolist(),up_axis=up.tolist(),normalized_lift=float(target@up/extent),
        thickness_over_extent=float(np.median(depth)/extent),
        thickness_min_max_over_extent=[float(np.min(depth)/extent),float(np.max(depth)/extent)],
        source_observed_height_over_extent=float(observed_height/extent),
        learned_prior_used=model is not None and not baseline,roi=roi,spoon=spoonmeta,
        source_point_smoothing=dict(method='finite observed-support normalized Gaussian convolution',
            sigma_pixels=.7,valid_source_pixels=int(valid.sum()),invalid_source_pixels_retained=int((~valid).sum()),
            selected_source_pixels=int(patch.sum()),selected_source_pixels_after_smoothing=int(np.isfinite(points[patch]).all(-1).sum()),
            missing_source_support_invented=False),
        hidden_prior_diagnostics=prior_stats,
        moved_visible_pixels=int(bitevis.sum()),spoon_visible_pixels=int(spoonvis.sum()),
        moved_projected_major_pixels=float(np.max(np.ptp(project(selected+target,K),axis=0))),
        source_closed_volume=float(abs(bite['complete'].volume)),carried_closed_volume=float(abs(moved_complete.volume)),
        exact_inverse_transport_max_error=float(np.max(np.abs((moved_complete.vertices-target)-bite['complete'].vertices))),
        relative_volume_transport_error=float(abs(abs(moved_complete.volume)-abs(bite['complete'].volume))/abs(bite['complete'].volume)),
        visible_correspondence_source_mae=float(np.mean(np.abs(source_check[binary_erosion(patch,iterations=2)&(source_depth>0)].astype(float)-source[binary_erosion(patch,iterations=2)&(source_depth>0)].astype(float)))),
        inferred_bite_watertight=bool(bite['complete'].is_watertight),
        source_target_mask_overlap=int((patch&bitevis).sum()),
        frozen_edit_mask_used=frozen_mask.exists(),
        pre_observation_projection_changes_outside_frozen_mask=pre_projection_outside,
        geometry_pixels_outside_frozen_mask=geometry_outside,
        core_change_pixels_outside_frozen_mask=int(np.any(result!=source,axis=-1)[~edit].sum()),
        assumptions=['visible MoGe surface; inferred underside and tangent plane',
            'rigid cohesive bite; noodle/granular deformation not modeled',
            'source UV persistent texture; new cut-face material posterior with source color calibration'],
        photographic_ground_truth=False)
    (folder/('geometry_'+suffix+'.json')).write_text(json.dumps(report,indent=2))
    return report
