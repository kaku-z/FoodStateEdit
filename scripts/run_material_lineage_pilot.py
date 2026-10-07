"""Execute a source-only material-lineage pilot, not the full proposed model.

Actual training: one small appearance epsilon MLP per real source photograph.
Geometry is a previously source-fitted monocular proxy. No previous generated
RGB result is read. An explicit image compositor retains the observed scene.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/host/space0/guo-z/tf-ufi/material_lineage_pilot_20261003')
OLD = Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
SEEDS = [41, 163, 907]
CASES = ['new_01_7442','new_02_7496','new_03_7443','new_04_7459',
         'prospective_01_7473','prospective_02_7441','prospective_03_11160','prospective_04_7498']

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save_json(p, obj):
    q = Path(str(p)+'.tmp')
    q.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    q.replace(p)

def linear(v):
    v = np.asarray(v, dtype=float)/255
    return np.where(v <= .04045, v/12.92, ((v+.055)/1.055)**2.4)

def srgb(v):
    v = np.clip(v, 0, 1)
    return np.uint8(np.clip(np.rint(255*np.where(v<=.0031308,12.92*v,1.055*v**(1/2.4)-.055)),0,255))

def raster(meshes, K):
    """One perspective-correct z-buffer for RGB material ownership and masks."""
    z = np.full((480,640), np.inf)
    normal = np.zeros((480,640,3), np.float64)
    labels = np.zeros((480,640),np.uint8)
    faces = np.full((480,640),-1,np.int32)
    for label, mesh in enumerate(meshes,1):
        v = np.asarray(mesh.vertices); p = v@K.T; uv = p[:,:2]/p[:,2:]
        for fid, f in enumerate(mesh.faces):
            pp, zz = uv[f], v[f,2]
            if np.any(zz<=0): raise ValueError('mesh crosses camera')
            low = np.maximum(np.floor(pp.min(0)).astype(int),[0,0])
            high = np.minimum(np.ceil(pp.max(0)).astype(int),[639,479])
            if np.any(high<low): continue
            x,y = np.meshgrid(np.arange(low[0],high[0]+1)+.5,np.arange(low[1],high[1]+1)+.5)
            d = (pp[1,1]-pp[2,1])*(pp[0,0]-pp[2,0])+(pp[2,0]-pp[1,0])*(pp[0,1]-pp[2,1])
            if abs(d)<1e-12: continue
            a = ((pp[1,1]-pp[2,1])*(x-pp[2,0])+(pp[2,0]-pp[1,0])*(y-pp[2,1]))/d
            b = ((pp[2,1]-pp[0,1])*(x-pp[2,0])+(pp[0,0]-pp[2,0])*(y-pp[2,1]))/d
            c = 1-a-b; iz = a/zz[0]+b/zz[1]+c/zz[2]
            nz = np.divide(1.,iz,out=np.full_like(iz,np.inf),where=iz>0)
            region = np.s_[low[1]:high[1]+1,low[0]:high[0]+1]
            take = (a>=-1e-7)&(b>=-1e-7)&(c>=-1e-7)&(nz<z[region])
            z[region][take]=nz[take]; normal[region][take]=mesh.face_normals[fid]
            labels[region][take]=label; faces[region][take]=fid
    return z, normal, labels, faces

def fresh_flags(mesh, full_planes, L):
    centers = np.asarray(mesh.triangles_center)
    normals = np.asarray(mesh.face_normals)
    dist = centers@full_planes[:,:3].T+full_planes[:,3]
    aligned = normals@full_planes[:,:3].T>.999
    return ~np.any((np.abs(dist)<L*2e-6)&aligned,axis=1)

def directional(normals, cal):
    """Positive angular interpolation of observed appearance; no albedo claim."""
    nn = np.asarray(cal['observed_plane_normals_camera'])
    cc = linear(cal['plane_median_rgb'])
    logits = 8*np.asarray(normals)@nn.T
    ww = np.exp(logits-logits.max(axis=-1,keepdims=True))
    return (ww@cc)/ww.sum(axis=-1,keepdims=True)

def query_grid(grid, xyz, low, high):
    norm = np.clip((np.asarray(xyz)-low)/(high-low),0,1)
    ijk = norm*(grid.shape[0]-1)
    return np.stack([map_coordinates(grid[...,c],ijk.T,order=1,mode='nearest',prefilter=False) for c in range(3)],axis=-1)

def hemisphere_visibility(mesh_paths, points, normals, L):
    import mitsuba as mi
    scene = mi.load_dict({'type':'scene', **{f'shape{i}':{'type':'ply','filename':str(p),'bsdf':{'type':'diffuse'}} for i,p in enumerate(mesh_paths)}})
    if not len(points): return np.empty(0)
    frame = mi.Frame3f(mi.Normal3f(normals.T))
    origin = mi.Point3f((points+normals*L*2e-5).T)
    blocked = np.zeros(len(points))
    for s in qmc.Sobol(d=2,scramble=True,seed=41).random_base2(7):
        ray = mi.Ray3f(origin,frame.to_world(mi.warp.square_to_cosine_hemisphere(mi.Point2f(s.tolist()))))
        ray.maxt = mi.Float(.75*L)
        blocked += np.asarray(scene.ray_test(ray),float)
    return 1-blocked/128

def mesh_copy(mesh, delta):
    m=mesh.copy();m.vertices=np.asarray(m.vertices)+delta;return m

def process_case(cid, cfg):
    started=time.time(); d=ROOT/cid; d.mkdir(exist_ok=False)
    g=OLD/'geometry_spoon_source_fit_ellipsoid_v3'/cid
    geom=json.loads((g/'geometry_report.json').read_text()); cal=json.loads((g/'appearance_calibration.json').read_text())
    R=np.asarray(geom['fit']['axes_camera_columns']);low=np.asarray(geom['fit']['low']);high=np.asarray(geom['fit']['high']);L=float(max(high-low))
    pose0=rigid_from_g37_report(geom['cut_and_support']); Q=pose0.rotation
    source=np.asarray(Image.open(g/'source.png').convert('RGB'))
    actual=OLD/'inputs'/cid/'original.jpg'
    mp=Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929/geometry')/cid/'maps.npz' if cid.startswith('new_') else OLD/'geometry_v3'/cid/'maps.npz'
    K=np.load(mp)['intrinsics'].copy();K[0]*=640;K[1]*=480
    meshes={n:trimesh.load(g/(n+'.ply'),process=False) for n in ['full','remaining','bite_source','bite_lifted','fork']}
    fullfood=np.asarray(meshes['full'].vertices)@R
    hull=ConvexHull(fullfood)
    # Oriented hull is used for positive quadrature only; rendered meshes stay unchanged.
    hullmesh=trimesh.Trimesh(vertices=fullfood,faces=hull.simplices,process=True);hullmesh.fix_normals()
    full_planes=hull.equations
    remainfood=meshes['remaining'].copy();remainfood.vertices=np.asarray(remainfood.vertices)@R
    bitefood=meshes['bite_source'].copy();bitefood.vertices=np.asarray(bitefood.vertices)@R
    rf=fresh_flags(remainfood,full_planes,L);bf=fresh_flags(bitefood,full_planes,L)
    assert rf.any() and bf.any()
    bitehull=ConvexHull(bitefood.vertices).equations
    def members(x):
        return np.max(x@bitehull[:,:3].T+bitehull[:,3],axis=1)<L*1e-7
    convergence=[]
    for level in [3,4,5]:
        state=sample_convex_mesh(hullmesh.vertices,hullmesh.faces,cid,refinement=level)
        selected=members(state.material_coordinates); parts=state.partition(selected)
        budget=MaterialLedger(state).validate(list(parts.values()))
        convergence.append({'refinement':level,'cells':len(state.material_ids),'carried_mass_fraction':parts['carried'].total_mass/state.total_mass})
    np.savez_compressed(d/'material_budget.npz',material_ids=np.asarray(state.material_ids),coords=state.material_coordinates,
                        root_mass=state.reference_masses,remaining_fraction=(~selected).astype(float),carried_fraction=selected.astype(float))
    budget.update(quadrature_convergence=convergence,exact_mesh_volume_bite_fraction=abs(float(bitefood.volume/hullmesh.volume)),
                  partition_scope='Centroid classification approximates cut volume; per-cell budget is exact. Uniform density and arbitrary scale.',
                  full_convex_hull_volume_delta=float(abs(hullmesh.volume)-abs(meshes['full'].volume)))
    save_json(d/'budget.json',budget)
    fullz,fulln,full_label,_=raster([meshes['full']],K)
    fy,fx=np.where(np.isfinite(fullz)); rays=np.c_[fx+.5,fy+.5,np.ones(len(fx))]@np.linalg.inv(K).T
    fullpts=rays*fullz[fy,fx,None]; fp=fullpts@R
    box=cal['source_material_box_canvas'];x0,y0,x1,y1=box
    base=np.median(linear(source[y0:y1,x0:x1]),axis=(0,1))
    foodmask=np.asarray(Image.open(OLD/'inputs'/cid/'food_mask.png'))>0
    validmask=binary_erosion(foodmask,iterations=3)[fy,fx]
    raw=linear(source[fy,fx]); illumination=directional(fulln[fy,fx],cal)
    residual=np.clip(raw/np.maximum(illumination,.05)*base,0,1)
    # Visible bare food only: exclude toppings using source color/chroma.
    validmask &= (np.max(np.abs(residual-base),axis=1)<.11)
    validmask &= np.all((fp>=low-L*1e-4)&(fp<=high+L*1e-4),axis=1)
    idx=np.flatnonzero(validmask)
    if len(idx)<128: raise ValueError((cid,'too few observed plain-source pixels',len(idx)))
    if len(idx)>30000:idx=idx[np.linspace(0,len(idx)-1,30000).round().astype(int)]
    coords=np.clip((fp[idx]-low)/(high-low),0,1).astype(np.float32)
    colors=residual[idx].astype(np.float32)
    np.savez_compressed(d/'train_samples.npz',coords=coords,colors=colors,base_color=base.astype(np.float32),
                        source_pixel_yx=np.c_[fy[idx],fx[idx]],source_linear_rgb=raw[idx],source_directional_rgb=illumination[idx])
    Image.fromarray(source).save(d/'source.png')
    inputs=[actual,g/'source.png',g/'geometry_report.json',g/'appearance_calibration.json',mp,OLD/'inputs'/cid/'food_mask.png']+[g/(n+'.ply') for n in meshes]
    save_json(d/'source_inputs.json',{'case_id':cid,'inputs':[{'path':str(p),'sha256':sha(p)} for p in inputs],
      'train_samples_sha256':sha(d/'train_samples.npz'),'source_bare_pixel_count':len(idx),
      'scope':'Only real source pixels, source-fit geometry and source-only angular appearance are read. No previous generated RGB image.',
      'appearance_proxy':'colors=clip(source_linearRGB/positive_angular_observed_planeRGB*bare_patch_base,0,1); not measured albedo.',
      'low':low.tolist(),'high':high.tolist(),'intrinsics_canvas':K.tolist(),'R_food_to_camera':R.tolist()})
    trained=train_field(d/'train_samples.npz',d/'training',seed=41,steps=cfg['optimizer_steps'],device='cuda',compute_block_size=2048,log_interval=500)
    field=load_field(trained['checkpoint_path'],device='cuda')
    G=cfg['grid_size']; nodes=np.stack(np.meshgrid(*[np.linspace(0,1,G)]*3,indexing='ij'),axis=-1).reshape(-1,3).astype(np.float32)
    ids=[f'{cid}:canonical-grid:{i}' for i in range(len(nodes))]
    fielddir=d/'fields';fielddir.mkdir()
    # Decoder constraint is frozen before field generation, from observed source.
    floor=np.quantile(colors,.005,axis=0);ceiling=np.quantile(colors,.995,axis=0)
    generated={}
    for seed in SEEDS:
        for stream in ['shared','independent_remaining']+[f'independent_carried_pose{p}' for p in range(3)]:
            noise=material_noise(ids,seed,3,stream).astype(np.float32)
            output=field.sample_material(nodes,noise,steps=24)
            bounded=np.clip(output,floor,ceiling)
            grid=bounded.reshape(G,G,G,3);generated[seed,stream]=grid
            np.savez_compressed(fielddir/f's{seed}_{stream}.npz',canonical_coords=nodes,persistent_ids=np.asarray(ids),noise=noise,
                                raw_linear_rgb=output,bounded_linear_rgb=bounded,source_floor=floor,source_ceiling=ceiling)
            print('FIELD',cid,seed,stream,flush=True)
    sharedpatch=trimesh.Trimesh(vertices=bitefood.vertices,faces=np.asarray(bitefood.faces)[bf][:,[0,2,1]],process=False)
    seam0=paired_seam_from_interface_mesh(sharedpatch.vertices,sharedpatch.faces,cid+':cut',carried_pose=pose0)
    seamcoords=seam0.remaining.material_coordinates
    # Source geometry alone fixes the action positions; no output selection.
    nominal=[np.zeros(3),np.array([.035,-.055,0])*L,np.array([-.035,.055,0])*L]
    pose_records=[]; outputrows=[]
    cropcase=next(c for c in json.loads((OLD/'inputs/manifest.json').read_text())['cases'] if c['case_id']==cid)
    left,top=cropcase['preprocessing']['pad_left_top'];w,h=cropcase['preprocessing']['resized'];rect=(left,top,left+w,top+h)
    # Remove only the projected chosen portion, never erase other source food.
    removed=np.asarray(Image.open(g/'source_bite_mask.png'))>0
    full_removed=binary_dilation(removed,iterations=1)
    basebackground=cv2.inpaint(source,np.uint8(full_removed)*255,3,cv2.INPAINT_TELEA)
    for pno,delta in enumerate(nominal):
        # Reject rather than silently clamp invalid branch geometry.
        pdir=d/f'pose{pno}';pdir.mkdir()
        meal=[meshes['remaining'],mesh_copy(meshes['bite_lifted'],delta),mesh_copy(meshes['fork'],delta)]
        paths=[]
        for n,m in zip(['remaining','carried','spoon'],meal):
            path=pdir/(n+'.ply');m.export(path);paths.append(path)
        depth,normals,labels,faceids=raster(meal,K)
        hit=labels>0; yy,xx=np.where(hit)
        pts=(np.c_[xx+.5,yy+.5,np.ones(len(xx))]@np.linalg.inv(K).T)*depth[hit][:,None]
        sourcecoords=pts@R
        carried=labels[hit]==2
        sourcecoords[carried]=pose0.inverse().apply((pts[carried]-delta)@R)
        food=(labels==1)|(labels==2)
        fresh=np.zeros(labels.shape,bool)
        rmask=labels==1;bmask=labels==2
        fresh[rmask]=rf[faceids[rmask]];fresh[bmask]=bf[faceids[bmask]]
        # Smooth the known ellipsoidal separation interface analytically.
        localnormals=normals[hit].copy();select=fresh[hit]
        center=np.asarray(geom['cut_and_support']['source_cutter_center']);radii=np.asarray(geom['cut_and_support']['cutter_radii'])
        cutoff=sourcecoords[select]-center;rz=np.where(cutoff[:,2]>=0,radii[0]*geom['cut_and_support']['upper_vertical_radius_over_bowl_radius'],radii[2])
        gradient=cutoff/np.c_[np.full(len(cutoff),radii[0]**2),np.full(len(cutoff),radii[1]**2),rz**2]
        gradient/=np.maximum(np.linalg.norm(gradient,axis=1,keepdims=True),1e-12)
        cutcar=carried[select];gradient[~cutcar]*=-1
        gradient[cutcar]=gradient[cutcar]@Q.T
        localnormals[select]=gradient@R.T
        normals[hit]=localnormals
        visibility=hemisphere_visibility(paths,pts[select],localnormals[select],L)
        ao=np.ones(labels.shape);ao[hit]=1;ao[fresh]=visibility
        # Source UV samples are used only for actually observed inherited faces.
        cp=sourcecoords@R.T; projected=cp@K.T;uv=projected[:,:2]/projected[:,2:]
        uvx=np.clip(uv[:,0]-.5,0,639);uvy=np.clip(uv[:,1]-.5,0,479)
        olddepth=map_coordinates(np.where(np.isfinite(fullz),fullz,1e9),[uvy,uvx],order=0,mode='nearest')
        observed=(np.abs(olddepth-cp[:,2])<.012*L)&~select&(labels[hit]!=3)
        observed &= (uv[:,0]>=0)&(uv[:,0]<640)&(uv[:,1]>=0)&(uv[:,1]<480)
        inherited=np.zeros(labels.shape,bool);inherited[hit]=observed
        photograph=np.stack([map_coordinates(linear(source)[...,c],[uvy,uvx],order=1,mode='nearest') for c in range(3)],axis=-1)
        # Preserve source appearance outside affected geometry, including garnish.
        edit=full_removed | (labels==2) | (labels==3) | fresh
        np.savez_compressed(pdir/'geometry_channels.npz',depth=depth,normals=normals,labels=labels,face_ids=faceids,
                            fresh=fresh,inherited=inherited,ao_visibility=ao,sourcecoords=sourcecoords,hit_yx=np.c_[yy,xx])
        Image.fromarray(np.uint8(edit)*255).save(pdir/'edit_mask.png')
        Image.fromarray(np.uint8(fresh)*255).save(pdir/'fresh_mask.png')
        Image.fromarray(labels*80).save(pdir/'labels.png')
        applied=pose0.then(RigidTransform(np.eye(3),delta@R))
        # Mass budget must survive the actual material movement as well.
        shifted=parts['carried'].transformed(applied)
        actualbudget=MaterialLedger(state).validate([parts['remaining'],shifted])
        seam=paired_seam_from_interface_mesh(sharedpatch.vertices,sharedpatch.faces,cid+':cut',carried_pose=applied)
        geometric_clearance=float(np.min(np.asarray(meal[1].vertices)@R[:,2])-high[2])
        pose_rec={'pose':pno,'joint_camera_translation':delta.tolist(),'lift_clearance_over_extent':geometric_clearance/L,
                  'visible_food_pixels':int(bmask.sum()),'visible_spoon_pixels':int((labels==3).sum()),
                  'paired_interface_material_correspondence_residual':seam.correspondence_residual(),
                  'budget_after_action':actualbudget,'contact':'Original inferred spoon support retained under joint translation; no measured contact.',
                  'unknown_source_faces':int((food&~fresh&~inherited).sum())}
        assert bmask.sum()>100 and (labels==3).sum()>100 and geometric_clearance>0
        pose_records.append(pose_rec);save_json(pdir/'pose.json',pose_rec)
        for seed in SEEDS:
            sharedcolor=query_grid(generated[seed,'shared'],seamcoords,low,high)
            remcolor=query_grid(generated[seed,'independent_remaining'],seamcoords,low,high)
            carcolor=query_grid(generated[seed,f'independent_carried_pose{pno}'],seamcoords,low,high)
            recovered=applied.inverse().apply(seam.carried.world_positions)
            actual_posecolor=query_grid(generated[seed,'shared'],recovered,low,high)
            np.savez_compressed(pdir/f'probe_s{seed}.npz',material_ids=np.asarray(seam.remaining.sample_ids),coords=seamcoords,
                                remaining_world=seam.remaining.world_positions,carried_world=seam.carried.world_positions,
                                recovered_coords=recovered,shared_remaining_rgb=sharedcolor,shared_carried_rgb=actual_posecolor,
                                independent_remaining_rgb=remcolor,independent_carried_rgb=carcolor,
                                rotation=applied.rotation,translation=applied.translation)
            for variant in ['shared','independent']:
                resultdir=pdir/f'{variant}_s{seed}';resultdir.mkdir()
                raw=np.ones((480,640,3),float)*.13
                restgrid=generated[seed,'shared' if variant=='shared' else 'independent_remaining']
                carrygrid=generated[seed,'shared' if variant=='shared' else f'independent_carried_pose{pno}']
                albedo=query_grid(restgrid,sourcecoords,low,high)
                albedo[carried]=query_grid(carrygrid,sourcecoords[carried],low,high)
                shade=directional(localnormals,cal)/np.maximum(base,.05)
                rgb=np.clip(albedo*shade,0,1)
                # Ambient visibility is an explicit shading prior, no recovered optics.
                rgb[select]*=(.60+.40*visibility)[:,None]
                rgb[observed]=photograph[observed]
                metal=labels[hit]==3
                mn=localnormals[metal];view=-pts[metal]/np.linalg.norm(pts[metal],axis=1,keepdims=True)
                facing=np.abs(np.sum(mn*view,axis=1))
                stripe=.5+.5*np.cos(10*mn[:,0]+7*mn[:,1])
                metalgray=np.clip(.13+.40*facing+.36*stripe,0,1)
                rgb[metal]=metalgray[:,None]*np.array([.98,.99,1])
                raw[hit]=rgb
                canvas=linear(source).copy();canvas[full_removed]=linear(basebackground)[full_removed]
                overlay=edit&hit;canvas[overlay]=raw[overlay]
                final=srgb(canvas)
                assert np.array_equal(final[~edit],source[~edit])
                Image.fromarray(final).save(resultdir/'composited.png')
                Image.fromarray(final).crop(rect).save(resultdir/'view.png')
                Image.fromarray(srgb(raw)).save(resultdir/'render.png')
                rec={'case_id':cid,'pose':pno,'seed':seed,'variant':variant,'final_sha256':sha(resultdir/'composited.png'),
                     'source_sha256':sha(d/'source.png'),'outside_action_source_exact':True,
                     'fresh_visible_pixels':int(fresh.sum()),'inherited_observed_pixels':int(inherited.sum()),
                     'learned_color_query':'Canonical 3D grid from actually trained epsilon MLP, with observed source quantile bounds.',
                     'scope':'3D surface shader plus explicit source-background compositor; no full-image generative model, hand, person or learned transition.',
                     'native_rect':rect}
                save_json(resultdir/'record.json',rec);outputrows.append(rec)
        print('POSE_RENDERED',cid,pno,flush=True)
    save_json(d/'manifest.json',{'status':'complete_unreviewed','case_id':cid,'poses':pose_records,'results':outputrows,
                                'training_checkpoint_sha256':sha(d/'training/material_denoiser.pt'),'seconds':time.time()-started})
    print('CASE_COMPLETE',cid,len(outputrows),time.time()-started,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--init',action='store_true');ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1);ap.add_argument('--gpu',type=int,default=0);a=ap.parse_args()
    global np,cv2,trimesh,Image,ConvexHull,map_coordinates,binary_erosion,binary_dilation,qmc
    global RigidTransform,MaterialLedger,sample_convex_mesh,rigid_from_g37_report,paired_seam_from_interface_mesh,material_noise,train_field,load_field
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    sys.path.insert(0,str(ROOT));sys.path.insert(0,'/mnt/tmp/guo-z_first_bite_structure_20260930_runtime/mitsuba_vendor')
    import numpy as np,cv2,trimesh
    from PIL import Image
    from scipy.spatial import ConvexHull
    from scipy.ndimage import map_coordinates,binary_erosion,binary_dilation
    from scipy.stats import qmc
    from foodstateedit.material_lineage import RigidTransform,MaterialLedger,sample_convex_mesh,rigid_from_g37_report,paired_seam_from_interface_mesh,material_noise
    from train_lineage_material_denoiser import train_field,load_field
    import torch,mitsuba as mi
    torch.set_num_threads(4);mi.set_variant('cuda_ad_rgb')
    if a.init:
        cfg={'status':'frozen_before_training','cases':CASES,'seeds':SEEDS,'poses':3,'variants':['shared','independent'],
             'optimizer_steps':6000,'grid_size':32,'diffusion_steps':1000,'ddim_steps':24,'training_seed':41,
             'joint_camera_translation_over_extent':[[0,0,0],[.035,-.055,0],[-.035,.055,0]],
             'expected_images':144,'scope':'Mechanism pilot, eight source-specific appearance MLPs. No full MLD, generalization, real mass, measured contact, hidden GT or photorealism claim.',
             'decoder_bounds':'Observed appearance proxy .5th/99.5th per-channel source quantiles, frozen before all field generation.',
             'background':'Original photograph; removed portion uses explicit source-only Telea reveal prior.',
             'input_geometry':'Existing source-fitted shared cut proxies; old generated RGB outputs are not inputs.',
             'script_sha256':sha(Path(__file__)),'created_unix':time.time()}
        assert not (ROOT/'config.json').exists();save_json(ROOT/'config.json',cfg);print('FROZEN',flush=True);return
    cfg=json.loads((ROOT/'config.json').read_text())
    assert cfg['script_sha256']==sha(Path(__file__)), 'Executed script differs from frozen recipe'
    rows=[]
    for i,cid in enumerate(CASES):
        if i%a.shards==a.shard:
            process_case(cid,cfg);rows.append(cid)
            save_json(ROOT/f'worker_{a.shard}.json',{'status':'running','completed_cases':rows,'gpu':a.gpu})
    save_json(ROOT/f'worker_{a.shard}.json',{'status':'complete','completed_cases':rows,'gpu':a.gpu})

if __name__=='__main__':main()
