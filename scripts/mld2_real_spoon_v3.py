"""Exploratory utensil-only correction; frozen v2 food state is reused exactly."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter
import trimesh
import pyrender
from OpenGL.arrays.ctypesparameters import CtypesParameterHandler

# The local Py3.12 OpenGL plugin registry omits ctypes byref texture IDs.
CtypesParameterHandler().register(CtypesParameterHandler.HANDLED_TYPES)

from mld2_real_geometry_v2 import closed_patch, Render, color_mesh


def utensil(bite,food_axes,translation,palette):
    up=food_axes[:,2].copy();view=np.array([0.,0.,-1.])
    tangent=view-up*(up@view);tangent/=np.linalg.norm(tangent)
    angle=min(np.deg2rad(12),np.arccos(np.clip(up@view,-1,1)))
    normal=np.cos(angle)*up+np.sin(angle)*tangent
    right=np.array([1.,0.,0.]);right-=normal*(right@normal);right/=np.linalg.norm(right)
    back=np.cross(normal,right);axes=np.stack([right,back,normal],axis=1)
    lower=(bite['low']+translation)@axes;center=lower[:,:2].mean(0)
    dimensions=np.ptp(lower[:,:2],axis=0);a=.64*max(dimensions);b=a/1.5
    radial=((lower[:,0]-center[0])/a)**2+((lower[:,1]-center[1])/b)**2
    inside=radial<.90;curvature=.14*b
    floor=float(np.min(lower[inside,2]-curvature*radial[inside]))
    x=np.linspace(-a,8*a,440);y=np.linspace(-b,b,129);xx,yy=np.meshgrid(x,y)
    u=np.clip((xx-a)/(7*a),0,1);cy=.055*b*np.sin(np.pi*u)
    width=b*(.11+.13*u**.7)
    tail=np.clip((u-.90)/.10,0,1);width*=np.sqrt(1-tail*tail)
    domain=(xx/a)**2+(yy/b)**2<=1
    domain|=(xx>=.85*a)&(np.abs(yy-cy)<=width)
    ids=np.full(xx.shape,-1,int);ids[domain]=np.arange(domain.sum())
    faces=[]
    for j in range(len(y)-1):
        for i in range(len(x)-1):
            for f in [[ids[j,i],ids[j+1,i],ids[j,i+1]],[ids[j,i+1],ids[j+1,i],ids[j+1,i+1]]]:
                if min(f)>=0:faces.append(f)
    faces=np.asarray(faces);xv=xx[domain];yv=yy[domain];uv=u[domain]
    bowl=floor+curvature*((np.minimum(xv,a)/a)**2+(yv/b)**2)
    handle=floor+curvature+.06*b*np.sin(np.pi*uv)+.06*b*uv
    local_width=np.maximum(width[domain],.001*b)
    handle+=.018*b*np.maximum(0,1-((yv-cy[domain])/local_width)**2)
    blend=np.clip((xv-.85*a)/(.25*a),0,1);blend=blend*blend*(3-2*blend)
    height=bowl*(1-blend)+handle*blend
    top=np.c_[xv+center[0],yv+center[1],height];thickness=.018*b
    bottom=top.copy();bottom[:,2]-=thickness;n=len(top)
    edges,counts=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0,return_counts=True)
    boundary=edges[counts==1]
    side=np.asarray([[p,q,q+n] for p,q in boundary]+[[p,q+n,p+n] for p,q in boundary])
    vertices=np.r_[top,bottom]@axes.T
    mesh=trimesh.Trimesh(vertices,np.r_[faces,faces[:,::-1]+n,side],process=False)
    mesh.remove_unreferenced_vertices();mesh.fix_normals(multibody=True)
    mesh.visual.vertex_colors=np.full((len(mesh.vertices),4),255,dtype=np.uint8)
    gaps=lower[inside,2]-(floor+curvature*radial[inside])
    info=dict(aspect_ratio=1.5,plate_tilt_degrees=float(np.rad2deg(angle)),bowl_normal_camera=normal.tolist(),
        radius_ab=[float(a),float(b)],shell_thickness=float(thickness),bowl_curvature_height=float(curvature),
        inferred_contact_min_gap=float(gaps.min()),inferred_penetration=float(max(0,-gaps.min())),
        food_lower_points_inside_bowl_fraction=float(inside.mean()),watertight=bool(mesh.is_watertight),
        connected_components=len(mesh.split(only_watertight=False)),
        lighting=dict(ambient=.18,strip_directional_lights=5,each_strip_intensity=.22,metallic=1.,roughness=.2),
        geometry='One joined closed shallow ellipse, narrow neck, curved widening handle and rounded tip',
        appearance='Separate smooth metallic1 roughness.2 pass, strip directional lights and source-photograph neutral palette. Inferred lighting, not measured HDR; no Qwen image pixels.')
    return mesh,info


def metal_pass(mesh,info,palette,K,w,h):
    rgb=np.asarray(palette).reshape(-1,3).astype(float)
    neutral=rgb[np.ptp(rgb,axis=1)<45]
    if len(neutral)==0:neutral=rgb
    tint=np.mean(neutral,axis=0);tint=np.clip(tint/max(1,tint.mean()),.92,1.08)
    material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[*(.74*tint),1.],metallicFactor=1.,roughnessFactor=.2)
    renderer=pyrender.OffscreenRenderer(w*2,h*2)
    scene=pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[.18,.18,.18])
    camera_pose=np.diag([1.,-1.,-1.,1.]);scene.add(pyrender.IntrinsicsCamera(K[0,0]*2,K[1,1]*2,K[0,2]*2,K[1,2]*2,znear=.001,zfar=100),pose=camera_pose)
    scene.add(pyrender.Mesh.from_trimesh(mesh,material=material,smooth=True))
    normal=np.asarray(info['bowl_normal_camera']);toward=-mesh.vertices.mean(0);toward/=np.linalg.norm(toward)
    light=2*(normal@toward)*normal-toward
    right=np.array([1.,0.,0.]);right-=normal*(normal@right);right/=np.linalg.norm(right)
    for offset in np.linspace(-.12,.12,5):
        direction=light+offset*right;direction/=np.linalg.norm(direction)
        pose=trimesh.geometry.align_vectors([0,0,-1],-direction)
        scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=.22),pose=pose)
    color,depth=renderer.render(scene,flags=pyrender.RenderFlags.RGBA|pyrender.RenderFlags.SKIP_CULL_FACES)
    renderer.delete()
    color=np.asarray(Image.fromarray(color[:,:,:3]).resize((w,h),Image.Resampling.LANCZOS))
    depth=np.asarray(Image.fromarray(depth).resize((w,h),Image.Resampling.NEAREST))
    return color,depth


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--indices',type=int,nargs='+');args=parser.parse_args()
    args.output.mkdir(exist_ok=True,parents=True);rows=[]
    for index,folder in enumerate(sorted(f for f in args.source.glob('real_*') if f.is_dir())):
        if args.indices is not None and index not in args.indices:continue
        output=args.output/folder.name;output.mkdir(exist_ok=True)
        source=np.asarray(Image.open(folder/'source.png').convert('RGB'));h,w=source.shape[:2]
        state=np.load(folder/'closure_raw.npz');metadata=json.loads((folder/'geometry_raw.json').read_text());R=state['world_axes'];translation=np.asarray(metadata['target_translation'])
        maps=dict(np.load(folder/'depth.npz'));points=maps['points'].copy();patch=np.asarray(Image.open(folder/'source_bite_mask.png'))>0
        points[patch]=state['source_camera_xyz'];K=maps['intrinsics'].copy();K[0]*=w;K[1]*=h
        bite=closed_patch(points,patch,state['floor_local'],R,state['interior_rgb'],source,state['bottom_rgb'])
        candidate_mask=np.asarray(Image.open(folder/'candidate_spoon_mask.png'))>0
        palette=source
        spoon,spoon_info=utensil(bite,R,translation,palette)
        renderer=Render(K,w,h)
        for key in ['top','side','bottom']:
            part=bite[key].copy();part.apply_translation(translation);renderer.add(part)
        food_rgb,food_depth=renderer.image()
        old_render=np.asarray(Image.open(folder/'moved_food.png').convert('RGB'))
        reconstructed_food_delta=float(np.abs(food_rgb.astype(float)-old_render.astype(float)).max())
        food_rgb=old_render
        renderer.add(bite['cavityfloor']);renderer.add(bite['cavityside']);cavity,cavity_depth=renderer.image()
        renderer.close();metal,metal_depth=metal_pass(spoon,spoon_info,palette,K,w,h)
        background=source.copy();background[patch]=cavity[patch]
        wall=(cavity_depth>0)&binary_dilation(patch,iterations=2);background[wall]=cavity[wall]
        old_food=np.asarray(Image.open(folder/'bite_mask.png'))>0;old_spoon=np.asarray(Image.open(folder/'spoon_mask.png'))>0
        valid=maps['mask']&np.isfinite(maps['points']).all(axis=-1);mask=(np.asarray(Image.open(folder/'food_mask.png'))>0)&valid
        shadow=gaussian_filter(np.roll((old_food|old_spoon).astype(float),int(.11*h),axis=0),max(2,metadata['source_bite_radius']*.30))
        shadow*=(mask|binary_dilation(mask,iterations=8))*.22
        background=np.clip(background*(1-shadow[...,None]),0,255).astype(np.uint8)
        spoon_visible=(metal_depth>0)&((food_depth==0)|(metal_depth<food_depth));food_visible=(food_depth>0)&((metal_depth==0)|(food_depth<=metal_depth))
        final=background.copy();final[spoon_visible]=metal[spoon_visible];final[food_visible]=food_rgb[food_visible]
        previous_edit=np.asarray(Image.open(folder/'frozen_edit_mask.png'))>0
        edit=previous_edit|binary_dilation(spoon_visible|food_visible,iterations=12);final[~edit]=source[~edit]
        Image.fromarray(source).save(output/'source.png');Image.fromarray(final).save(output/'final_v3.png')
        for name,array in [('spoon_mask',spoon_visible),('food_mask',food_visible),('frozen_edit_mask',edit)]:Image.fromarray(array.astype(np.uint8)*255).save(output/(name+'.png'))
        spoon.export(output/'spoon_v3.ply')
        frozen=np.load(folder/'transport_raw.npz');food_geometry_delta=float(np.abs(bite['complete'].vertices-frozen['source_points']).max())
        food_render_delta=float(np.abs(food_rgb.astype(float)-old_render.astype(float)).max())
        old_raw=np.asarray(Image.open(folder/'raw.png').convert('RGB'));mutual=food_visible&old_food
        cavity_visible=patch&~spoon_visible&~food_visible
        row=dict(case_id=folder.name,index=index,round='Exploratory v3; all first outcomes inspected',spoon=spoon_info,
            food_state_source=str(folder),food_geometry_max_change=food_geometry_delta,food_material_render_max_change=food_render_delta,
            reconstructed_food_render_max_difference=reconstructed_food_delta,
            actual_food_rgb='Frozen v2 full moved-food rendering reused exactly; no diffusion food pixels',
            mutual_visible_food_vs_v2_mae=float(np.abs(final.astype(float)-old_raw.astype(float))[mutual].mean()),
            visible_source_cavity_vs_v2_mae=float(np.abs(final.astype(float)-old_raw.astype(float))[cavity_visible].mean()) if cavity_visible.any() else 0.,
            outside_context_source_mae=float(np.abs(final.astype(float)-source.astype(float))[~edit].mean()),
            source_uv_geometry_changed=False,food_neural_generation_used=False,candidate_spoon_palette_used=False,
            Qwen_RGB_used=False,analytic_PBR_spoon_fraction=1.,food_RGB_source='Observed photograph plus frozen bounded MLD2 hidden material posterior',
            generated_food_pixels_used=0,food_visibility_changed_pixels=int((food_visible!=old_food).sum()),
            source_shadow='Fixed v2 shadow cue retained',spoon_model_role='Source-photograph neutral palette; geometry and PBR strip-light setup analytic. Qwen is a separate baseline and v2 metal-projection control, unused in final v3.')
        (output/'final_v3.json').write_text(json.dumps(row,indent=2));rows.append(row);print(json.dumps(row),flush=True)
        (args.output/'v3_manifest.json').write_text(json.dumps(dict(status='processing',cases=rows),indent=2))
    (args.output/'v3_manifest.json').write_text(json.dumps(dict(status='complete',cases=rows),indent=2))


if __name__=='__main__':main()
