"""Geometry diagnostics; renders are explicitly not generated photographs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
from PIL import Image, ImageDraw
import manifold3d as manifold
import pyrender
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.transform import Rotation
import trimesh


def pose(parameters):
    result = np.eye(4)
    result[:3,:3] = Rotation.from_euler("xyz", parameters[:3]).as_matrix() * parameters[3]
    result[:3,3] = [parameters[4],parameters[5],0]
    return result


class Renderer:
    def __init__(self, width=640, height=480):
        self.renderer = pyrender.OffscreenRenderer(width,height)
        self.scene = pyrender.Scene(bg_color=[0.965,0.975,0.99,1],ambient_light=[0.35]*3)
        self.camera_pose = np.eye(4)
        self.camera_pose[2,3] = 4
        self.scene.add(pyrender.OrthographicCamera(xmag=1.2*width/height,ymag=1.2,znear=0.01,zfar=10),pose=self.camera_pose)
        light_pose = np.eye(4)
        light_pose[:3,:3] = Rotation.from_euler('xyz',[0.5,-0.5,0]).as_matrix()
        self.scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2.8),pose=light_pose)
        self.nodes = []

    def add(self, mesh, color=(0.62,0.71,0.8,1), transform=None):
        material = pyrender.MetallicRoughnessMaterial(baseColorFactor=color,metallicFactor=0,roughnessFactor=0.85)
        node = self.scene.add(pyrender.Mesh.from_trimesh(mesh,material=material,smooth=False),pose=transform)
        self.nodes.append(node)
        return node

    def clear(self):
        for node in self.nodes:
            self.scene.remove_node(node)
        self.nodes=[]

    def render(self):
        rgb, depth = self.renderer.render(self.scene)
        return rgb[:,:,:3],depth

    def close(self):
        self.renderer.delete()


def register(mesh, target_mask):
    # A fitted silhouette is a reprojection diagnostic, not a calibrated camera.
    small = Renderer(160,120)
    node = small.add(mesh)
    target = np.asarray(target_mask.resize((160,120),Image.Resampling.NEAREST)) > 0
    evaluations=0
    def objective(p):
        nonlocal evaluations
        small.scene.set_pose(node,pose(p))
        depth = small.renderer.render(small.scene,flags=pyrender.RenderFlags.DEPTH_ONLY)
        pred = depth>0
        evaluations += 1
        return 1-np.logical_and(pred,target).sum()/max(1,np.logical_or(pred,target).sum())
    bounds = [(-np.pi,np.pi)]*3+[(0.5,1.5),(-0.3,0.7),(-0.4,0.4)]
    global_fit = differential_evolution(objective,bounds,seed=17,popsize=7,maxiter=28,polish=False,workers=1)
    local_fit = minimize(objective,global_fit.x,method='Powell',bounds=bounds,options={'maxiter':30,'maxfev':600,'xtol':0.0005})
    best = local_fit if local_fit.fun<global_fit.fun else global_fit
    small.close()
    return pose(best.x),{'parameters':best.x.tolist(),'fit_iou_160x120':1-float(best.fun),
                        'evaluations':evaluations,'camera':'fitted orthographic; no calibration or metric scale'}


def as_trimesh(solid):
    mesh = solid.to_mesh64()
    return trimesh.Trimesh(vertices=np.asarray(mesh.vert_properties)[:,:3],faces=np.asarray(mesh.tri_verts),process=False)


def split_and_lift(mesh):
    derived = mesh.copy()
    reversed_winding = bool(derived.volume < 0)
    if reversed_winding:
        derived.invert()
    assert derived.is_watertight, 'Raw mesh is not watertight; no silent repair allowed'
    solid = manifold.Manifold(manifold.Mesh64(np.asarray(derived.vertices,dtype=np.float64),np.asarray(derived.faces,dtype=np.uint64)))
    assert solid.status() == manifold.Error.NoError, str(solid.status())
    normal = np.array([0,-0.65,0.76]); normal /= np.linalg.norm(normal)
    projection = derived.vertices @ normal
    lo,hi = projection.min(),projection.max()
    total = solid.volume()
    for _ in range(24):
        offset = (lo+hi)/2
        bite, remaining = solid.split_by_plane(normal,offset)
        if bite.volume()/total > 0.06:
            lo=offset
        else:
            hi=offset
    bite, remaining = solid.split_by_plane(normal,(lo+hi)/2)
    b = as_trimesh(bite); remain=as_trimesh(remaining)
    width = float(max(derived.extents))
    elevation=np.deg2rad(49)
    assumed_up=np.array([0,np.cos(elevation),np.sin(elevation)])
    translation=0.45*width*assumed_up+np.array([-0.55*width,0,0])
    lifted=b.copy(); lifted.apply_translation(translation)
    floor=float((derived.vertices@assumed_up).min())
    clearance=float((lifted.vertices@assumed_up).min()-floor)
    report={'original_volume':float(total),'bite_volume':float(bite.volume()),'remaining_volume':float(remaining.volume()),
            'bite_fraction':float(bite.volume()/total),
            'relative_volume_residual':abs(bite.volume()+remaining.volume()-total)/total,
            'partition_overlap_volume':float((bite ^ remaining).volume()),
            'bite_watertight':bool(b.is_watertight),'remaining_watertight':bool(remain.is_watertight),
            'bite_components':len(b.split(only_watertight=False,repair=False)),
            'manifold_constructor_relative_volume_change':abs(total-float(derived.volume))/abs(float(derived.volume)),
            'plane_normal':normal.tolist(),'plane_offset':float((lo+hi)/2),
            'translation':translation.tolist(),'assumed_up':assumed_up.tolist(),
            'clearance_over_assumed_support_plane':clearance,
            'clearance_over_food_extent':clearance/width,
            'winding_reversed_for_boolean':reversed_winding,
            'source_correspondence':'Remaining and bite are complementary subsets of the same reconstructed mesh.',
            'limits':'Plane and lift are hand-designed geometry commands. No utensil support, deformation, calibrated plate plane, or food mass measurement.'}
    return remain,b,lifted,report


def label(image,title):
    panel=Image.new('RGB',(640,520),'white')
    panel.paste(Image.fromarray(image) if isinstance(image,np.ndarray) else image,(0,40))
    ImageDraw.Draw(panel).text((14,12),title,fill=(20,34,54))
    return panel


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output-name',default='geometry_review_v2')
    parser.add_argument('--reuse-registration-from',type=Path,
                        help='Optional prior report with the same source and raw mesh hashes.')
    args=parser.parse_args()
    root=args.root
    run=json.loads((root/'results/shape_v1/run_manifest.json').read_text())
    out=root/'results'/args.output_name;out.mkdir(exist_ok=False)
    source=Image.open(root/'inputs/source.jpg').convert('RGB')
    mask=Image.open(root/'inputs/food_mask.png').convert('L')
    prior=json.loads(args.reuse_registration_from.read_text()) if args.reuse_registration_from else None
    reports=[]
    for job in run['jobs']:
        if job['status']!='complete':continue
        seed=job['seed'];file=root/f'results/shape_v1/seed_{seed}_raw.ply'
        assert hashlib.sha256(file.read_bytes()).hexdigest()==job['output_sha256']
        mesh=trimesh.load(file,force='mesh',process=False)
        report={'seed':seed,'raw_sha256':job['output_sha256'],'raw_watertight':bool(mesh.is_watertight)}
        full=None
        try:
            areas=mesh.area_faces
            raw_volume=float(mesh.volume)
            mesh.update_faces(areas>0.0)
            mesh.remove_unreferenced_vertices()
            report['postprocessing']={'policy':'Explicit supplementary pass: remove exactly zero-area faces only; no hole filling or positive-area face removal.',
                'removed_zero_area_faces':int((areas==0.0).sum()),'watertight_after_cleanup':bool(mesh.is_watertight),
                'relative_volume_change':abs(float(mesh.volume)-raw_volume)/abs(raw_volume),
                'components_after_cleanup':len(mesh.split(only_watertight=False,repair=False))}
            if prior is None:
                transform,fit=register(mesh,mask)
            else:
                old=next(p for p in prior['results'] if p['seed']==seed)
                assert old['raw_sha256']==job['output_sha256']
                fit=old['registration'].copy();transform=pose(fit['parameters'])
                fit['reused_from']=str(args.reuse_registration_from)
            report['registration']=fit
            registered=mesh.copy();registered.apply_transform(transform)
            registered.export(out/f'seed_{seed}_registered.glb')
            full=Renderer()
            full.clear();full.add(registered);front,depth=full.render()
            pred=depth>0;truth=np.asarray(mask)>0
            fit['fit_iou_640x480']=float(np.logical_and(pred,truth).sum()/np.logical_or(pred,truth).sum())
            # Projection overlay only; never presented as a generated real photo.
            overlay=np.asarray(source).copy();overlay[pred]=(0.5*overlay[pred]+0.5*np.array([20,160,220])).astype(np.uint8)
            Image.fromarray(overlay).save(out/f'seed_{seed}_projection_overlay.png')
            center=registered.centroid
            rotated=registered.copy();rotated.vertices=(rotated.vertices-center)@Rotation.from_euler('y',65,degrees=True).as_matrix().T
            full.clear();full.add(rotated);side,_=full.render()
            Image.fromarray(front).save(out/f'seed_{seed}_front.png')
            Image.fromarray(side).save(out/f'seed_{seed}_side.png')
            remain,bite,lifted,cut=split_and_lift(registered);report['cut']=cut
            remain.export(out/f'seed_{seed}_remaining.ply');bite.export(out/f'seed_{seed}_bite_at_source.ply');lifted.export(out/f'seed_{seed}_bite_lifted.ply')
            full.clear();full.add(remain);full.add(lifted,(0.95,0.48,0.14,1));cut_image,cut_depth=full.render()
            scene=trimesh.Scene()
            remain.visual.face_colors=[130,165,195,255];lifted.visual.face_colors=[245,135,45,255]
            scene.add_geometry(remain,node_name='remaining');scene.add_geometry(lifted,node_name='lifted_bite')
            scene.export(out/f'seed_{seed}_cut_lift.glb')
            for name,array in [('front',front),('side',side),('cut_lift',cut_image)]:Image.fromarray(array).save(out/f'seed_{seed}_{name}.png')
            panels=[label(source,'REAL INPUT | original UECFOOD photograph'),label(front,f'PREDICTED MESH | seed {seed} | fitted view'),label(side,'PREDICTED MESH | rotated 65 degrees'),label(cut_image,'GEOMETRY DEMO | same bite cut and translated | no texture')]
            board=Image.new('RGB',(1280,1040),'white')
            for i,panel in enumerate(panels):board.paste(panel,((i%2)*640,(i//2)*520))
            board.save(out/f'seed_{seed}_overview.jpg',quality=95)
            report['status']='complete'
        except Exception:
            report.update(status='failed',error=traceback.format_exc());print(report['error'],flush=True)
        finally:
            if full is not None:
                full.close()
        reports.append(report)
        (out/'geometry_report.json').write_text(json.dumps({'scope':'Single-image development geometry diagnostic, not 3D accuracy or final image evaluation.','analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':reports},indent=2)+'\n')
        print('ANALYZED',seed,report['status'],flush=True)


if __name__=='__main__':main()
