"""Freeze lossless stress controls, including a stronger 2-D similarity surrogate."""
import argparse, copy, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import build_multimaterial_pilot as base

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fit_similarity(x,y):
    u,s,vt=np.linalg.svd(x.T@y)
    fix=np.diag([1.,np.linalg.det(u@vt)])
    rotation=u@fix@vt
    scale=np.trace(np.diag(s)@fix)/np.sum(x*x)
    return scale*rotation

def render(source,case,job,arm):
    old_motion,old_transform=base.motion,base.transform
    old_spec=copy.deepcopy(base.SPECS[case])
    base.SPECS[case]['yaw']=job['yaw_radians']
    def motion(i,spec,camera):
        original,amount=old_motion(i,spec,camera)
        if i<6:return original,amount
        c=np.asarray(spec['contact'])*[camera.width,camera.height]
        end=np.asarray(spec['final'])*[camera.width,camera.height]
        return camera.backproject(c[None],1.1)[0]*(1-amount)+camera.backproject(end[None],job['end_depth'])[0]*amount,amount
    # All arms share the same contact anchor, smoothstep, source update and mask.
    # The stronger 2D arm additionally receives best-fit scale and in-plane rotation.
    _,reference_geometry=base.render_solid_case(source,case,'relative3d')
    vertices=np.array(reference_geometry['vertices_uv'])
    depths=np.array(reference_geometry['vertex_relative_depths'])
    def transform(uv,z,target,amount,spec,camera,unused):
        center=np.asarray(spec['contact'])*[camera.width,camera.height]
        anchor=camera.project(target[None])[0][0]
        if arm=='planar_translation':return np.asarray(uv)+anchor-center
        if arm=='relative3d':return old_transform(uv,z,target,amount,spec,camera,'relative3d')
        projected=old_transform(vertices,depths,target,amount,spec,camera,'relative3d')
        matrix=fit_similarity(vertices-center,projected-anchor)
        return (np.asarray(uv)-center)@matrix+anchor
    base.motion,base.transform=motion,transform
    try:
        frames,geometry=base.render_solid_case(source,case,'relative3d')
        geometry.update(treatment=arm,end_depth=job['end_depth'],yaw_radians=job['yaw_radians'])
        # Explicitly distinguish control-space diagnostics from output evaluation.
        residuals=[]
        camera=base.PinholeCamera.normalized_relative(source.shape[1],source.shape[0])
        for i in range(21):
            target,amount=motion(i,base.SPECS[case],camera)
            actual=transform(vertices,depths,target,amount,base.SPECS[case],camera,arm)
            intended=old_transform(vertices,depths,target,amount,base.SPECS[case],camera,'relative3d')
            if not np.isfinite(actual).all():raise ValueError('Nonfinite projection')
            if np.any(actual[:,0]<0) or np.any(actual[:,0]>=source.shape[1]) or np.any(actual[:,1]<0) or np.any(actual[:,1]>=source.shape[0]):raise ValueError('Payload clipped')
            residuals.append(float(np.sqrt(np.mean(np.sum((actual-intended)**2,axis=1)))))
        geometry['payload_landmark_residual_to_intended_projection_px']=residuals
        return frames,geometry
    finally:
        base.motion,base.transform=old_motion,old_transform
        base.SPECS[case]=old_spec

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    cfg=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=False)
    dataset=ROOT/'artifacts/day19_multimaterial_dataset_v3'
    meta=json.loads((dataset/'dataset_manifest.json').read_text())
    report=dict(schema='foodstateedit.spatial_stress_controls.v1',status='controls_only_not_generation',config=cfg,config_sha256=sha(a.config),builder_sha256=sha(Path(__file__)),jobs=[])
    for job in cfg['jobs']:
        folder=a.output/job['id'];folder.mkdir()
        ref=dataset/job['case']/'reference.png';source=np.array(Image.open(ref).convert('RGB'))
        rec=next(x for x in meta['cases'] if x['case_id']==job['case'])
        expected=rec['files']['reference.png']['sha256'];assert sha(ref)==expected
        Image.fromarray(source).save(folder/'reference.png')
        controls={};geoms={}
        for arm in cfg['arms']:
            frames,geoms[arm]=render(source,job['case'],job,arm);controls[arm]=np.stack(frames)
            assert len(frames)==21 and np.array_equal(frames[0],source)
        union=np.logical_or.reduce([np.any(v!=source,axis=(0,3)) for v in controls.values()])
        alpha=np.array(Image.fromarray(union.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(33)))
        Image.fromarray(alpha).save(folder/'shared_alpha.png')
        np.savez_compressed(folder/'controls.npz',**controls)
        for arm,frames in controls.items():
            assert np.all(frames[:,alpha==0]==source[alpha==0])
            Image.fromarray(frames[-1]).save(folder/(arm+'_control_final.png'))
            sheet=np.concatenate([np.concatenate([frames[i] for i in row],axis=1) for row in [[0,3,6],[10,15,20]]],axis=0)
            Image.fromarray(sheet).save(folder/(arm+'_control_review.png'))
        (folder/'geometry.json').write_text(json.dumps(geoms,indent=2))
        report['jobs'].append(dict(**job,prompt=rec['prompt']+' '+job['prompt_suffix'],negative_prompt=rec['negative_prompt'],source_sha256=expected,support_fraction=float((alpha>0).mean()),files={p.name:dict(sha256=sha(p),size_bytes=p.stat().st_size) for p in folder.iterdir()},final_control_residuals={arm:g['payload_landmark_residual_to_intended_projection_px'][-1] for arm,g in geoms.items()}))
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({j['id']:j['final_control_residuals'] for j in report['jobs']},indent=2))
if __name__=='__main__':main()
