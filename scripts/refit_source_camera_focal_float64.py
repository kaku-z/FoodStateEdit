"""Float64 intrinsics for finite-difference focal derivatives.

Source-only camera/frustum diagnostic with a jointly optimized focal scale.

Landmarks are approximate and reused for fitting and diagnostics. These fits
do not provide calibrated cameras or independent reconstruction accuracy.
"""
import json, hashlib, zipfile, time
from pathlib import Path
import numpy as np, cv2
from PIL import Image, ImageDraw
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
import source_frustum_geometry as frustum
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def project(points,K):
    p=points@K.T;return p[:,:2]/p[:,2:]
def main():
    out=ROOT/'gate_v43_source_camera_float64';out.mkdir(exist_ok=False)
    (out/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    cfg={'created_unix':time.time(),'source_stage':'geometry_spoon_source_fit_ellipsoid_v3','focal_scale_bounds':[.4,2.5],'perturbation_seeds':[101,211,307,401,503],'landmark_perturbation_std_pixels':3,'new_neural_calls':0,'scope':'In-sample sensitivity diagnostic with source-only approximate annotations. No heldout or measured camera.'}
    (out/'frozen_plan.json').write_text(json.dumps(cfg,indent=2));rows=[]
    for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases']:
        cid=c['case_id'];geo=ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid;report=json.loads((geo/'geometry_report.json').read_text())['fit']
        if 'target_landmarks' not in report:
            rows.append({'case_id':cid,'status':'retained_observability_fallback','reason':'Only two bottom landmarks observable. Established constrained fit remains; no extra side annotation invented.'});continue
        R0=np.asarray(report['axes_camera_columns']);lo=np.asarray(report['low']);hi=np.asarray(report['high']);center=(lo+hi)/2;extent=hi-lo;L=max(extent);taper=np.asarray(report['bottom_taper_xy']);target=np.asarray(report['target_landmarks']);weights=np.asarray(report['landmark_weights']);indices=np.r_[np.arange(4),np.arange(4,7)]
        maps=(Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929/geometry')/cid/'maps.npz') if cid.startswith('new_') else ROOT/'geometry_v3'/cid/'maps.npz'
        K0=np.load(maps)['intrinsics'].astype(np.float64).copy();K0[0]*=640;K0[1]*=480;depth=float((center@R0.T)[2]);p0=np.r_[Rotation.from_matrix(R0).as_rotvec(),center,np.log(extent),np.log(taper),0.]
        def decode(p):
            R=Rotation.from_rotvec(p[:3]).as_matrix();e=np.exp(p[6:9]);cen=p[3:6];t=np.exp(p[9:11]);K=K0.copy();K[0,0]*=np.exp(p[11]);K[1,1]*=np.exp(p[11]);return R,cen-e/2,cen+e/2,t,K
        def solve(annotations):
            def residual(p):
                R,a,b,t,K=decode(p);cam=frustum.vertices(a,b,t)@R.T;pred=project(cam,K)[indices]
                return np.r_[((pred-annotations)*weights[:,None]/640).ravel(),.005*(R-R0).ravel(),.015*(cam[:,2].mean()-depth)/depth,.01*np.log(t),.003*p[11]]
            lower=np.r_[p0[:3]-.6,center-.7*L,np.log(extent*.35),np.log([.55,.55]),np.log(.4)]
            upper=np.r_[p0[:3]+.6,center+.7*L,np.log(extent*2.8),np.log([1.3,1.3]),np.log(2.5)]
            sol=least_squares(residual,p0,bounds=(lower,upper),max_nfev=1200,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            R,a,b,t,K=decode(sol.x);uv=project(frustum.vertices(a,b,t)@R.T,K);err=np.linalg.norm(uv[indices]-annotations,axis=1)
            return {'focal_scale':float(np.exp(sol.x[11])),'axes_camera_columns':R.tolist(),'low':a.tolist(),'high':b.tolist(),'bottom_taper_xy':t.tolist(),'intrinsics_canvas':K.tolist(),'projected_landmarks':uv[indices].tolist(),'landmark_errors_pixels':err.tolist(),'weighted_rmse_pixels':float(np.sqrt(np.mean((err*weights)**2))),'solver_success':bool(sol.success),'evaluations':int(sol.nfev)}
        fit=solve(target);perturbed=[]
        for seed in cfg['perturbation_seeds']:
            q=solve(target+np.random.default_rng(seed).normal(0,3,target.shape));perturbed.append({'seed':seed,'focal_scale':q['focal_scale'],'weighted_rmse_pixels':q['weighted_rmse_pixels']})
        old_errors=np.asarray(report['landmark_reprojection_pixels']);old_rmse=float(np.sqrt(np.mean((old_errors*weights)**2)))
        row={'case_id':cid,'status':'diagnostic_complete','old_weighted_rmse_pixels':old_rmse,'joint_fit':fit,'perturbed_annotation_fits':perturbed,'source_geometry_report_sha256':sha(geo/'geometry_report.json'),'scope':'Focal scale and solid shape are jointly inferred from reused approximate image landmarks. In-sample residual reduction is not physical correctness or model validation.'};rows.append(row)
        board=Image.new('RGB',(1280,520),'white');draw=ImageDraw.Draw(board)
        for col,(label,uv) in enumerate([('previous fixed focal',np.asarray(report['projected_landmarks'])),('joint focal '+str(round(fit['focal_scale'],3)),np.asarray(fit['projected_landmarks']))]):
            im=Image.open(geo/'source.png').convert('RGB');d=ImageDraw.Draw(im)
            for a,b in zip(target,uv):
                d.line([tuple(a),tuple(b)],fill='red',width=2);x,y=b;d.ellipse((x-3,y-3,x+3,y+3),outline='cyan',width=2);x,y=a;d.ellipse((x-3,y-3,x+3,y+3),outline='red',width=2)
            board.paste(im,(640*col,35));draw.text((640*col+5,8),label,fill='black')
        board.save(out/(cid+'_landmark_diagnostic.jpg'),quality=95);print('CAMERA_REFIT',cid,round(old_rmse,3),round(fit['weighted_rmse_pixels'],3),round(fit['focal_scale'],3),flush=True)
    (out/'report.json').write_text(json.dumps({'status':'complete_unreviewed','cases':rows,'neural_calls':0,'not_adopted_geometry':True,'script_sha256':sha(Path(__file__))},indent=2))
    with zipfile.ZipFile(ROOT/'gate_v43_source_camera_float64.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file():z.write(p,p.relative_to(ROOT))
if __name__=='__main__':main()
