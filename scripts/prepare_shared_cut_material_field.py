"""Freeze a source-color material field on an actual shared cut mesh."""
import argparse,os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gate',required=True);ap.add_argument('--geometry',required=True);ap.add_argument('--channels',required=True);ap.add_argument('--output',default='shared_cut_material_field_v3');a=ap.parse_args()
 os.environ['CUDA_VISIBLE_DEVICES']='6';sys.path.insert(0,'/mnt/tmp/guo-z_first_bite_structure_20260930_runtime/mitsuba_vendor');import mitsuba as mi,numpy as np
 from PIL import Image
 from scipy.ndimage import binary_dilation,gaussian_filter
 from scipy.stats import qmc
 mi.set_variant('cuda_ad_rgb');out=ROOT/a.gate/a.output;out.mkdir(exist_ok=False);(out/'executed_script.py').write_bytes(Path(__file__).read_bytes());rows=[]
 (out/'frozen_plan.json').write_text(json.dumps({'created_unix':time.time(),'args':vars(a),'source_only':True,'material':'65th percentile visible plain source RGB medians; fixed 70% ambient plus 30% directional appearance, bounded [.90,1.08] of that color and 10% ambient visibility shading.','scope':'Uncalibrated appearance prior, not recovered hidden material or physical scattering.'},indent=2))
 for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases']:
  cid=c['case_id'];geo=ROOT/a.geometry/cid;data=np.load(ROOT/a.channels/cid/'geometry_channels.npz');rep=json.loads((geo/'geometry_report.json').read_text());L=max(np.asarray(rep['fit']['high'])-np.asarray(rep['fit']['low']));z=data['remaining_depth'];full=data['full_depth'];finite=np.isfinite(z)&np.isfinite(full);delta=np.zeros(z.shape);np.subtract(z,full,out=delta,where=finite);cut=binary_dilation(np.asarray(Image.open(geo/'source_bite_mask.png'))>0,iterations=2);mask=finite&cut&(delta>L*1e-5)&(data['labels']==1);assert mask.sum()>50
  mp=(Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929/geometry')/cid/'maps.npz') if cid.startswith('new_') else ROOT/'geometry_v3'/cid/'maps.npz';K=np.load(mp)['intrinsics'].copy();K[0]*=640;K[1]*=480;yy,xx=np.where(mask);p=np.c_[xx+.5,yy+.5,np.ones(len(xx))]@np.linalg.inv(K).T;p*=z[mask][:,None];n=data['scene_normals'][mask].copy();n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-8);scene=mi.load_dict({'type':'scene','remaining':{'type':'ply','filename':str(geo/'remaining.ply'),'bsdf':{'type':'diffuse'}}});frame=mi.Frame3f(mi.Normal3f(n.T));origin=mi.Point3f((p+n*L*1e-5).T);occluded=np.zeros(len(p))
  for sample in qmc.Sobol(d=2,scramble=True,seed=41).random_base2(8):
   ray=mi.Ray3f(origin,frame.to_world(mi.warp.square_to_cosine_hemisphere(mi.Point2f(sample.tolist()))));ray.maxt=mi.Float(.75*L);occluded+=np.asarray(scene.ray_test(ray),float)
  ao=np.zeros(mask.shape);ao[mask]=occluded/256;ao=gaussian_filter(ao,.6)/np.maximum(gaussian_filter(mask.astype(float),.6),1e-8);ao*=mask;cal=json.loads((geo/'appearance_calibration.json').read_text());coef=np.asarray(cal['coefficient_rgb']);white=np.quantile(np.asarray(cal['plane_median_rgb']),.65,axis=0);predicted=np.c_[np.ones(len(n)),n]@coef*255;color=np.clip(.70*white+.30*predicted,.90*white,1.08*white)*(1-.10*ao[mask])[:,None];d=out/(cid+'_field');d.mkdir();np.savez_compressed(d/'field.npz',mask=mask,ao=ao,pixels_rgb=color);Image.fromarray(np.uint8(mask)*255).save(d/'cavity_mask.png');row={'case_id':cid,'fresh_surface_pixels':int(mask.sum()),'source_fitted_rgb':white.tolist(),'geometry_sha256':sha(geo/'geometry_report.json'),'scope':'Source-only RGB and ambient-visibility prior on a shared monocular cut surface.'};(d/'result.json').write_text(json.dumps(row,indent=2));rows.append(row);print('MATERIAL_FIELD',cid,flush=True)
 (out/'manifest.json').write_text(json.dumps({'status':'complete','fields_frozen_before_projection':True,'cases':rows,'script_sha256':sha(Path(__file__))},indent=2))
if __name__=='__main__':main()
