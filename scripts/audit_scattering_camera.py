"""Verify actual Mitsuba sensor rays against the CPU shared-geometry depth.
This checks implementation agreement, not agreement with measured real 3D.
"""
import os,sys,json,hashlib
from pathlib import Path
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def main():
 os.environ['CUDA_VISIBLE_DEVICES']='6';sys.path.insert(0,'/mnt/tmp/guo-z_first_bite_structure_20260930_runtime/mitsuba_vendor')
 from run_qwen_image_edit_direct_baseline import gpu_snapshot
 assert not gpu_snapshot(6)['compute_processes']
 import mitsuba as mi,numpy as np
 mi.set_variant('cuda_ad_rgb');rows=[]
 for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases']:
  cid=c['case_id'];g=ROOT/'geometry_spoon_open_corner_v1'/cid;mp=Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929/geometry')/cid/'maps.npz' if cid.startswith('new_') else ROOT/'geometry_v3'/cid/'maps.npz';K=np.load(mp)['intrinsics'].copy();K[0]*=640;K[1]*=480
  report=json.loads((g/'geometry_report.json').read_text());L=max(np.asarray(report['fit']['high'])-np.asarray(report['fit']['low']))
  sensor={'type':'perspective','fov':float(np.degrees(2*np.arctan(320/K[0,0]))),'fov_axis':'x','to_world':mi.ScalarTransform4f().scale([-1,-1,1]),'near_clip':.001*L,'far_clip':100*L,'film':{'type':'hdrfilm','width':640,'height':480,'pixel_format':'rgba','rfilter':{'type':'box'}}}
  config={'type':'scene','sensor':sensor}
  for name in ['remaining','bite_lifted','fork']:config[name]={'type':'ply','filename':str(g/(name+'.ply')),'bsdf':{'type':'diffuse'}}
  scene=mi.load_dict(config);data=np.load(ROOT/'spoon_open_corner_channels_v1'/cid/'geometry_channels.npz');ids=np.flatnonzero(data['labels'].reshape(-1)>0);selected=ids[np.linspace(0,len(ids)-1,min(1500,len(ids))).astype(int)];yy,xx=np.unravel_index(selected,(480,640))
  ray,_=scene.sensors()[0].sample_ray(0,.5,mi.Point2f([(xx+.5)/640,(yy+.5)/480]),mi.Point2f(.5,.5));direction=np.asarray(ray.d).T;expected=np.c_[xx+.5,yy+.5,np.ones(len(xx))]@np.linalg.inv(K).T;expected/=np.linalg.norm(expected,axis=1,keepdims=True)
  direction_error=float(np.max(np.linalg.norm(direction-expected,axis=1)));assert direction_error<2e-6,(cid,direction_error)
  hit=scene.ray_intersect(ray);valid=np.asarray(hit.is_valid());assert valid.all(),(cid,valid.mean());z=np.asarray(hit.p)[2];depth_error=np.abs(z-data['scene_depth'][yy,xx])/L
  assert depth_error.max()<1e-4,(cid,float(depth_error.max()))
  row={'case_id':cid,'sampled_visible_geometry_pixels':len(xx),'actual_sensor_ray_direction_error_max':direction_error,'camera_depth_error_max_over_food_extent':float(depth_error.max()),'camera_depth_error_median_over_food_extent':float(np.median(depth_error)),'implementation_agreement_pass':True,'real_geometry_accuracy_evaluated':False};rows.append(row);print('CAMERA_PASS',cid,flush=True)
 (ROOT/'gate_v24/camera_audit.json').write_text(json.dumps({'status':'complete','rows':rows,'all_eight_pass':True,'scope':'Actual Mitsuba camera and ray/mesh intersections versus the same inferred CPU geometry. No measured 3D or real paired truth.','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
if __name__=='__main__':main()
