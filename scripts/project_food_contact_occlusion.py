"""Differential food-occlusion shading on the already generated utensil.
Cosine hemisphere visibility is computed from the shared 3D food/utensil pose.
This is an ambient shading prior, not measured lighting or real contact truth.
"""
import os,sys,json,hashlib
from pathlib import Path
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 os.environ['CUDA_VISIBLE_DEVICES']='6';sys.path.insert(0,'/mnt/tmp/guo-z_first_bite_structure_20260930_runtime/mitsuba_vendor')
 from run_qwen_image_edit_direct_baseline import gpu_snapshot
 assert not gpu_snapshot(6)['compute_processes']
 import mitsuba as mi,numpy as np
 from PIL import Image,ImageDraw
 from scipy.ndimage import gaussian_filter,distance_transform_edt
 from scipy.stats import qmc
 mi.set_variant('cuda_ad_rgb');out=ROOT/'food_contact_occlusion_v1';out.mkdir(exist_ok=False);fields={};rows=[]
 cases={c['case_id']:c for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases']}
 for cid in cases:
  geo=ROOT/'geometry_spoon_open_corner_v1'/cid;data=np.load(ROOT/'spoon_open_corner_channels_v1'/cid/'geometry_channels.npz');metal=data['labels']==3;food=data['labels']==2
  mp=Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929/geometry')/cid/'maps.npz' if cid.startswith('new_') else ROOT/'geometry_v3'/cid/'maps.npz';K=np.load(mp)['intrinsics'].copy();K[0]*=640;K[1]*=480
  report=json.loads((geo/'geometry_report.json').read_text());L=max(np.asarray(report['fit']['high'])-np.asarray(report['fit']['low']))
  eligible=metal&(distance_transform_edt(~food)<35);yy,xx=np.where(eligible);p=np.c_[xx+.5,yy+.5,np.ones(len(xx))]@np.linalg.inv(K).T;p*=data['scene_depth'][eligible][:,None];n=data['scene_normals'][eligible].copy();n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-8)
  scene=mi.load_dict({'type':'scene','food':{'type':'ply','filename':str(geo/'bite_lifted.ply'),'bsdf':{'type':'diffuse'}}})
  samples=qmc.Sobol(d=2,scramble=True,seed=41).random_base2(8);occluded=np.zeros(len(p),float);frame=mi.Frame3f(mi.Normal3f(n.T));origin=mi.Point3f((p+n*L*1e-5).T)
  for sample in samples:
   local=mi.warp.square_to_cosine_hemisphere(mi.Point2f(sample.tolist()));direction=frame.to_world(local);ray=mi.Ray3f(origin,direction);ray.maxt=mi.Float(.75*L);occluded+=np.asarray(scene.ray_test(ray),float)
  field=np.zeros((480,640),float);field[eligible]=occluded/len(samples);field=gaussian_filter(field,.6);field*=eligible;fields[cid]=field
  folder=out/cid;folder.mkdir();Image.fromarray(np.uint8(np.clip(field,0,1)*255)).save(folder/'ambient_occlusion.png');np.save(folder/'ambient_occlusion.npy',field)
  audit={'case_id':cid,'samples_per_visible_metal_pixel':256,'eligible_visible_metal_pixels':len(p),'occlusion_fraction_max':float(field.max()),'shade_coefficient':.28,'maxt_over_food_extent':.75,'ray_epsilon_over_extent':1e-5,'scope':'Ambient visibility of the added food, excluding self-spoon and original scene occluders. Conservative scalar darkening; indirect bounce, actual lighting and friction are not measured.'};(folder/'audit.json').write_text(json.dumps(audit,indent=2));print('OCCLUSION',cid,flush=True)
 for gate_name in ['gate_v20','gate_v21']:
  gate=ROOT/gate_name;variant=gate/'contact_occlusion_v1';variant.mkdir(exist_ok=False);gate_rows=[]
  for j in json.loads((gate/'config.json').read_text())['jobs']:
   if gate_name=='gate_v20' and j['method']!='silken_food_only_canny':continue
   cid=j['case_id'];parent=gate/'collection_complete'/j['id']/'composited.png';image=np.asarray(Image.open(parent).convert('RGB'),float);field=fields[cid];shade=1-.28*field[...,None];result=np.uint8(np.clip(np.rint(image*shade),0,255));assert np.array_equal(result[field==0],image[field==0].astype(np.uint8))
   d=variant/j['id'];d.mkdir();Image.fromarray(result).save(d/'composited.png');c=cases[cid];l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h);Image.fromarray(result).crop(rect).save(d/'view.png')
   board=Image.new('RGB',(1280,h+28),'white');draw=ImageDraw.Draw(board)
   for i,(title,im) in enumerate([('food-only parent',Image.fromarray(np.uint8(image))),('shared-geometry contact shading prior',Image.fromarray(result))]):board.paste(im.crop(rect).resize((640,h),Image.Resampling.LANCZOS),(i*640,28));draw.text((i*640+5,6),title,fill='black')
   board.save(d/'comparison.jpg',quality=95);row={'id':j['id'],'raw_generation':False,'parent_sha256':sha(parent),'additional_neural_calls':0,'outside_visible_metal_shading_parent_exact':True,'photorealism_guaranteed':False};gate_rows.append(row);rows.append(row)
  (variant/'manifest.json').write_text(json.dumps({'status':'complete_unreviewed','rows':gate_rows,'additional_neural_calls':0},indent=2))
 (out/'manifest.json').write_text(json.dumps({'status':'complete_unreviewed','geometry_cases':8,'derived_primary_composites':len(rows),'shade_coefficient':.28,'scope':'3D-derived ambient contact shading prior, not measured photo illumination.','script_sha256':sha(Path(__file__))},indent=2))
if __name__=='__main__':main()
