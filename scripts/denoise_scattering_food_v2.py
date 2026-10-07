"""Explicit OptiX-derived render variant; noisy transport outputs stay intact."""
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
 from scipy.ndimage import distance_transform_edt
 mi.set_variant('cuda_ad_rgb');gate=ROOT/'gate_v24';assert json.loads((gate/'manifest.json').read_text())['status']=='complete_unreviewed'
 out=gate/'optix_source_grid_v2';out.mkdir(exist_ok=False);rows=[]
 for j in json.loads((ROOT/'gate_v20/config.json').read_text())['jobs']:
  if j['method']!='silken_food_only_canny':continue
  cid=j['case_id'];geo=ROOT/'geometry_spoon_open_corner_v1'/cid;data=np.load(ROOT/'spoon_open_corner_channels_v1'/cid/'geometry_channels.npz');food=data['labels']==2
  parent=ROOT/'gate_v20/collection_complete'/j['id']/'composited.png';base=np.asarray(Image.open(parent).convert('RGB'),float);prior=np.asarray(Image.open(geo/'rgb_control.png').convert('RGB'),float)
  report=json.loads((geo/'geometry_report.json').read_text());fit=report['fit'];action=report['cut_and_support'];R=np.asarray(fit['axes_camera_columns']);Q=np.asarray(action['rotation_food_frame']);dest=np.asarray(action['destination_center']);src=np.asarray(action['source_center']);hi=np.asarray(fit['high']);L=max(hi-np.asarray(fit['low']))
  mp=Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929/geometry')/cid/'maps.npz' if cid.startswith('new_') else ROOT/'geometry_v3'/cid/'maps.npz';K=np.load(mp)['intrinsics'].copy();K[0]*=640;K[1]*=480
  yy,xx=np.where(food);rays=np.c_[xx+.5,yy+.5,np.ones(len(xx))]@np.linalg.inv(K).T;cam=rays*data['scene_depth'][food][:,None];back=(cam@R-dest)@Q+src;n=data['scene_normals'][food]@R@Q;selected=(np.abs(back[:,2]-hi[2])<L*1e-4)&(n[:,2]>.99);top=np.zeros(food.shape,bool);top[yy[selected],xx[selected]]=True
  case=next(c for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases'] if c['case_id']==cid);l,t=case['preprocessing']['pad_left_top'];w,h=case['preprocessing']['resized'];rect=(l,t,l+w,t+h);original_size=tuple(case['source_size'])
  folder=out/cid;folder.mkdir();tiles=[('neural baseline',Image.fromarray(np.uint8(base)))]
  for optical in [30,100,300]:
   d=folder/('optical_'+str(optical));d.mkdir();noisy_path=gate/cid/('optical_'+str(optical))/'render_rgba.exr';noisy=mi.TensorXf(mi.Bitmap(str(noisy_path)));denoiser=mi.OptixDenoiser(input_size=(noisy.shape[1],noisy.shape[0]),albedo=False,normals=False,temporal=False);denoised=denoiser(noisy);mi.util.write_bitmap(str(d/'denoised.exr'),denoised,write_async=False)
   radiance=np.maximum(np.asarray(denoised)[:,:,:3],0);target=prior/255.;target=np.where(target<=.04045,target/12.92,((target+.055)/1.055)**2.4);gains=np.clip(np.median(target[food],axis=0)/np.maximum(np.median(radiance[food],axis=0),1e-6),.05,20);linear=radiance*gains;material=np.clip(np.where(linear<=.0031308,12.92*linear,1.055*np.maximum(linear,0)**(1/2.4)-.055)*255,0,255);material[top]=prior[top]
   alpha=np.clip(distance_transform_edt(food),0,1)[...,None];result=np.uint8(np.clip(np.rint(base*(1-alpha)+material*alpha),0,255));im=Image.fromarray(result);native=im.crop(rect).resize(original_size,Image.Resampling.LANCZOS);sampled=np.asarray(native.resize((w,h),Image.Resampling.LANCZOS),float);canvas=result.astype(float).copy();canvas[t:t+h,l:l+w]=sampled;result=np.uint8(np.clip(np.rint(base*(1-alpha)+canvas*alpha),0,255));assert np.array_equal(result[~food],base[~food].astype(np.uint8))
   Image.fromarray(result).save(d/'composited.png');Image.fromarray(result).crop(rect).save(d/'view.png');native.save(d/'native_resolution.png');tiles.append(('OptiX + source grid '+str(optical),Image.fromarray(result)))
   row={'case_id':cid,'optical_depth':optical,'raw_render_sha256':sha(noisy_path),'denoiser':'Mitsuba 3.9.1 OptixDenoiser color only','denoised_raw_neural_generation':False,'outside_food_parent_exact':True,'additional_neural_calls':0,'linear_exposure_rgb':gains.tolist(),'normalization_domain':'Linear radiance before sRGB and clipping','optical_constants_and_lighting_inferred':True,'camera_sampling':'Fixed source original pixel grid; no result-selected crop.'};(d/'result.json').write_text(json.dumps(row,indent=2));rows.append(row)
  board=Image.new('RGB',(640*len(tiles),h+28),'white');draw=ImageDraw.Draw(board)
  for i,(title,im) in enumerate(tiles):board.paste(im.crop(rect).resize((640,h),Image.Resampling.LANCZOS),(i*640,28));draw.text((i*640+5,6),title,fill='black')
  board.save(folder/'comparison.jpg',quality=95);print('DENOISED',cid,flush=True)
 (out/'manifest.json').write_text(json.dumps({'status':'complete_unreviewed','rows':rows,'additional_neural_calls':0,'source_top_is_flat_surface_prior':True,'script_sha256':sha(Path(__file__))},indent=2))
if __name__=='__main__':main()



