"""Label-consistent material composition on the fixed G37 shared cutter.

CPU food visibility and GPU color raster boundaries differed by a few pixels.
Those pixels previously copied metal RGB into food. Evaluate unknown-food
material on the same CPU food normals instead; keep all original experiments.

Keep source top details. Preserve side chroma and bounded source microtexture,
and use a smooth foundation-image illumination prior. No calibrated albedo,
true-world contact, or source-exact RGB guarantee is implied by this arm.
"""
import json,time,hashlib,zipfile
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw
from scipy.ndimage import gaussian_filter,distance_transform_edt
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def smooth(values,mask,sigma):
 den=gaussian_filter(mask.astype(float),sigma)
 return np.stack([np.divide(gaussian_filter(values[...,i]*mask,sigma),den,out=np.zeros(den.shape),where=den>1e-6) for i in range(values.shape[-1])],axis=-1)
def main():
 g=ROOT/'gate_v37';out=ROOT/'gate_v40';out.mkdir(exist_ok=False)
 (out/'executed_script.py').write_bytes(Path(__file__).read_bytes())
 cfg=json.loads((g/'parent_config.json').read_text());cases={c['case_id']:c for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases']}
 parents={json.loads(p.read_text())['id']:p.parent for p in g.glob('parent_worker_*/*/result.json')};rows=[]
 (out/'frozen_plan.json').write_text(json.dumps({'created_unix':time.time(),'arms':['source_photo_cut','source_locked_cut','band_photo_cut','band_locked_cut'],'raw_calls':0,'source_side_blur_sigma':2,'side_source_detail_bound_rgb':4,'target_smooth_sigma':6,'side_illumination_gain_bounds':[.85,1.15],'top_scalar_gain_bounds':[.9,1.1],'scope':'Post-hoc development experiment. All eight cases and three seeds retained. Source consumption membership applies to appearance samples; photometric priors are not physical measurements.'},indent=2))
 for j in cfg['jobs']:
  cid=j['case_id'];geo=ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid;fp=ROOT/'source_fit_ellipsoid_food_fields_consumptive_v4'/cid/'field.npz';f=np.load(fp);a=np.load(ROOT/'spoon_source_fit_ellipsoid_channels_v3'/cid/'geometry_channels.npz');source=np.asarray(Image.open(geo/'source.png').convert('RGB'),float);proxy=np.asarray(Image.open(geo/'rgb_control.png').convert('RGB'),float);parent=parents[j['id']]/'composited.png';generated=np.asarray(Image.open(parent).convert('RGB'),float);food=f['food'];xx=f['xx'];yy=f['yy'];valid=f['valid'];top=f['top'];uv=f['source_uv'];sample=f['sampled'];face=f['face'];colors=sample.copy();lum=np.array([.2126,.7152,.0722]);consumed=np.asarray(Image.open(geo/'source_bite_mask.png'))>0;fg=np.asarray(Image.open(geo/'food_mask.png'))>0;support=consumed&fg
  # Normalized filtering cannot import appearance from unconsumed source pixels.
  low=smooth(source,support,2);mx=uv[:,0].astype(np.float32).reshape(-1,1);my=uv[:,1].astype(np.float32).reshape(-1,1);blur=cv2.remap(low.astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).reshape(-1,3);detail=np.clip(sample-blur,-4,4);fits=[]
  for plane in np.unique(face[valid]):
   region=valid&(face==plane);mask=np.zeros(food.shape,bool);mask[yy[region],xx[region]]=True
   sl=sample[region]@lum;target_luma=generated@lum;median_source=float(np.median(sl));median_target=float(np.median(target_luma[mask]));normalized=smooth(target_luma[...,None],mask,6)[...,0];gain=np.clip(normalized[yy,xx]/max(median_target,1),.85,1.15)
   side=region&~top;colors[side]=(blur[side]+detail[side])*gain[side,None]
   tp=region&top;topgain=float(np.clip(median_target/max(median_source,1),.9,1.1));colors[tp]=sample[tp]*topgain
   fits.append({'plane':int(plane),'side_pixels':int(side.sum()),'top_pixels':int(tp.sum()),'top_gain':topgain,'side_gain_median':float(np.median(gain[side])) if side.any() else None})
  c=cases[cid];l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h);edit=np.asarray(Image.open(geo/'edit_mask.png'))>0
  for arm in ['source_photo_cut','source_locked_cut','band_photo_cut','band_locked_cut']:
   base_path=parent if arm.endswith('photo_cut') else g/'geometry_locked_action_v1/photographic_cut'/(cid+'__'+str(j['seed']))/'pre_sampling.png'
   pre=np.asarray(Image.open(base_path).convert('RGB'),float);coef=np.asarray(json.loads((geo/'appearance_calibration.json').read_text())['coefficient_rgb']);predicted=np.maximum(np.c_[np.ones(int(food.sum())),a['scene_normals'][food]]@coef,.05)*255;pre[food]=predicted;selected=sample if arm.startswith('source_') else colors;pre[yy[valid],xx[valid]]=selected[valid]
   pre=np.uint8(np.clip(np.rint(pre),0,255));native=Image.fromarray(pre).crop(rect).resize(tuple(c['source_size']),Image.Resampling.LANCZOS).resize((w,h),Image.Resampling.LANCZOS);grid=source.copy();grid[t:t+h,l:l+w]=np.asarray(native);blend=np.clip(distance_transform_edt(edit)/5,0,1)[...,None];final=np.uint8(np.clip(np.rint(source*(1-blend)+grid*blend),0,255));assert np.array_equal(final[~edit],source.astype(np.uint8)[~edit]);dest=out/arm/(cid+'__'+str(j['seed']));dest.mkdir(parents=True);Image.fromarray(pre).save(dest/'pre_sampling.png');Image.fromarray(final).save(dest/'composited.png');Image.fromarray(final).crop(rect).save(dest/'view.png')
   row={'case_id':cid,'seed':j['seed'],'arm':arm,'parent_sha256':sha(parent),'base_sha256':sha(base_path),'fields_sha256':sha(fp),'source_consumed_mask_sha256':sha(geo/'source_bite_mask.png'),'raw_calls':0,'side_details_rgb_bound':4,'gpu_proxy_rgb_used_inside_food':False,'food_unknown_appearance':'Source calibrated affine RGB evaluated at CPU food normals, not GPU mixed edge colors','outside_geometry_source_exact':True,'fits':fits,'scope':'Proxy carried-food silhouette, source top texture, smoothed consumed-source side appearance with a foundation smooth-light prior. Photo-cut arm retains generated cut geometry; locked-cut arm has inferred source-cut masks. No realism acceptance by construction.'};(dest/'result.json').write_text(json.dumps(row,indent=2));rows.append(row)
  print('LABEL_CONSISTENT',cid,j['seed'],flush=True)
 for cid,c in cases.items():
  l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h);board=Image.new('RGB',(2048,1560),'white');draw=ImageDraw.Draw(board)
  for ri,arm in enumerate(['source_photo_cut','source_locked_cut','band_photo_cut','band_locked_cut']):
   paths=[('source',ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid/'source.png')]+[(arm+' '+str(s),out/arm/(cid+'__'+str(s))/'composited.png') for s in [41,163,907]]
   for i,(label,p) in enumerate(paths):
    im=Image.open(p).convert('RGB').crop(rect);im.thumbnail((504,354));board.paste(im,(i*512+(504-im.width)//2,ri*390+28));draw.text((i*512+5,ri*390+6),label,fill='black')
  board.save(out/(cid+'_review.jpg'),quality=95)
 (out/'manifest.json').write_text(json.dumps({'status':'complete_unreviewed','derived_composites':len(rows),'raw_calls':0,'rows':rows,'script_sha256':sha(Path(__file__))},indent=2))
 with zipfile.ZipFile(ROOT/'gate_v40_complete.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in out.rglob('*'):
   if p.is_file():z.write(p,p.relative_to(ROOT))
 print('LABEL_CONSISTENT_COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
