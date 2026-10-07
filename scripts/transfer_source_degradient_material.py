"""Remove a fitted smooth source illumination field before target-light transfer.

These are constrained RGB lighting approximations, not recovered physical
albedo or measured relighting. Texture still samples the same consumed source.
"""
import json,time,hashlib,zipfile
from pathlib import Path
import cv2,numpy as np,trimesh
from PIL import Image,ImageDraw
from scipy.ndimage import gaussian_filter,distance_transform_edt,binary_erosion
from scipy.spatial import ConvexHull
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 for i in range(10):
  try:return json.loads(p.read_text())
  except (OSError,json.JSONDecodeError):time.sleep(.3)
 raise RuntimeError(str(p))
def main():
 while read(ROOT/'gate_v37/postprocessing_execution.json')['status']!='complete_unreviewed':time.sleep(20)
 out=ROOT/'gate_v38';out.mkdir(exist_ok=False);(out/'executed_script.py').write_bytes(Path(__file__).read_bytes());cases={c['case_id']:c for c in read(ROOT/'inputs/manifest.json')['cases']};rows=[]
 (out/'frozen_plan.json').write_text(json.dumps({'created_unix':time.time(),'additional_raw_calls':0,'seeds':[41,163,907],'cases':8,'source_illumination_sigma_canvas':12,'target_illumination_sigma_canvas':6,'scalar_gain_bounds':[.65,1.4],'scope':'RGB low-frequency gradient correction with a fixed smooth-light prior. Source texture/chroma is preserved; no calibrated albedo/lighting claim. All cells retained.'},indent=2))
 for gate,geometry,field_name in [('gate_v36','geometry_spoon_source_fit_v2','source_fit_food_fields_consumptive_v3'),('gate_v37','geometry_spoon_source_fit_ellipsoid_v3','source_fit_ellipsoid_food_fields_consumptive_v4')]:
  g=ROOT/gate;parents={read(p)['id']:p.parent for p in g.glob('parent_worker_*/*/result.json')}
  for cid,c in cases.items():
   geo=ROOT/geometry/cid;field=np.load(ROOT/field_name/cid/'field.npz');source=np.asarray(Image.open(geo/'source.png').convert('RGB'),float);food=field['food'];valid=field['valid'];xx=field['xx'];yy=field['yy'];uv=field['source_uv'];face=field['face'];sampled=field['sampled'];rep=read(geo/'geometry_report.json');R=np.asarray(rep['fit']['axes_camera_columns']);planes=ConvexHull(np.asarray(trimesh.load(geo/'full.ply',process=False).vertices)@R).equations
   mp=(Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929/geometry')/cid/'maps.npz') if cid.startswith('new_') else ROOT/'geometry_v3'/cid/'maps.npz';maps=np.load(mp);norm=np.asarray(maps['normal']);foreground=binary_erosion(np.asarray(Image.open(geo/'food_mask.png'))>0,iterations=2)&maps['mask'];ref=read(ROOT/'gate_v13_run03'/cid/'reference_selection.json');a,b,d,e=ref['material_box_canvas'];white=np.median(source[b:e,a:d].reshape(-1,3),axis=0);chroma=source/np.maximum(source.sum(2)[...,None],1);white_chroma=white/white.sum();plain=foreground&(np.linalg.norm(chroma-white_chroma,axis=2)<.045);lum=source@np.array([.2126,.7152,.0722]);source_light={};source_counts={}
   for plane in np.unique(face[valid]):
    n=planes[plane,:3]@R.T;weight=plain&(norm@n>.90);count=int(weight.sum());median=float(np.median(lum[weight])) if count>=20 else float(np.median(field['before'][valid&(face==plane)]@np.array([.2126,.7152,.0722])));weight&=(lum>.65*median)&(lum<min(254.,1.3*median));den=gaussian_filter(weight.astype(float),12);num=gaussian_filter(lum*weight,12);smooth=np.divide(num,den,out=np.full(lum.shape,median),where=den>1e-4);smooth=np.clip(smooth,.65*median,1.3*median);value=cv2.remap(smooth.astype(np.float32),uv[:,0].astype(np.float32).reshape(-1,1),uv[:,1].astype(np.float32).reshape(-1,1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).ravel();source_light[plane]=value;source_counts[int(plane)]=int(weight.sum())
   for seed in [41,163,907]:
    if gate=='gate_v36':jid=cid+'__source_fit_food_intrinsic__'+str(seed);base_path=g/'source_fit_surface_consumptive_v3/source_rgb_unrelit'/jid/'transported.png';target_path=g/'collection_complete'/jid/'composited.png'
    else:
     jid=cid+'__source_fit_ellipsoid_spoon_photo_canny__'+str(seed);base_path=g/'geometry_locked_action_v1/photographic_cut'/(cid+'__'+str(seed))/'pre_sampling.png';target_path=parents[jid]/'composited.png'
    base=np.asarray(Image.open(base_path).convert('RGB'),float);target=np.asarray(Image.open(target_path).convert('RGB'),float);tlum=target@np.array([.2126,.7152,.0722]);tchroma=target/np.maximum(target.sum(2)[...,None],1);gain=np.ones(len(xx));fits=[]
    for plane in np.unique(face[valid]):
     region=valid&(face==plane);mask=np.zeros(food.shape,bool);mask[yy[region],xx[region]]=True;median=float(np.median(tlum[mask]));weight=mask&(np.linalg.norm(tchroma-white_chroma,axis=2)<.045)&(tlum>.75*median)&(tlum<min(254.,1.25*median));den=gaussian_filter(weight.astype(float),6);num=gaussian_filter(tlum*weight,6);fallback=float(np.quantile(tlum[mask],.7));smooth=np.divide(num,den,out=np.full(tlum.shape,fallback),where=den>1e-4);predicted=np.clip(smooth[yy,xx],.7*fallback,min(255.,1.3*fallback));gain[region]=np.clip(predicted[region]/np.maximum(source_light[plane][region],1),.65,1.4);fits.append({'plane':int(plane),'source_plain_pixels':source_counts[int(plane)],'target_plain_pixels':int(weight.sum()),'gain_median':float(np.median(gain[region]))})
    pre=base.copy();pre[yy[valid],xx[valid]]=sampled[valid]*gain[valid,None];pre=np.uint8(np.clip(np.rint(pre),0,255));observed=np.zeros(food.shape,bool);observed[yy[valid],xx[valid]]=True;assert np.array_equal(pre[~observed],base.astype(np.uint8)[~observed]);l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h);edit=np.asarray(Image.open(geo/'edit_mask.png'))>0;native=Image.fromarray(pre).crop(rect).resize(tuple(c['source_size']),Image.Resampling.LANCZOS).resize((w,h),Image.Resampling.LANCZOS);grid=source.copy();grid[t:t+h,l:l+w]=np.asarray(native);blend=np.clip(distance_transform_edt(edit)/5,0,1)[...,None];final=np.uint8(np.clip(np.rint(source*(1-blend)+grid*blend),0,255));assert np.array_equal(final[~edit],source.astype(np.uint8)[~edit]);dest=out/gate/(cid+'__'+str(seed));dest.mkdir(parents=True);Image.fromarray(pre).save(dest/'pre_sampling.png');Image.fromarray(final).save(dest/'composited.png');Image.fromarray(final).crop(rect).save(dest/'view.png');row={'case_id':cid,'seed':seed,'geometry_gate':gate,'base':str(base_path),'base_sha256':sha(base_path),'target_light_parent':str(target_path),'target_light_parent_sha256':sha(target_path),'raw_calls':0,'source_fields_sha256':sha(ROOT/field_name/cid/'field.npz'),'source_consumption_supported_samples':int(valid.sum()),'source_chroma_preserved_by_scalar_gain':True,'outside_observed_surface_base_exact_before_sampling':True,'outside_geometry_source_exact':True,'source_and_target_light_fits':fits,'scope':'Approximate scalar RGB gradient correction on known consumed source samples; not measured albedo, relighting or physical realism.'};(dest/'result.json').write_text(json.dumps(row,indent=2));rows.append(row)
   paths=[('source',geo/'source.png')]+[('source gradient corrected '+str(s),out/gate/(cid+'__'+str(s))/'composited.png') for s in [41,163,907]];l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h);board=Image.new('RGB',(2560,h+28),'white');dr=ImageDraw.Draw(board)
   for i,(title,p) in enumerate(paths):board.paste(Image.open(p).convert('RGB').crop(rect).resize((640,h),Image.Resampling.LANCZOS),(i*640,28));dr.text((i*640+5,6),title,fill='black')
   board.save(out/(gate+'_'+cid+'_three_seed.jpg'),quality=95);print('RELIGHT',gate,cid,flush=True)
 (out/'manifest.json').write_text(json.dumps({'status':'complete_unreviewed','additional_raw_calls':0,'derived_composites':len(rows),'rows':rows,'script_sha256':sha(Path(__file__))},indent=2))
 with zipfile.ZipFile(ROOT/'gate_v38_complete.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in out.rglob('*'):
   if p.is_file():z.write(p,p.relative_to(ROOT))
 print('RGB_GRADIENT_CORRECTION_COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
