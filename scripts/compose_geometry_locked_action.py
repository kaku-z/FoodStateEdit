"""Geometry fixes the action; source pixels fix food; a model supplies appearance.

Each arm uses every seed and the same frozen shape. Masks guarantee only proxy
geometry composition, not actual 3D, true contact or photographic realism.
"""
import argparse,json,time,hashlib,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFilter
from scipy.ndimage import distance_transform_edt
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gate',default='gate_v37');ap.add_argument('--geometry',default='geometry_spoon_source_fit_ellipsoid_v3');ap.add_argument('--channels',default='spoon_source_fit_ellipsoid_channels_v3');ap.add_argument('--fields',default='source_fit_ellipsoid_food_fields_consumptive_v4');ap.add_argument('--cut-output',default='coupled_source_cut_reconstruction_v3');a=ap.parse_args();g=ROOT/a.gate;out=g/'geometry_locked_action_v1';out.mkdir(exist_ok=False);(out/'executed_script.py').write_bytes(Path(__file__).read_bytes());cfg=json.loads((g/'parent_config.json').read_text());parents={json.loads(p.read_text())['id']:p.parent for p in g.glob('parent_worker_*/*/result.json')};assert set(parents)=={j['id'] for j in cfg['jobs']};cases={c['case_id']:c for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases']};rows=[]
 (out/'frozen_plan.json').write_text(json.dumps({'created_unix':time.time(),'additional_raw_calls':0,'args':vars(a),'arms':['photographic_cut','source_color_cut'],'rule':'Source controls all unchanged pixels, expected geometry masks define carried food and spoon; consumed source surface pixels define the food. The full-scene model supplies only spoon RGB and, in the photographic-cut arm, new cut RGB. Inferred source-color/plate priors define the other cut arm. Every seed preserved.','scope':'Geometry masks and source correspondences are inferred from a single photo with source landmarks. This is an algorithmic guarantee on proxy geometry, not independent real-world validation.'},indent=2))
 for j in cfg['jobs']:
  cid=j['case_id'];geo=ROOT/a.geometry/cid;data=np.load(ROOT/a.channels/cid/'geometry_channels.npz');fields=np.load(ROOT/a.fields/cid/'field.npz');cut=np.load(g/a.cut_output/(cid+'_field')/'field.npz');source=np.asarray(Image.open(geo/'source.png').convert('RGB'));proxy=np.asarray(Image.open(geo/'rgb_control.png').convert('RGB'));food=data['labels']==2;spoon=data['labels']==3;valid=fields['valid'];xx=fields['xx'];yy=fields['yy'];parent=parents[j['id']]/'composited.png';generated=np.asarray(Image.open(parent).convert('RGB'));c=cases[cid];l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h);edit=np.asarray(Image.open(geo/'edit_mask.png'))>0
  for arm in ['photographic_cut','source_color_cut']:
   canvas=source.astype(float).copy();reconstructed=cut['rgb'].copy()
   if arm=='photographic_cut':reconstructed[cut['fresh']]=generated[cut['fresh']]
   canvas[cut['reconstruct']]=reconstructed[cut['reconstruct']];canvas[spoon]=generated[spoon];canvas[food]=proxy[food];canvas[yy[valid],xx[valid]]=fields['sampled'][valid]
   action=cut['reconstruct']|food|spoon;alpha=np.clip(distance_transform_edt(action)/1,0,1)[...,None];pre=np.uint8(np.clip(np.rint(source*(1-alpha)+canvas*alpha),0,255));assert np.array_equal(pre[~action],source[~action]);native=Image.fromarray(pre).crop(rect).resize(tuple(c['source_size']),Image.Resampling.LANCZOS).resize((w,h),Image.Resampling.LANCZOS);grid=source.copy();grid[t:t+h,l:l+w]=np.asarray(native);sampling_mask=np.asarray(Image.fromarray(np.uint8(action)*255).filter(ImageFilter.MaxFilter(5)))>0;sampling_mask&=edit;blend=np.clip(distance_transform_edt(sampling_mask)/2,0,1)[...,None];final=np.uint8(np.clip(np.rint(source*(1-blend)+grid*blend),0,255));assert np.array_equal(final[~sampling_mask],source[~sampling_mask]);dest=out/arm/(cid+'__'+str(j['seed']));dest.mkdir(parents=True);Image.fromarray(pre).save(dest/'pre_sampling.png');Image.fromarray(final).save(dest/'composited.png');Image.fromarray(final).crop(rect).save(dest/'view.png');row={'id':cid+'__'+str(j['seed']),'case_id':cid,'seed':j['seed'],'arm':arm,'parent':str(parent),'parent_sha256':sha(parent),'geometry_sha256':sha(geo/'geometry_report.json'),'source_fields_sha256':sha(ROOT/a.fields/cid/'field.npz'),'raw_generation':False,'additional_raw_calls':0,'food_only_generation_used':False,'observed_food_fraction':float(valid.mean()),'spoon_pixels':int(spoon.sum()),'food_pixels':int(food.sum()),'outside_proxy_action_source_exact_before_sampling':True,'outside_sampling_action_source_exact':True,'scope':'Proxy geometry composition with source-visible food appearance, model spoon/cut appearance and explicit hidden-material/background priors. Physical and perceptual correctness is not established by construction.'};(dest/'result.json').write_text(json.dumps(row,indent=2));rows.append(row)
 for cid,c in cases.items():
  l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h)
  for arm in ['photographic_cut','source_color_cut']:
   paths=[('source',ROOT/a.geometry/cid/'source.png')]+[(arm+' seed'+str(s),out/arm/(cid+'__'+str(s))/'composited.png') for s in [41,163,907]];board=Image.new('RGB',(2560,h+28),'white');dr=ImageDraw.Draw(board)
   for i,(title,p) in enumerate(paths):board.paste(Image.open(p).convert('RGB').crop(rect).resize((640,h),Image.Resampling.LANCZOS),(640*i,28));dr.text((640*i+5,6),title,fill='black')
   board.save(out/(cid+'_'+arm+'_three_seed.jpg'),quality=95)
 (out/'manifest.json').write_text(json.dumps({'status':'complete_unreviewed','additional_raw_calls':0,'derived_composites':len(rows),'rows':rows,'script_sha256':sha(Path(__file__))},indent=2))
 with zipfile.ZipFile(ROOT/(a.gate+'_geometry_locked_action_v1.zip'),'w',zipfile.ZIP_DEFLATED) as z:
  for p in out.rglob('*'):
   if p.is_file():z.write(p,p.relative_to(ROOT))
 print('GEOMETRY_LOCKED_ACTION',len(rows),flush=True)
if __name__=='__main__':main()
