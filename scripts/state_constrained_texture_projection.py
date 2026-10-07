"""Explicit state-layer projection baseline; neural appearance cannot remove planned food.
This is a derived compositor, not a raw generator output or proven photorealism.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import gaussian_filter,distance_transform_edt,binary_dilation
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def alpha(m,width):return np.clip(distance_transform_edt(m)/width,0,1)[...,None]
def main():
 g=ROOT/'gate_v18';cfg=json.loads((g/'config.json').read_text());cases={c['case_id']:c for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases']};out=g/'state_projection_v1';out.mkdir(exist_ok=False);rows=[]
 for j in cfg['jobs']:
  cid=j['case_id'];geo=ROOT/'geometry_spoon_open_corner_v1'/cid;ch=ROOT/'spoon_open_corner_channels_v1'/cid
  ps=list(g.glob('worker_*/'+j['id']+'/composited.png'));assert len(ps)==1,j['id'];p=ps[0]
  source=np.asarray(Image.open(geo/'source.png').convert('RGB'),float);generated=np.asarray(Image.open(p).convert('RGB'),float);proxy=np.asarray(Image.open(geo/'rgb_control.png').convert('RGB'),float)
  channels=np.load(ch/'geometry_channels.npz');labels=channels['labels'];full=channels['full_depth'];remaining=channels['remaining_depth']
  report=json.loads((geo/'geometry_report.json').read_text());fit=report['fit'];L=max(np.asarray(fit['high'])-np.asarray(fit['low']))
  delta=np.zeros_like(full);valid=np.isfinite(full)&np.isfinite(remaining);np.subtract(remaining,full,out=delta,where=valid)
  fresh=valid&(delta>.003*L)&(labels==1);bite=labels==2;utensil=labels==3
  editable=np.asarray(Image.open(j['files']['edit_mask']['path']).convert('L'))>127
  fresh &= editable;bite &= editable;utensil &= editable
  select=json.loads((ROOT/'gate_v13_run03'/cid/'reference_selection.json').read_text());x0,y0,x1,y1=select['material_box_canvas'];patch=source[y0:y1,x0:x1]
  grain=patch-gaussian_filter(patch,[1.2,1.2,0]);bound=np.clip(3*np.median(np.abs(grain.reshape(-1,3)),axis=0)*1.4826,2.,8.)
  # Only a bounded high-frequency appearance residual crosses the neural/state boundary.
  high=generated-gaussian_filter(generated,[1.2,1.2,0]);high=np.clip(high,-bound,bound);material=np.clip(proxy+high,0,255)
  target=binary_dilation(bite|utensil,iterations=3)&editable
  source_food=np.asarray(Image.open(geo/'food_mask.png'))>0
  clearing=target&~source_food
  result=generated.copy();result[clearing]=source[clearing]
  metal_alpha=alpha(utensil,1.);result=result*(1-metal_alpha)+generated*metal_alpha
  for mask in [fresh,bite]:
   a=alpha(mask,1.);result=result*(1-a)+material*a
  final=np.uint8(np.clip(np.rint(result),0,255));assert np.array_equal(final[~editable],source.astype(np.uint8)[~editable])
  d=out/j['id'];d.mkdir();Image.fromarray(final).save(d/'composited.png')
  c=cases[cid];l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h);Image.fromarray(final).crop(rect).save(d/'view.png')
  semantic=np.zeros_like(final);semantic[fresh]=[255,120,30];semantic[bite]=[0,220,80];semantic[utensil]=[30,120,255];Image.fromarray(semantic).save(d/'projected_semantic_layers.png')
  board=Image.new('RGB',(640*3,h+28),'white');draw=ImageDraw.Draw(board)
  for i,(title,im) in enumerate([('geometry appearance prior',Image.fromarray(np.uint8(proxy))),('raw-derived neural composite',Image.fromarray(np.uint8(generated))),('explicit state projection baseline',Image.fromarray(final))]):
   board.paste(im.crop(rect).resize((640,h),Image.Resampling.LANCZOS),(640*i,28));draw.text((640*i+6,6),title,fill='black')
  board.save(d/'comparison.jpg',quality=95)
  row={'id':j['id'],'raw_generation':False,'parent_composite':str(p),'parent_composite_sha256':sha(p),'visible_bite_pixels':int(bite.sum()),'fresh_cut_pixels':int(fresh.sum()),'visible_utensil_pixels':int(utensil.sum()),'source_calibrated_residual_bound_rgb':bound.tolist(),'outside_edit_source_exact':True,'geometry_masks_source':'CPU visibility/depth of the shared Boolean edit and rigidly carried portion','scope':'A deterministic state projection/compositor. Visible geometric layers are preserved by construction; photographic realism is not guaranteed. Hidden material, shape and lighting remain priors.'}
  (d/'result.json').write_text(json.dumps(row,indent=2));rows.append(row)
 (out/'manifest.json').write_text(json.dumps({'status':'complete_unreviewed','rows':rows,'all_cells_same_rule':True,'additional_raw_calls':0,'script_sha256':sha(Path(__file__))},indent=2))
 print('PROJECTED',len(rows))
if __name__=='__main__':main()
