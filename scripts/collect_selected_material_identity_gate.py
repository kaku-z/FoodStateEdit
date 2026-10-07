"""Collect food-only crop generations over their fixed g18 full-image parents."""
import argparse,json,hashlib,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--require-complete',action='store_true');args=ap.parse_args();g=ROOT/'gate_v28';cfg=json.loads((g/'config.json').read_text());cases={c['case_id']:c for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases']};found={}
 for p in g.glob('worker_*/*/result.json'):found[json.loads(p.read_text())['id']]=p.parent
 complete=set(found)=={j['id'] for j in cfg['jobs']}
 if args.require_complete:assert complete
 out=g/('collection_'+args.tag);out.mkdir(exist_ok=False);rows=[]
 for cid in sorted({j['case_id'] for j in cfg['jobs']}):
  context=Path(next(j['transform'] for j in cfg['jobs'] if j['case_id']==cid)).parent;transform=json.loads((context/'transform.json').read_text());parent=Path(transform['full_composite_parent']);assert sha(parent)==transform['parent_sha256'];base=np.asarray(Image.open(parent).convert('RGB'));c=cases[cid];l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h);tiles=[('fixed g20 neural parent',Image.fromarray(base))]
  for j in cfg['jobs']:
   if j['case_id']!=cid or j['id'] not in found:continue
   d=found[j['id']];x0,y0,x1,y1=transform['box'];raw=Image.open(d/'raw.png').convert('RGBA').resize((x1-x0,y1-y0),Image.Resampling.LANCZOS);under=Image.fromarray(base).crop((x0,y0,x1,y1)).convert('RGBA');patch=np.asarray(Image.alpha_composite(under,raw).convert('RGB'),float)
   mask=np.asarray(Image.open(context/'full_mask.png'))>0;cropmask=np.zeros(mask.shape,bool);cropmask[y0:y1,x0:x1]=True;alpha=np.minimum(distance_transform_edt(mask),distance_transform_edt(cropmask))/2;alpha=np.clip(alpha,0,1)[...,None];canvas=base.astype(float).copy();canvas[y0:y1,x0:x1]=patch
   result=np.uint8(np.clip(np.rint(base*(1-alpha)+canvas*alpha),0,255));assert np.array_equal(result[~mask],base[~mask]);dest=out/j['id'];dest.mkdir();Image.fromarray(result).save(dest/'composited.png');Image.fromarray(result).crop(rect).save(dest/'view.png');tiles.append((j['method'],Image.fromarray(result)))
   row={'id':j['id'],'raw_crop':str(d/'raw.png'),'raw_crop_sha256':sha(d/'raw.png'),'parent':str(parent),'parent_sha256':sha(parent),'outside_food_mask_parent_exact':True,'new_full_image_raw_generation':False,'transform':transform};(dest/'result.json').write_text(json.dumps(row,indent=2));rows.append(row)
  board=Image.new('RGB',(640*len(tiles),h+28),'white');draw=ImageDraw.Draw(board)
  for i,(title,im) in enumerate(tiles):board.paste(im.crop(rect).resize((640,h),Image.Resampling.LANCZOS),(i*640,28));draw.text((i*640+5,6),title,fill='black')
  board.save(out/(cid+'_food_only.jpg'),quality=95)
 status={'status':'complete_unreviewed' if complete else 'partial_snapshot','completed_raw_crop_outputs':len(found),'expected_raw_crop_outputs':len(cfg['jobs']),'derived_full_image_compositions':len(rows),'rows':rows,'all_cells_retained':True,'script_sha256':sha(Path(__file__))};(out/'manifest.json').write_text(json.dumps(status,indent=2))
 if complete:(g/'completion.json').write_text(json.dumps(status,indent=2))
 with zipfile.ZipFile(ROOT/('gate_v28_'+args.tag+'.zip'),'w',zipfile.ZIP_DEFLATED) as z:
  for p in g.glob('*.json'):z.write(p,p.relative_to(ROOT))
  for folder in list(found.values())+[out]:
   for p in folder.rglob('*'):
    if p.is_file():z.write(p,p.relative_to(ROOT))
 print(status['status'],len(found),flush=True)
if __name__=='__main__':main()


