"""Collect every two-stage stress seed with its own parent and fixed food transform."""
import argparse,json,hashlib,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--require-complete',action='store_true');a=ap.parse_args();g=ROOT/'gate_v35';cfg=json.loads((g/'config.json').read_text());cases={c['case_id']:c for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases']};found={json.loads(p.read_text())['id']:p.parent for p in g.glob('worker_*/*/result.json')};complete=set(found)=={j['id'] for j in cfg['jobs']}
 if a.require_complete:assert complete
 out=g/('collection_'+a.tag);out.mkdir(exist_ok=False);rows=[]
 for j in cfg['jobs']:
  if j['id'] not in found:continue
  transform=json.loads(Path(j['transform']).read_text());parent=Path(transform['full_composite_parent']);assert sha(parent)==transform['parent_sha256'];base=np.asarray(Image.open(parent).convert('RGB'));x0,y0,x1,y1=transform['box'];d=found[j['id']];raw=Image.open(d/'raw_rgba.png').convert('RGBA').resize((x1-x0,y1-y0),Image.Resampling.LANCZOS);patch=np.asarray(Image.alpha_composite(Image.fromarray(base).crop((x0,y0,x1,y1)).convert('RGBA'),raw).convert('RGB'),float)
  mask=np.asarray(Image.open(Path(j['transform']).parent/'full_mask.png'))>0;rectmask=np.zeros(mask.shape,bool);rectmask[y0:y1,x0:x1]=True;alpha=np.clip(np.minimum(distance_transform_edt(mask),distance_transform_edt(rectmask))/2,0,1)[...,None];canvas=base.astype(float).copy();canvas[y0:y1,x0:x1]=patch;final=np.uint8(np.clip(np.rint(base*(1-alpha)+canvas*alpha),0,255));assert np.array_equal(final[~mask],base[~mask]);dest=out/j['id'];dest.mkdir();Image.fromarray(final).save(dest/'composited.png');c=cases[j['case_id']];l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h);Image.fromarray(final).crop(rect).save(dest/'view.png')
  row={'id':j['id'],'raw_crop_sha256':sha(d/'raw_rgba.png'),'parent_sha256':sha(parent),'parent':str(parent),'seed':j['seed'],'new_full_image_raw_generation':False,'outside_food_mask_parent_exact':True,'transform':transform};(dest/'result.json').write_text(json.dumps(row,indent=2));rows.append(row)
 for cid,c in cases.items():
  tiles=[('source',Image.open(ROOT/'geometry_spoon_observed_surface_v1'/cid/'source.png')),('two-stage seed41',Image.open(out/(cid+'__observed_surface_food_intrinsic__41')/'composited.png'))]
  for seed in [163,907]:
   p=out/(cid+'__observed_surface_food_intrinsic__'+str(seed))/'composited.png'
   if p.exists():tiles.append(('two-stage seed'+str(seed),Image.open(p)))
  l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h);board=Image.new('RGB',(640*len(tiles),h+28),'white');draw=ImageDraw.Draw(board)
  for i,(title,im) in enumerate(tiles):board.paste(im.convert('RGB').crop(rect).resize((640,h),Image.Resampling.LANCZOS),(640*i,28));draw.text((640*i+5,6),title,fill='black')
  board.save(out/(cid+'_three_seed.jpg'),quality=95)
 manifest={'status':'complete_unreviewed' if complete else 'partial_snapshot','expected_parent_raw':24,'completed_parent_raw':len(list(g.glob('parent_worker_*/*/result.json'))),'expected_food_raw_crops':24,'completed_food_raw_crops':len(found),'full_image_compositions':len(rows),'rows':rows,'primary_three_seed_cells':'Twenty-four new matching-seed coupled parents plus twenty-four food crops; shared rectangular upper cut and shallow ellipsoidal lower contact; supported pose maximizes source-observed surface visibility before synthesis','all_cells_same_method':True,'script_sha256':sha(Path(__file__))};(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 if complete:(g/'completion.json').write_text(json.dumps(manifest,indent=2))
 with zipfile.ZipFile(ROOT/('gate_v35_'+a.tag+'.zip'),'w',zipfile.ZIP_DEFLATED) as z:
  for folder in [g]:
   for p in folder.rglob('*'):
    if p.is_file() and (p.suffix in ['.json','.png','.py','.jpg']):z.write(p,p.relative_to(ROOT))
 print(manifest['status'],len(found),flush=True)
if __name__=='__main__':main()


