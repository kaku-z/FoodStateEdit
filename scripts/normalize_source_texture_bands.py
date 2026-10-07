"""Source-observed texture-band attenuation on neural food, all cells with one fixed rule."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import gaussian_filter,distance_transform_edt,binary_erosion
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def blur(img,mask,sigma):
 weight=gaussian_filter(mask.astype(float),sigma,mode='nearest')
 return gaussian_filter(img*mask[...,None],[sigma,sigma,0],mode='nearest')/np.maximum(weight[...,None],1e-6)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gate',required=True);args=ap.parse_args();g=ROOT/args.gate;cfg=json.loads((g/'config.json').read_text());collection=g/'collection_complete';assert collection.exists();out=g/'source_texture_bands_v1';out.mkdir(exist_ok=False);rows=[];sigmas=[.6,1.2,2.4,4.8]
 for j in cfg['jobs']:
  cid=j['case_id'];p=collection/j['id']/'composited.png';generated=np.asarray(Image.open(p).convert('RGB'),float);source=np.asarray(Image.open(ROOT/'geometry_spoon_open_corner_v1'/cid/'source.png').convert('RGB'),float)
  mask=np.load(ROOT/'spoon_open_corner_channels_v1'/cid/'geometry_channels.npz')['labels']==2;safe=binary_erosion(mask,iterations=5)
  if safe.sum()<100:safe=binary_erosion(mask,iterations=2)
  assert safe.any()
  selection=json.loads((ROOT/'gate_v13_run03'/cid/'reference_selection.json').read_text());x0,y0,x1,y1=selection['material_box_canvas'];patch=source[y0:y1,x0:x1]
  base=generated.copy();patchbase=patch.copy();bands=[];audit=[]
  for sigma in sigmas:
   smooth=blur(generated,mask,sigma);patchsmooth=gaussian_filter(patch,[sigma,sigma,0],mode='reflect')
   band=base-smooth;source_band=patchbase-patchsmooth
   # Match robust source material variation; permit less attenuation where the source supports that texture.
   source_sigma=np.median(np.abs(source_band.reshape(-1,3)-np.median(source_band.reshape(-1,3),axis=0)),axis=0)*1.4826
   neural_sigma=np.median(np.abs(band[safe]-np.median(band[safe],axis=0)),axis=0)*1.4826
   ratio=np.minimum(1,np.maximum(source_sigma,.25)/np.maximum(neural_sigma,1e-6));bands.append(band*ratio);audit.append(dict(sigma_canvas_pixels=sigma,source_band_mad_rgb=source_sigma.tolist(),neural_band_mad_rgb=neural_sigma.tolist(),attenuation_ratio_rgb=ratio.tolist()))
   base=smooth;patchbase=patchsmooth
  normalized=base+sum(bands);alpha=np.clip(distance_transform_edt(mask)/4,0,1)[...,None]
  result=np.uint8(np.clip(np.rint(generated*(1-alpha)+normalized*alpha),0,255));assert np.array_equal(result[~mask],generated.astype(np.uint8)[~mask]);d=out/j['id'];d.mkdir();Image.fromarray(result).save(d/'composited.png')
  c=next(c for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases'] if c['case_id']==cid);l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h);Image.fromarray(result).crop(rect).save(d/'view.png');board=Image.new('RGB',(1280,h+28),'white');draw=ImageDraw.Draw(board)
  for i,(title,im) in enumerate([('food-only neural composite',Image.fromarray(np.uint8(generated))),('source texture-band attenuation',Image.fromarray(result))]):board.paste(im.crop(rect).resize((640,h),Image.Resampling.LANCZOS),(i*640,28));draw.text((i*640+6,6),title,fill='black')
  board.save(d/'comparison.jpg',quality=95);row={'id':j['id'],'raw_generation':False,'parent_sha256':sha(p),'source_only_material_box':selection['material_box_canvas'],'source_material_low_observability':selection['material_reference_low_observability'],'bands':audit,'outside_carried_food_parent_exact':True,'scope':'Attenuate unsupported image-frequency variation using source-visible material statistics; low-frequency neural shading and geometric outline remain. This is a derived material-consistency hypothesis, not recovered hidden texture or human realism score.'};(d/'result.json').write_text(json.dumps(row,indent=2));rows.append(row)
 (out/'manifest.json').write_text(json.dumps({'status':'complete_unreviewed','rows':rows,'all_cells_same_rule':True,'additional_raw_calls':0,'script_sha256':sha(Path(__file__))},indent=2));print('NORMALIZED',len(rows))
if __name__=='__main__':main()
