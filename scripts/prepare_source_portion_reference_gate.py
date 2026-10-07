"""Source-visible selected-portion reference, not an entire meal or hidden cut texture."""
import json,hashlib,copy
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from prepare_spoon_geometry_channels_v3 import rasterize
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def entry(p):return {'path':str(p),'sha256':sha(p)}
def main():
 g=ROOT/'gate_v22';g.mkdir(exist_ok=False);base=json.loads((ROOT/'gate_v20/config.json').read_text());jobs=[]
 for prototype in base['jobs']:
  if prototype['method']!='silken_food_only_canny':continue
  j=copy.deepcopy(prototype);cid=j['case_id'];geometry=ROOT/'geometry_spoon_open_corner_v1'/cid;d=g/cid;d.mkdir()
  mp=Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929/geometry')/cid/'maps.npz' if cid.startswith('new_') else ROOT/'geometry_v3'/cid/'maps.npz';maps=np.load(mp);K=maps['intrinsics'].copy();K[0]*=640;K[1]*=480
  channels=np.load(ROOT/'spoon_open_corner_channels_v1'/cid/'geometry_channels.npz');full=channels['full_depth'];piece,_,_=rasterize(geometry,K,['bite_source']);report=json.loads((geometry/'geometry_report.json').read_text());L=max(np.asarray(report['fit']['high'])-np.asarray(report['fit']['low']))
  good=np.isfinite(full)&np.isfinite(piece);delta=np.full_like(full,np.inf);np.subtract(piece,full,out=delta,where=good)
  food=np.asarray(Image.open(geometry/'food_mask.png'))>0;visible=good&(np.abs(delta)<.002*L)&food
  assert visible.sum()>100,(cid,visible.sum());yy,xx=np.where(visible);box=(max(0,int(xx.min())-4),max(0,int(yy.min())-4),min(640,int(xx.max())+5),min(480,int(yy.max())+5))
  source=np.asarray(Image.open(geometry/'source.png').convert('RGB'));reference=np.full_like(source,127);reference[visible]=source[visible]
  Image.fromarray(reference).crop(box).resize((512,512),Image.Resampling.LANCZOS).save(d/'source_portion_reference.png');Image.fromarray(np.uint8(visible)*255).save(d/'source_visible_portion_mask.png')
  (d/'reference_provenance.json').write_text(json.dumps({'source_sha256':sha(geometry/'source.png'),'source_camera_crop_box':box,'observed_selected_portion_pixels':int(visible.sum()),'selection':'Same shared Boolean bite at source; visible only where its depth matches the full food, within original food mask. No output used to select reference pixels.','background':'Neutral127 outside observed portion, not recovered hidden texture.','scope':'Source-visible portion material/identity cue; source garnish depth and unobserved cut material remain priors.'},indent=2))
  j.update(id=cid+'__silken_food_source_portion_reference__41',method='silken_food_source_portion_reference',reference_keys=['reference_portion']);j['files']['reference_portion']=entry(d/'source_portion_reference.png');j['prompt']='The reference contains only source-photographed pixels belonging to the selected first portion, isolated on neutral gray. Preserve this food and its actual sauce or garnish appearance; use the edge control for its new shape and pose on the existing spoon. '+j['prompt'];jobs.append(j)
 cfg=dict(base,jobs=jobs,stage='development_source_visible_portion_reference_identity');(g/'config.json').write_text(json.dumps(cfg,indent=2));(g/'freeze.json').write_text(json.dumps({'expected_raw_crops':8,'config_sha256':sha(g/'config.json'),'script_sha256':sha(Path(__file__)),'comparison':'Same fixed g18 parents, food masks, Canny, seed41 and gel prompt as v20 no-reference primary; only source-visible selected-portion reference is added.','all_cells_retained':True},indent=2));print('FROZEN8')
if __name__=='__main__':main()
