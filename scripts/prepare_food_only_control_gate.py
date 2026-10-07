"""Freeze food-only regional generation over a fixed full-image spoon/removal parent."""
import json,hashlib,copy
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def entry(p):return {'path':str(p),'sha256':sha(p)}
def main():
 g=ROOT/'gate_v19';g.mkdir(exist_ok=False);cases={c['case_id']:c for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases']};previous=json.loads((ROOT/'gate_v18/config.json').read_text());jobs=[]
 for proto in previous['jobs']:
  cid=proto['case_id'];c=cases[cid];geometry=ROOT/'geometry_spoon_open_corner_v1'/cid;channels=ROOT/'spoon_open_corner_channels_v1'/cid
  parents=list((ROOT/'gate_v18').glob('worker_*/'+proto['id']+'/composited.png'));assert len(parents)==1;parent=parents[0]
  labels=np.load(channels/'geometry_channels.npz')['labels'];bite=labels==2;edit=binary_dilation(bite,iterations=2)
  yy,xx=np.where(bite);cx=(xx.min()+xx.max())/2;cy=(yy.min()+yy.max())/2;size=192
  l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];x0=int(np.clip(round(cx-size/2),l,l+w-size));y0=int(np.clip(round(cy-size/2),t,t+h-size));box=(x0,y0,x0+size,y0+size)
  full=np.zeros((480,640),bool);full[y0:y0+size,x0:x0+size]=edit[y0:y0+size,x0:x0+size]
  d=g/cid;d.mkdir();Image.fromarray(np.uint8(full)*255).save(d/'full_mask.png')
  Image.open(parent).convert('RGB').crop(box).resize((640,640),Image.Resampling.LANCZOS).save(d/'source.png')
  Image.fromarray(np.uint8(full)*255).crop(box).resize((640,640),Image.Resampling.NEAREST).save(d/'edit_mask.png')
  for name in ['photo_canny','depth']:Image.open(channels/(name+'.png')).convert('RGB').crop(box).resize((640,640),Image.Resampling.NEAREST).save(d/(name+'.png'))
  transform={'box':box,'input_resolution':[640,640],'full_composite_parent':str(parent),'parent_sha256':sha(parent),'source_only_crop':True,'food_only_edit':True,'utensil_and_source_removal_parent':'Fixed g18 seed41; do not regenerate these components','geometry_report_sha256':sha(geometry/'geometry_report.json')};(d/'transform.json').write_text(json.dumps(transform,indent=2))
  prompt='A close-up real food photograph showing one solid bite of smooth moist silken tofu seated on the existing eating spoon. The editable shape is entirely a solid piece of edible white tofu, with a soft dense interior, fine natural moist texture and gently imperfect freshly separated edges. Keep the original tofu color and the small amount of sauce or topping on its upper surface when present. The portion fills the indicated food silhouette and is visibly raised above the metal rim. The existing spoon, its handle, its reflections and the photograph background already exist and remain in place. Natural source lighting, photographic food texture and a subtle food-to-spoon contact shadow. Only food and tableware are visible.'
  for method,key,ref in [('food_only_canny','photo_canny',False),('food_only_material_reference','photo_canny',True),('food_only_depth','depth',False)]:
   j={'id':cid+'__'+method+'__41','case_id':cid,'method':method,'seed':41,'hint_scope':'target','control_scale':.7,'prompt':prompt,'reference_keys':[],'transform':str(d/'transform.json'),'files':{'source':entry(d/'source.png'),'edit_mask':entry(d/'edit_mask.png'),'control':entry(d/(key+'.png'))}}
   if ref:j['reference_keys']=['reference_material'];j['files']['reference_material']=entry(ROOT/'gate_v13_run03'/cid/'material_reference.png');j['prompt']='The reference supplies only original food color and material. '+j['prompt']
   jobs.append(j)
 cfg={'stage':'development_factorized_food_only_refinement','not_formal':True,'inference':dict(width=768,height=768,steps=40,true_cfg_scale=1.,use_kv_cache=False),'jobs':jobs,'scope':'Eight development real-food inputs; one fixed g18 parent per case. No output selection. Original full-image state and utensil are held outside the local food mask.'};(g/'config.json').write_text(json.dumps(cfg,indent=2))
 (g/'freeze.json').write_text(json.dumps({'config_sha256':sha(g/'config.json'),'expected_raw_crop_outputs':24,'script_sha256':sha(Path(__file__)),'all_outputs_retained':True,'hypothesis':'Generate only carried food in the observed context of a fixed spoon, avoiding simultaneous utensil/food competition. Compare Canny, material-reference Canny, and depth at the same seed.'},indent=2));print('FROZEN24')
if __name__=='__main__':main()
