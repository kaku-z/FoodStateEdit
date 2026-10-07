"""Source-only material identity state: select references for the actual portion.
Avoid the full-meal reference encouraging garnish from an uncut source area.
"""
import json,copy,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def entry(p):return {'path':str(p),'sha256':sha(p)}
def main():
 g=ROOT/'gate_v28';g.mkdir(exist_ok=False);cfg=json.loads((ROOT/'gate_v26/config.json').read_text());jobs=[]
 for prototype in cfg['jobs']:
  if prototype['start_raw_sigma']!=.8:continue
  j=copy.deepcopy(prototype);cid=j['case_id'];geometry=ROOT/'geometry_spoon_open_corner_v1'/cid;d=g/cid;d.mkdir()
  source=np.asarray(Image.open(geometry/'source.png').convert('RGB'),float);visible=np.asarray(Image.open(ROOT/'gate_v22'/cid/'source_visible_portion_mask.png'))>0
  select=json.loads((ROOT/'gate_v13_run03'/cid/'reference_selection.json').read_text());x0,y0,x1,y1=select['material_box_canvas'];bare=np.median(source[y0:y1,x0:x1].reshape(-1,3),axis=0);bare_chroma=bare/bare.sum();rgb=source[visible];chroma=rgb/np.maximum(rgb.sum(1,keepdims=True),1);colored=np.linalg.norm(chroma-bare_chroma,axis=1)>.06;fraction=float(colored.mean());plain=fraction<.05
  Image.fromarray(np.uint8(source[y0:y1,x0:x1])).resize((512,512),Image.Resampling.LANCZOS).save(d/'bare_material_reference.png')
  state={'case_id':cid,'source_only':True,'actual_source_fragment_sha256':sha(ROOT/'gate_v22'/cid/'source_portion_reference.png'),'visible_fragment_pixels':int(visible.sum()),'bare_material_source_box':[x0,y0,x1,y1],'chromatic_seasoning_fraction':fraction,'chroma_distance_threshold':.06,'seasoning_fraction_threshold':.05,'plain_portion_prior':plain,'scope':'Chromatic appearance cue from source-visible portion pixels, not measured garnish segmentation or hidden material truth.'};(d/'material_identity_state.json').write_text(json.dumps(state,indent=2))
  j['files'].pop('original');j['files']['material']=entry(d/'bare_material_reference.png');j['input_order']=['layout','material','portion'];j['method']='selected_material_identity';j['id']=cid+'__selected_material_identity__41'
  identity='The selected source portion is plain tofu. Keep its top plain tofu, with no leaf, seasoning, brown spot or added topping.' if plain else 'The selected source portion has visible seasoning in Image 3. Preserve only the seasoning appearance shown in that selected portion; do not add a garnish from elsewhere in the meal.'
  j['prompt']='Image 1 shows one filled small mouthful already resting on a spoon. Image 2 is the actual unseasoned tofu material photographed in this meal. Image 3 contains only photographed pixels of the selected source portion on a neutral gray background. '+identity+' Replace the artificial-looking food surfaces in Image 1 with the real matching cold silken tofu: delicate dense compact moist soybean gel, subtle natural grain, tiny wet irregularities, soft fresh cut surfaces and realistic photographic highlights. Keep the exact solid silhouette, scale, pose, spoon contact and illumination in Image 1. Keep the spoon and entire background unchanged. This is unbaked and untoasted tofu, not custard, cheese, cake or a hollow cup. No person or hand.'
  j['material_identity_state']=str(d/'material_identity_state.json');jobs.append(j)
 cfg.update(jobs=jobs,stage='development_native_selected_portion_material_identity',frozen_raw_cells=len(jobs),selection_rule='All eight cases, global sigma .8, all cells retained. Seasoning/plain cue is derived from the original source fragment before these outputs exist. No full-meal reference.');(g/'config.json').write_text(json.dumps(cfg,indent=2));print('PREPARED',len(jobs))
if __name__=='__main__':main()
