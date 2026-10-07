"""Food-only material replacement with photographed scene and portion references.
The original scene and derived visible fragment are identity cues, not paired truth.
"""
import copy,json,hashlib
from pathlib import Path
from PIL import Image
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def entry(p):return {'path':str(p),'sha256':sha(p)}
def main():
 g=ROOT/'gate_v26';g.mkdir(exist_ok=False);cfg=json.loads((ROOT/'gate_v23/config.json').read_text());cfg['stage']='development_food_only_source_conditioned_material_replacement';jobs=[]
 for p in json.loads((ROOT/'gate_v20/config.json').read_text())['jobs']:
  if p['method']!='silken_food_only_canny':continue
  cid=p['case_id'];d=g/cid;d.mkdir();context=Path(p['transform']).parent;transform=json.loads(Path(p['transform']).read_text());parent=ROOT/'gate_v20/collection_complete'/p['id']/'composited.png';image=Image.open(parent).convert('RGB');image.crop(tuple(transform['box'])).resize((640,640),Image.Resampling.LANCZOS).save(d/'layout.png')
  (d/'full_mask.png').write_bytes((context/'full_mask.png').read_bytes());(d/'edit_mask.png').write_bytes((context/'edit_mask.png').read_bytes())
  transform.update(full_composite_parent=str(parent),parent_sha256=sha(parent),material_initialization='Existing filled neural food and generated spoon, not a flat proxy.')
  (d/'transform.json').write_text(json.dumps(transform,indent=2));geo=ROOT/'geometry_spoon_open_corner_v1'/cid
  for sigma in [.6,.8]:
   method='source_conditioned_material_'+str(round(100*sigma));jobs.append({'id':cid+'__'+method+'__41','case_id':cid,'method':method,'seed':41,'start_raw_sigma':sigma,'lock_context':True,'input_order':['layout','original','portion'],
   'transform':str(d/'transform.json'),'files':{'layout':entry(d/'layout.png'),'original':entry(geo/'source.png'),'portion':entry(ROOT/'gate_v22'/cid/'source_portion_reference.png'),'edit_mask':entry(d/'edit_mask.png')},
   'prompt':'Image 1 is a composite mock-up of one small spoon-borne portion. Replace the artificial-looking food piece in Image 1 with the actual cold silken tofu photographed in Image 2. Image 3 isolates the observed pixels of the selected portion from Image 2 on gray; it is an appearance reference, not an extra object. Render one compact solid mouthful of that same moist tofu with its matching photographed sauce, seasoning or leaf only where present in this selected portion. Its freshly scooped surfaces have delicate dense compact soybean gel texture, subtle wet irregularities and natural soft photographic light. Use the existing small solid silhouette and pose in Image 1; keep it filled and resting on the existing spoon. Keep the metal spoon, its highlights, and the entire background exactly unchanged. Avoid a miniature complete meal, a baked dessert, toasted patch, hollow cup or added toppings. No person, hand or arm.'})
 cfg['jobs']=jobs;cfg['frozen_raw_cells']=len(jobs);cfg['selection_rule']='All eight development cases times two global predeclared noise levels. Source/portion references selected without viewing these generated cells.';(g/'config.json').write_text(json.dumps(cfg,indent=2));print('PREPARED',len(jobs))
if __name__=='__main__':main()
