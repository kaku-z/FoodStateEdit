"""Mechanistic paired-noise/initialization-retention audit; no realism scoring."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 g=ROOT/'gate_v30';cfg=json.loads((g/'config.json').read_text())
 found={json.loads(p.read_text())['id']:p.parent for p in g.glob('worker_*/*/result.json')}
 assert set(found)=={j['id'] for j in cfg['jobs']}
 rows=[]
 for cid in sorted({j['case_id'] for j in cfg['jobs']}):
  jobs=[j for j in cfg['jobs'] if j['case_id']==cid];requests=[json.loads((found[j['id']]/'request.json').read_text()) for j in jobs]
  assert len({x['noise_sha256'] for x in requests})==1
  assert len({x['initial_latents_sha256'] for x in requests})==1
  assert len({x['actual_start_sigma'] for x in requests})==1
  transform=json.loads(Path(jobs[0]['transform']).read_text());initial=np.asarray(Image.open(transform['appearance_initialization']).convert('RGB'),float)
  top=np.asarray(Image.open(ROOT/'gate_v29/observed_material_neural_light_v1'/(cid+'__silken_food_intrinsic_geometry__41')/'transport_mask.png'))>0
  mask=np.asarray(Image.open(Path(jobs[0]['transform']).parent/'full_mask.png'))>0
  parent=np.asarray(Image.open(transform['full_composite_parent']).convert('RGB'))
  values=[]
  for j,request in zip(jobs,requests):
   p=g/'collection_complete'/j['id']/'composited.png';result=np.asarray(Image.open(p).convert('RGB'));assert np.array_equal(result[~mask],parent[~mask])
   audit=json.loads((found[j['id']]/'context_audit.json').read_text());assert len(audit['steps'])==cfg['inference']['steps']
   values.append({'method':j['method'],'composite_sha256':sha(p),
                  'transported_top_initialization_rgb_mae_0_255':float(np.abs(result.astype(float)[top]-initial[top]).mean()),
                  'outside_food_mask_parent_exact':True,
                  'last_structural_strength':audit['steps'][-1]['structural_strength']})
  rows.append({'case_id':cid,'same_noise_and_initial_latents':True,'actual_start_sigma':requests[0]['actual_start_sigma'],'observed_top_pixels':int(top.sum()),'arms':values})
 report={'status':'complete','all_eight_pairs_matched':True,'rows':rows,
         'scope':'Retaining an inferred source-UV initialization is a mechanistic diagnostic, not ground-truth appearance accuracy or photorealism. Pixel equality outside the mask is produced by collection compositing.',
         'config_sha256':sha(g/'config.json'),'script_sha256':sha(Path(__file__))}
 (g/'paired_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
