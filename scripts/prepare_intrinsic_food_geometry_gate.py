"""Reproduce/verify the G29 single-factor control preparation.

The original G29 was prepared with an inline remote Python command. This
utility was saved afterward and verifies its exact recorded control arrays;
it is not represented as the originally executed preparation script.
"""
import argparse,copy,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--verify-existing',action='store_true');ap.add_argument('--output',type=Path);a=ap.parse_args()
 if a.verify_existing:
  rows=[]
  for j in json.loads((ROOT/'gate_v29/config.json').read_text())['jobs']:
   transform=json.loads(Path(j['transform']).read_text());p=ROOT/'spoon_open_corner_channels_v1'/j['case_id']/'intrinsic.png'
   expected=Image.open(p).convert('RGB').crop(tuple(transform['box'])).resize((640,640),Image.Resampling.NEAREST)
   actual=Path(j['files']['control']['path']);assert sha(actual)==j['files']['control']['sha256']
   assert np.array_equal(np.asarray(expected),np.asarray(Image.open(actual)))
   rows.append({'id':j['id'],'recorded_control_sha256':sha(actual),'decoded_pixels_equal_reproduction':True})
  report={'rows':rows,'original_preparation':'Inline remote Python; this reproducibility utility was saved after that preparation.',
          'script_sha256':sha(Path(__file__))}
  (ROOT/'gate_v29/preparation_reproduction_audit.json').write_text(json.dumps(report,indent=2));print('VERIFIED',len(rows));return
 assert a.output is not None
 a.output.mkdir(exist_ok=False);cfg=json.loads((ROOT/'gate_v20/config.json').read_text());jobs=[]
 for old in cfg['jobs']:
  if old['method']!='silken_food_only_canny':continue
  j=copy.deepcopy(old);cid=j['case_id'];d=a.output/cid;d.mkdir();transform=json.loads(Path(j['transform']).read_text())
  p=d/'intrinsic.png';Image.open(ROOT/'spoon_open_corner_channels_v1'/cid/'intrinsic.png').convert('RGB').crop(tuple(transform['box'])).resize((640,640),Image.Resampling.NEAREST).save(p)
  j['method']='silken_food_intrinsic_geometry';j['id']=cid+'__'+j['method']+'__41';j['files']['control']={'path':str(p),'sha256':sha(p)};jobs.append(j)
 cfg.update(jobs=jobs,stage='reproduction_food_only_internal_geometric_edges',frozen_raw_cells=8,not_formal=True)
 (a.output/'config.json').write_text(json.dumps(cfg,indent=2));print('PREPARED',len(jobs))
if __name__=='__main__':main()
