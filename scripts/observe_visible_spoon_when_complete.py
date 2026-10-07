"""Observe all frozen g17 cells once complete; no oracle boxes supplied to SAM3."""
from pathlib import Path
import json,time,hashlib,subprocess
import numpy as np
from PIL import Image
R=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def load(p):
 for _ in range(10):
  try:return json.loads(p.read_text())
  except (OSError,json.JSONDecodeError):time.sleep(.3)
 raise RuntimeError(str(p))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cfg=load(R/'gate_v17/config.json');ids={j['id'] for j in cfg['jobs']}
while True:
 outputs={}
 for p in (R/'gate_v17').glob('worker_*/*/result.json'):
  j=load(p)
  if j['id'] in ids:outputs[j['id']]=p.parent
 if set(outputs)==ids:break
 time.sleep(20)
g=R/'spoon_observer_v2';g.mkdir(exist_ok=False)
prior=load(R/'spoon_observer_v1/config.json');observe={'sam3':prior['sam3'],'images':[],'scope':'All 32 fixed g17 development cells, eight unedited-source controls and three visually empty-spoon controls. No automatic success score.'}
cases=load(R/'inputs/manifest.json')['cases'];target={}
for c in cases:
 cid=c['case_id'];folder=R/'geometry_spoon_visibility_v1'/cid
 bite=(np.asarray(Image.open(folder/'bite_mask.png'))>0)&~(np.asarray(Image.open(folder/'visible_fork_mask.png'))>0)
 p=g/(cid+'__visible_target.png');Image.fromarray(np.uint8(bite)*255).save(p);target[cid]=str(p)
 src=folder/'source.png';observe['images'].append(dict(id=cid+'__source',role='unedited_source',case_id=cid,path=str(src),sha256=sha(src),target_mask=str(p)))
for j in cfg['jobs']:
 p=outputs[j['id']]/'composited.png';observe['images'].append(dict(id=j['id'],role='development_algorithm_output',case_id=j['case_id'],path=str(p),sha256=sha(p),target_mask=target[j['case_id']]))
observe['images'] += [x for x in prior['images'] if x['role']=='visually_empty_spoon_control']
(g/'config.json').write_text(json.dumps(observe,indent=2))
subprocess.run(['/host/space0/guo-z/tf-ufi/food3d_pilot_20260928/venv/bin/python',str(R/'observe_supported_spoon.py'),'--config',str(g/'config.json'),'--output',str(g/'observations'),'--gpu','1'],cwd=str(R),check=True)
