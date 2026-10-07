"""Observe all new cells plus controls, without success auto-assignment."""
import json,time,subprocess,hashlib,zipfile
from pathlib import Path
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930');PY='/host/space0/guo-z/tf-ufi/food3d_pilot_20260928/venv/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 for i in range(10):
  try:return json.loads(p.read_text())
  except (OSError,json.JSONDecodeError):time.sleep(.3)
 raise RuntimeError(str(p))
def main():
 for gate in ['gate_v36','gate_v37']:
  while not (ROOT/gate/'postprocessing_execution.json').exists() or read(ROOT/gate/'postprocessing_execution.json')['status']!='complete_unreviewed':time.sleep(20)
 subprocess.run([PY,'-u',str(ROOT/'audit_source_registered_action_outputs.py')],cwd=str(ROOT),check=True)
 import numpy as np
 from PIL import Image
 out=ROOT/'spoon_observer_v10';out.mkdir(exist_ok=False);base=read(ROOT/'spoon_observer_v9/config.json');images=[i for i in base['images'] if i['role'] in ['unedited_source','visually_empty_spoon_control']]
 for gate,channels in [('gate_v35','spoon_observed_surface_channels_v1'),('gate_v36','spoon_source_fit_channels_v2'),('gate_v37','spoon_source_fit_ellipsoid_channels_v3')]:
  g=ROOT/gate
  for c in read(ROOT/'inputs/manifest.json')['cases']:
   cid=c['case_id'];target=out/(gate+'__'+cid+'__target.png');labels=np.load(ROOT/channels/cid/'geometry_channels.npz')['labels'];Image.fromarray(np.uint8(labels==2)*255).save(target)
   if gate=='gate_v35':paths=[p for arm in ['cavity_source_material_v2','coupled_source_cut_reconstruction_v3'] for p in (g/arm).glob('*/'+cid+'*/composited.png')]
   else:paths=[p for p in g.glob('*/'+cid+'*/composited.png') if p.parent.parent.name.startswith('parent_worker_') or p.parent.parent.name in ['collection_complete','source_geometry_food_ablation_v1']]+[p for arm in ['source_fit_surface_consumptive_v3','coupled_source_cut_reconstruction_v3','geometry_locked_action_v1'] for p in (g/arm).glob('*/'+cid+'*/composited.png')]
   for p in paths:
    relative=p.relative_to(g);role='development_'+gate+'_'+str(relative.parent.parent).replace('/','__');images.append({'id':gate+'__'+str(relative.parent).replace('/','__'),'case_id':cid,'role':role,'path':str(p),'sha256':sha(p),'target_mask':str(target)})
 assert len(images)==395,(len(images),[i['role'] for i in images]);cfg={'sam3':base['sam3'],'images':images,'automatic_success_assignment':False,'scope':'All new development seed/method cells, including full-scene model parents, with unedited-source and visibly-empty-spoon controls. SAM3 is a fallible text-only 2D diagnostic and cannot decide realism or prove actual lift/contact.'};(out/'config.json').write_text(json.dumps(cfg,indent=2));subprocess.run([PY,'-u',str(ROOT/'observe_supported_spoon.py'),'--config',str(out/'config.json'),'--output',str(out/'observations'),'--gpu','1'],cwd=str(ROOT),check=True)
 with zipfile.ZipFile(ROOT/'continuation_diagnostics_v10.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in [ROOT/'source_registered_action_output_audit.json',ROOT/'sam3_observer_provenance.json',out/'config.json',out/'observations/observations.json',ROOT/'geometry_spoon_source_fit_v2/contact_audit.json',ROOT/'geometry_spoon_source_fit_ellipsoid_v3/contact_audit.json']:z.write(p,p.relative_to(ROOT))
 print('V10_DIAGNOSTICS_COMPLETE',len(images),flush=True)
if __name__=='__main__':main()
