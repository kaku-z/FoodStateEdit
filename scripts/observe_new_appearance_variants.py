"""Observe all new appearance arms with unedited/empty-spoon controls."""
from pathlib import Path
import json,hashlib,subprocess,zipfile
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930');PY='/host/space0/guo-z/tf-ufi/food3d_pilot_20260928/venv/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 import numpy as np
 from PIL import Image
 subprocess.run([PY,'-u',str(ROOT/'audit_label_consistent_food.py')],check=True,cwd=str(ROOT));out=ROOT/'spoon_observer_v11';out.mkdir(exist_ok=False);base=json.loads((ROOT/'spoon_observer_v10/config.json').read_text());images=[i for i in base['images'] if i['role'] in ['unedited_source','visually_empty_spoon_control']]
 for gate in ['gate_v38','gate_v39','gate_v40']:
  for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases']:
   cid=c['case_id']
   for p in (ROOT/gate).glob('*/'+cid+'*/composited.png'):
    arm=p.parent.parent.name;family='spoon_source_fit_channels_v2' if gate=='gate_v38' and arm=='gate_v36' else 'spoon_source_fit_ellipsoid_channels_v3';target=out/(family+'__'+cid+'.png')
    if not target.exists():Image.fromarray(np.uint8(np.load(ROOT/family/cid/'geometry_channels.npz')['labels']==2)*255).save(target)
    images.append({'id':gate+'__'+arm+'__'+p.parent.name,'case_id':cid,'role':'development_'+gate+'_'+arm,'path':str(p),'sha256':sha(p),'target_mask':str(target)})
 assert len(images)==203,len(images);cfg={'sam3':base['sam3'],'images':images,'automatic_success_assignment':False,'scope':'Same source and empty-spoon controls; text-only 2D diagnostics of all derived appearance arms. No photographic realism or actual 3D guarantee.'};(out/'config.json').write_text(json.dumps(cfg,indent=2));subprocess.run([PY,'-u',str(ROOT/'observe_supported_spoon.py'),'--config',str(out/'config.json'),'--output',str(out/'observations'),'--gpu','1'],check=True,cwd=str(ROOT))
 with zipfile.ZipFile(ROOT/'continuation_diagnostics_v11.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in [ROOT/'label_consistent_food_audit.json',ROOT/'sam3_observer_provenance.json',out/'config.json',out/'observations/observations.json']:z.write(p,p.relative_to(ROOT))
 print('V11_DIAGNOSTICS_COMPLETE',len(images),flush=True)
if __name__=='__main__':main()
