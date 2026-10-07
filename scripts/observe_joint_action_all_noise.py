"""Same fallible 2D observer and controls on every G46/G47/G48 composition."""
import json,time,subprocess,zipfile,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930');PY='/host/space0/guo-z/tf-ufi/food3d_pilot_20260928/venv/bin/python'
def read(p):
    for _ in range(10):
        try:return json.loads(p.read_text())
        except (OSError,json.JSONDecodeError):time.sleep(.3)
    raise RuntimeError(str(p))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=ROOT/'spoon_observer_v14';out.mkdir(exist_ok=False);(out/'executed_orchestrator.py').write_bytes(Path(__file__).read_bytes())
    base=read(ROOT/'spoon_observer_v11/config.json');images=[i for i in base['images'] if i['role'] in ['unedited_source','visually_empty_spoon_control']]
    for gate in ['gate_v46','gate_v47','gate_v48','gate_v49']:
        while read(ROOT/gate/'execution.json')['status']!='complete_unreviewed':time.sleep(20)
    for gate in ['gate_v46','gate_v47','gate_v48','gate_v49']:
        for c in read(ROOT/'inputs/manifest.json')['cases']:
            cid=c['case_id'];target=out/(cid+'_food_target.png')
            if not target.exists():Image.fromarray(np.uint8(np.load(ROOT/'spoon_source_fit_ellipsoid_channels_v3'/cid/'geometry_channels.npz')['labels']==2)*255).save(target)
            for p in (ROOT/gate).glob('*/'+cid+'*/composited.png'):
                arm=p.parent.parent.name;images.append({'id':gate+'__'+arm+'__'+p.parent.name,'case_id':cid,'role':'development_'+gate+'_'+arm,'path':str(p),'sha256':sha(p),'target_mask':str(target)})
    assert len(images)==347,len(images)
    cfg={'sam3':base['sam3'],'images':images,'automatic_success_assignment':False,'scope':'All material cells, with duplicated unedited-source and empty-spoon controls. Text-only two-dimensional diagnostics do not rate realism or verify real contact, hands or physics.'};(out/'config.json').write_text(json.dumps(cfg,indent=2))
    subprocess.run([PY,'-u',str(ROOT/'observe_supported_spoon.py'),'--config',str(out/'config.json'),'--output',str(out/'observations'),'--gpu','1'],check=True,cwd=str(ROOT))
    with zipfile.ZipFile(ROOT/'continuation_diagnostics_v14.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in [ROOT/'projected_material_audit_gate_v46.json',ROOT/'joint_light_audit_gate_v47.json',ROOT/'joint_action_audit_gate_v48.json',ROOT/'joint_action_audit_gate_v49.json',ROOT/'sam3_observer_provenance.json',out/'config.json',out/'observations/observations.json']:
            # G41 audit can finish while observation runs; never silently omit it.
            while not p.exists():time.sleep(2)
            z.write(p,p.relative_to(ROOT))
    print('V14_DIAGNOSTICS_COMPLETE',len(images),flush=True)
if __name__=='__main__':main()
