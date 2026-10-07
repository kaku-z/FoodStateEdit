"""Wait for full generation, apply both frozen source-transfer arms, then observe."""
import json,time,subprocess,hashlib,zipfile
from pathlib import Path
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
PY='/host/space0/guo-z/tf-ufi/food3d_pilot_20260928/venv/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    for i in range(10):
        try:return json.loads(p.read_text())
        except (OSError,json.JSONDecodeError):time.sleep(.3)
    raise RuntimeError(str(p))
def run(script,args=[]):subprocess.run([PY,'-u',str(ROOT/script)]+args,cwd=str(ROOT),check=True)
def main():
    receipt=ROOT/'gate_v35/postprocessing_execution.json'
    state={'status':'waiting_for_complete_generation','created_unix':time.time(),'script_sha256':sha(Path(__file__))};receipt.write_text(json.dumps(state,indent=2))
    while not (ROOT/'gate_v35/completion.json').exists():
        if read(ROOT/'gate_v35/execution.json')['status']=='technical_failure':raise RuntimeError('Generation requires repair')
        time.sleep(20)
    assert read(ROOT/'gate_v35/completion.json')['completed_food_raw_crops']==24
    state.update(status='transport_running',updated_unix=time.time());receipt.write_text(json.dumps(state,indent=2))
    run('transport_all_observed_food_surfaces.py');run('transport_consumed_observed_food_surfaces.py')
    if not (ROOT/'gate_v35/source_geometry_food_ablation_v1/manifest.json').exists():run('compose_source_geometry_food_ablation.py')
    run('audit_observed_surface_outputs.py')
    out=ROOT/'spoon_observer_v9';out.mkdir(exist_ok=False);base=read(ROOT/'spoon_observer_v8/config.json');images=[x for x in base['images'] if x['role'] in ['visually_empty_spoon_control','unedited_source']]
    import numpy as np
    from PIL import Image
    for c in read(ROOT/'inputs/manifest.json')['cases']:
        cid=c['case_id'];labels=np.load(ROOT/'spoon_observed_surface_channels_v1'/cid/'geometry_channels.npz')['labels'];Image.fromarray(np.uint8(labels==2)*255).save(out/(cid+'__new_target.png'))
    for j in read(ROOT/'gate_v35/config.json')['jobs']:
        cid=j['case_id'];variants=[('g35_neural',ROOT/'gate_v35/collection_complete'/j['id']/'composited.png')]
        for arm in ['all_observed_surface_v1','all_observed_surface_consumptive_v2']:
            for rule in ['source_rgb_unrelit','source_rgb_scalar_neural_light']:variants.append((arm+'__'+rule,ROOT/'gate_v35'/arm/rule/j['id']/'composited.png'))
        for role,p in variants:images.append({'id':role+'__'+j['id'],'case_id':cid,'role':'development_'+role,'path':str(p),'sha256':sha(p),'target_mask':str(out/(cid+'__new_target.png'))})
    for gate_name in ['gate_v33','gate_v34']:
        for j in read(ROOT/gate_name/'config.json')['jobs']:
            p=ROOT/gate_name/'collection_complete'/j['id']/'composited.png';images.append({'id':gate_name+'__'+j['id'],'case_id':j['case_id'],'role':'development_'+gate_name+'_'+j['method'],'path':str(p),'sha256':sha(p),'target_mask':str(ROOT/'spoon_observer_v8'/(j['case_id']+'__visible_target.png'))})
    for row in read(ROOT/'gate_v35/source_geometry_food_ablation_v1/manifest.json')['rows']:
        p=ROOT/'gate_v35/source_geometry_food_ablation_v1'/row['id']/'composited.png';images.append({'id':'source_geometry_food__'+row['id'],'case_id':row['case_id'],'role':'development_no_food_only_crop_ablation','path':str(p),'sha256':sha(p),'target_mask':str(out/(row['case_id']+'__new_target.png'))})
    cfg={'sam3':base['sam3'],'images':images,'automatic_success_assignment':False,'scope':'All eight development cases and every seed/arm, plus original source and empty-spoon controls. Text-only fallible observer, no ground truth or realism judgment.'};assert len(images)==187
    (out/'config.json').write_text(json.dumps(cfg,indent=2));state.update(status='observer_running',observer_images=len(images));receipt.write_text(json.dumps(state,indent=2))
    run('observe_supported_spoon.py',['--config',str(out/'config.json'),'--output',str(out/'observations'),'--gpu','1'])
    with zipfile.ZipFile(ROOT/'continuation_diagnostics_v9.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in [ROOT/'observed_surface_output_audit.json',ROOT/'coupled_box_cap_output_audit.json',ROOT/'sam3_observer_provenance.json',out/'config.json',out/'observations/observations.json',ROOT/'geometry_spoon_observed_surface_v1/contact_audit.json',ROOT/'observed_food_surface_fields_consumptive_v2/manifest.json']:
            z.write(p,p.relative_to(ROOT))
    state.update(status='complete_unreviewed',finished_unix=time.time());receipt.write_text(json.dumps(state,indent=2));print('POSTPROCESSING_COMPLETE',flush=True)
if __name__=='__main__':main()
