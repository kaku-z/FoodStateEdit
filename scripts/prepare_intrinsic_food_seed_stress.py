"""Repeat the frozen intrinsic food recipe on each existing seed's own parent."""
import copy,hashlib,json
from pathlib import Path
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    gate=ROOT/'gate_v31';gate.mkdir(exist_ok=False)
    cfg=json.loads((ROOT/'gate_v29/config.json').read_text())
    controls={j['case_id']:j['files']['control'] for j in cfg['jobs']}
    jobs=[]
    for old in json.loads((ROOT/'gate_v21/config.json').read_text())['jobs']:
        j=copy.deepcopy(old);j['method']='silken_food_intrinsic_geometry'
        j['id']=j['case_id']+'__'+j['method']+'__'+str(j['seed'])
        j['files']['control']=copy.deepcopy(controls[j['case_id']])
        transform=json.loads(Path(j['transform']).read_text())
        assert sha(Path(transform['full_composite_parent']))==transform['parent_sha256']
        assert str(j['seed']) in str(transform['full_composite_parent'])
        jobs.append(j)
    assert len(jobs)==16
    cfg['jobs']=jobs;cfg['stage']='development_intrinsic_food_control_seed_stress'
    cfg['frozen_raw_cells']=16
    cfg['selection_rule']='Frozen G29 recipe, seeds163/907 for every case. Reuse completed G21 first-stage parents with their own matching seeds; no new parent generation, no per-case selection.'
    cfg['base_seed41_config_sha256']=sha(ROOT/'gate_v29/config.json')
    (gate/'config.json').write_text(json.dumps(cfg,indent=2));print('PREPARED',len(jobs),flush=True)
if __name__=='__main__':main()
