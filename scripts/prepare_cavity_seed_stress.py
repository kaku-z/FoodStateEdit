"""Freeze two more seeds for the same cavity-only rule over matching parents."""
import copy,json,hashlib,time
from pathlib import Path
from PIL import Image
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    gate=ROOT/'gate_v34';gate.mkdir(exist_ok=False)
    cfg=json.loads((ROOT/'gate_v33/config.json').read_text())
    prototypes=[j for j in cfg['jobs'] if j['method']=='cavity_unanchored'];jobs=[]
    for prototype in prototypes:
        cid=prototype['case_id'];old=Path(prototype['transform']).parent;trans=json.loads(Path(prototype['transform']).read_text())
        for seed in [163,907]:
            d=gate/cid/str(seed);d.mkdir(parents=True)
            parent=ROOT/'gate_v32/cavity_geometry_projection_v1'/(cid+'__box_cap_food_intrinsic__'+str(seed))/'projected.png'
            Image.open(parent).convert('RGB').crop(tuple(trans['box'])).resize((640,640),Image.Resampling.LANCZOS).save(d/'layout.png')
            for name in ['full_mask.png','edit_mask.png']:(d/name).write_bytes((old/name).read_bytes())
            t=dict(trans,full_composite_parent=str(parent),parent_sha256=sha(parent),parent_seed=seed,mask_sha256=sha(d/'full_mask.png'))
            (d/'transform.json').write_text(json.dumps(t,indent=2))
            j=copy.deepcopy(prototype);j.update(id=cid+'__cavity_unanchored__'+str(seed),seed=seed,transform=str(d/'transform.json'))
            j['files']['layout']={'path':str(d/'layout.png'),'sha256':sha(d/'layout.png')}
            j['files']['edit_mask']={'path':str(d/'edit_mask.png'),'sha256':sha(d/'edit_mask.png')};jobs.append(j)
    cfg.update(stage='development_cavity_only_seed_stress',jobs=jobs,frozen_raw_cells=16,frozen_unix=time.time(),selection_rule='Same global unanchored cavity rule, all eight cases, seeds163/907 on their own matching G32 parents. Paired seed41 arms remain in G33; this is a robustness test, not an automatic method adoption.')
    (gate/'config.json').write_text(json.dumps(cfg,indent=2));print('PREPARED',len(jobs),flush=True)
if __name__=='__main__':main()
