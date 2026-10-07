"""Development-only typed structural anchoring. Prior trials remain immutable."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion

OLD=Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929')
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def file(p):return {'path':str(p),'sha256':sha(p)}

def main():
    g=ROOT/'gate_v1';g.mkdir(exist_ok=False)
    cfg=json.loads((OLD/'validation/frozen.json').read_text())
    oldjobs=cfg.pop('jobs');cfg.update(stage='development_typed_structure',not_formal=True,jobs=[])
    chosen=[('new_03_7443','target_locked',41),('new_01_7442','target_locked',907),
            ('new_02_7496','hole_locked',41),('new_04_7459','hole_locked',41)]
    variants=[('early',.65,.35,.65,0.,False),('late',.85,.50,.85,0.,False),
              ('persistent',.90,.55,.85,.20,False),('low_frequency',.90,.65,.90,0.,True)]
    for cid,kind,seed in chosen:
        prior=next(j for j in oldjobs if (j['case_id'],j['method'],j['seed'])==(cid,kind,seed))
        prep=OLD/'validation/prepared'/cid;geo=OLD/'geometry_v4'/cid
        region='target' if kind.startswith('target') else 'hole'
        tr=json.loads((prep/'transforms.json').read_text())[region]
        def mask(name):return np.asarray(Image.open(geo/(name+'.png')))>0
        if region=='target':
            bite=mask('bite_mask');fork=mask('fork_mask')&~binary_dilation(bite,iterations=1)
            boundary=binary_dilation(bite,iterations=3)^binary_erosion(bite,iterations=3)
            whole=binary_dilation(bite|fork,iterations=3)
        else:
            bite=mask('source_bite_mask');fork=np.zeros_like(bite)
            boundary=binary_dilation(bite,iterations=3)^binary_erosion(bite,iterations=3)
            whole=binary_dilation(bite,iterations=3)
        for name,fw,bw,end,tail,lowpass in variants:
            j=copy.deepcopy(prior);j['id']=f'{cid}__{region}_{name}__{seed}';j['method']=f'{region}_{name}'
            d=g/j['id'];d.mkdir()
            weight=np.maximum(fork*fw,(whole if lowpass else boundary)*bw)
            im=Image.fromarray(np.uint8(np.clip(weight,0,1)*255)).crop(tr['box_xyxy']).resize((512,512),Image.Resampling.BILINEAR)
            im.save(d/'structure_weight.png');j['files']['structure_weight']=file(d/'structure_weight.png')
            j['structure']={'end_fraction':end,'tail_weight':tail,'lowpass':lowpass,'kernel':3,'begin_taper_fraction':end-.20}
            cfg['jobs'].append(j)
    (g/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (g/'freeze.json').write_text(json.dumps({'frozen_unix':time.time(),'config_sha256':sha(g/'config.json'),
        'scope':'All four sources and selected failures are development data. No held-out claim.',
        'hypothesis':'Typed spatial latent anchoring may retain fork continuity and food boundaries while late denoising synthesizes material.',
        'quality_gate':'Every predetermined seed and failure retained. A mode must improve structural fidelity without introducing visible proxy artifacts; otherwise revise.',
        'variants':variants,'raw_calls':len(cfg['jobs'])},indent=2)+'\n')
    print(json.dumps({'jobs':len(cfg['jobs']),'config_sha256':sha(g/'config.json')}))

if __name__=='__main__':main()
