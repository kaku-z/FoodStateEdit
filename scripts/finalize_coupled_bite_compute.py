"""Predeclared fresh-process replay and advisory observer after frozen workers stop."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import zipfile
import numpy as np
from PIL import Image

ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929')
OLD=Path('/host/space0/guo-z/tf-ufi/first_bite_complete_20260929')
GEOPY='/host/space0/guo-z/tf-ufi/food3d_pilot_20260928/venv/bin/python'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    for i in range(8):
        try:return json.loads(p.read_text())
        except OSError as e:
            if e.errno!=116 or i==7:raise
            time.sleep(.3)


def main():
    receipt=ROOT/'compute_completion.json';state={'status':'waiting_for_frozen_workers','started_unix':time.time()}
    receipt.write_text(json.dumps(state,indent=2))
    while True:
        p=ROOT/'validation/supervisor.json'
        if p.exists():
            s=read(p)
            if s['status']=='workers_finished':break
        time.sleep(30)
    assert all(x==0 for x in s['returncodes'])
    subprocess.run([GEOPY,str(ROOT/'collect_coupled_bite_validation.py'),'--root',str(ROOT)],check=True)
    inv=read(ROOT/'validation/inventory.json');assert inv['status']=='complete'
    cell=next(x for x in inv['cells'] if x['case_id']=='new_01_7442' and x['method']=='L1_free_hole' and x['seed']==41)
    cfg=read(ROOT/'validation/frozen.json');cfg['jobs']=[j for j in cfg['jobs'] if j['id'] in cell['dependencies']]
    cfg['stage']='predeclared_replay';cfg['not_formal']=True
    rep=ROOT/'reproducibility';v=rep/'root/validation';v.mkdir(parents=True,exist_ok=False)
    (v/'frozen.json').write_text(json.dumps(cfg,indent=2)+'\n')
    expected=dict(cell,status='scheduled')
    (v/'expected_cells.json').write_text(json.dumps([expected],indent=2)+'\n')
    replay_cmd=[cfg['backend']['python'],str(ROOT/'run_coupled_bite_qwen.py'),'--config',str(v/'frozen.json'),
                '--output',str(v/'worker_0'),'--gpus','0,1']
    with (ROOT/'logs/replay.log').open('wb') as log:replay=subprocess.Popen(replay_cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    sam=read(OLD/'sam3_runtime.json');images=[]
    for p in sorted((ROOT/'inputs').glob('*/source.png')):
        images.append({'id':p.parent.name+'__source','role':'unedited_source','case_id':p.parent.name,'path':str(p),'sha256':sha(p)})
    for x in inv['cells']:
        if x['status']=='generated':
            images.append({'id':x['id'],'role':'algorithm_output','case_id':x['case_id'],'path':x['output_path'],
                'sha256':x['output_sha256'],'target_mask':str(ROOT/'geometry_v4'/x['case_id']/'bite_mask.png')})
    oc=ROOT/'validation/observer_frozen.json';oc.write_text(json.dumps({'sam3':sam['sam3'],'images':images},indent=2))
    shutil.copy2(OLD/'observe_complete_bite.py',ROOT/'observe_complete_bite.py')
    observer_cmd=[sam['python'],str(ROOT/'observe_complete_bite.py'),'--config',str(oc),'--output',str(ROOT/'observer'),'--gpu','2']
    with (ROOT/'logs/observer.log').open('wb') as log:observer=subprocess.Popen(observer_cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    state.update(status='replay_and_observer',replay_pid=replay.pid,observer_pid=observer.pid)
    receipt.write_text(json.dumps(state,indent=2))
    while replay.poll() is None or observer.poll() is None:time.sleep(20)
    assert replay.returncode==0 and observer.returncode==0,(replay.returncode,observer.returncode)
    subprocess.run([GEOPY,str(ROOT/'collect_coupled_bite_validation.py'),'--root',str(rep/'root')],check=True)
    ri=read(v/'inventory.json');rc=ri['cells'][0];checks=[]
    for original in cell['raw_components']:
        rerun=next(x for x in ri['raw_results'] if x['id']==original['id'])
        checks.append({'id':original['id'],'original_sha256':original['sha256'],'replay_sha256':rerun['raw_sha256'],
                       'exact_file_match':original['sha256']==rerun['raw_sha256']})
    a=np.asarray(Image.open(cell['output_path'])).astype(float);b=np.asarray(Image.open(rc['output_path'])).astype(float)
    audit={'status':'complete','raw_checks':checks,'endpoint':cell['id'],
        'original_output_sha256':cell['output_sha256'],'replay_output_sha256':rc['output_sha256'],
        'exact_composited_file_match':cell['output_sha256']==rc['output_sha256'],
        'pixel_mae_255':float(np.abs(a-b).mean()),'pixel_max_difference':float(np.abs(a-b).max()),
        'scope':'Two raw components and their L1 composition, one predeclared source and seed, fresh process on same server; excluded from primary endpoints.'}
    (rep/'replay_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    state.update(status='archiving',replay_returncode=0,observer_returncode=0)
    receipt.write_text(json.dumps(state,indent=2)+'\n')
    subprocess.run([GEOPY,str(ROOT/'collect_coupled_bite_validation.py'),'--root',str(ROOT),'--archive'],check=True)
    state.update(status='compute_complete_review_required',finished_unix=time.time())
    receipt.write_text(json.dumps(state,indent=2)+'\n')
    arc=ROOT/'postcompute_evidence.zip'
    with zipfile.ZipFile(arc,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in ['reproducibility','observer']:
            for p in (ROOT/folder).rglob('*'):
                if p.is_file() and p.suffix!='.npz':z.write(p,p.relative_to(ROOT))
        for p in [receipt,ROOT/'validation/observer_frozen.json',ROOT/'logs/replay.log',ROOT/'logs/observer.log']:
            z.write(p,p.relative_to(ROOT))
    (ROOT/'postcompute_evidence_receipt.json').write_text(json.dumps({'sha256':sha(arc),'size_bytes':arc.stat().st_size},indent=2)+'\n')


if __name__=='__main__':main()
