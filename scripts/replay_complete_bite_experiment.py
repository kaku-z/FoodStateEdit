"""Execute the two replay checks declared before the formal outputs existed."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_complete_20260929')


def main():
    import numpy as np
    from PIL import Image
    inv=json.loads((ROOT/'formal/inventory.json').read_text());assert inv['status']=='complete'
    out=ROOT/'reproducibility';out.mkdir(exist_ok=False)
    jid='test_01_7445__C_rgb3d__41';processes=[];rows=[]
    for backend,devices in [('qwen','2,3'),('vace','0')]:
        c=json.loads((ROOT/'formal'/f'{backend}_frozen.json').read_text())
        c['jobs']=[j for j in c['jobs'] if j['id']==jid];assert len(c['jobs'])==1
        c.pop('shared_claim_root');c['stage']='predeclared_reproducibility_check'
        cp=out/(backend+'_replay.json');cp.write_text(json.dumps(c,indent=2)+'\n')
        python=c['backend']['python'] if backend=='qwen' else c['runtime']['python']
        cmd=[python,str(ROOT/f'run_complete_bite_{backend}.py'),'--config',str(cp),'--output',str(out/backend),'--gpus' if backend=='qwen' else '--gpu',devices]
        with (ROOT/'logs'/(backend+'_replay.log')).open('w') as log:proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(proc);rows.append({'backend':backend,'pid':proc.pid,'command':cmd,'id':jid,'status':'running'})
    receipt=out/'replay_audit.json'
    receipt.write_text(json.dumps({'status':'running','checks':rows},indent=2)+'\n')
    for proc,row in zip(processes,rows):
        code=proc.wait();row['returncode']=code
        if code!=0:row.update(status='technical_failure');continue
        original=next(x for x in inv['cells'] if x['backend']==row['backend'] and x['id']==jid)
        first=Path(original['raw_path']);second=out/row['backend']/jid/'raw.png'
        a=np.asarray(Image.open(first).convert('RGB')).astype(np.int16);b=np.asarray(Image.open(second).convert('RGB')).astype(np.int16);assert a.shape==b.shape
        delta=np.abs(a-b)
        row.update(status='complete',original_sha256=hashlib.sha256(first.read_bytes()).hexdigest(),replay_sha256=hashlib.sha256(second.read_bytes()).hexdigest(),
                   exact_file_match=first.read_bytes()==second.read_bytes(),exact_pixel_match=bool(np.array_equal(a,b)),mae_255=float(delta.mean()),maximum_absolute_pixel_channel_difference=int(delta.max()),different_pixel_fraction=float(np.any(delta>0,axis=2).mean()))
        if row['backend']=='qwen':
            ra=json.loads(first.with_name('request.json').read_text());rb=json.loads(second.with_name('request.json').read_text());row['initial_noise_hash_equal']=ra['noise_sha256']==rb['noise_sha256']
    receipt.write_text(json.dumps({'status':'complete','checks':rows,'excluded_from_primary_denominator':True,'scope':'One predeclared cell per backend, same server/GPU model and cached model bytes; not a cross-platform reproducibility claim.'},indent=2)+'\n')
    print(json.dumps(rows))


if __name__=='__main__':main()
