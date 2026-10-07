"""Launch frozen model calls, monitor owned jobs, collect without changing settings."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929')
GEOPY='/host/space0/guo-z/tf-ufi/food3d_pilot_20260928/venv/bin/python'


def main():
    v=ROOT/'validation';f=json.loads((v/'FROZEN.json').read_text())
    assert hashlib.sha256((v/'frozen.json').read_bytes()).hexdigest()==f['configuration_sha256']
    for name,h in f['code_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
    cfg=json.loads((v/'frozen.json').read_text());workers=[]
    status={'status':'launching','started_unix':time.time(),'workers':[],'resource_samples':[],
        'time_policy':'User authorized unrestricted duration; elapsed time and progress monitored, no arbitrary hard timeout or silent rerun.'}
    p=v/'supervisor.json'
    for i in range(4):
        cmd=[cfg['backend']['python'],str(ROOT/'run_coupled_bite_qwen.py'),'--config',str(v/'frozen.json'),
             '--output',str(v/f'worker_{i}'),'--gpus',f'{i*2},{i*2+1}','--shard',str(i),'--shards','4']
        with (ROOT/'logs'/f'validation_worker_{i}.log').open('wb') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(proc);status['workers'].append({'pid':proc.pid,'command':cmd})
        p.write_text(json.dumps(status,indent=2)+'\n');print('STARTED',i,proc.pid,flush=True)
        if i<3:time.sleep(8)
    status['status']='running';last_collect=0
    while True:
        alive=[x.pid for x in workers if x.poll() is None]
        mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
        status['resource_samples'].append({'unix':time.time(),'owned_alive':alive,'mem_available_mib':mem['MemAvailable']//1024})
        status['elapsed_seconds']=time.time()-status['started_unix']
        p.write_text(json.dumps(status,indent=2)+'\n')
        if time.time()-last_collect>120 or not alive:
            r=subprocess.run([GEOPY,str(ROOT/'collect_coupled_bite_validation.py'),'--root',str(ROOT)],capture_output=True,text=True)
            status['collection']={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr};last_collect=time.time()
        if not alive:break
        time.sleep(30)
    status.update(status='workers_finished',finished_unix=time.time(),returncodes=[x.returncode for x in workers])
    p.write_text(json.dumps(status,indent=2)+'\n')
    if any(x.returncode!=0 for x in workers):raise RuntimeError('A frozen worker failed; preserve records, do not retry silently')
    print('FROZEN_WORKERS_FINISHED',flush=True)


if __name__=='__main__':main()
