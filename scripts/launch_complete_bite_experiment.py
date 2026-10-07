"""Own-process supervisor for the frozen offline experiment; never evicts GPU users."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_complete_20260929')


def main():
    from run_qwen_image_edit_direct_baseline import gpu_snapshot
    assert (ROOT/'formal/FROZEN.json').exists()
    assert not (ROOT/'formal/supervisor.json').exists()
    status={'pid':os.getpid(),'started_unix':time.time(),'status':'starting','workers':[],'resource_samples':[]}
    processes=[]
    def save():
        p=ROOT/'formal/supervisor.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(p)
    save()
    try:
        for backend,devices in [('qwen','2,3'),('qwen','4,5'),('qwen','6,7'),('vace','0'),('vace','1')]:
            cfg=ROOT/'formal'/f'{backend}_frozen.json';c=json.loads(cfg.read_text())
            gpuids=[int(x) for x in devices.split(',')]
            for g in gpuids:
                snap=gpu_snapshot(g)
                assert not snap['compute_processes'] and snap['memory_free_mib']>47500,snap
            python=c['backend']['python'] if backend=='qwen' else c['runtime']['python']
            output=ROOT/'formal'/f'{backend}_worker_{devices.replace(",","_")}'
            cmd=[python,str(ROOT/f'run_complete_bite_{backend}.py'),'--config',str(cfg),'--output',str(output),'--gpus' if backend=='qwen' else '--gpu',devices]
            logpath=ROOT/'logs'/(output.name+'.log')
            with logpath.open('w') as log:
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append(proc);status['workers'].append({'backend':backend,'devices':devices,'pid':proc.pid,'output':str(output),'command':cmd,'log':str(logpath)})
            save();time.sleep(15)
        status['status']='running';save()
        while any(p.poll() is None for p in processes):
            mem={l.split(':')[0]:int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines() if ':' in l and len(l.split())>=2 and l.split()[1].isdigit()}
            status['resource_samples'].append({'unix':time.time(),'mem_available_mib':mem['MemAvailable']//1024,'swap_free_mib':mem['SwapFree']//1024,
                'running_owned_pids':[p.pid for p in processes if p.poll() is None]})
            save();time.sleep(30)
        for p,row in zip(processes,status['workers']):row['returncode']=p.returncode
        status.update(status='workers_finished' if all(p.returncode==0 for p in processes) else 'worker_failure',finished_unix=time.time());save()
        # Receipt construction verifies every generated file hash. No automatic reruns.
        python=json.loads((ROOT/'formal/vace_frozen.json').read_text())['runtime']['python']
        result=subprocess.run([python,str(ROOT/'collect_complete_bite_experiment.py')],capture_output=True,text=True)
        status['collection']={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr};save()
    except BaseException as exc:
        status.update(status='supervisor_failure',error=repr(exc),traceback=traceback.format_exc());save()
        # Already running owned jobs may finish; do not kill unrelated work or silently retry.
        raise


if __name__=='__main__':main()
