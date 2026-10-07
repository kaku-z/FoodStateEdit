"""Finish the already authorized replay, observer and archive steps after all workers exit."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_complete_20260929')
PY='/host/space0/guo-z/envs/geoedit/bin/python'


def read_json(p):
    for attempt in range(8):
        try:return json.loads(p.read_text())
        except OSError as exc:
            if exc.errno!=116 or attempt==7:raise
            time.sleep(.25*(attempt+1))


def main():
    receipt=ROOT/'compute_completion.json';assert not receipt.exists()
    state={'status':'waiting_for_formal_workers','pid':os.getpid(),'started_unix':time.time()}
    def save():
        tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(receipt)
    save()
    try:
        while True:
            s=read_json(ROOT/'formal/supervisor.json')
            if s['status']=='workers_finished':break
            if s['status'] in ['worker_failure','supervisor_failure']:raise RuntimeError('Worker supervisor needs attention: '+s['status'])
            time.sleep(30)
        subprocess.run([PY,str(ROOT/'collect_complete_bite_experiment.py')],check=True)
        inv=json.loads((ROOT/'formal/inventory.json').read_text());assert inv['status']=='complete'
        state['status']='replays_and_observer';save()
        subprocess.run(['/usr/bin/python3',str(ROOT/'prepare_complete_bite_observer.py')],check=True)
        with (ROOT/'logs/replay_supervisor.log').open('w') as log:
            replay=subprocess.Popen([PY,str(ROOT/'replay_complete_bite_experiment.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        sam=json.loads((ROOT/'sam3_runtime.json').read_text())
        with (ROOT/'logs/observer_formal.log').open('w') as log:
            observer=subprocess.Popen([sam['python'],str(ROOT/'observe_complete_bite.py'),'--config',str(ROOT/'formal/observer_frozen.json'),'--output',str(ROOT/'observer_formal'),'--gpu','4'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(replay_pid=replay.pid,observer_pid=observer.pid);save()
        for name,p in [('replays',replay),('observer',observer)]:
            state[name+'_returncode']=p.wait();save()
            assert p.returncode==0,(name,p.returncode)
        state['status']='archiving_compute_artifacts';save()
        subprocess.run([PY,str(ROOT/'collect_complete_bite_experiment.py'),'--archive'],check=True)
        state.update(status='compute_complete_visual_review_required',finished_unix=time.time());save()
    except BaseException as exc:
        state.update(status='attention_required',error=repr(exc),traceback=traceback.format_exc());save();raise


if __name__=='__main__':main()
