"""Wait for the predecessor to release GPUs, then run two balanced workers."""
import json,subprocess,time,hashlib
from pathlib import Path
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
MODEL_PY='/home/yanai-lab/guo-z/.conda/envs/flux_pure_env/bin/python'
GEOPY='/host/space0/guo-z/tf-ufi/food3d_pilot_20260928/venv/bin/python'
def main():
    g=ROOT/'gate_v34';(g/'executed_orchestrator.py').write_bytes(Path(__file__).read_bytes())
    state={'status':'waiting_for_g33','started_unix':time.time(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};(g/'execution.json').write_text(json.dumps(state,indent=2))
    while True:
        previous=json.loads((ROOT/'gate_v33/execution.json').read_text())
        if previous['status']=='complete_unreviewed':break
        if 'repair' in previous['status']:raise RuntimeError('Predecessor requires repair before a dependent stress experiment')
        time.sleep(20)
    processes=[];launch=[]
    for shard,gpus in enumerate(['0,1','4,5']):
        command=[MODEL_PY,'-u',str(ROOT/'run_bite_material_sdedit.py'),'--config',str(g/'config.json'),'--output',str(g/('worker_'+str(shard))),'--gpus',gpus,'--shard',str(shard),'--shards','2']
        with open(ROOT/'logs'/('gate_v34_worker_'+str(shard)+'.log'),'wb') as log:p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=str(ROOT),start_new_session=True)
        processes.append(p);launch.append({'pid':p.pid,'command':command,'shard':shard,'gpus':gpus})
    state.update(status='running',launch=launch,workers_started_unix=time.time());(g/'execution.json').write_text(json.dumps(state,indent=2))
    while any(p.poll() is None for p in processes):time.sleep(20)
    state['exit_codes']=[p.returncode for p in processes]
    if all(p.returncode==0 for p in processes):
        p=subprocess.run([GEOPY,'-u',str(ROOT/'collect_cavity_seed_stress.py')],cwd=str(ROOT));state['collection_exit_code']=p.returncode;state['status']='complete_unreviewed' if p.returncode==0 else 'collection_needs_repair'
    else:state['status']='worker_needs_repair'
    state['finished_unix']=time.time();(g/'execution.json').write_text(json.dumps(state,indent=2));print(state['status'],flush=True)
if __name__=='__main__':main()
