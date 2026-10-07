"""Run two independent balanced model workers and collect only a complete gate."""
import subprocess,json,time,hashlib
from pathlib import Path
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
MODEL_PY='/home/yanai-lab/guo-z/.conda/envs/flux_pure_env/bin/python'
GEOMETRY_PY='/host/space0/guo-z/tf-ufi/food3d_pilot_20260928/venv/bin/python'
def main():
    gate=ROOT/'gate_v33';(gate/'executed_orchestrator.py').write_bytes(Path(__file__).read_bytes())
    processes=[];launch=[]
    for shard,gpus in enumerate(['0,1','4,5']):
        command=[MODEL_PY,'-u',str(ROOT/'run_bite_material_sdedit.py'),'--config',str(gate/'config.json'),'--output',str(gate/('worker_'+str(shard))),'--gpus',gpus,'--shard',str(shard),'--shards','2']
        log=open(ROOT/'logs'/('gate_v33_worker_'+str(shard)+'.log'),'wb')
        p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=str(ROOT),start_new_session=True);log.close();processes.append(p)
        launch.append({'pid':p.pid,'command':command,'shard':shard,'gpus':gpus})
    state={'status':'running','launch':launch,'started_unix':time.time(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (gate/'execution.json').write_text(json.dumps(state,indent=2))
    while any(p.poll() is None for p in processes):time.sleep(20)
    state['exit_codes']=[p.returncode for p in processes]
    if all(p.returncode==0 for p in processes):
        p=subprocess.run([GEOMETRY_PY,'-u',str(ROOT/'collect_cavity_surface_gate.py')],cwd=str(ROOT))
        state['collection_exit_code']=p.returncode
        state['status']='complete_unreviewed' if p.returncode==0 else 'collection_needs_repair'
    else:state['status']='worker_needs_repair'
    state['finished_unix']=time.time();(gate/'execution.json').write_text(json.dumps(state,indent=2));print(state['status'],flush=True)
if __name__=='__main__':main()
