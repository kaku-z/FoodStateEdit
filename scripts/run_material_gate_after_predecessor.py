"""In-task dependency watcher; only our completed predecessor releases GPUs."""
import time,json,subprocess
from pathlib import Path
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def main():
 g=ROOT/'gate_v26';state={'status':'waiting_for_frozen_predecessor','predecessor':'gate_v23'};p=g/'dependency_watcher_state.json'
 while True:
  statuses=[]
  for i in range(2):
   try:statuses.append(json.loads((ROOT/'gate_v23'/('worker_'+str(i))/'manifest.json').read_text())['status'])
   except (OSError,json.JSONDecodeError):statuses.append('unreadable')
  state.update(predecessor_status=statuses,updated_unix=time.time());p.write_text(json.dumps(state,indent=2))
  if statuses==['complete_unreviewed','complete_unreviewed']:break
  if 'technical_failure' in statuses:raise RuntimeError('Predecessor failed; do not silently abandon incomplete cells.')
  time.sleep(20)
 workers=[]
 for i,gpus in enumerate(['0,2','3,4']):
  cmd=['/home/yanai-lab/guo-z/.conda/envs/flux_pure_env/bin/python','-u',str(ROOT/'run_bite_material_sdedit.py'),'--config',str(g/'config.json'),'--output',str(g/('worker_'+str(i))),'--gpus',gpus,'--shard',str(i),'--shards','2'];log=open(ROOT/'logs'/('gate_v26_worker_'+str(i)+'.log'),'w');worker=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);(g/('worker_'+str(i)+'_launch.json')).write_text(json.dumps({'pid':worker.pid,'command':cmd}));workers.append(worker)
 state.update(status='running',worker_pids=[w.pid for w in workers]);p.write_text(json.dumps(state,indent=2))
 exits=[w.wait() for w in workers];state.update(status='workers_complete' if exits==[0,0] else 'technical_failure',exit_codes=exits,finished_unix=time.time());p.write_text(json.dumps(state,indent=2))
if __name__=='__main__':main()
