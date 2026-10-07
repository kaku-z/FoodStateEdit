"""Memory-bounded continuation of exact frozen cells; no new seeds or selection."""
import json,hashlib,os,time,subprocess
from pathlib import Path
R=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930');G=R/'gate_v17'
def load(p):
 for _ in range(10):
  try:return json.loads(p.read_text())
  except (OSError,json.JSONDecodeError):time.sleep(.3)
 raise RuntimeError(str(p))
def live(pid):
 p=Path('/proc')/str(pid)/'stat'
 return p.exists() and p.read_text().split()[2]!='Z'
def memory():
 return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))//1024
cfg=load(G/'config.json');parent=hashlib.sha256((G/'config.json').read_bytes()).hexdigest();completed=set()
for p in G.glob('worker_*/*/result.json'):completed.add(load(p)['id'])
active={i:load(G/('worker_'+str(i)+'_launch.json'))['pid'] for i in [0,1]};waiting=[];history=[]
for i in range(2,8):
 folder=G/('worker_'+str(i));mp=folder/'manifest.json';j=load(mp)
 (folder/'manifest_before_resource_interrupt.json').write_text(json.dumps(j,indent=2));j.update(status='interrupted_for_memory_scheduler',interruption='Preserve partial artifacts, resume uncompleted fixed cells in a new folder')
 mp.write_text(json.dumps(j,indent=2))
 jobs=[j for index,j in enumerate(cfg['jobs']) if index%8==i and j['id'] not in completed]
 if jobs:
  derived=dict(cfg,jobs=jobs,parent_frozen_config_sha256=parent)
  cp=G/('resume_'+str(i)+'_config.json');cp.write_text(json.dumps(derived,indent=2));waiting.append((i,cp))
while waiting or active:
 for i,pid in list(active.items()):
  if not live(pid):history.append({'gpu':i,'pid':pid,'event':'worker_exit','unix':time.time()});del active[i]
 if waiting and len(active)<2 and memory()>90000:
  i,cp=waiting.pop(0);out=G/('worker_'+str(i)+'_resume01')
  cmd=['/mnt/tmp/guo-z_first_bite_structure_20260930_runtime/venv/bin/python','-u',str(R/'run_depth_bite_qwen21_v2.py'),'--config',str(cp),'--output',str(out),'--gpu',str(i)]
  f=open(str(R/'logs'/('gate_v17_worker_'+str(i)+'_resume01.log')),'wb');p=subprocess.Popen(cmd,cwd=str(R),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  active[i]=p.pid;(G/('worker_'+str(i)+'_resume01_launch.json')).write_text(json.dumps({'pid':p.pid,'command':cmd,'parent_frozen_config_sha256':parent}));history.append({'gpu':i,'pid':p.pid,'event':'resume_launch','unix':time.time()})
 (G/'scheduler_status.json').write_text(json.dumps({'status':'running','active':active,'waiting_gpu_shards':[x[0] for x in waiting],'mem_available_mib':memory(),'history':history,'maximum_simultaneous_workers':2},indent=2))
 time.sleep(15)
(G/'scheduler_status.json').write_text(json.dumps({'status':'queue_exited_review_required','history':history},indent=2))
subprocess.run(['/host/space0/guo-z/tf-ufi/food3d_pilot_20260928/venv/bin/python',str(R/'collect_visible_spoon_gate.py'),'--tag','complete','--require-complete'],cwd=str(R),check=True)
