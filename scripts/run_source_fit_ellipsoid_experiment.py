"""All development sources x three fixed seeds, shared ellipsoidal spoon scoop.

One full-scene generative prior is used per cell. The carried food surface is
then constructed by the source-registered geometry algorithm, separately saved.
"""
import json,copy,time,subprocess,hashlib,traceback
from pathlib import Path
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930');G=ROOT/'gate_v37';PY='/mnt/tmp/guo-z_first_bite_structure_20260930_runtime/venv/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 for i in range(10):
  try:return json.loads(p.read_text())
  except (OSError,json.JSONDecodeError):time.sleep(.3)
 raise RuntimeError(str(p))
def write(p,j):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(j,indent=2));t.replace(p)
def main():
 G.mkdir(exist_ok=False);(G/'executed_orchestrator.py').write_bytes(Path(__file__).read_bytes());cfg=read(ROOT/'gate_v18/config.json');protos={j['case_id']:j for j in cfg['jobs']};jobs=[]
 gm=ROOT/'geometry_spoon_source_fit_ellipsoid_v3/manifest.json';cm=ROOT/'spoon_source_fit_ellipsoid_channels_v3/manifest.json';assert all(c['status']=='geometry_ready' for c in read(gm)['cases']);assert len(read(cm)['cases'])==8
 for c in read(ROOT/'inputs/manifest.json')['cases']:
  cid=c['case_id'];geo=ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid;ch=ROOT/'spoon_source_fit_ellipsoid_channels_v3'/cid
  for seed in [41,163,907]:
   j=copy.deepcopy(protos[cid]);j.update(id=cid+'__source_fit_ellipsoid_spoon_photo_canny__'+str(seed),method='source_fit_ellipsoid_spoon_photo_canny',seed=seed)
   j['files']={k:{'path':str(p),'sha256':sha(p)} for k,p in [('source',geo/'source.png'),('edit_mask',ch/'edit_mask.png'),('control',ch/'photo_canny.png')]};jobs.append(j)
 cfg['jobs']=jobs;cfg['stage']='development_source_registered_shared_ellipsoidal_scoop';cfg['not_formal']=True;write(G/'parent_config.json',cfg);write(G/'frozen_plan.json',{'status':'frozen_before_generation','case_count':8,'seeds':[41,163,907],'full_scene_raw_cells':24,'food_only_raw_cells':0,'geometry_sha256':sha(gm),'channels_sha256':sha(cm),'source_fields_sha256':sha(ROOT/'source_fit_ellipsoid_food_fields_consumptive_v4/manifest.json'),'scope':'Previously viewed development photos. The unknown scene and hidden material remain priors; no paired truth. All cells retained.'});write(G/'execution.json',{'status':'waiting_for_gate_v36','created_unix':time.time()})
 while read(ROOT/'gate_v36/execution.json')['status']!='all_48_raw_complete_unreviewed':
  if read(ROOT/'gate_v36/execution.json')['status']=='technical_failure':raise RuntimeError('Predecessor needs repair')
  time.sleep(20)
 write(G/'execution.json',{'status':'parent_running','unix':time.time()});ps=[]
 for shard,gpu in enumerate([0,4]):
  args=[PY,'-u',str(ROOT/'run_multireference_geometry_qwen21.py'),'--config',str(G/'parent_config.json'),'--output',str(G/('parent_worker_'+str(shard))),'--gpu',str(gpu),'--shard',str(shard),'--shards','2'];log=ROOT/'logs'/('gate_v37_parent_'+str(shard)+'.log')
  with log.open('w') as f:proc=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=str(ROOT))
  write(G/('parent_worker_'+str(shard)+'_launch.json'),{'pid':proc.pid,'argv':args,'log':str(log),'unix':time.time()});ps.append(proc)
 while any(p.poll() is None for p in ps):
  if any(p.poll() not in [None,0] for p in ps):raise RuntimeError('Worker needs repair')
  time.sleep(20)
 assert all(p.returncode==0 for p in ps);assert len(list(G.glob('parent_worker_*/*/result.json')))==24
 write(G/'execution.json',{'status':'all_24_raw_complete_unreviewed','completed_parent_raw':24,'food_only_raw_cells':0,'unix':time.time()});print('ALL_RAW_COMPLETE',24,flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  if G.exists():write(G/'execution.json',{'status':'technical_failure','traceback':traceback.format_exc(),'unix':time.time()})
  raise
