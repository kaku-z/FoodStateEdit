"""Execute the predeclared second food stage after all frozen seed-stress parents complete."""
import json,hashlib,time,subprocess,copy
from pathlib import Path
from PIL import Image
R=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930');G=R/'gate_v21'
def load(p):
 for _ in range(10):
  try:return json.loads(p.read_text())
  except (OSError,json.JSONDecodeError):time.sleep(.3)
 raise RuntimeError(str(p))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
parents=load(G/'parent_config.json');recipe=load(G/'recipe.json');expected={j['id'] for j in parents['jobs']}
while True:
 found={load(p)['id']:p.parent for p in G.glob('parent_worker_*/*/result.json')}
 states=[load(p) for p in G.glob('parent_worker_*/manifest.json')]
 if set(found)==expected and len(states)==3 and all(s['status']=='complete_unreviewed' for s in states):break
 if any(s['status']=='technical_failure' for s in states):raise RuntimeError('Parent worker requires technical repair; preserve frozen cells')
 time.sleep(15)
prototypes={j['case_id']:j for j in load(R/'gate_v20/config.json')['jobs'] if j['method']=='silken_food_only_canny'};jobs=[]
for parent in parents['jobs']:
 cid=parent['case_id'];seed=parent['seed'];old=copy.deepcopy(prototypes[cid]);previous_context=Path(old['transform']).parent;transform=load(previous_context/'transform.json');box=transform['box'];p=found[parent['id']]/'composited.png';d=G/'contexts'/cid/str(seed);d.mkdir(parents=True,exist_ok=False)
 Image.open(p).convert('RGB').crop(box).resize((640,640),Image.Resampling.LANCZOS).save(d/'source.png')
 transform.update(full_composite_parent=str(p),parent_sha256=sha(p),frozen_parent_id=parent['id'],seed=seed);(d/'transform.json').write_text(json.dumps(transform,indent=2))
 # All masks/control maps remain those fixed before either stress seed was generated.
 (d/'full_mask.png').write_bytes((previous_context/'full_mask.png').read_bytes())
 old.update(id=cid+'__silken_food_only_canny__'+str(seed),seed=seed,transform=str(d/'transform.json'))
 old['files']['source']={'path':str(d/'source.png'),'sha256':sha(d/'source.png')};assert old['prompt']==recipe['food_stage_prompt'] and not old['reference_keys'];jobs.append(old)
cfg={'stage':'development_predeclared_full_two_stage_seed_stress_food','jobs':jobs,'inference':recipe['food_stage_inference'],'not_formal':True,'parent_recipe_sha256':sha(G/'recipe.json'),'scope':'Exact predeclared food recipe. Each seed has its own parent image. No per-seed re-selection or regeneration.'};(G/'config.json').write_text(json.dumps(cfg,indent=2))
(G/'food_stage_freeze.json').write_text(json.dumps({'config_sha256':sha(G/'config.json'),'recipe_sha256':sha(G/'recipe.json'),'expected_raw_crops':16,'predeclared_before_parent_generation':True},indent=2))
processes=[]
for shard,gpu in enumerate([0,2,3]):
 cmd=['/mnt/tmp/guo-z_first_bite_structure_20260930_runtime/venv/bin/python','-u',str(R/'run_multireference_geometry_qwen21.py'),'--config',str(G/'config.json'),'--output',str(G/('worker_'+str(shard))),'--gpu',str(gpu),'--shard',str(shard),'--shards','3'];f=open(str(R/'logs'/('gate_v21_food_'+str(shard)+'.log')),'wb');p=subprocess.Popen(cmd,cwd=str(R),stdout=f,stderr=subprocess.STDOUT,start_new_session=True);processes.append(p);(G/('worker_'+str(shard)+'_launch.json')).write_text(json.dumps({'pid':p.pid,'command':cmd}));print('FOOD_LAUNCH',shard,p.pid,flush=True)
for p in processes:assert p.wait()==0
(G/'two_stage_workers_complete.json').write_text(json.dumps({'status':'all_workers_exited_successfully_review_required','expected_parent_raw':16,'expected_food_raw':16,'script_sha256':sha(Path(__file__))},indent=2))
