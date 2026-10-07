"""Execute a predeclared two-stage development experiment, all eight x 3 seeds.

Two workers maximum. Each food crop uses its own completed matching-seed parent.
No best-of selection, no case-specific prompt changes, no source-photo overwrite.
"""
import copy,hashlib,json,pathlib,subprocess,time,traceback
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
ROOT=pathlib.Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
PY='/mnt/tmp/guo-z_first_bite_structure_20260930_runtime/venv/bin/python'
G=ROOT/'gate_v35'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def entry(p):return {'path':str(p),'sha256':sha(p)}
def write(p,c):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(c,indent=2));tmp.replace(p)
def read(p):
 for i in range(10):
  try:return json.loads(p.read_text())
  except (OSError,json.JSONDecodeError):time.sleep(.3)
 raise RuntimeError('Cannot read '+str(p))
def workers(config,prefix):
 ps=[]
 for shard,gpu in enumerate([0,4]):
  args=[PY,'-u',str(ROOT/'run_multireference_geometry_qwen21.py'),'--config',str(config),
        '--output',str(G/(prefix+str(shard))),'--gpu',str(gpu),'--shard',str(shard),'--shards','2']
  log=ROOT/'logs'/('gate_v35_'+prefix+str(shard)+'.log')
  with log.open('w') as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=str(ROOT))
  write(G/(prefix+str(shard)+'_launch.json'),{'pid':p.pid,'argv':args,'log':str(log),'unix':time.time()});ps.append(p)
 while any(p.poll() is None for p in ps):
  time.sleep(20)
  if any(p.poll() not in [None,0] for p in ps):raise RuntimeError('Worker technical failure; stop phase transition and preserve all cells.')
 assert all(p.returncode==0 for p in ps)
 for s in range(2):assert read(G/(prefix+str(s))/'manifest.json')['status']=='complete_unreviewed'

def main():
 G.mkdir(exist_ok=False)
 (G/'executed_orchestrator.py').write_bytes(pathlib.Path(__file__).read_bytes())
 cases=read(ROOT/'inputs/manifest.json')['cases']
 geom=read(ROOT/'geometry_spoon_observed_surface_v1/manifest.json')
 channels=read(ROOT/'spoon_observed_surface_channels_v1/manifest.json')
 assert len(geom['cases'])==len(channels['cases'])==8
 assert all(c['status']=='geometry_ready' for c in geom['cases'])
 parent=read(ROOT/'gate_v18/config.json');parents=[];specs=[]
 proto={j['case_id']:j for j in parent['jobs']}
 foodproto={j['case_id']:j for j in read(ROOT/'gate_v29/config.json')['jobs']}
 for c in cases:
  cid=c['case_id'];geometry=ROOT/'geometry_spoon_observed_surface_v1'/cid;channel=ROOT/'spoon_observed_surface_channels_v1'/cid
  labels=np.load(channel/'geometry_channels.npz')['labels'];mask=binary_dilation(labels==2,iterations=2)
  yy,xx=np.where(mask);cx=xx.mean();cy=yy.mean();size=192
  l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized']
  x0=int(np.clip(round(cx-size/2),l,l+w-size));y0=int(np.clip(round(cy-size/2),t,t+h-size));box=[x0,y0,x0+size,y0+size]
  assert mask[y0:y0+size,x0:x0+size].sum()==mask.sum()
  d=G/'static'/cid;d.mkdir(parents=True)
  Image.fromarray(np.uint8(mask)*255).save(d/'full_mask.png')
  Image.fromarray(np.uint8(mask)*255).crop(tuple(box)).resize((640,640),Image.Resampling.NEAREST).save(d/'edit_mask.png')
  Image.open(channel/'intrinsic.png').crop(tuple(box)).resize((640,640),Image.Resampling.NEAREST).save(d/'intrinsic.png')
  for seed in [41,163,907]:
   j=copy.deepcopy(proto[cid]);j.update(seed=seed,method='observed_surface_spoon_photo_canny',id=cid+'__observed_surface_spoon_photo_canny__'+str(seed))
   j['prompt']=j['prompt'].replace('small plump solid bite-sized scoop','small solid bite-sized piece with a flat top and coherent cut sides').replace('A small open rounded scoop is missing','A small open corner portion is missing').replace('a gently curved fresh tofu surface continues down to the bottom of the small recess','fresh vertical cut sides lead into a shallow rounded lower cut')
   j['files']={'source':entry(geometry/'source.png'),'edit_mask':entry(channel/'edit_mask.png'),'control':entry(channel/'photo_canny.png')};parents.append(j)
   f=copy.deepcopy(foodproto[cid]);f.update(seed=seed,method='observed_surface_food_intrinsic',id=cid+'__observed_surface_food_intrinsic__'+str(seed))
   f['files']={'edit_mask':entry(d/'edit_mask.png'),'control':entry(d/'intrinsic.png')}
   specs.append({'job_without_parent_source':f,'box':box,'full_mask':entry(d/'full_mask.png'),'parent_id':j['id']})
 parent['jobs']=parents;parent['stage']='development_observed_surface_pose_parent_three_seed';parent['not_formal']=True
 write(G/'parent_config.json',parent)
 plan={'status':'frozen_before_generation','scope':'All eight photos are development, no paired ground truth.',
       'parent_raw_cells':24,'food_raw_cells':24,'seeds':[41,163,907],'food_specs':specs,
       'geometry_manifest_sha256':sha(ROOT/'geometry_spoon_observed_surface_v1/manifest.json'),
       'channels_manifest_sha256':sha(ROOT/'spoon_observed_surface_channels_v1/manifest.json'),
       'selection_rule':'Source-observed surface visibility determines supported pose before generation. Same global source cutter and two-stage prompts, all three matching seeds, every output retained.'}
 write(G/'frozen_plan.json',plan);write(G/'execution.json',{'status':'parent_running','unix':time.time()})
 # Freeze all jobs first; wait for the prior two model workers to exit.
 write(G/'execution.json',{'status':'waiting_for_gate_v34','unix':time.time()})
 while read(ROOT/'gate_v34/execution.json')['status']!='complete_unreviewed':
  if 'repair' in read(ROOT/'gate_v34/execution.json')['status']:raise RuntimeError('Predecessor requires repair')
  time.sleep(20)
 write(G/'execution.json',{'status':'parent_running','unix':time.time()})
 workers(G/'parent_config.json','parent_worker_')
 found={read(p)['id']:p.parent for p in G.glob('parent_worker_*/*/result.json')};assert set(found)=={j['id'] for j in parents}
 food=read(ROOT/'gate_v29/config.json');jobs=[]
 for s in specs:
  j=s['job_without_parent_source'];cid=j['case_id'];seed=j['seed'];d=G/'contexts'/cid/str(seed);d.mkdir(parents=True)
  p=found[s['parent_id']]/'composited.png';Image.open(p).convert('RGB').crop(tuple(s['box'])).resize((640,640),Image.Resampling.LANCZOS).save(d/'source.png')
  (d/'full_mask.png').write_bytes(pathlib.Path(s['full_mask']['path']).read_bytes())
  transform={'box':s['box'],'source_crop_size':192,'model_input_size':640,'model_generated_size':768,
             'full_composite_parent':str(p),'parent_sha256':sha(p),'parent_seed':seed,
             'crop_rule':'Fixed 192 canvas-pixel source-geometry food-centred crop, clamped to original image content. No generated-output selection.'}
  write(d/'transform.json',transform);j['transform']=str(d/'transform.json');j['files']['source']=entry(d/'source.png');jobs.append(j)
 food['jobs']=jobs;food['stage']='development_observed_surface_pose_food_three_seed';food['not_formal']=True;food['frozen_raw_cells']=24
 food['selection_rule']=plan['selection_rule'];write(G/'config.json',food)
 write(G/'execution.json',{'status':'food_running','completed_parent_raw':24,'unix':time.time()})
 workers(G/'config.json','worker_')
 assert len(list(G.glob('worker_*/*/result.json')))==24
 write(G/'execution.json',{'status':'all_48_raw_complete_unreviewed','completed_parent_raw':24,'completed_food_raw':24,'unix':time.time()})
 print('ALL_RAW_COMPLETE',48,flush=True)
 subprocess.run(['/host/space0/guo-z/tf-ufi/food3d_pilot_20260928/venv/bin/python','-u',str(ROOT/'collect_observed_surface_pose_experiment.py'),'--tag','complete','--require-complete'],cwd=str(ROOT),check=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  if G.exists():write(G/'execution.json',{'status':'technical_failure','traceback':traceback.format_exc(),'unix':time.time()})
  raise
