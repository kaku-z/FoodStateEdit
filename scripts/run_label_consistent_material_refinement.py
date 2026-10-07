"""Frozen G40 priors, all development cases x seeds x two noise levels.

The Qwen2511 prior edits only plain carried-food appearance. Source garnish,
silhouette and outside-food pixels are restored by a disclosed compositor.
"""
import json,copy,time,hashlib,subprocess,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion,distance_transform_edt
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
PY='/home/yanai-lab/guo-z/.conda/envs/flux_pure_env/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def entry(p):return {'path':str(p),'sha256':sha(p)}
def read(p):
 for _ in range(10):
  try:return json.loads(p.read_text())
  except (OSError,json.JSONDecodeError):time.sleep(.3)
 raise RuntimeError(str(p))
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def main():
 g=ROOT/'gate_v41';g.mkdir(exist_ok=False);(g/'executed_orchestrator.py').write_bytes(Path(__file__).read_bytes());cfg=read(ROOT/'gate_v23/config.json');cfg['stage']='development_label_consistent_source_garnish_locked_material';cfg['not_formal']=True;cfg['inference']['steps']=24;cfg['jobs']=[];cases={c['case_id']:c for c in read(ROOT/'inputs/manifest.json')['cases']}
 for cid,c in cases.items():
  geo=ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid;f=np.load(ROOT/'source_fit_ellipsoid_food_fields_consumptive_v4'/cid/'field.npz');food=f['food'];yy,xx=np.where(food);l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];x0=int(max(l,min(l+w-192,round(xx.mean()-96))));y0=int(max(t,min(t+h-192,round(yy.mean()-96))));box=(x0,y0,x0+192,y0+192);selection=read(ROOT/'gate_v13_run03'/cid/'reference_selection.json');source=np.asarray(Image.open(geo/'source.png').convert('RGB'));a,b,d,e=selection['material_box_canvas'];white=np.median(source[b:e,a:d].reshape(-1,3),axis=0);white_chroma=white/white.sum();region=source[b:e,a:d];reference=Image.fromarray(region).resize((512,512),Image.Resampling.LANCZOS)
  for seed in [41,163,907]:
   dest=g/'contexts'/cid/str(seed);dest.mkdir(parents=True);parent=ROOT/'gate_v40/source_photo_cut'/(cid+'__'+str(seed))/'pre_sampling.png';base=np.asarray(Image.open(parent).convert('RGB'));chroma=base/np.maximum(base.sum(2)[...,None],1);lum=base@np.array([.2126,.7152,.0722]);plain=(np.linalg.norm(chroma-white_chroma,axis=2)<.04)&(lum>.72*(white@np.array([.2126,.7152,.0722])));editable=binary_erosion(food,iterations=2)&plain;assert editable.any();Image.fromarray(np.uint8(editable)*255).save(dest/'full_mask.png');Image.fromarray(np.uint8(editable)*255).crop(box).resize((640,640),Image.Resampling.NEAREST).save(dest/'edit_mask.png');Image.fromarray(base).crop(box).resize((640,640),Image.Resampling.LANCZOS).save(dest/'layout.png');reference.save(dest/'reference.png');write(dest/'transform.json',{'box':box,'parent':str(parent),'parent_sha256':sha(parent),'editable_food_pixels':int(editable.sum()),'fixed_food_silhouette':True,'source_garnish_color_prior_frozen_before_generation':True,'scope':'Approximate source-chroma plain-material mask, not a human garnish segmentation. Source-only expected silhouette selects the crop.'})
   for sigma in [.35,.60]:
    method='label_consistent_material_sigma'+str(round(sigma*100));cfg['jobs'].append({'id':cid+'__'+method+'__'+str(seed),'case_id':cid,'method':method,'seed':seed,'input_order':['layout','reference'],'lock_context':True,'start_raw_sigma':sigma,'transform':str(dest/'transform.json'),'files':{k:entry(dest/(k+'.png')) for k in ['layout','reference','edit_mask']},'prompt':'Image 1 shows a first bite of cold silken tofu already resting on a raised stainless steel spoon, before eating. Image 2 is the actual plain tofu photographed in this meal. Refine only the existing plain food surface in Image 1 to natural photographic soybean gel matching Image 2: dense moist tofu, fine compact grain, soft translucent edges, continuous gentle lighting and subtle wet highlights. Keep its exact solid silhouette, size, position and spoon contact. Preserve every existing photographed sauce and garnish on top, the spoon and the background. Keep a filled solid top. No new topping, brown toasted spot, person or hand.'})
 cfg['frozen_raw_cells']=48;cfg['selection_rule']='Every eight development case, three fixed seeds, both predefined sigma levels; no per-case best selection.';write(g/'config.json',cfg);write(g/'frozen_plan.json',{'status':'frozen_before_generation','raw_cells':48,'source_prior_gate':'gate_v40/source_photo_cut','seeds':[41,163,907],'requested_raw_sigmas':[.35,.60],'script_sha256':sha(Path(__file__)),'worker_sha256':sha(ROOT/'run_bite_material_sdedit.py'),'config_sha256':sha(g/'config.json'),'scope':'New post-hoc development cycle. No heldout, actual paired first-bite photograph or independent quality acceptance.'});write(g/'execution.json',{'status':'running','unix':time.time()});procs=[]
 for i,gpus in enumerate(['0,2','3,4']):
  cmd=[PY,'-u',str(ROOT/'run_bite_material_sdedit.py'),'--config',str(g/'config.json'),'--output',str(g/('worker_'+str(i))),'--gpus',gpus,'--shard',str(i),'--shards','2'];log=ROOT/'logs'/('gate_v41_worker_'+str(i)+'.log')
  with log.open('w') as h:p=subprocess.Popen(cmd,stdout=h,stderr=subprocess.STDOUT,cwd=str(ROOT),start_new_session=True)
  write(g/('worker_'+str(i)+'_launch.json'),{'pid':p.pid,'argv':cmd,'log':str(log)});procs.append(p)
 while any(p.poll() is None for p in procs):
  if any(p.poll() not in [None,0] for p in procs):raise RuntimeError('Worker requires technical repair; retain completed cells')
  time.sleep(20)
 found={read(p)['id']:p.parent for p in g.glob('worker_*/*/result.json')};assert set(found)=={j['id'] for j in cfg['jobs']};out=g/'collection_complete';out.mkdir();rows=[]
 for j in cfg['jobs']:
  tr=read(Path(j['transform']));base=np.asarray(Image.open(tr['parent']).convert('RGB'));assert sha(Path(tr['parent']))==tr['parent_sha256'];x0,y0,x1,y1=tr['box'];patch=np.asarray(Image.open(found[j['id']]/'raw.png').convert('RGB').resize((192,192),Image.Resampling.LANCZOS),float);mask=np.asarray(Image.open(Path(j['transform']).parent/'full_mask.png'))>0;canvas=base.astype(float).copy();canvas[y0:y1,x0:x1]=patch;alpha=np.clip(distance_transform_edt(mask)/3,0,1)[...,None];pre=np.uint8(np.clip(np.rint(base*(1-alpha)+canvas*alpha),0,255));assert np.array_equal(pre[~mask],base[~mask]);c=cases[j['case_id']];l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h);geo=ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/j['case_id'];source=np.asarray(Image.open(geo/'source.png').convert('RGB'));edit=np.asarray(Image.open(geo/'edit_mask.png'))>0;native=Image.fromarray(pre).crop(rect).resize(tuple(c['source_size']),Image.Resampling.LANCZOS).resize((w,h),Image.Resampling.LANCZOS);grid=source.copy();grid[t:t+h,l:l+w]=np.asarray(native);blend=np.clip(distance_transform_edt(edit)/5,0,1)[...,None];final=np.uint8(np.clip(np.rint(source*(1-blend)+grid*blend),0,255));assert np.array_equal(final[~edit],source[~edit]);dest=out/j['id'];dest.mkdir();Image.fromarray(pre).save(dest/'pre_sampling.png');Image.fromarray(final).save(dest/'composited.png');Image.fromarray(final).crop(rect).save(dest/'view.png');row={'id':j['id'],'case_id':j['case_id'],'seed':j['seed'],'method':j['method'],'raw':str(found[j['id']]/'raw.png'),'raw_sha256':sha(found[j['id']]/'raw.png'),'parent':tr['parent'],'parent_sha256':tr['parent_sha256'],'outside_editable_food_parent_exact_before_sampling':True,'outside_geometry_source_exact':True,'actual_start_sigma':read(found[j['id']]/'request.json')['actual_start_sigma'],'scope':'Qwen2511 material prior inside the source-chroma plain-food mask. All fixed geometry/garnish claims refer to the disclosed compositor, not raw model behavior.'};write(dest/'result.json',row);rows.append(row)
 for cid,c in cases.items():
  l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h);board=Image.new('RGB',(2048,1170),'white');dr=ImageDraw.Draw(board)
  for ri,method in enumerate(['prior','label_consistent_material_sigma35','label_consistent_material_sigma60']):
   paths=[('source',ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid/'source.png')]+[(method+' '+str(s),ROOT/'gate_v40/source_photo_cut'/(cid+'__'+str(s))/'composited.png' if method=='prior' else out/(cid+'__'+method+'__'+str(s))/'composited.png') for s in [41,163,907]]
   for i,(label,p) in enumerate(paths):
    im=Image.open(p).convert('RGB').crop(rect);im.thumbnail((504,354));board.paste(im,(i*512+(504-im.width)//2,ri*390+28));dr.text((i*512+4,ri*390+5),label,fill='black')
  board.save(g/(cid+'_review.jpg'),quality=95)
 write(g/'execution.json',{'status':'complete_unreviewed','raw_cells':48,'compositions':len(rows),'rows':rows,'unix':time.time()})
 with zipfile.ZipFile(ROOT/'gate_v41_complete.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in g.rglob('*'):
   if p.is_file():z.write(p,p.relative_to(ROOT))
 print('MATERIAL_REFINEMENT_COMPLETE',48,flush=True)
if __name__=='__main__':main()
