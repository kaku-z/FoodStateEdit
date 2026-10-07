"""Audit new raw receipts, actual model inputs and deterministic mask boundaries."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter
from scipy.ndimage import binary_erosion
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 for i in range(10):
  try:return json.loads(p.read_text())
  except (OSError,json.JSONDecodeError):time.sleep(.3)
 raise RuntimeError(str(p))
def main():
 reports=[]
 for gate,geometry,channels,fields in [('gate_v36','geometry_spoon_source_fit_v2','spoon_source_fit_channels_v2','source_fit_food_fields_consumptive_v3'),('gate_v37','geometry_spoon_source_fit_ellipsoid_v3','spoon_source_fit_ellipsoid_channels_v3','source_fit_ellipsoid_food_fields_consumptive_v4')]:
  g=ROOT/gate;raw_count=0;checks=[]
  for config,prefix in [('parent_config.json','parent_worker_'),('config.json','worker_')]:
   if not (g/config).exists():continue
   cfg=read(g/config);found={read(p)['id']:p.parent for p in g.glob(prefix+'*/*/result.json')};assert len(found)==24 and set(found)=={j['id'] for j in cfg['jobs']}
   for j in cfg['jobs']:
    d=found[j['id']];rec=read(d/'result.json');req=read(d/'request.json');assert rec['seed']==req['seed']==j['seed'] and rec['case_id']==req['case_id']==j['case_id']
    for n,expected in rec['files'].items():assert sha(d/n)==expected,(gate,j['id'],n)
    for info in j['files'].values():assert sha(Path(info['path']))==info['sha256']
   raw_count+=24
  field_checks=[]
  for c in read(ROOT/'inputs/manifest.json')['cases']:
   cid=c['case_id'];geo=ROOT/geometry/cid;source=np.asarray(Image.open(geo/'source.png').convert('RGB'));edit=np.asarray(Image.open(geo/'edit_mask.png'))>0;field=np.load(ROOT/fields/cid/'field.npz');uv=field['source_uv'][field['valid']];floor=np.floor(uv).astype(int);x=floor[:,0];y=floor[:,1];assert (x>=0).all() and (x<639).all() and (y>=0).all() and (y<479).all();consumed=(np.asarray(Image.open(geo/'source_bite_mask.png'))>0)&binary_erosion(np.asarray(Image.open(geo/'food_mask.png'))>0,iterations=1);assert all(consumed[y+dy,x+dx].all() for dx,dy in [(0,0),(1,0),(0,1),(1,1)]);field_checks.append({'case_id':cid,'source_samples':len(uv),'consumption_contributor_violations':0,'observed_food_fraction':float(field['valid'].mean())})
   paths=[p for p in g.glob('*/'+cid+'*/composited.png')]+[p for p in g.glob('*/*/'+cid+'*/composited.png')];seen=set()
   for p in paths:
    if p in seen or p.parent.parent.name.startswith("worker_"):continue
    seen.add(p);arr=np.asarray(Image.open(p).convert('RGB'));assert np.array_equal(arr[~edit],source[~edit]),str(p);checks.append({'path':str(p),'sha256':sha(p),'outside_geometry_source_exact':True})
   if (g/'geometry_locked_action_v1').exists():
    data=np.load(ROOT/channels/cid/'geometry_channels.npz');cf=np.load(g/'coupled_source_cut_reconstruction_v3'/(cid+'_field')/'field.npz');action=cf['reconstruct']|(data['labels']==2)|(data['labels']==3);sampling=np.asarray(Image.fromarray(np.uint8(action)*255).filter(ImageFilter.MaxFilter(5)))>0;sampling&=edit
    for p in (g/'geometry_locked_action_v1').glob('*/'+cid+'*/pre_sampling.png'):
     pre=np.asarray(Image.open(p).convert('RGB'));assert np.array_equal(pre[~action],source[~action]);final=np.asarray(Image.open(p.with_name('composited.png')).convert('RGB'));assert np.array_equal(final[~sampling],source[~sampling]);rec=read(p.with_name('result.json'));assert sha(Path(rec['parent']))==rec['parent_sha256'] and rec['food_only_generation_used'] is False
   for row in read(g/'source_geometry_food_ablation_v1/manifest.json')['rows']:
    if row['case_id']!=cid:continue
    assert row['food_only_generated_crop_used'] is False
  if (g/'config.json').exists():
   for j in read(g/'config.json')['jobs']:
    t=read(Path(j['transform']));parent=Path(t['full_composite_parent']);req=read(parent.parent/'request.json');assert sha(parent)==t['parent_sha256'] and req['seed']==t['parent_seed']==j['seed'];mask=np.asarray(Image.open(Path(j['transform']).parent/'full_mask.png'))>0;arr=np.asarray(Image.open(g/'collection_complete'/j['id']/'composited.png').convert('RGB'));base=np.asarray(Image.open(parent).convert('RGB'));assert np.array_equal(arr[~mask],base[~mask])
  reports.append({'gate':gate,'raw_images_verified':raw_count,'compositions_verified':len(checks),'source_fields':field_checks,'image_checks':checks,'actual_contact_audit_sha256':sha(ROOT/geometry/'contact_audit.json')});print('AUDITED',gate,raw_count,len(checks),flush=True)
 out={'status':'verified','reports':reports,'script_sha256':sha(Path(__file__)),'scope':'Input/output file integrity and proxy-geometry/source-membership compositor invariants. These checks do not establish photographic realism, measured 3D or real physical performance.'};(ROOT/'source_registered_action_output_audit.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
