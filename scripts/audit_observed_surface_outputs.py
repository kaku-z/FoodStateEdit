"""Integrity and source-consumption checks for the pose-aware experiment."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    for i in range(10):
        try:return json.loads(p.read_text())
        except (OSError,json.JSONDecodeError):time.sleep(.3)
    raise RuntimeError(str(p))
def main():
    gate=ROOT/'gate_v35';parents={read(p)['id']:p.parent for p in gate.glob('parent_worker_*/*/result.json')};foods={read(p)['id']:p.parent for p in gate.glob('worker_*/*/result.json')}
    configs=[read(gate/'parent_config.json'),read(gate/'config.json')]
    for cfg,found in zip(configs,[parents,foods]):
        assert len(found)==24 and set(found)=={j['id'] for j in cfg['jobs']}
        for j in cfg['jobs']:
            d=found[j['id']];receipt=read(d/'result.json');request=read(d/'request.json');assert request['seed']==j['seed']==receipt['seed'];assert request['case_id']==j['case_id']==receipt['case_id']
            for name,expected in receipt['files'].items():assert sha(d/name)==expected,(j['id'],name)
            for info in j['files'].values():assert sha(Path(info['path']))==info['sha256']
    rows=[];field_rows=[]
    for c in read(ROOT/'inputs/manifest.json')['cases']:
        cid=c['case_id'];g=ROOT/'geometry_spoon_observed_surface_v1'/cid;field=np.load(ROOT/'observed_food_surface_fields_consumptive_v2'/cid/'field.npz');uv=field['source_uv'][field['valid']];floor=np.floor(uv).astype(int);a=floor[:,0];b=floor[:,1];assert (a>=0).all() and (a<639).all() and (b>=0).all() and (b<479).all()
        consumed=np.asarray(Image.open(g/'source_bite_mask.png'))>0
        assert consumed[b,a].all() and consumed[b,a+1].all() and consumed[b+1,a].all() and consumed[b+1,a+1].all()
        field_rows.append({'case_id':cid,'source_bilinear_samples_verified':len(uv),'contributors_outside_inferred_removed_region':0,'observed_food_fraction':float(field['valid'].mean()),'source_correspondence_scope':'Approximate source-only camera and landmark correspondence within the inferred consumed footprint; not real before/after truth.'})
    for j in configs[1]['jobs']:
        cid=j['case_id'];g=ROOT/'geometry_spoon_observed_surface_v1'/cid;source=np.asarray(Image.open(g/'source.png').convert('RGB'));edit=np.asarray(Image.open(g/'edit_mask.png'))>0;trans=read(Path(j['transform']));parent=Path(trans['full_composite_parent']);parent_request=read(parent.parent/'request.json');assert parent_request['seed']==trans['parent_seed']==j['seed'];assert parent_request['case_id']==cid;assert sha(parent)==trans['parent_sha256']
        base=np.asarray(Image.open(parent).convert('RGB'));mask=np.asarray(Image.open(Path(j['transform']).parent/'full_mask.png'))>0;neural=np.asarray(Image.open(gate/'collection_complete'/j['id']/'composited.png').convert('RGB'));assert np.array_equal(neural[~mask],base[~mask]);assert np.array_equal(neural[~edit],source[~edit]);checks={}
        for variant in ['all_observed_surface_v1','all_observed_surface_consumptive_v2']:
            for rule in ['source_rgb_unrelit','source_rgb_scalar_neural_light']:
                p=gate/variant/rule/j['id']/'composited.png';arr=np.asarray(Image.open(p).convert('RGB'));assert np.array_equal(arr[~edit],source[~edit]);checks[variant+'/'+rule]={'sha256':sha(p),'outside_geometry_source_exact':True}
        rows.append({'id':j['id'],'matching_parent_seed':True,'raw_food_receipt_verified':True,'outside_food_compositor_parent_exact':True,'variants':checks})
    ablation=read(gate/'source_geometry_food_ablation_v1/manifest.json');assert len(ablation['rows'])==24
    for row in ablation['rows']:
        cid=row['case_id'];source=np.asarray(Image.open(ROOT/'geometry_spoon_observed_surface_v1'/cid/'source.png').convert('RGB'));mask=np.asarray(Image.open(ROOT/'geometry_spoon_observed_surface_v1'/cid/'edit_mask.png'))>0;food=np.load(ROOT/'observed_food_surface_fields_consumptive_v2'/cid/'field.npz')['food'];parent=parents[row['parent_id']]/'composited.png';base=np.asarray(Image.open(parent).convert('RGB'));pre=np.asarray(Image.open(gate/'source_geometry_food_ablation_v1'/row['id']/'pre_sampling.png').convert('RGB'));final=np.asarray(Image.open(gate/'source_geometry_food_ablation_v1'/row['id']/'composited.png').convert('RGB'));assert np.array_equal(pre[~food],base[~food]);assert np.array_equal(final[~mask],source[~mask]);assert sha(parent)==row['parent_sha256'];assert row['food_only_generated_crop_used'] is False
    out={'status':'verified','raw_parent_images':24,'raw_food_crops':24,'derived_surface_composites':96,'no_food_only_crop_ablation_composites':24,'source_consumption_fields':field_rows,'rows':rows,'geometry_contact_audit_sha256':sha(ROOT/'geometry_spoon_observed_surface_v1/contact_audit.json'),'script_sha256':sha(Path(__file__)),'scope':'Raw receipt integrity, matched source-only geometry, exact compositor boundaries and strict source pixel contributor membership. No perceptual realism, independent 3D reconstruction or human-rating claim.'}
    (ROOT/'observed_surface_output_audit.json').write_text(json.dumps(out,indent=2));print('VERIFIED',48,96,8,flush=True)
if __name__=='__main__':main()
