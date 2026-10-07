"""Verify persisted raw receipts, matching seeds and compositor boundaries."""
import json,hashlib,time,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    for attempt in range(10):
        try:return json.loads(p.read_text())
        except (OSError,json.JSONDecodeError):
            if attempt==9:raise
            time.sleep(.3)
def main():
    gate=ROOT/'gate_v32';pconfig=read(gate/'parent_config.json');fconfig=read(gate/'config.json')
    raw_parents={read(p)['id']:p.parent for p in gate.glob('parent_worker_*/*/result.json')}
    raw_food={read(p)['id']:p.parent for p in gate.glob('worker_*/*/result.json')}
    assert set(raw_parents)=={j['id'] for j in pconfig['jobs']}
    assert set(raw_food)=={j['id'] for j in fconfig['jobs']}
    assert len(raw_parents)==len(raw_food)==24
    for jobs,found in [(pconfig['jobs'],raw_parents),(fconfig['jobs'],raw_food)]:
        for j in jobs:
            d=found[j['id']];result=read(d/'result.json');request=read(d/'request.json')
            assert request['seed']==j['seed']==result['seed']
            assert request['case_id']==j['case_id']==result['case_id']
            for name,expected in result['files'].items():assert sha(d/name)==expected,(j['id'],name)
            for name,info in j['files'].items():assert sha(Path(info['path']))==info['sha256']
    rows=[]
    for j in fconfig['jobs']:
        cid=j['case_id'];geometry=ROOT/'geometry_spoon_box_cap_v1'/cid
        source=np.asarray(Image.open(geometry/'source.png').convert('RGB'))
        edit=np.asarray(Image.open(geometry/'edit_mask.png'))>0
        transform=read(Path(j['transform']));parent=Path(transform['full_composite_parent'])
        parent_request=read(parent.parent/'request.json')
        assert parent_request['seed']==j['seed']==transform['parent_seed']
        assert parent_request['case_id']==cid
        assert sha(parent)==transform['parent_sha256']
        parent_rgb=np.asarray(Image.open(parent).convert('RGB'))
        assert np.array_equal(parent_rgb[~edit],source[~edit])
        fullmask=np.asarray(Image.open(Path(j['transform']).parent/'full_mask.png'))>0
        final=np.asarray(Image.open(gate/'collection_complete'/j['id']/'composited.png').convert('RGB'))
        assert np.array_equal(final[~fullmask],parent_rgb[~fullmask])
        checks={}
        for variant in ['collection_complete','observed_material_box_cap_uv_v1','observed_material_box_cap_uv_v1_source_grid_v2','cavity_geometry_projection_v1']:
            image=np.asarray(Image.open(gate/variant/j['id']/'composited.png').convert('RGB'))
            assert np.array_equal(image[~edit],source[~edit]),(j['id'],variant)
            checks[variant]={'outside_geometry_source_exact':True,'sha256':sha(gate/variant/j['id']/'composited.png')}
        rows.append({'id':j['id'],'case_id':cid,'seed':j['seed'],'matching_seed_parent':True,'raw_food_receipt_verified':True,'outside_food_compositor_parent_exact':True,'variants':checks})
    paired=[]
    g33=ROOT/'gate_v33'
    if (g33/'completion.json').exists():
        cfg=read(g33/'config.json');found={read(p)['id']:p.parent for p in g33.glob('worker_*/*/result.json')}
        assert set(found)=={j['id'] for j in cfg['jobs']}
        for j in cfg['jobs']:
            d=found[j['id']];r=read(d/'result.json')
            for name,expected in r['files'].items():assert sha(d/name)==expected
            for name,info in j['files'].items():assert sha(Path(info['path']))==info['sha256']
            trans=read(Path(j['transform']));parent=np.asarray(Image.open(trans['full_composite_parent']).convert('RGB'))
            mask=np.asarray(Image.open(Path(j['transform']).parent/'full_mask.png'))>0
            composed=np.asarray(Image.open(g33/'collection_complete'/j['id']/'pre_sampling.png').convert('RGB'))
            assert np.array_equal(composed[~mask],parent[~mask])
        for cid in {j['case_id'] for j in cfg['jobs']}:
            requests=[read(found[j['id']]/'request.json') for j in cfg['jobs'] if j['case_id']==cid]
            for key in ['noise_sha256','initial_latents_sha256','actual_start_sigma','prompt','seed','input_order']:
                assert requests[0][key]==requests[1][key],(cid,key)
            paired.append({'case_id':cid,'noise_initialization_sigma_prompt_equal':True,'actual_start_sigma':requests[0]['actual_start_sigma']})
    audit={'status':'verified','g32_raw_parent_receipts':24,'g32_raw_food_receipts':24,'g32_final_cells':24,'rows':rows,'g33_paired_cases':paired,'created_unix':time.time(),'geometry_contact_audit_sha256':sha(ROOT/'geometry_spoon_box_cap_v1/contact_audit.json'),'script_sha256':sha(Path(__file__)),
           'scope':'Integrity, matched parents/noise and exact compositor boundaries. These checks do not establish perceptual realism, true reconstruction or final image physical support.'}
    (ROOT/'coupled_box_cap_output_audit.json').write_text(json.dumps(audit,indent=2))
    print('VERIFIED',24,24,len(paired),flush=True)
if __name__=='__main__':main()
