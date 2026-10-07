"""Check G41 receipts and composition invariants, not photographic realism."""
import json, hashlib, time, zipfile, argparse
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    for _ in range(10):
        try: return json.loads(p.read_text())
        except (OSError,json.JSONDecodeError): time.sleep(.3)
    raise RuntimeError(str(p))
def rgb(p): return np.asarray(Image.open(p).convert('RGB'))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gate',choices=['gate_v41','gate_v42'],default='gate_v41');args=ap.parse_args()
    expected_count=48 if args.gate=='gate_v41' else 24
    g=ROOT/args.gate; execution=read(g/'execution.json')
    assert execution['status']=='complete_unreviewed'
    cfg=read(g/'config.json'); cases={c['case_id']:c for c in read(ROOT/'inputs/manifest.json')['cases']}
    raw={read(p)['id']:p.parent for p in g.glob('worker_*/*/result.json')}
    assert len(cfg['jobs'])==len(raw)==expected_count
    rows=[]; pairs={}
    for j in cfg['jobs']:
        p=raw[j['id']]; req=read(p/'request.json'); tr=read(Path(j['transform']))
        ctx=Path(j['transform']).parent; base=rgb(Path(tr['parent'])); assert sha(Path(tr['parent']))==tr['parent_sha256']
        mask=np.asarray(Image.open(ctx/'full_mask.png'))>0; x0,y0,x1,y1=tr['box']
        in_crop=np.zeros(mask.shape,bool); in_crop[y0:y1,x0:x1]=True; assert not np.any(mask&~in_crop)
        cid=j['case_id']; geo=ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid
        f=np.load(ROOT/'source_fit_ellipsoid_food_fields_consumptive_v4'/cid/'field.npz')
        assert not np.any(mask&~binary_erosion(f['food'],iterations=2))
        patch=np.asarray(Image.open(p/'raw.png').convert('RGB').resize((192,192),Image.Resampling.LANCZOS),float)
        target=base.astype(float).copy(); target[y0:y1,x0:x1]=patch
        alpha=np.clip(distance_transform_edt(mask)/3,0,1)[...,None]
        expected=np.uint8(np.clip(np.rint(base*(1-alpha)+target*alpha),0,255))
        d=g/'collection_complete'/j['id']; pre=rgb(d/'pre_sampling.png'); final=rgb(d/'composited.png')
        assert np.array_equal(pre,expected); assert np.array_equal(pre[~mask],base[~mask])
        source=rgb(geo/'source.png'); edit=np.asarray(Image.open(geo/'edit_mask.png'))>0
        assert np.array_equal(final[~edit],source[~edit])
        receipt=read(d/'result.json'); assert receipt['raw_sha256']==sha(p/'raw.png')
        for name, item in j['files'].items(): assert sha(Path(item['path']))==item['sha256']
        rawreceipt=read(p/'result.json')
        for filename, expected_sha in rawreceipt['files'].items():assert sha(p/filename)==expected_sha
        context=read(p/'context_audit.json');assert context['enabled'] and len(context['steps'])==cfg['inference']['steps']
        assert all(s['editable_latent_update_max']==0 and s['structural_strength']==0 for s in context['steps'])
        # Request receipts contain actual scheduler sigma and reproducible latent hashes.
        pairkey=(cid,j['seed']); pairs.setdefault(pairkey,[]).append(req)
        rows.append({'id':j['id'],'pre_sampling_recomputed_exact':True,'outside_plain_material_parent_exact':True,
          'outer_two_pixel_food_boundary_parent_exact':True,'mask_inside_declared_crop':True,'outside_geometry_source_exact':True,
          'actual_start_sigma':req['actual_start_sigma'],'requested_start_sigma':j['start_raw_sigma'],
          'editable_pixels':int(mask.sum()),'raw_sha256':sha(p/'raw.png'),'final_sha256':sha(d/'composited.png'),
          'scope':'Image composition and provenance checks. A plain-material chroma mask approximates garnish; latent context and geometry are inferred priors.'})
    pairrows=[]
    for (cid,seed),requests in pairs.items():
        assert len(requests)==(2 if args.gate=='gate_v41' else 1)
        if len(requests)==2:assert requests[0]['noise_sha256']==requests[1]['noise_sha256']
        hashes=[{k:v for k,v in q.items() if 'sha' in k and ('noise' in k or 'latent' in k)} for q in requests]
        pairrows.append({'case_id':cid,'seed':seed,'receipt_latent_hashes':hashes,'actual_sigmas':[q['actual_start_sigma'] for q in requests]})
    report={'status':'verified','raw_cells':expected_count,'compositions':expected_count,'rows':rows,'paired_receipts':pairrows,
       'source_exact_after_sampling_claim':'Only outside the declared geometry edit mask. Inside it, one native-resolution sampling step is disclosed.',
       'acceptance_by_construction':False,'realism_or_actual_3d_verified':False,'script_sha256':sha(Path(__file__))}
    (ROOT/('material_refinement_audit_'+args.gate+'.json')).write_text(json.dumps(report,indent=2))
    print('MATERIAL_REFINEMENT_AUDIT',len(rows),flush=True)
if __name__=='__main__':main()
