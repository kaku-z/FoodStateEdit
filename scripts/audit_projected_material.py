"""Check source-color and spatial preservation in G44 without rating realism."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def rgb(p):return np.asarray(Image.open(p).convert('RGB'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    g=ROOT/'gate_v44';execution=json.loads((g/'execution.json').read_text());assert execution['status']=='complete_unreviewed' and len(execution['rows'])==72
    cfg=json.loads((ROOT/'gate_v42/config.json').read_text());jobs={(j['case_id'],j['seed']):j for j in cfg['jobs']};rows=[]
    for row in execution['rows']:
        cid=row['case_id'];seed=row['seed'];tr=json.loads(Path(jobs[cid,seed]['transform']).read_text());base=rgb(Path(tr['parent']));assert sha(Path(tr['parent']))==row['source_prior_sha256']
        mask=np.asarray(Image.open(Path(jobs[cid,seed]['transform']).parent/'full_mask.png'))>0;d=g/row['arm']/(cid+'__'+str(seed));pre=rgb(d/'pre_sampling.png');final=rgb(d/'composited.png')
        assert np.array_equal(pre[~mask],base[~mask]);assert sha(d/'composited.png')==row['composited_sha256']
        # Each modified pixel should retain the parent's color direction up to
        # integer rounding. Exclude saturated pixels from this independent check.
        check=mask&(pre.max(2)<254)&(base.min(2)>8);a=base[check].astype(float);b=pre[check].astype(float);gain=np.sum(a*b,axis=1)/np.sum(a*a,axis=1);res=np.max(np.abs(a*gain[:,None]-b),axis=1);maximum=float(res.max()) if len(res) else None
        assert maximum is not None and maximum<1.01,(cid,seed,row['arm'],maximum)
        geo=ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid;source=rgb(geo/'source.png');edit=np.asarray(Image.open(geo/'edit_mask.png'))>0;assert np.array_equal(final[~edit],source[~edit])
        field=np.load(ROOT/'source_fit_ellipsoid_food_fields_consumptive_v4'/cid/'field.npz');assert not np.any(mask&~field['food'])
        rows.append({'case_id':cid,'seed':seed,'arm':row['arm'],'outside_plain_material_parent_exact_before_sampling':True,'plain_mask_inside_expected_food':True,'outside_geometry_source_exact':True,'unsaturated_color_direction_checked_pixels':int(check.sum()),'maximum_color_direction_rgb_residual':maximum,'scope':'No new raw color pasted and approximate source-garnish pixels preserved before resampling. Image constraints do not prove natural shape, accurate 3D, garnish relief or real-world eating physics.'})
    report={'status':'verified','compositions_verified':len(rows),'rows':rows,'photographic_realism_verified':False,'script_sha256':sha(Path(__file__))};(ROOT/'projected_material_audit.json').write_text(json.dumps(report,indent=2));print('PROJECTED_MATERIAL_AUDIT',len(rows),flush=True)
if __name__=='__main__':main()
