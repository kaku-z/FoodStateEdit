"""Construct the food from source pixels/geometry without a food-only model crop.

The full-scene parent is still a generated image and may contain generated food
outside the expected 3-D silhouette. This is a disclosed ablation, not a claim
that no foundation model contributed food anywhere in the image.
"""
import json,hashlib,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    gate=ROOT/'gate_v35';cfg=json.loads((gate/'parent_config.json').read_text());parents={json.loads(p.read_text())['id']:p.parent for p in gate.glob('parent_worker_*/*/result.json')};assert set(parents)=={j['id'] for j in cfg['jobs']}
    out=gate/'source_geometry_food_ablation_v1';out.mkdir(exist_ok=False);(out/'executed_script.py').write_bytes(Path(__file__).read_bytes());cases={c['case_id']:c for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases']};rows=[]
    for j in cfg['jobs']:
        cid=j['case_id'];geo=ROOT/'geometry_spoon_observed_surface_v1'/cid;fields=np.load(ROOT/'observed_food_surface_fields_consumptive_v2'/cid/'field.npz');food=fields['food'];valid=fields['valid'];xx=fields['xx'];yy=fields['yy'];source=np.asarray(Image.open(geo/'source.png').convert('RGB'));proxy=np.asarray(Image.open(geo/'rgb_control.png').convert('RGB'));parent=parents[j['id']]/'composited.png';base=np.asarray(Image.open(parent).convert('RGB'))
        body=base.astype(float).copy();body[food]=proxy[food];body[yy[valid],xx[valid]]=fields['sampled'][valid]
        alpha=np.clip(distance_transform_edt(food)/2,0,1)[...,None];composed=np.uint8(np.clip(np.rint(base*(1-alpha)+body*alpha),0,255));assert np.array_equal(composed[~food],base[~food])
        c=cases[cid];l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h);edit=np.asarray(Image.open(geo/'edit_mask.png'))>0;native=Image.fromarray(composed).crop(rect).resize(tuple(c['source_size']),Image.Resampling.LANCZOS).resize((w,h),Image.Resampling.LANCZOS);grid=source.copy();grid[t:t+h,l:l+w]=np.asarray(native);blend=np.clip(distance_transform_edt(edit)/5,0,1)[...,None];final=np.uint8(np.clip(np.rint(source*(1-blend)+grid*blend),0,255));assert np.array_equal(final[~edit],source[~edit])
        jid=cid+'__source_geometry_food__'+str(j['seed']);dest=out/jid;dest.mkdir();Image.fromarray(composed).save(dest/'pre_sampling.png');Image.fromarray(final).save(dest/'composited.png');Image.fromarray(final).crop(rect).save(dest/'view.png')
        row={'id':jid,'case_id':cid,'seed':j['seed'],'parent_id':j['id'],'parent_sha256':sha(parent),'raw_generation':False,'additional_raw_calls':0,'food_only_generated_crop_used':False,'source_observed_food_fraction':float(valid.mean()),'compositor_outside_food_parent_exact':True,'sampling_outside_geometry_source_exact':True,'unknown_food_material':'Source-calibrated analytic geometry render, not observed hidden material.','scope':'Food-only second-stage generation is omitted. The first full-scene parent still uses a foundation model and may contain food outside the expected silhouette.'};(dest/'result.json').write_text(json.dumps(row,indent=2));rows.append(row)
    for cid,c in cases.items():
        l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h);paths=[('source',ROOT/'geometry_spoon_observed_surface_v1'/cid/'source.png')]+[('source/geometry food, no food-only crop seed'+str(s),out/(cid+'__source_geometry_food__'+str(s))/'composited.png') for s in [41,163,907]];board=Image.new('RGB',(2560,h+28),'white');draw=ImageDraw.Draw(board)
        for i,(title,p) in enumerate(paths):board.paste(Image.open(p).convert('RGB').crop(rect).resize((640,h),Image.Resampling.LANCZOS),(640*i,28));draw.text((640*i+5,6),title,fill='black')
        board.save(out/(cid+'_three_seed.jpg'),quality=95)
    (out/'manifest.json').write_text(json.dumps({'status':'complete_unreviewed','derived_composites':24,'additional_raw_calls':0,'rows':rows,'script_sha256':sha(Path(__file__))},indent=2))
    with zipfile.ZipFile(ROOT/'source_geometry_food_ablation_v1.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(ROOT))
    print('COMPOSED',24,flush=True)
if __name__=='__main__':main()
