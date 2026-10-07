"""Preserve cavity raw outputs and compose only the declared fixed region."""
import json,hashlib,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    g=ROOT/'gate_v33';cfg=json.loads((g/'config.json').read_text())
    cases={c['case_id']:c for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases']}
    found={json.loads(p.read_text())['id']:p.parent for p in g.glob('worker_*/*/result.json')}
    assert set(found)=={j['id'] for j in cfg['jobs']}
    out=g/'collection_complete';out.mkdir(exist_ok=False);rows=[]
    for j in cfg['jobs']:
        context=Path(j['transform']).parent;trans=json.loads(Path(j['transform']).read_text())
        parent=Path(trans['full_composite_parent']);assert sha(parent)==trans['parent_sha256']
        base=np.asarray(Image.open(parent).convert('RGB'));x0,y0,x1,y1=trans['box']
        raw=Image.open(found[j['id']]/'raw.png').convert('RGB').resize((x1-x0,y1-y0),Image.Resampling.LANCZOS)
        candidate=base.astype(float).copy();candidate[y0:y1,x0:x1]=np.asarray(raw)
        mask=np.asarray(Image.open(context/'full_mask.png'))>0
        alpha=np.clip(distance_transform_edt(mask)/2,0,1)[...,None]
        composite=np.uint8(np.clip(np.rint(base*(1-alpha)+candidate*alpha),0,255))
        assert np.array_equal(composite[~mask],base[~mask])
        c=cases[j['case_id']];l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h)
        geometry=ROOT/'geometry_spoon_box_cap_v1'/j['case_id'];source=np.asarray(Image.open(geometry/'source.png').convert('RGB'))
        edit=np.asarray(Image.open(geometry/'edit_mask.png'))>0
        native=Image.fromarray(composite).crop(rect).resize(tuple(c['source_size']),Image.Resampling.LANCZOS).resize((w,h),Image.Resampling.LANCZOS)
        sampled=source.copy();sampled[t:t+h,l:l+w]=np.asarray(native)
        blend=np.clip(distance_transform_edt(edit)/5,0,1)[...,None]
        final=np.uint8(np.clip(np.rint(source*(1-blend)+sampled*blend),0,255));assert np.array_equal(final[~edit],source[~edit])
        d=out/j['id'];d.mkdir();Image.fromarray(composite).save(d/'pre_sampling.png');Image.fromarray(final).save(d/'composited.png');Image.fromarray(final).crop(rect).save(d/'view.png')
        row={'id':j['id'],'raw_crop_sha256':sha(found[j['id']]/'raw.png'),'parent_sha256':sha(parent),'compositor_outside_cavity_parent_exact':True,'sampling_outside_geometry_source_exact':True,'raw_generation':False,'transform':trans};(d/'result.json').write_text(json.dumps(row,indent=2));rows.append(row)
    for cid,c in cases.items():
        l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h)
        parent=Path(json.loads((g/cid/'transform.json').read_text())['full_composite_parent'])
        tiles=[('cavity appearance prior',parent)]+[(j['method'],out/j['id']/'composited.png') for j in cfg['jobs'] if j['case_id']==cid]
        board=Image.new('RGB',(640*len(tiles),h+28),'white');draw=ImageDraw.Draw(board)
        for i,(title,p) in enumerate(tiles):board.paste(Image.open(p).convert('RGB').crop(rect).resize((640,h),Image.Resampling.LANCZOS),(i*640,28));draw.text((i*640+5,6),title,fill='black')
        board.save(out/(cid+'_paired.jpg'),quality=95)
    summary={'status':'complete_unreviewed','expected_raw_crops':16,'completed_raw_crops':len(found),'rows':rows,'additional_parent_generations':0,'script_sha256':sha(Path(__file__))};(g/'completion.json').write_text(json.dumps(summary,indent=2));(out/'manifest.json').write_text(json.dumps(summary,indent=2))
    with zipfile.ZipFile(ROOT/'gate_v33_complete.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in g.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(ROOT))
    print('COMPLETE',len(found),flush=True)
if __name__=='__main__':main()
