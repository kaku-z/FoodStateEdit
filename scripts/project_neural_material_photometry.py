"""Project G42 appearance statistics onto source-grounded food regions.

No raw neural RGB is pasted into the food. Approximate source garnish pixels
remain frozen, and neural illumination/detail are bounded material priors.
"""
import json, time, hashlib, zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter,distance_transform_edt
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
LUMA=np.array([.2126,.7152,.0722])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    for _ in range(10):
        try:return json.loads(p.read_text())
        except (OSError,json.JSONDecodeError):time.sleep(.3)
    raise RuntimeError(str(p))
def smooth(v,m,sigma):
    den=gaussian_filter(m.astype(float),sigma)
    return np.divide(gaussian_filter(v*m,sigma),den,out=np.zeros(v.shape),where=den>1e-6)
def main():
    out=ROOT/'gate_v44';out.mkdir(exist_ok=False);(out/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    plan={'status':'frozen_before_composition','arms':['illumination_only','source_detail','neural_detail'],'derived_cells':72,'new_raw_calls':0,'source_stage':'gate_v40/source_photo_cut','neural_stage':'gate_v42','seeds':[41,163,907],'detail_rgb_luminance_bound':4,'plain_chroma_distance':.04,'illumination_field':'Per inherited-plane fixed quadratic IRLS; unsupported pixels assigned the nearest observed plane as a disclosed image-space prior.','selection_rule':'All eight development cases and three seeds in every predefined arm. No case or seed filtering.','scope':'Neural illumination and microtexture statistics are projected, not raw neural color or inferred shape. Approximate source-chroma garnish protection and single-image geometry do not guarantee realism.'}
    (out/'frozen_plan.json').write_text(json.dumps(plan,indent=2));(out/'execution.json').write_text(json.dumps({'status':'waiting_for_frozen_raw_cells','unix':time.time()}))
    cfg=read(ROOT/'gate_v42/config.json');cases={c['case_id']:c for c in read(ROOT/'inputs/manifest.json')['cases']}
    while read(ROOT/'gate_v42/execution.json')['status']!='complete_unreviewed':time.sleep(20)
    raw={read(p)['id']:p.parent for p in (ROOT/'gate_v42').glob('worker_*/*/result.json')};assert set(raw)=={j['id'] for j in cfg['jobs']};rows=[]
    for j in cfg['jobs']:
        cid=j['case_id'];tr=read(Path(j['transform']));base=np.asarray(Image.open(tr['parent']).convert('RGB'));assert sha(Path(tr['parent']))==tr['parent_sha256']
        ctx=Path(j['transform']).parent;editable=np.asarray(Image.open(ctx/'full_mask.png'))>0;x0,y0,x1,y1=tr['box']
        rawpath=raw[j['id']]/'raw.png';tile=np.asarray(Image.open(rawpath).convert('RGB').resize((192,192),Image.Resampling.LANCZOS),float)
        candidate=base.astype(float).copy();candidate[y0:y1,x0:x1]=tile
        f=np.load(ROOT/'source_fit_ellipsoid_food_fields_consumptive_v4'/cid/'field.npz');food=f['food'];valid=f['valid'];xx=f['xx'];yy=f['yy'];face=f['face']
        observed=np.zeros(food.shape,bool);observed[yy[valid],xx[valid]]=True
        assigned=np.zeros(food.shape,int);assigned[yy[valid],xx[valid]]=face[valid]+1
        nearest=distance_transform_edt(~observed,return_distances=False,return_indices=True);assigned[food&~observed]=assigned[tuple(nearest[:,food&~observed])]
        white=np.median(np.asarray(Image.open(ctx/'reference.png').convert('RGB')).reshape(-1,3),axis=0);wchroma=white/white.sum()
        rawlum=candidate@LUMA;baselum=base@LUMA;rawchroma=candidate/np.maximum(candidate.sum(2)[...,None],1)
        rawplain=(np.linalg.norm(rawchroma-wchroma,axis=2)<.04)&(rawlum>.60*(white@LUMA))&(rawlum<254.5)
        fit_support=food&rawplain&(distance_transform_edt(food)>6)
        illumination=baselum.copy();source_detail=np.zeros(food.shape);neural_detail=np.zeros(food.shape);fits=[]
        for region_id in np.unique(assigned[food]):
            region=food&(assigned==region_id);support=region&fit_support;ry,rx=np.where(region)
            if support.sum()>=20:
                u=(rx-rx.mean())/max(1,np.ptp(rx));v=(ry-ry.mean())/max(1,np.ptp(ry));X=np.c_[np.ones(len(rx)),u,v,u*u,u*v,v*v];keep=support[ry,rx];A=X[keep];values=rawlum[ry[keep],rx[keep]];weights=np.ones(len(values));penalty=np.diag([.001,2,2,4,4,4])
                for _ in range(8):
                    beta=np.linalg.solve(A.T@(weights[:,None]*A)+penalty,A.T@(weights*values));res=A@beta-values;scale=max(1.,1.4826*np.median(np.abs(res-np.median(res))));weights=np.minimum(1.,1.5*scale/np.maximum(np.abs(res),1e-6))
                median=float(np.median(values));pred=np.clip(X@beta,.75*median,min(255.,1.25*median));mode='fixed_quadratic_IRLS'
            else:
                values=rawlum[region&rawplain];median=float(np.median(values)) if len(values) else float(np.median(baselum[region]));pred=np.full(len(rx),median);mode='declared_median_fallback'
            illumination[ry,rx]=pred
            srcsupport=region&editable;rawsupport=region&rawplain
            source_detail[region]=np.clip(baselum[region]-smooth(baselum,srcsupport,3)[region],-4,4)
            micro=smooth(rawlum,rawsupport,.7)-smooth(rawlum,rawsupport,2)
            neural_detail[region&rawplain]=np.clip(micro[region&rawplain],-4,4)
            fits.append({'region_id':int(region_id),'region_pixels':int(region.sum()),'plain_fit_pixels':int(support.sum()),'median_neural_luminance':median,'mode':mode})
        c=cases[cid];l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h);geo=ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid;source=np.asarray(Image.open(geo/'source.png').convert('RGB'));edit=np.asarray(Image.open(geo/'edit_mask.png'))>0
        for arm in plan['arms']:
            delta=np.zeros(food.shape) if arm=='illumination_only' else source_detail if arm=='source_detail' else neural_detail
            desired=np.clip(illumination+delta,0,255);gain=np.clip(desired/np.maximum(baselum,1),.50,1.6)
            projected=base.astype(float)*gain[...,None];alpha=np.clip(distance_transform_edt(editable)/3,0,1)[...,None]
            pre=np.uint8(np.clip(np.rint(base*(1-alpha)+projected*alpha),0,255));assert np.array_equal(pre[~editable],base[~editable])
            native=Image.fromarray(pre).crop(rect).resize(tuple(c['source_size']),Image.Resampling.LANCZOS).resize((w,h),Image.Resampling.LANCZOS);grid=source.copy();grid[t:t+h,l:l+w]=np.asarray(native);blend=np.clip(distance_transform_edt(edit)/5,0,1)[...,None];final=np.uint8(np.clip(np.rint(source*(1-blend)+grid*blend),0,255));assert np.array_equal(final[~edit],source[~edit])
            d=out/arm/(cid+'__'+str(j['seed']));d.mkdir(parents=True);Image.fromarray(pre).save(d/'pre_sampling.png');Image.fromarray(final).save(d/'composited.png');Image.fromarray(final).crop(rect).save(d/'view.png')
            row={'case_id':cid,'seed':j['seed'],'arm':arm,'raw_calls':0,'raw_parent_sha256':sha(rawpath),'source_prior_sha256':sha(Path(tr['parent'])),'composited_sha256':sha(d/'composited.png'),'outside_plain_material_parent_exact_before_sampling':True,'outside_geometry_source_exact':True,'raw_neural_rgb_pasted':False,'fits':fits,'scope':plan['scope']};(d/'result.json').write_text(json.dumps(row,indent=2));rows.append(row)
        print('PHOTOMETRY_PROJECTED',cid,j['seed'],flush=True)
    for cid,c in cases.items():
        l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h);board=Image.new('RGB',(2048,1560),'white');draw=ImageDraw.Draw(board)
        for ri,arm in enumerate(['prior']+plan['arms']):
            paths=[('source',ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid/'source.png')]+[(arm+' '+str(s),ROOT/'gate_v40/source_photo_cut'/(cid+'__'+str(s))/'composited.png' if arm=='prior' else out/arm/(cid+'__'+str(s))/'composited.png') for s in [41,163,907]]
            for col,(label,p) in enumerate(paths):
                im=Image.open(p).convert('RGB').crop(rect);im.thumbnail((504,354));board.paste(im,(col*512+(504-im.width)//2,ri*390+28));draw.text((col*512+5,ri*390+5),label,fill='black')
        board.save(out/(cid+'_review.jpg'),quality=95)
    (out/'execution.json').write_text(json.dumps({'status':'complete_unreviewed','derived_composites':len(rows),'raw_calls':0,'rows':rows,'unix':time.time()},indent=2))
    with zipfile.ZipFile(ROOT/'gate_v44_complete.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(ROOT))
    print('PHOTOMETRY_COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
