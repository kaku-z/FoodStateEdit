"""G53: integrate bounded material gradients with source-continuous cut edges.

Screen-space screened Poisson integration is an appearance prior, not physical
subsurface scattering. Geometry and food silhouettes remain the G37 proxy.
"""
import json,time,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import gaussian_filter,distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from run_joint_action_full_noise import ROOT,read,write,sha,rgb
from compose_positive_light_neural_band import linear,srgb,LUMA

def masked_smooth(a,m,sigma):
    den=gaussian_filter(m.astype(float),sigma)
    if a.ndim==3:
        return np.stack([gaussian_filter(a[...,i]*m,sigma)/np.maximum(den,1e-8) for i in range(3)],-1)
    return gaussian_filter(a*m,sigma)/np.maximum(den,1e-8)

def integrate(target,detail,mask,boundary,edge_weight):
    yy,xx=np.nonzero(mask);ids=np.full(mask.shape,-1,int);ids[mask]=np.arange(len(yy))
    rows=[];cols=[];vals=[];rhs=.15*target[mask]+edge_weight[mask]*boundary[mask]
    diag=.15+edge_weight[mask];h,w=mask.shape
    for dy,dx in [(1,0),(-1,0),(0,1),(0,-1)]:
        yn=yy+dy;xn=xx+dx;inside=(yn>=0)&(yn<h)&(xn>=0)&(xn<w);yn=np.clip(yn,0,h-1);xn=np.clip(xn,0,w-1)
        valid=inside&mask[yn,xn];j=ids[yn[valid],xn[valid]];i=np.flatnonzero(valid)
        diag[i]+=1;rhs[i]+=detail[yy[valid],xx[valid]]-detail[yn[valid],xn[valid]]
        rows.extend(i);cols.extend(j);vals.extend(-np.ones(len(i)))
    rows.extend(np.arange(len(yy)));cols.extend(np.arange(len(yy)));vals.extend(diag)
    mat=csr_matrix((vals,(rows,cols)),shape=(len(yy),len(yy)));result=target.copy();result[mask]=spsolve(mat,rhs)
    residual=float(np.max(np.abs(mat@result[mask]-rhs)))
    return result,residual

def main():
    g=ROOT/'gate_v53';g.mkdir(exist_ok=False);(g/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    arms=['geometry_soft','source_gradient','neural_gradient']
    scope='All eight repeatedly used development images and fixed seeds. Source-continuous screen-space gradient transport; not physical subsurface transport, measured interior texture, real-world volume/contact, or realism acceptance.'
    write(g/'frozen_plan.json',dict(arms=arms,new_raw_calls=0,derived_composites=72,seeds=[41,163,907],screening=.15,low_frequency_sigma_canvas=3,detail_highpass_sigma_canvas=5,log_detail_limit=.12,boundary_weight=3,scope=scope))
    write(g/'execution.json',dict(status='running',unix=time.time()));cases=read(ROOT/'inputs/manifest.json')['cases'];rows=[]
    raws={read(p)['id']:p.parent/'raw.png' for p in (ROOT/'gate_v50').glob('worker_*/*/result.json')}
    for c in cases:
        cid=c['case_id'];geo=ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid;source=rgb(geo/'source.png');edit=np.asarray(Image.open(geo/'edit_mask.png'))>0
        food=np.load(ROOT/'source_fit_ellipsoid_food_fields_consumptive_v4'/cid/'field.npz')['food'];fresh=np.load(ROOT/'gate_v37/coupled_source_cut_reconstruction_v3'/(cid+'_field')/'field.npz')['fresh']
        l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h)
        _,nearest=distance_transform_edt(fresh,return_indices=True);edge=np.clip(2-distance_transform_edt(fresh),0,1)*3
        for seed in [41,163,907]:
            ctx=ROOT/'gate_v50/contexts'/cid/str(seed);tr=read(ctx/'transform.json');protected=np.asarray(Image.open(ctx/'protected_source_color.png'))>0
            prior=ROOT/'gate_v52/geometry_light_only'/(cid+'__'+str(seed))/'pre_sampling.png';base=rgb(prior);outside=rgb(ROOT/'gate_v46/source_detail'/(cid+'__'+str(seed))/'pre_sampling.png');a=linear(base.astype(float));smooth=masked_smooth(a,fresh,3);a[fresh]=smooth[fresh]
            target=np.log(np.maximum(a@LUMA,1e-5));outside_lin=linear(outside.astype(float));boundary=np.log(np.maximum((outside_lin@LUMA)[nearest[0],nearest[1]],1e-5))
            white=np.asarray(tr['source_white_chroma']);oc=outside/np.maximum(outside.sum(2)[...,None],1);nearcolor=oc[nearest[0],nearest[1]];plain=np.linalg.norm(nearcolor-white,axis=2)<.06;weights=edge*plain
            rawpath=raws[cid+'__paired_patch_action__'+str(seed)];raw=rgb(rawpath);neural=np.zeros(food.shape,float)
            for i,box in enumerate(tr['boxes']):
                x,y,x1,y1=box;panel=linear(raw[:,i*384:(i+1)*384].astype(float))@LUMA
                small=np.asarray(Image.fromarray(np.log(np.maximum(panel,1e-5)).astype(np.float32)).resize((x1-x,y1-y),Image.Resampling.LANCZOS));dest=np.zeros(food.shape,float);dest[y:y1,x:x1]=small;m=food if i==0 else fresh
                high=dest-masked_smooth(dest,m,5);neural[m]=np.clip(high[m],-.12,.12)
            ref=rgb(ctx/'reference.png').astype(float);ref_log=np.log(np.maximum(linear(ref)@LUMA,1e-5));ref_detail=ref_log-gaussian_filter(ref_log,8);ref_detail=np.clip(ref_detail,-.12,.12)
            # Actual source patch, tiled and shifted as a stochastic material prior.
            tile=np.tile(ref_detail,(1,2))[:480,:640];tile=np.roll(tile,seed%512,axis=1);source_detail=tile-gaussian_filter(tile,5)
            confidence=np.load(ROOT/'gate_v52/geometry_light_only'/(cid+'__'+str(seed))/'material_confidence.npy');eligible=food&~protected
            for arm in arms:
                detail=np.zeros(food.shape,float) if arm=='geometry_soft' else source_detail if arm=='source_gradient' else neural
                u,residual=integrate(target,target+detail,fresh,boundary,weights)
                candidate=a.copy();ratio=np.exp(np.clip(u-target,-.30,.30));candidate[fresh]*=ratio[fresh,None]
                if arm!='geometry_soft':candidate[eligible]*=np.exp(detail[eligible]*confidence[eligible])[:,None]
                pre=np.uint8(np.clip(np.rint(srgb(np.clip(candidate,0,1))),0,255));action=food|fresh;pre[~action]=base[~action];pre[protected]=base[protected]
                assert np.array_equal(pre[~action],base[~action]);assert np.array_equal(pre[protected],base[protected]);assert residual<1e-8
                native=Image.fromarray(pre).crop(rect).resize(tuple(c['source_size']),Image.Resampling.LANCZOS).resize((w,h),Image.Resampling.LANCZOS);grid=source.copy();grid[t:t+h,l:l+w]=np.asarray(native);blend=np.clip(distance_transform_edt(edit)/5,0,1)[...,None];final=np.uint8(np.clip(np.rint(source*(1-blend)+grid*blend),0,255));assert np.array_equal(final[~edit],source[~edit])
                d=g/arm/(cid+'__'+str(seed));d.mkdir(parents=True);Image.fromarray(pre).save(d/'pre_sampling.png');Image.fromarray(final).save(d/'composited.png');Image.fromarray(final).crop(rect).save(d/'view.png')
                row=dict(case_id=cid,seed=seed,arm=arm,new_raw_calls=0,parent_sha256=sha(prior),raw_sha256=sha(rawpath),output_sha256=sha(d/'composited.png'),outside_action_parent_exact_before_sampling=True,protected_source_color_parent_exact_before_sampling=True,outside_geometry_source_exact=True,poisson_max_absolute_residual=residual,maximum_log_gain=.30,scope=scope);write(d/'result.json',row);rows.append(row)
            print('SCREENED_TRANSPORT',cid,seed,flush=True)
        board=Image.new('RGB',(2048,1560),'white');dr=ImageDraw.Draw(board)
        for ri,arm in enumerate(['prior']+arms):
            paths=[('source',geo/'source.png')]+[(arm+' '+str(s),ROOT/'gate_v52/geometry_light_only'/(cid+'__'+str(s))/'composited.png' if arm=='prior' else g/arm/(cid+'__'+str(s))/'composited.png') for s in [41,163,907]]
            for col,(title,p) in enumerate(paths):
                im=Image.open(p).convert('RGB').crop(rect);im.thumbnail((504,354));board.paste(im,(col*512+(504-im.width)//2,ri*390+28));dr.text((col*512+5,ri*390+5),title,fill='black')
        board.save(g/(cid+'_review.jpg'),quality=95)
    write(g/'execution.json',dict(status='complete_unreviewed',raw_calls=0,compositions=len(rows),rows=rows,unix=time.time()))
    with zipfile.ZipFile(ROOT/'gate_v53_complete.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in g.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(ROOT))
    print('SCREENED_TRANSPORT_COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
