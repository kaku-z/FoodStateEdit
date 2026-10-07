"""Development G50: magnified paired spoonful/recess appearance synthesis.

Compound change to view layout and spatial resolution, not a pure resolution
ablation. Source-based masks and all development seeds are frozen in advance.
"""
import copy, hashlib, json, subprocess, time, zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, distance_transform_edt
from run_joint_action_full_noise import ROOT, PY, read, write, sha, rgb

SIDE = 384

def square(mask, c):
    yy, xx = np.where(mask)
    size = int(np.ceil((max(np.ptp(xx)+1, np.ptp(yy)+1)+32)/32)*32)
    l,t = c['preprocessing']['pad_left_top']; w,h=c['preprocessing']['resized']
    size=min(size,w,h)
    x=int(np.clip(round((xx.min()+xx.max()+1-size)/2),l,l+w-size))
    y=int(np.clip(round((yy.min()+yy.max()+1-size)/2),t,t+h-size))
    assert mask[y:y+size,x:x+size].sum()==mask.sum()
    return [x,y,x+size,y+size]

def main():
    g=ROOT/'gate_v50';g.mkdir(exist_ok=False)
    (g/'executed_orchestrator.py').write_bytes(Path(__file__).read_bytes())
    cfg=copy.deepcopy(read(ROOT/'gate_v49/config.json'));cfg.update(jobs=[],stage='development_paired_patch_joint_action')
    cases={c['case_id']:c for c in read(ROOT/'inputs/manifest.json')['cases']}
    for cid,c in cases.items():
        geo=ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid
        f=np.load(ROOT/'source_fit_ellipsoid_food_fields_consumptive_v4'/cid/'field.npz')
        cut=np.load(ROOT/'gate_v37/coupled_source_cut_reconstruction_v3'/(cid+'_field')/'field.npz')
        source=rgb(geo/'source.png'); edit=np.asarray(Image.open(geo/'edit_mask.png'))>0
        spoon=np.load(ROOT/'spoon_source_fit_ellipsoid_channels_v3'/cid/'geometry_channels.npz')['labels']==3
        sel=read(ROOT/'gate_v13_run03'/cid/'reference_selection.json');x,y,x1,y1=sel['material_box_canvas']
        white=np.median(source[y:y1,x:x1].reshape(-1,3),0);white/=white.sum()
        observed=np.zeros(edit.shape,bool);observed[f['yy'][f['valid']],f['xx'][f['valid']]]=True
        for seed in [41,163,907]:
            d=g/'contexts'/cid/str(seed);d.mkdir(parents=True)
            parent=ROOT/'gate_v46/source_detail'/(cid+'__'+str(seed))/'pre_sampling.png';base=rgb(parent)
            chroma=base/np.maximum(base.sum(2)[...,None],1)
            protected=f['food']&observed&(np.linalg.norm(chroma-white,axis=2)>.06)
            masks=[binary_dilation(m,iterations=4)&edit&~spoon&~binary_dilation(protected,iterations=1) for m in [f['food'],cut['reconstruct']]]
            assert not (masks[0]&masks[1]).any()
            boxes=[square(m,c) for m in masks]
            layout=Image.new('RGB',(2*SIDE,SIDE));mask_image=Image.new('L',(2*SIDE,SIDE))
            for i,(box,m) in enumerate(zip(boxes,masks)):
                layout.paste(Image.fromarray(base).crop(box).resize((SIDE,SIDE),Image.Resampling.LANCZOS),(i*SIDE,0))
                mask_image.paste(Image.fromarray(np.uint8(m)*255).crop(box).resize((SIDE,SIDE),Image.Resampling.NEAREST),(i*SIDE,0))
                Image.fromarray(np.uint8(m)*255).save(d/('mask_'+str(i)+'.png'))
            layout.save(d/'layout.png');mask_image.save(d/'edit_mask.png');Image.fromarray(source).save(d/'source.png')
            Image.fromarray(source[y:y1,x:x1]).resize((512,512),Image.Resampling.LANCZOS).save(d/'reference.png')
            Image.fromarray(np.uint8(protected)*255).save(d/'protected_source_color.png')
            tr=dict(parent=str(parent),parent_sha256=sha(parent),boxes=boxes,patch_side=SIDE,original_source=str(geo/'source.png'),source_white_chroma=white.tolist(),scope='Two independently magnified context crops, left spoonful and right recess; image coordinates are restored by a separate disclosed compositor. Bounded silhouette edits do not prove 3D volume/contact.')
            write(d/'transform.json',tr)
            cfg['jobs'].append(dict(id=cid+'__paired_patch_action__'+str(seed),case_id=cid,method='paired_patch_action',seed=seed,start_raw_sigma=1.,input_order=['layout','source','reference'],lock_context=True,transform=str(d/'transform.json'),files={k:dict(path=str(d/(k+'.png')),sha256=sha(d/(k+'.png'))) for k in ['layout','source','reference','edit_mask']},prompt='Image 1 contains two magnified views of the SAME real meal: LEFT the first spoonful of soft tofu on its raised stainless steel spoon, RIGHT the corresponding scooped recess in the remaining tofu. Image 2 is the actual original meal. Image 3 is its plain tofu material. Jointly refine both magnified views into natural photographic moist soybean gel with the same source lighting, continuous gentle shading, dense fine grain, subtle wet highlights and soft irregular scooped surfaces. Keep the existing spoonful filled, its overall shape and secure contact; keep a small concave recess, never a protrusion. Preserve the exact two-panel arrangement, locations, utensil, photographed toppings, all other food and background. No new topping, new ingredient, duplicate food, person or hand.'))
    write(g/'config.json',cfg)
    write(g/'frozen_plan.json',dict(status='frozen_before_generation',raw_cells=24,compositions=48,seeds=[41,163,907],worker_sha256=sha(ROOT/'run_bite_material_sdedit_full_noise.py'),config_sha256=sha(g/'config.json'),selection_rule='All eight repeatedly reviewed development cases and every fixed seed; no best-case selection.',comparison_scope='Post-hoc compound paired-view/resolution change relative to whole-image G49; not an isolated resolution ablation.',perfect_result_claim=False))
    write(g/'execution.json',dict(status='waiting_for_predecessor',unix=time.time()))
    while read(ROOT/'gate_v49/execution.json')['status']!='complete_unreviewed':time.sleep(20)
    # SAM v14 uses GPU 1. These three pairs leave it free and never interrupt it.
    write(g/'execution.json',dict(status='running',unix=time.time()));procs=[]
    for shard,gpus in enumerate(['0,2','3,4','5,7']):
        args=[PY,'-u',str(ROOT/'run_bite_material_sdedit_full_noise.py'),'--config',str(g/'config.json'),'--output',str(g/('worker_'+str(shard))),'--gpus',gpus,'--shard',str(shard),'--shards','3']
        log=ROOT/'logs'/('gate_v50_'+str(shard)+'.log')
        with log.open('w') as h:p=subprocess.Popen(args,stdout=h,stderr=subprocess.STDOUT,cwd=str(ROOT),start_new_session=True)
        write(g/('worker_'+str(shard)+'_launch.json'),dict(pid=p.pid,argv=args,log=str(log)));procs.append(p)
    while any(p.poll() is None for p in procs):
        if any(p.poll() not in [None,0] for p in procs):raise RuntimeError('Retain completed cells for repair')
        time.sleep(20)
    found={read(p)['id']:p.parent for p in g.glob('worker_*/*/result.json')};assert set(found)=={j['id'] for j in cfg['jobs']};rows=[]
    for j in cfg['jobs']:
        cid=j['case_id'];tr=read(Path(j['transform']));ctx=Path(j['transform']).parent;base=rgb(Path(tr['parent']));assert sha(Path(tr['parent']))==tr['parent_sha256']
        rawpath=found[j['id']]/'raw.png';raw=rgb(rawpath);assert raw.shape==(SIDE,2*SIDE,3)
        masks=[np.asarray(Image.open(ctx/('mask_'+str(i)+'.png')))>0 for i in range(2)];mask=masks[0]|masks[1]
        candidate=base.astype(float).copy()
        for i,(box,m) in enumerate(zip(tr['boxes'],masks)):
            x,y,x1,y1=box;part=Image.fromarray(raw[:,i*SIDE:(i+1)*SIDE]).resize((x1-x,y1-y),Image.Resampling.LANCZOS)
            temp=base.astype(float).copy();temp[y:y1,x:x1]=np.asarray(part);candidate[m]=temp[m]
        f=np.load(ROOT/'source_fit_ellipsoid_food_fields_consumptive_v4'/cid/'field.npz');cut=np.load(ROOT/'gate_v37/coupled_source_cut_reconstruction_v3'/(cid+'_field')/'field.npz')
        white=np.asarray(tr['source_white_chroma']);lum=candidate@np.array([.2126,.7152,.0722]);bc=base/np.maximum(base.sum(2)[...,None],1)
        plain=(f['food']|cut['fresh'])&(distance_transform_edt(mask)>2)&(np.linalg.norm(bc-white,axis=2)<.06)
        c=cases[cid];l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h)
        geo=Path(tr['original_source']).parent;source=rgb(geo/'source.png');edit=np.asarray(Image.open(geo/'edit_mask.png'))>0
        for variant in ['direct_masked_raw','source_chroma_projected']:
            a=candidate.copy()
            if variant=='source_chroma_projected':a[plain]=white/(white@np.array([.2126,.7152,.0722]))*lum[plain,None]
            alpha=np.clip(distance_transform_edt(mask)/2,0,1)[...,None];pre=np.uint8(np.clip(np.rint(base*(1-alpha)+a*alpha),0,255))
            assert np.array_equal(pre[~mask],base[~mask]);protected=np.asarray(Image.open(ctx/'protected_source_color.png'))>0;assert np.array_equal(pre[protected],base[protected])
            native=Image.fromarray(pre).crop(rect).resize(tuple(c['source_size']),Image.Resampling.LANCZOS).resize((w,h),Image.Resampling.LANCZOS);grid=source.copy();grid[t:t+h,l:l+w]=np.asarray(native)
            blend=np.clip(distance_transform_edt(edit)/5,0,1)[...,None];final=np.uint8(np.clip(np.rint(source*(1-blend)+grid*blend),0,255));assert np.array_equal(final[~edit],source[~edit])
            d=g/variant/j['id'];d.mkdir(parents=True);Image.fromarray(pre).save(d/'pre_sampling.png');Image.fromarray(final).save(d/'composited.png');Image.fromarray(final).crop(rect).save(d/'view.png')
            row=dict(id=j['id'],case_id=cid,seed=j['seed'],variant=variant,raw_sha256=sha(rawpath),parent_sha256=tr['parent_sha256'],output_sha256=sha(d/'composited.png'),actual_start_sigma=read(found[j['id']]/'request.json')['actual_start_sigma'],scope='Paired crop model output followed by disclosed coordinate restoration and source-color projection, not native single-photo model output. Generated silhouette/contact/volume not verified.')
            write(d/'result.json',row);rows.append(row)
    for cid,c in cases.items():
        l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h);board=Image.new('RGB',(2048,1170),'white');dr=ImageDraw.Draw(board)
        for ri,v in enumerate(['prior','direct_masked_raw','source_chroma_projected']):
            paths=[('source',ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid/'source.png')]+[(v+' '+str(s),ROOT/'gate_v46/source_detail'/(cid+'__'+str(s))/'composited.png' if v=='prior' else g/v/(cid+'__paired_patch_action__'+str(s))/'composited.png') for s in [41,163,907]]
            for col,(title,p) in enumerate(paths):
                im=Image.open(p).convert('RGB').crop(rect);im.thumbnail((504,354));board.paste(im,(col*512+(504-im.width)//2,ri*390+28));dr.text((col*512+5,ri*390+5),title,fill='black')
        board.save(g/(cid+'_review.jpg'),quality=95)
    write(g/'execution.json',dict(status='complete_unreviewed',raw_cells=24,compositions=len(rows),rows=rows,unix=time.time()))
    with zipfile.ZipFile(ROOT/'gate_v50_complete.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in g.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(ROOT))
    print('PAIRED_PATCH_ACTION_COMPLETE',24,len(rows),flush=True)

if __name__=='__main__':main()
