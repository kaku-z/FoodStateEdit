"""G56: shared action geometry, separately conditioned food and recess nodes.

All samples retained. Compound node/geometry-context/anchor/prompt experiment;
not an isolated architecture ablation, calibrated geometry, or new training.
"""
import copy,time,subprocess,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
from run_joint_action_full_noise import ROOT,PY,read,write,sha,rgb

def main():
    g=ROOT/'gate_v56';g.mkdir(exist_ok=False);(g/'executed_orchestrator.py').write_bytes(Path(__file__).read_bytes())
    cfg=copy.deepcopy(read(ROOT/'gate_v54/config.json'));cfg.update(jobs=[],stage='development_separate_action_nodes')
    cases={c['case_id']:c for c in read(ROOT/'inputs/manifest.json')['cases']}
    prompts={
        'food':'Image 1 is the actual original meal, Image 2 is its plain tofu material. Produce a photographic close-up of one small solid first portion of THIS tofu resting securely on the raised stainless steel spoon at the location fixed by the surrounding image context. Match the real source camera lighting, moist dense soybean gel texture, subtle translucency and soft irregular freshly scooped edges. Keep the existing spoon filled and preserve its angle, location and contact. Preserve only photographed toppings already in the fixed context. No second spoon, new topping, new ingredient, duplicate food, person or hand.',
        'cut':'Image 1 is the actual original meal, Image 2 is its plain tofu material. Produce a photographic close-up of the REMAINING tofu on its original plate after removing its first small spoonful. A small concave scooped recess opens at the front edge at the location fixed by the surrounding image context. Show the inward curving moist cut surface and its naturally shaded interior, continuous with this exact original tofu. Preserve the surrounding original garnish, sauce, plate, lighting and food. The spoon has already left this crop: this view contains ONLY remaining tofu and its original plate, with no utensil in or near the recess. Keep the opening empty and concave. No spoon, metal, fork, knife, refilled food, outward bump, added sauce in the recess, new ingredient, person or hand.'}
    scope='Separate photographic food/recess nodes with shared inferred G37 cut geometry and G53 appearance context. Compound node layout, context, spatial anchoring and semantic prompt change; no per-image or seed selection. Geometry is an uncalibrated prior; generated food/contact/volume and realism require separate review.'
    for cid,c in cases.items():
        food=np.load(ROOT/'source_fit_ellipsoid_food_fields_consumptive_v4'/cid/'field.npz')['food'];fresh=np.load(ROOT/'gate_v37/coupled_source_cut_reconstruction_v3'/(cid+'_field')/'field.npz')['fresh']
        for seed in [41,163,907]:
            oldctx=ROOT/'gate_v50/contexts'/cid/str(seed);tr=read(oldctx/'transform.json');parent=ROOT/'gate_v53/geometry_soft'/(cid+'__'+str(seed))/'pre_sampling.png';base=rgb(parent)
            for i,node in enumerate(['food','cut']):
                d=g/'contexts'/cid/str(seed)/node;d.mkdir(parents=True);box=tr['boxes'][i];m=np.asarray(Image.open(oldctx/('mask_'+str(i)+'.png')))>0;core=food if node=='food' else fresh
                Image.fromarray(base).crop(box).resize((384,384),Image.Resampling.LANCZOS).save(d/'layout.png')
                Image.fromarray(np.uint8(m)*255).crop(box).resize((384,384),Image.Resampling.NEAREST).save(d/'edit_mask.png')
                for name in ['source','reference']:(d/(name+'.png')).write_bytes((oldctx/(name+'.png')).read_bytes())
                weight=np.zeros(m.shape,float);weight[m]=.15;weight[m&core]=.15 if node=='food' else 1.;weight[m&core&(distance_transform_edt(core)<4)]=1
                Image.fromarray(np.uint8(weight*255)).crop(box).resize((384,384),Image.Resampling.NEAREST).save(d/'structure_weight.png')
                write(d/'transform.json',dict(parent=str(parent),parent_sha256=sha(parent),box=box,node=node,source_context=str(oldctx),scope=scope))
                job=dict(id=cid+'__node_'+node+'__'+str(seed),case_id=cid,method='node_'+node,seed=seed,start_raw_sigma=1.,input_order=['layout','source','reference'],conditioning_input_order=['source','reference'],lock_context=True,structure=dict(lowpass=True,kernel=3,begin_taper_fraction=.25,end_fraction=.65,tail_weight=.10 if node=='food' else .35),transform=str(d/'transform.json'),files={k:dict(path=str(d/(k+'.png')),sha256=sha(d/(k+'.png'))) for k in ['layout','source','reference','edit_mask','structure_weight']},prompt=prompts[node])
                if node=='cut':job['negative_prompt']='spoon, fork, knife, metal, utensil, food filling the recess, protrusion, hand, person, extra garnish, added ingredient'
                cfg['jobs'].append(job)
    write(g/'config.json',cfg);write(g/'frozen_plan.json',dict(status='frozen_before_generation',raw_cells=48,compositions=48,seeds=[41,163,907],nodes=['food','cut'],worker_sha256=sha(ROOT/'run_separate_action_node_worker.py'),config_sha256=sha(g/'config.json'),scope=scope,perfect_result_claim=False))
    write(g/'execution.json',dict(status='waiting_for_predecessor',unix=time.time()))
    while read(ROOT/'gate_v55/execution.json')['status']!='complete_unreviewed':time.sleep(20)
    write(g/'execution.json',dict(status='running',unix=time.time()));procs=[]
    for shard,gpus in enumerate(['0,2','3,4','5,7']):
        args=[PY,'-u',str(ROOT/'run_separate_action_node_worker.py'),'--config',str(g/'config.json'),'--output',str(g/('worker_'+str(shard))),'--gpus',gpus,'--shard',str(shard),'--shards','3'];log=ROOT/'logs'/('gate_v56_'+str(shard)+'.log')
        with log.open('w') as h:p=subprocess.Popen(args,stdout=h,stderr=subprocess.STDOUT,cwd=str(ROOT),start_new_session=True)
        write(g/('worker_'+str(shard)+'_launch.json'),dict(pid=p.pid,argv=args,log=str(log)));procs.append(p)
    while any(p.poll() is None for p in procs):
        if any(p.poll() not in [None,0] for p in procs):raise RuntimeError('Preserve completed node outputs for repair')
        time.sleep(20)
    found={read(p)['id']:p.parent for p in g.glob('worker_*/*/result.json')};assert set(found)=={j['id'] for j in cfg['jobs']};rows=[]
    for cid,c in cases.items():
        geo=ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid;source=rgb(geo/'source.png');edit=np.asarray(Image.open(geo/'edit_mask.png'))>0;food=np.load(ROOT/'source_fit_ellipsoid_food_fields_consumptive_v4'/cid/'field.npz')['food'];fresh=np.load(ROOT/'gate_v37/coupled_source_cut_reconstruction_v3'/(cid+'_field')/'field.npz')['fresh']
        l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h)
        for seed in [41,163,907]:
            oldctx=ROOT/'gate_v50/contexts'/cid/str(seed);tr=read(oldctx/'transform.json');parent=ROOT/'gate_v53/geometry_soft'/(cid+'__'+str(seed))/'pre_sampling.png';base=rgb(parent);candidate=base.astype(float).copy();masks=[];rawhash=[]
            for i,node in enumerate(['food','cut']):
                jid=cid+'__node_'+node+'__'+str(seed);rawpath=found[jid]/'raw.png';raw=rgb(rawpath);assert raw.shape==(384,384,3);box=tr['boxes'][i];x,y,x1,y1=box;m=np.asarray(Image.open(oldctx/('mask_'+str(i)+'.png')))>0;temp=base.astype(float).copy();temp[y:y1,x:x1]=np.asarray(Image.fromarray(raw).resize((x1-x,y1-y),Image.Resampling.LANCZOS));candidate[m]=temp[m];masks.append(m);rawhash.append(sha(rawpath))
            mask=masks[0]|masks[1];white=np.asarray(tr['source_white_chroma']);lum=candidate@np.array([.2126,.7152,.0722]);bc=base/np.maximum(base.sum(2)[...,None],1);plain=(food|fresh)&(distance_transform_edt(mask)>2)&(np.linalg.norm(bc-white,axis=2)<.06);protected=np.asarray(Image.open(oldctx/'protected_source_color.png'))>0
            for variant in ['direct_masked_raw','source_chroma_projected']:
                a=candidate.copy()
                if variant=='source_chroma_projected':a[plain]=white/(white@np.array([.2126,.7152,.0722]))*lum[plain,None]
                alpha=np.clip(distance_transform_edt(mask)/2,0,1)[...,None];pre=np.uint8(np.clip(np.rint(base*(1-alpha)+a*alpha),0,255));assert np.array_equal(pre[~mask],base[~mask]);assert np.array_equal(pre[protected],base[protected])
                native=Image.fromarray(pre).crop(rect).resize(tuple(c['source_size']),Image.Resampling.LANCZOS).resize((w,h),Image.Resampling.LANCZOS);grid=source.copy();grid[t:t+h,l:l+w]=np.asarray(native);blend=np.clip(distance_transform_edt(edit)/5,0,1)[...,None];final=np.uint8(np.clip(np.rint(source*(1-blend)+grid*blend),0,255));assert np.array_equal(final[~edit],source[~edit]);d=g/variant/(cid+'__'+str(seed));d.mkdir(parents=True);Image.fromarray(pre).save(d/'pre_sampling.png');Image.fromarray(final).save(d/'composited.png');Image.fromarray(final).crop(rect).save(d/'view.png');row=dict(case_id=cid,seed=seed,variant=variant,parent_sha256=sha(parent),raw_node_sha256=rawhash,output_sha256=sha(d/'composited.png'),scope=scope);write(d/'result.json',row);rows.append(row)
        board=Image.new('RGB',(2048,1170),'white');dr=ImageDraw.Draw(board)
        for ri,variant in enumerate(['prior','direct_masked_raw','source_chroma_projected']):
            paths=[('source',geo/'source.png')]+[(variant+' '+str(s),ROOT/'gate_v53/geometry_soft'/(cid+'__'+str(s))/'composited.png' if variant=='prior' else g/variant/(cid+'__'+str(s))/'composited.png') for s in [41,163,907]]
            for col,(title,p) in enumerate(paths):
                im=Image.open(p).convert('RGB').crop(rect);im.thumbnail((504,354));board.paste(im,(col*512+(504-im.width)//2,ri*390+28));dr.text((col*512+5,ri*390+5),title,fill='black')
        board.save(g/(cid+'_review.jpg'),quality=95)
    write(g/'execution.json',dict(status='complete_unreviewed',raw_cells=48,compositions=len(rows),rows=rows,unix=time.time()))
    with zipfile.ZipFile(ROOT/'gate_v56_complete.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in g.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(ROOT))
    print('SEPARATE_ACTION_NODES_COMPLETE',48,len(rows),flush=True)

if __name__=='__main__':main()
