"""Whole-image, jointly conditioned bite/cut appearance under bounded edit masks.

This tests whether releasing local material boundaries while anchoring coarse
action structure improves realism. Every development source/seed is retained.
"""
import copy,json,time,hashlib,subprocess,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_dilation,distance_transform_edt
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930');PY='/home/yanai-lab/guo-z/.conda/envs/flux_pure_env/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    for _ in range(10):
        try:return json.loads(p.read_text())
        except (OSError,json.JSONDecodeError):time.sleep(.3)
    raise RuntimeError(str(p))
def write(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def rgb(p):return np.asarray(Image.open(p).convert('RGB'))
def main():
    g=ROOT/'gate_v48';g.mkdir(exist_ok=False);(g/'executed_orchestrator.py').write_bytes(Path(__file__).read_bytes());cfg=read(ROOT/'gate_v23/config.json');cfg.update(stage='development_joint_action_refinement',not_formal=True,jobs=[]);cfg['inference']['steps']=24;cases={c['case_id']:c for c in read(ROOT/'inputs/manifest.json')['cases']}
    for cid,c in cases.items():
        geo=ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid;f=np.load(ROOT/'source_fit_ellipsoid_food_fields_consumptive_v4'/cid/'field.npz');cut=np.load(ROOT/'gate_v37/coupled_source_cut_reconstruction_v3'/(cid+'_field')/'field.npz');food=f['food'];source=rgb(geo/'source.png');edit=np.asarray(Image.open(geo/'edit_mask.png'))>0;a=np.load(ROOT/'spoon_source_fit_ellipsoid_channels_v3'/cid/'geometry_channels.npz');spoon=a['labels']==3
        selection=read(ROOT/'gate_v13_run03'/cid/'reference_selection.json');x,y,x1,y1=selection['material_box_canvas'];white=np.median(source[y:y1,x:x1].reshape(-1,3),0);white/=white.sum();ref=Image.fromarray(source[y:y1,x:x1]).resize((512,512),Image.Resampling.LANCZOS);observed=np.zeros(food.shape,bool);observed[f['yy'][f['valid']],f['xx'][f['valid']]]=True
        for seed in [41,163,907]:
            d=g/'contexts'/cid/str(seed);d.mkdir(parents=True);parent=ROOT/'gate_v46/source_detail'/(cid+'__'+str(seed))/'pre_sampling.png';base=rgb(parent);chroma=base/np.maximum(base.sum(2)[...,None],1);garnish=food&observed&(np.linalg.norm(chroma-white,axis=2)>.06);action=food|cut['reconstruct'];mask=binary_dilation(action,iterations=4)&edit&~spoon&~binary_dilation(garnish,iterations=1)
            weight=np.zeros(food.shape,float);weight[mask]=.15;boundary=mask&((distance_transform_edt(food)<4)&food|((distance_transform_edt(cut['reconstruct'])<4)&cut['reconstruct']));weight[boundary]=1
            Image.fromarray(base).save(d/'layout.png');Image.fromarray(source).save(d/'source.png');ref.save(d/'reference.png');Image.fromarray(np.uint8(mask)*255).save(d/'edit_mask.png');Image.fromarray(np.uint8(weight*255)).save(d/'structure_weight.png');Image.fromarray(np.uint8(garnish)*255).save(d/'protected_source_color.png');write(d/'transform.json',dict(parent=str(parent),parent_sha256=sha(parent),original_source=str(geo/'source.png'),mask_sha256=sha(d/'edit_mask.png'),editable_pixels=int(mask.sum()),boundary_freedom_canvas_pixels=4,scope='Approximate color garnish protection and bounded edit area. A changed model silhouette is not guaranteed to preserve the proxy volume/contact.'))
            for anchor in ['free','coarse_anchor']:
                method='joint_action_'+anchor;job=dict(id=cid+'__'+method+'__'+str(seed),case_id=cid,method=method,seed=seed,start_raw_sigma=.65,input_order=['layout','source','reference'],lock_context=True,transform=str(d/'transform.json'),files={k:dict(path=str(d/(k+'.png')),sha256=sha(d/(k+'.png'))) for k in ['layout','source','reference','edit_mask','structure_weight']},prompt='Image 1 is a layout of this actual meal after lifting its very first spoonful, before eating. Image 2 is the real original meal before lifting. Image 3 is the plain silken tofu from this exact meal. Make Image 1 a natural photograph of the same moment and same food: one small moist soft tofu portion resting securely on the already raised stainless steel spoon, visibly separate from the plate, with a matching small smoothly scooped recess in the tofu on the plate. Refine both the spoonful and its matching recess together with the original camera lighting, dense fine soybean gel texture, gentle translucency and soft irregular cut edges. Preserve the existing spoon location and angle, portion size and overall shape, existing source sauce and garnish, all other food, plate and background. The lifted portion contains only the exact toppings already visible in Image 1. Keep a solid filled spoonful. No extra herb, new ingredient, duplicate food, hand or person.')
                if anchor=='coarse_anchor':job['structure']=dict(lowpass=True,kernel=3,begin_taper_fraction=.25,end_fraction=.65,tail_weight=.10)
                cfg['jobs'].append(job)
    write(g/'config.json',cfg);write(g/'frozen_plan.json',dict(status='frozen_before_generation',new_raw_cells=48,compositions=96,seeds=[41,163,907],arms=['free','coarse_anchor'],requested_sigma=.65,selection_rule='All eight repeatedly viewed development images, every fixed seed and both arms; no individual best selection.',outputs=['direct_masked_raw','source_chroma_projected'],worker_sha256=sha(ROOT/'run_bite_material_sdedit.py'),config_sha256=sha(g/'config.json'),scope='Post-hoc joint-region rendering-prior experiment. Coarse latent anchoring and a bounded image mask do not prove 3D conservation or real contact after generated boundary refinement.'))
    write(g/'execution.json',dict(status='running',unix=time.time()));procs=[]
    for shard,gpus in enumerate(['0,2','3,4','1,6','5,7']):
        args=[PY,'-u',str(ROOT/'run_bite_material_sdedit.py'),'--config',str(g/'config.json'),'--output',str(g/('worker_'+str(shard))),'--gpus',gpus,'--shard',str(shard),'--shards','4'];log=ROOT/'logs'/('gate_v48_'+str(shard)+'.log')
        with log.open('w') as h:p=subprocess.Popen(args,stdout=h,stderr=subprocess.STDOUT,cwd=str(ROOT),start_new_session=True)
        write(g/('worker_'+str(shard)+'_launch.json'),dict(pid=p.pid,argv=args,log=str(log)));procs.append(p)
    while any(p.poll() is None for p in procs):
        if any(p.poll() not in [None,0] for p in procs):raise RuntimeError('A worker requires repair; preserve completed cells')
        time.sleep(20)
    found={read(p)['id']:p.parent for p in g.glob('worker_*/*/result.json')};assert set(found)=={j['id'] for j in cfg['jobs']};rows=[]
    for j in cfg['jobs']:
        cid=j['case_id'];s=j['seed'];ctx=Path(j['transform']).parent;tr=read(Path(j['transform']));base=rgb(Path(tr['parent']));rawpath=found[j['id']]/'raw.png';raw=rgb(rawpath);assert raw.shape==base.shape;mask=np.asarray(Image.open(ctx/'edit_mask.png'))>0;alpha=np.clip(distance_transform_edt(mask)/2,0,1)[...,None];reference=rgb(ctx/'reference.png');white=np.median(reference.reshape(-1,3),0);white/=white.sum();f=np.load(ROOT/'source_fit_ellipsoid_food_fields_consumptive_v4'/cid/'field.npz');cut=np.load(ROOT/'gate_v37/coupled_source_cut_reconstruction_v3'/(cid+'_field')/'field.npz');interior=(f['food']|cut['fresh'])&(distance_transform_edt(mask)>2);luma=raw@np.array([.2126,.7152,.0722]);original_chroma=base/np.maximum(base.sum(2)[...,None],1);plain=interior&(np.linalg.norm(original_chroma-white,axis=2)<.06);c=cases[cid];l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h);source=rgb(Path(tr['original_source']));edit=np.asarray(Image.open(Path(tr['original_source']).parent/'edit_mask.png'))>0
        for variant in ['direct_masked_raw','source_chroma_projected']:
            candidate=raw.astype(float).copy()
            if variant=='source_chroma_projected':candidate[plain]=white/(white@np.array([.2126,.7152,.0722]))*luma[plain,None]
            pre=np.uint8(np.clip(np.rint(base*(1-alpha)+candidate*alpha),0,255));assert np.array_equal(pre[~mask],base[~mask]);native=Image.fromarray(pre).crop(rect).resize(tuple(c['source_size']),Image.Resampling.LANCZOS).resize((w,h),Image.Resampling.LANCZOS);grid=source.copy();grid[t:t+h,l:l+w]=np.asarray(native);blend=np.clip(distance_transform_edt(edit)/5,0,1)[...,None];final=np.uint8(np.clip(np.rint(source*(1-blend)+grid*blend),0,255));assert np.array_equal(final[~edit],source[~edit]);d=g/variant/j['id'];d.mkdir(parents=True);Image.fromarray(pre).save(d/'pre_sampling.png');Image.fromarray(final).save(d/'composited.png');Image.fromarray(final).crop(rect).save(d/'view.png');row=dict(id=j['id'],case_id=cid,seed=s,arm=j['method'],variant=variant,raw_sha256=sha(rawpath),parent_sha256=tr['parent_sha256'],output_sha256=sha(d/'composited.png'),outside_bounded_edit_parent_exact_before_sampling=True,outside_geometry_source_exact=True,actual_start_sigma=read(found[j['id']]/'request.json')['actual_start_sigma'],scope='Model geometry refinement inside a bounded region; source chroma projection changes color, not shape. Geometry masks cannot establish the actual generated food silhouette, contact or volume.');write(d/'result.json',row);rows.append(row)
    for cid,c in cases.items():
        l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h);board=Image.new('RGB',(2048,1950),'white');dr=ImageDraw.Draw(board)
        choices=[('prior',None)]+[(anchor,v) for anchor in ['free','coarse_anchor'] for v in ['direct_masked_raw','source_chroma_projected']]
        for ri,(anchor,v) in enumerate(choices):
            paths=[('source',ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid/'source.png')]+[(anchor+' '+str(s)+((' '+v) if v else ''),ROOT/'gate_v46/source_detail'/(cid+'__'+str(s))/'composited.png' if v is None else g/v/(cid+'__joint_action_'+anchor+'__'+str(s))/'composited.png') for s in [41,163,907]]
            for col,(title,p) in enumerate(paths):
                im=Image.open(p).convert('RGB').crop(rect);im.thumbnail((504,354));board.paste(im,(col*512+(504-im.width)//2,ri*390+28));dr.text((col*512+5,ri*390+5),title,fill='black')
        board.save(g/(cid+'_review.jpg'),quality=95)
    write(g/'execution.json',dict(status='complete_unreviewed',raw_cells=48,compositions=len(rows),rows=rows,unix=time.time()))
    with zipfile.ZipFile(ROOT/'gate_v48_complete.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in g.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(ROOT))
    print('JOINT_ACTION_COMPLETE',48,len(rows),flush=True)
if __name__=='__main__':main()
