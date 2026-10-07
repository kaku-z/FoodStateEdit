"""G51: ray-traced cavity visibility on the unchanged shared scoop geometry.

Differential linear-RGB ambient shading prior, not recovered lighting, true
scattering or an independently measured surface. All coefficients are retained.
"""
import os,sys,json,time,zipfile
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='6'
sys.path.insert(0,'/mnt/tmp/guo-z_first_bite_structure_20260930_runtime/mitsuba_vendor')
import numpy as np
from PIL import Image,ImageDraw
from scipy.stats import qmc
from scipy.ndimage import gaussian_filter,distance_transform_edt
from run_joint_action_full_noise import ROOT,read,write,sha,rgb

def linear(x):
    x=x/255.;return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
def srgb(x):return 255*np.where(x<=.0031308,12.92*x,1.055*np.maximum(x,0)**(1/2.4)-.055)

def main():
    from run_qwen_image_edit_direct_baseline import gpu_snapshot
    assert not gpu_snapshot(6)['compute_processes']
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')
    g=ROOT/'gate_v51';g.mkdir(exist_ok=False);(g/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    plan=dict(status='frozen_before_raytracing',samples=1024,sobol_seed=41,strengths=[.25,.5,.75],max_ray_length_over_body_extent=.75,ray_origin_epsilon_over_body_extent=1e-5,new_raw_calls=0,derived_composites=72,parent='G47/neural_source_grain',scope='Post-hoc geometric ambient visibility prior. No source light measurement, true scans, deformable food, real material scattering or photographic acceptance. Source RGB and geometry are inferred. Every development case/seed/coefficient retained.')
    write(g/'frozen_plan.json',plan);write(g/'execution.json',dict(status='running',unix=time.time()));cases={c['case_id']:c for c in read(ROOT/'inputs/manifest.json')['cases']};rows=[];fields={}
    samples=qmc.Sobol(d=2,scramble=True,seed=41).random_base2(10)
    for cid in cases:
        geo=ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid;a=np.load(ROOT/'spoon_source_fit_ellipsoid_channels_v3'/cid/'geometry_channels.npz');cut=np.load(ROOT/'gate_v37/coupled_source_cut_reconstruction_v3'/(cid+'_field')/'field.npz');mask=cut['fresh'];assert (a['labels'][mask]==1).all()
        rep=read(geo/'geometry_report.json');L=max(np.asarray(rep['fit']['high'])-np.asarray(rep['fit']['low']))
        mp=Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929/geometry')/cid/'maps.npz' if cid.startswith('new_') else ROOT/'geometry_v3'/cid/'maps.npz'
        K=np.load(mp)['intrinsics'].copy();K[0]*=640;K[1]*=480
        yy,xx=np.where(mask);p=np.c_[xx+.5,yy+.5,np.ones(len(xx))]@np.linalg.inv(K).T;p*=a['scene_depth'][mask,None];n=a['scene_normals'][mask].astype(float);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-9)
        scene=mi.load_dict(dict(type='scene',remaining=dict(type='ply',filename=str(geo/'remaining.ply'),bsdf=dict(type='diffuse'))))
        frame=mi.Frame3f(mi.Normal3f(n.T));origin=mi.Point3f((p+n*L*1e-5).T);count=np.zeros(len(p),float);short=None
        for i,s in enumerate(samples):
            ray=mi.Ray3f(origin,frame.to_world(mi.warp.square_to_cosine_hemisphere(mi.Point2f(s.tolist()))));ray.maxt=mi.Float(.75*L);count+=np.asarray(scene.ray_test(ray),float)
            if i==255:short=count/256
        raw=count/1024;field=np.zeros(mask.shape,float);field[mask]=raw
        # Normalize smoothing within the same material mask, without erasing
        # inner-edge visibility or contaminating other image layers.
        den=gaussian_filter(mask.astype(float),.6);field=np.divide(gaussian_filter(field,.6),den,out=np.zeros_like(field),where=den>1e-9);field[~mask]=0
        assert np.isfinite(field).all() and field.min()>=0 and field.max()<=1.00001
        d=g/'fields'/cid;d.mkdir(parents=True);np.save(d/'ambient_occlusion.npy',field);Image.fromarray(np.uint8(np.clip(field,0,1)*255)).save(d/'ambient_occlusion.png')
        write(d/'audit.json',dict(case_id=cid,fresh_pixels=int(mask.sum()),remaining_mesh_sha256=sha(geo/'remaining.ply'),geometry_channels_sha256=sha(ROOT/'spoon_source_fit_ellipsoid_channels_v3'/cid/'geometry_channels.npz'),occlusion_mean=float(raw.mean()),occlusion_max=float(raw.max()),sobol_256_vs_1024_mean_absolute_difference=float(np.abs(short-raw).mean()),sobol_256_vs_1024_p95_absolute_difference=float(np.quantile(np.abs(short-raw),.95)),scope='Numerical convergence diagnostic within an inferred geometry. Not measured physical accuracy.'))
        fields[cid]=field;print('CUT_VISIBILITY',cid,float(raw.mean()),flush=True)
    for cid,c in cases.items():
        geo=ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid;source=rgb(geo/'source.png');edit=np.asarray(Image.open(geo/'edit_mask.png'))>0;cut=np.load(ROOT/'gate_v37/coupled_source_cut_reconstruction_v3'/(cid+'_field')/'field.npz');mask=cut['fresh'];ao=fields[cid];l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h)
        for seed in [41,163,907]:
            parent=ROOT/'gate_v47/neural_source_grain'/(cid+'__'+str(seed))/'pre_sampling.png';base=rgb(parent)
            for strength in plan['strengths']:
                arm='visibility'+str(round(strength*100));candidate=base.astype(float).copy();candidate[mask]=srgb(linear(base[mask].astype(float))*(1-strength*ao[mask,None]));pre=np.uint8(np.clip(np.rint(candidate),0,255));assert np.array_equal(pre[~mask],base[~mask])
                native=Image.fromarray(pre).crop(rect).resize(tuple(c['source_size']),Image.Resampling.LANCZOS).resize((w,h),Image.Resampling.LANCZOS);grid=source.copy();grid[t:t+h,l:l+w]=np.asarray(native);blend=np.clip(distance_transform_edt(edit)/5,0,1)[...,None];final=np.uint8(np.clip(np.rint(source*(1-blend)+grid*blend),0,255));assert np.array_equal(final[~edit],source[~edit])
                d=g/arm/(cid+'__'+str(seed));d.mkdir(parents=True);Image.fromarray(pre).save(d/'pre_sampling.png');Image.fromarray(final).save(d/'composited.png');Image.fromarray(final).crop(rect).save(d/'view.png')
                row=dict(case_id=cid,seed=seed,arm=arm,ambient_strength=strength,parent_sha256=sha(parent),output_sha256=sha(d/'composited.png'),outside_fresh_cut_parent_exact_before_sampling=True,outside_geometry_source_exact=True,neural_calls=0,scope=plan['scope']);write(d/'result.json',row);rows.append(row)
        board=Image.new('RGB',(2048,1560),'white');dr=ImageDraw.Draw(board)
        for ri,arm in enumerate(['prior','visibility25','visibility50','visibility75']):
            paths=[('source',geo/'source.png')]+[(arm+' '+str(s),ROOT/'gate_v47/neural_source_grain'/(cid+'__'+str(s))/'composited.png' if arm=='prior' else g/arm/(cid+'__'+str(s))/'composited.png') for s in [41,163,907]]
            for col,(title,p) in enumerate(paths):
                im=Image.open(p).convert('RGB').crop(rect);im.thumbnail((504,354));board.paste(im,(col*512+(504-im.width)//2,ri*390+28));dr.text((col*512+5,ri*390+5),title,fill='black')
        board.save(g/(cid+'_review.jpg'),quality=95)
    write(g/'execution.json',dict(status='complete_unreviewed',raw_calls=0,compositions=len(rows),rows=rows,unix=time.time()))
    with zipfile.ZipFile(ROOT/'gate_v51_complete.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in g.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(ROOT))
    print('CUT_RAY_VISIBILITY_COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
