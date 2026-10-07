"""Source-color cavity projection with a weak ambient visibility prior.

High scattering is a qualitative material prior here, not a solved physical
renderer or calibrated inverse-lighting reconstruction.
"""
import os,sys,json,hashlib,zipfile,time
from pathlib import Path
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    os.environ['CUDA_VISIBLE_DEVICES']='6';sys.path.insert(0,'/mnt/tmp/guo-z_first_bite_structure_20260930_runtime/mitsuba_vendor')
    import mitsuba as mi,numpy as np
    from PIL import Image,ImageDraw
    from scipy.ndimage import binary_dilation,distance_transform_edt,gaussian_filter
    from scipy.stats import qmc
    mi.set_variant('cuda_ad_rgb');g=ROOT/'gate_v35';out=g/'cavity_source_material_v2';out.mkdir(exist_ok=False);(out/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    cases={c['case_id']:c for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases']};rows=[]
    (out/'frozen_plan.json').write_text(json.dumps({'created_unix':time.time(),'status':'frozen_before_projection','raw_calls':0,'cases':8,'seeds':[41,163,907],'parents':['consumptive_surface_neural_light','no_food_only_crop_ablation'],'source_color_rule':'65th percentile of source-observed plain plane RGB medians, 70% ambient plus 30% directional fit, bounded [.90,1.08] relative to that source color; 10% cosine visibility shadow.','scope':'Source RGB/ambient scattering appearance prior; not measured albedo, lighting or hidden texture.'},indent=2))
    for cid,c in cases.items():
        geo=ROOT/'geometry_spoon_observed_surface_v1'/cid;data=np.load(ROOT/'spoon_observed_surface_channels_v1'/cid/'geometry_channels.npz');rep=json.loads((geo/'geometry_report.json').read_text());L=max(np.asarray(rep['fit']['high'])-np.asarray(rep['fit']['low']));z=data['remaining_depth'];full=data['full_depth'];finite=np.isfinite(z)&np.isfinite(full);delta=np.zeros(z.shape);np.subtract(z,full,out=delta,where=finite);cut=binary_dilation(np.asarray(Image.open(geo/'source_bite_mask.png'))>0,iterations=2);mask=finite&cut&(delta>L*1e-5)&(data['labels']==1);assert mask.sum()>100
        mp=(Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929/geometry')/cid/'maps.npz') if cid.startswith('new_') else ROOT/'geometry_v3'/cid/'maps.npz';K=np.load(mp)['intrinsics'].copy();K[0]*=640;K[1]*=480;yy,xx=np.where(mask);p=np.c_[xx+.5,yy+.5,np.ones(len(xx))]@np.linalg.inv(K).T;p*=z[mask][:,None];n=data['scene_normals'][mask].copy();n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-8)
        scene=mi.load_dict({'type':'scene','remaining':{'type':'ply','filename':str(geo/'remaining.ply'),'bsdf':{'type':'diffuse'}}});frame=mi.Frame3f(mi.Normal3f(n.T));origin=mi.Point3f((p+n*L*1e-5).T);samples=qmc.Sobol(d=2,scramble=True,seed=41).random_base2(8);occluded=np.zeros(len(p))
        for sample in samples:
            ray=mi.Ray3f(origin,frame.to_world(mi.warp.square_to_cosine_hemisphere(mi.Point2f(sample.tolist()))));ray.maxt=mi.Float(.75*L);occluded+=np.asarray(scene.ray_test(ray),float)
        ao=np.zeros(mask.shape);ao[mask]=occluded/256;ao=gaussian_filter(ao,.6)/np.maximum(gaussian_filter(mask.astype(float),.6),1e-8);ao*=mask
        appearance=json.loads((geo/'appearance_calibration.json').read_text());coefficient=np.asarray(appearance['coefficient_rgb']);white=np.quantile(np.asarray(appearance['plane_median_rgb']),.65,axis=0);predicted=np.c_[np.ones(len(n)),n]@coefficient*255;material=np.clip(.70*white+.30*predicted,.90*white,1.08*white)*(1-.10*ao[mask])[:,None]
        f=out/(cid+'_field');f.mkdir();np.savez_compressed(f/'field.npz',mask=mask,ao=ao,pixels_rgb=material);Image.fromarray(np.uint8(mask)*255).save(f/'cavity_mask.png');source=np.asarray(Image.open(geo/'source.png').convert('RGB'));edit=np.asarray(Image.open(geo/'edit_mask.png'))>0;l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h)
        for seed in [41,163,907]:
            for arm in ['consumptive_surface_neural_light','no_food_only_crop_ablation']:
                if arm=='consumptive_surface_neural_light':jid=cid+'__observed_surface_food_intrinsic__'+str(seed);parent=g/'all_observed_surface_consumptive_v2/source_rgb_scalar_neural_light'/jid/'composited.png'
                else:jid=cid+'__source_geometry_food__'+str(seed);parent=g/'source_geometry_food_ablation_v1'/jid/'composited.png'
                base=np.asarray(Image.open(parent).convert('RGB'),float);gray=base@np.array([.2126,.7152,.0722]);detail=np.clip(gray-gaussian_filter(gray,2),-3,3);candidate=base.copy();candidate[mask]=material+detail[mask][:,None];alpha=np.clip(distance_transform_edt(mask)/2,0,1)[...,None];pre=np.uint8(np.clip(np.rint(base*(1-alpha)+candidate*alpha),0,255));assert np.array_equal(pre[~mask],base.astype(np.uint8)[~mask])
                native=Image.fromarray(pre).crop(rect).resize(tuple(c['source_size']),Image.Resampling.LANCZOS).resize((w,h),Image.Resampling.LANCZOS);grid=source.copy();grid[t:t+h,l:l+w]=np.asarray(native);blend=np.clip(distance_transform_edt(edit)/5,0,1)[...,None];final=np.uint8(np.clip(np.rint(source*(1-blend)+grid*blend),0,255));assert np.array_equal(final[~edit],source[~edit]);dest=out/arm/(cid+'__'+str(seed));dest.mkdir(parents=True);Image.fromarray(pre).save(dest/'pre_sampling.png');Image.fromarray(final).save(dest/'composited.png');Image.fromarray(final).crop(rect).save(dest/'view.png')
                row={'id':cid+'__'+str(seed),'case_id':cid,'seed':seed,'arm':arm,'parent_sha256':sha(parent),'parent':str(parent),'raw_calls':0,'source_fitted_color_rgb':white.tolist(),'outside_cavity_parent_exact_before_sampling':True,'outside_geometry_source_exact':True,'cavity_pixels':int(mask.sum()),'scope':'Only new hidden cavity appearance changes, using a source-color and weak ambient visibility prior. Flat garnish and geometric shape assumptions remain.'};(dest/'result.json').write_text(json.dumps(row,indent=2));rows.append(row)
        for arm in ['consumptive_surface_neural_light','no_food_only_crop_ablation']:
            paths=[('source',geo/'source.png')]+[('source-color cavity seed'+str(s),out/arm/(cid+'__'+str(s))/'composited.png') for s in [41,163,907]];board=Image.new('RGB',(2560,h+28),'white');draw=ImageDraw.Draw(board)
            for i,(title,path) in enumerate(paths):board.paste(Image.open(path).convert('RGB').crop(rect).resize((640,h),Image.Resampling.LANCZOS),(i*640,28));draw.text((i*640+5,6),title,fill='black')
            board.save(out/(cid+'_'+arm+'_three_seed.jpg'),quality=95)
        print('CAVITY',cid,flush=True)
    (out/'manifest.json').write_text(json.dumps({'status':'complete_unreviewed','derived_composites':len(rows),'all_cells_same_rule':True,'additional_raw_calls':0,'rows':rows,'script_sha256':sha(Path(__file__))},indent=2))
    with zipfile.ZipFile(ROOT/'cavity_source_material_v2.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(ROOT))
if __name__=='__main__':main()
