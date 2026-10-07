"""A single directional light prior for carried food and its shared cut.

Source RGB and geometry fix identity/action; source or robust neural luminance
fixes one scalar irradiance field. Hidden cut grain is explicitly synthesized
from a bare source patch's spectrum, not observed interior texture.
"""
import json,hashlib,time,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import gaussian_filter,distance_transform_edt
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930');LUMA=np.array([.2126,.7152,.0722])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rgb(p):return np.asarray(Image.open(p).convert('RGB'))
def main():
    out=ROOT/'gate_v47';out.mkdir(exist_ok=False);(out/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    arms=['source_diffuse','neural_diffuse','neural_source_grain'];plan=dict(status='frozen_before_composition',arms=arms,new_raw_calls=0,derived_cells=72,seeds=[41,163,907],camera_geometry_stage='G37 unchanged',source_stage='G46 source_detail',directional_ridge=20,neural_fit_weight=.35,cavity_ambient_occlusion_prior=.94,source_texture_spectrum_rgb_bound=3,scope='Post-hoc development study on all eight previously reviewed sources. Shared scalar directional lighting and inferred cut geometry; no measured albedo, hidden texture, scattering or deformable food physics. No per-case or per-seed selection.')
    (out/'frozen_plan.json').write_text(json.dumps(plan,indent=2));cfg=json.loads((ROOT/'gate_v42/config.json').read_text());cases={c['case_id']:c for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases']};raw={json.loads(p.read_text())['id']:p.parent for p in (ROOT/'gate_v42').glob('worker_*/*/result.json')};rows=[]
    for j in cfg['jobs']:
        cid,s=j['case_id'],j['seed'];c=cases[cid];geo=ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid;source=rgb(geo/'source.png');basepath=ROOT/'gate_v46/source_detail'/(cid+'__'+str(s))/'pre_sampling.png';base=rgb(basepath);f=np.load(ROOT/'source_fit_ellipsoid_food_fields_consumptive_v4'/cid/'field.npz');a=np.load(ROOT/'spoon_source_fit_ellipsoid_channels_v3'/cid/'geometry_channels.npz');cut=np.load(ROOT/'gate_v37/coupled_source_cut_reconstruction_v3'/(cid+'_field')/'field.npz');rep=json.loads((geo/'geometry_report.json').read_text());cal=json.loads((geo/'appearance_calibration.json').read_text());R=np.asarray(rep['fit']['axes_camera_columns']);Q=np.asarray(rep['cut_and_support']['rotation_food_frame']);origin=np.asarray(rep['cut_and_support']['source_center']);dest=np.asarray(rep['cut_and_support']['destination_center']);center=np.asarray(rep['cut_and_support']['source_cutter_center']);radius=rep['cut_and_support']['cutter_radii'][0];lower=rep['cut_and_support']['cutter_radii'][2];upper=rep['cut_and_support']['upper_vertical_radius_over_bowl_radius']*radius
        mp=Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929/geometry')/cid/'maps.npz' if cid.startswith('new_') else ROOT/'geometry_v3'/cid/'maps.npz';K=np.load(mp)['intrinsics'].copy();K[0]*=640;K[1]*=480;yy,xx=np.indices((480,640));rays=np.stack([xx+.5,yy+.5,np.ones_like(xx)],-1)@np.linalg.inv(K).T;food=f['food'];fresh=cut['fresh'];normals=a['scene_normals'].astype(float).copy()
        # Smooth analytic normals on the same ellipsoidal cut, not changed RGB
        # labels or the GPU raster boundary. Inherited plane normals remain.
        for mask,carried in [(food,True),(fresh,False)]:
            points=(rays[mask]*a['scene_depth'][mask,None])@R
            if carried:points=(points-dest)@Q+origin
            q=points-center;rz=np.where(q[:,2]>=0,upper,lower);implicit=(q[:,0]**2+q[:,1]**2)/radius**2+q[:,2]**2/rz**2;curved=np.abs(implicit-1)<.045
            gradient=np.c_[q[:,0]/radius**2,q[:,1]/radius**2,q[:,2]/rz**2];gradient/=np.maximum(np.linalg.norm(gradient,axis=1)[:,None],1e-9)
            newnorm=gradient@(Q.T if carried else np.eye(3))@R.T*(1 if carried else -1);old=normals[mask];old[curved]=newnorm[curved];normals[mask]=old
        ctx=Path(j['transform']).parent;white=np.median(rgb(ctx/'reference.png').reshape(-1,3),0);chroma=white/white.sum();tr=json.loads(Path(j['transform']).read_text());x0,y0,x1,y1=tr['box'];candidate=base.astype(float).copy();rawpath=raw[j['id']]/'raw.png';candidate[y0:y1,x0:x1]=np.asarray(Image.open(rawpath).convert('RGB').resize((192,192),Image.Resampling.LANCZOS));rawlum=candidate@LUMA;rawchroma=candidate/np.maximum(candidate.sum(2)[...,None],1);support=food&(distance_transform_edt(food)>5)&(np.linalg.norm(rawchroma-chroma,axis=2)<.04)&(rawlum>.6*(white@LUMA))&(rawlum<254.5)
        N=np.asarray(cal['observed_plane_normals_camera']);Y=np.asarray(cal['plane_median_rgb'])@LUMA;A=np.c_[np.ones(len(N)),N];ridge=np.diag([.02,20,20,20]);prior=np.r_[float(np.median(Y)),0,0,0];source_beta=np.linalg.solve(A.T@A+ridge*.01,A.T@Y+ridge*.01@prior)
        nf=normals[support];nl=rawlum[support];B=np.c_[np.ones(len(nf)),nf];weights=np.ones(len(nf));beta=source_beta.copy()
        if len(nf)>=20:
            for _ in range(8):
                scaleweight=.35/max(1,len(nl));beta=np.linalg.solve(A.T@A+scaleweight*(B.T@(weights[:,None]*B))+ridge,A.T@Y+scaleweight*(B.T@(weights*nl))+ridge@source_beta);res=B@beta-nl;scale=max(1.,1.4826*np.median(np.abs(res-np.median(res))));weights=np.minimum(1,1.5*scale/np.maximum(np.abs(res),1e-6))
        observed=np.zeros(food.shape,bool);observed[f['yy'][f['valid']],f['xx'][f['valid']]]=True;original=np.zeros(source.shape,float);original[f['yy'][f['valid']],f['xx'][f['valid']]]=f['sampled'][f['valid']];oldnorm=a['scene_normals'][food]@R@Q@R.T;oldlum=np.c_[np.ones(int(food.sum())),oldnorm]@source_beta
        sourcechroma=original/np.maximum(original.sum(2)[...,None],1);distance=np.linalg.norm(sourcechroma-chroma,axis=2);q=np.clip((.09-distance)/.06,0,1);confidence=q*q*(3-2*q);confidence[food&~observed]=1;confidence[~food]=0
        # Phase-synthesized luminance microtexture; its distribution comes from
        # the actual bare reference. It is an interior appearance prior.
        box=cal['source_material_box_canvas'];bx,by,ex,ey=box;patch=source[by:ey,bx:ex]@LUMA;detail=patch-gaussian_filter(patch,2);spectrum=np.abs(np.fft.rfft2(detail));rng=np.random.default_rng(s);phase=np.exp(1j*rng.uniform(-np.pi,np.pi,spectrum.shape));grain=np.fft.irfft2(spectrum*phase,s=detail.shape).real;grain=np.asarray(Image.fromarray(grain.astype(np.float32),mode='F').resize((640,480),Image.Resampling.BILINEAR));grain=np.clip(grain,-3,3)
        l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h);edit=np.asarray(Image.open(geo/'edit_mask.png'))>0
        for arm in arms:
            coefficient=source_beta if arm=='source_diffuse' else beta;light=np.clip(np.c_[np.ones(480*640),normals.reshape(-1,3)]@coefficient,.45*np.median(Y),min(255,1.35*np.median(Y))).reshape(480,640);withgrain=arm=='neural_source_grain';desired=light+grain if withgrain else light
            canvas=base.astype(float).copy();predicted=chroma/(chroma@LUMA)*desired[...,None];target=predicted.copy();gain=np.clip(desired[food]/np.maximum(oldlum,1),.5,1.6);transferred=original[food]*gain[:,None];target[food]=np.where(observed[food,None],transferred,predicted[food]);alpha=confidence[...,None];canvas[food]=(original*(1-alpha)+target*alpha)[food];canvas[fresh]=predicted[fresh]*.94;canvas[cut['background']]=cut['rgb'][cut['background']];inherited=cut['reconstruct']&~fresh&~cut['background'];canvas[inherited]=source[inherited]
            pre=np.uint8(np.clip(np.rint(canvas),0,255));action=food|cut['reconstruct'];assert np.array_equal(pre[~action],base[~action]);native=Image.fromarray(pre).crop(rect).resize(tuple(c['source_size']),Image.Resampling.LANCZOS).resize((w,h),Image.Resampling.LANCZOS);grid=source.copy();grid[t:t+h,l:l+w]=np.asarray(native);blend=np.clip(distance_transform_edt(edit)/5,0,1)[...,None];final=np.uint8(np.clip(np.rint(source*(1-blend)+grid*blend),0,255));assert np.array_equal(final[~edit],source[~edit]);d=out/arm/(cid+'__'+str(s));d.mkdir(parents=True);Image.fromarray(pre).save(d/'pre_sampling.png');Image.fromarray(final).save(d/'composited.png');Image.fromarray(final).crop(rect).save(d/'view.png');row=dict(case_id=cid,seed=s,arm=arm,raw_calls=0,base_sha256=sha(basepath),raw_material_sha256=sha(rawpath),output_sha256=sha(d/'composited.png'),source_directional_coefficients=source_beta.tolist(),joint_directional_coefficients=coefficient.tolist(),neural_plain_fit_pixels=int(support.sum()),outside_food_and_cut_parent_exact_before_sampling=True,outside_geometry_source_exact=True,scope=plan['scope']);(d/'result.json').write_text(json.dumps(row,indent=2));rows.append(row)
        print('JOINT_LIGHT',cid,s,flush=True)
    for cid,c in cases.items():
        l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h);board=Image.new('RGB',(2048,1560),'white');dr=ImageDraw.Draw(board)
        for ri,arm in enumerate(['prior']+arms):
            paths=[('source',ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid/'source.png')]+[(arm+' '+str(s),ROOT/'gate_v46/source_detail'/(cid+'__'+str(s))/'composited.png' if arm=='prior' else out/arm/(cid+'__'+str(s))/'composited.png') for s in [41,163,907]]
            for col,(title,p) in enumerate(paths):
                im=Image.open(p).convert('RGB').crop(rect);im.thumbnail((504,354));board.paste(im,(col*512+(504-im.width)//2,ri*390+28));dr.text((col*512+5,ri*390+5),title,fill='black')
        board.save(out/(cid+'_review.jpg'),quality=95)
    (out/'execution.json').write_text(json.dumps(dict(status='complete_unreviewed',derived_composites=len(rows),raw_calls=0,rows=rows,unix=time.time()),indent=2))
    with zipfile.ZipFile(ROOT/'gate_v47_complete.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(ROOT))
    print('JOINT_LIGHT_COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
