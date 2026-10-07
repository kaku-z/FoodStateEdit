"""G52: positive source-light interpolation, cut visibility and neural grain.

Low-frequency appearance comes from the shared inferred geometry and source
planes. The paired diffusion output supplies only bounded high-frequency
luminance, so it cannot replace the spoonful or reverse cavity shading.
"""
import json,time,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import gaussian_filter,distance_transform_edt
from run_joint_action_full_noise import ROOT,read,write,sha,rgb

LUMA=np.array([.2126,.7152,.0722])
def linear(x):
    x=x/255.;return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
def srgb(x):return 255*np.where(x<=.0031308,12.92*x,1.055*np.maximum(x,0)**(1/2.4)-.055)

def source_light(normals,N,Y):
    # Positive angular interpolation avoids RGB affine extrapolation outside
    # the measured-in-image plane colors. These are RGB observations, not albedo.
    scores=8*np.asarray(normals)@N.T;scores-=scores.max(axis=-1,keepdims=True)
    weights=np.exp(scores);weights/=weights.sum(axis=-1,keepdims=True)
    return weights@Y

def normals_for(cid,food,fresh,a):
    geo=ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid;rep=read(geo/'geometry_report.json');R=np.asarray(rep['fit']['axes_camera_columns']);Q=np.asarray(rep['cut_and_support']['rotation_food_frame']);origin=np.asarray(rep['cut_and_support']['source_center']);dest=np.asarray(rep['cut_and_support']['destination_center']);center=np.asarray(rep['cut_and_support']['source_cutter_center']);radius=rep['cut_and_support']['cutter_radii'][0];lower=rep['cut_and_support']['cutter_radii'][2];upper=rep['cut_and_support']['upper_vertical_radius_over_bowl_radius']*radius
    mp=Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929/geometry')/cid/'maps.npz' if cid.startswith('new_') else ROOT/'geometry_v3'/cid/'maps.npz';K=np.load(mp)['intrinsics'].copy();K[0]*=640;K[1]*=480;yy,xx=np.indices(food.shape);rays=np.stack([xx+.5,yy+.5,np.ones_like(xx)],-1)@np.linalg.inv(K).T;n=a['scene_normals'].astype(float).copy()
    for mask,carried in [(food,True),(fresh,False)]:
        p=(rays[mask]*a['scene_depth'][mask,None])@R
        if carried:p=(p-dest)@Q+origin
        q=p-center;rz=np.where(q[:,2]>=0,upper,lower);implicit=(q[:,0]**2+q[:,1]**2)/radius**2+q[:,2]**2/rz**2;curved=np.abs(implicit-1)<.045
        grad=np.c_[q[:,0]/radius**2,q[:,1]/radius**2,q[:,2]/rz**2];grad/=np.maximum(np.linalg.norm(grad,axis=1)[:,None],1e-9);smooth=grad@(Q.T if carried else np.eye(3))@R.T*(1 if carried else -1);old=n[mask];old[curved]=smooth[curved];n[mask]=old
    return n,R,Q

def main():
    g=ROOT/'gate_v52';g.mkdir(exist_ok=False);(g/'executed_script.py').write_bytes(Path(__file__).read_bytes());plan=dict(status='frozen_before_composition',arms=['geometry_light_only','paired_neural_grain'],new_raw_calls=0,derived_composites=48,seeds=[41,163,907],positive_normal_kernel=8,cavity_ambient_strength=.5,neural_bandpass_sigma_in_magnified_panel=3,detail_rgb_luminance_cap=6,scope='Post-hoc development on all eight repeatedly viewed sources. Source RGB directional observations, positive interpolation and geometric visibility are appearance priors, not recovered albedo or calibrated lighting. G50 raw contributes only a bounded high-frequency luminance residual; its geometry and low-frequency shading are discarded. Every case/seed/arm retained.')
    write(g/'frozen_plan.json',plan);write(g/'execution.json',dict(status='waiting_for_predecessor',unix=time.time()))
    while read(ROOT/'gate_v50/execution.json')['status']!='complete_unreviewed':time.sleep(20)
    write(g/'execution.json',dict(status='running',unix=time.time()));cases={c['case_id']:c for c in read(ROOT/'inputs/manifest.json')['cases']};raws={read(p)['id']:p.parent/'raw.png' for p in (ROOT/'gate_v50').glob('worker_*/*/result.json')};rows=[]
    for cid,c in cases.items():
        geo=ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid;source=rgb(geo/'source.png');edit=np.asarray(Image.open(geo/'edit_mask.png'))>0;a=np.load(ROOT/'spoon_source_fit_ellipsoid_channels_v3'/cid/'geometry_channels.npz');f=np.load(ROOT/'source_fit_ellipsoid_food_fields_consumptive_v4'/cid/'field.npz');cut=np.load(ROOT/'gate_v37/coupled_source_cut_reconstruction_v3'/(cid+'_field')/'field.npz');food=f['food'];fresh=cut['fresh'];n,R,Q=normals_for(cid,food,fresh,a);cal=read(geo/'appearance_calibration.json');N=np.asarray(cal['observed_plane_normals_camera']);Y=linear(np.asarray(cal['plane_median_rgb']));light=source_light(n,N,Y);oldlight=source_light(a['scene_normals'][food]@R@Q@R.T,N,Y)
        ao=np.load(ROOT/'gate_v51/fields'/cid/'ambient_occlusion.npy');light[fresh]*=(1-.5*ao[fresh,None])
        observed=np.zeros(food.shape,bool);observed[f['yy'][f['valid']],f['xx'][f['valid']]]=True;original=np.zeros(source.shape,float);original[f['yy'][f['valid']],f['xx'][f['valid']]]=f['sampled'][f['valid']]
        l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h)
        for seed in [41,163,907]:
            ctx=ROOT/'gate_v50/contexts'/cid/str(seed);tr=read(ctx/'transform.json');rawpath=raws[cid+'__paired_patch_action__'+str(seed)];raw=rgb(rawpath);parent=ROOT/'gate_v46/source_detail'/(cid+'__'+str(seed))/'pre_sampling.png';base=rgb(parent);white=np.asarray(tr['source_white_chroma']);chroma=original/np.maximum(original.sum(2)[...,None],1);q=np.clip((.09-np.linalg.norm(chroma-white,axis=2))/.06,0,1);confidence=q*q*(3-2*q);confidence[food&~observed]=1;confidence[~food]=0;protected=np.asarray(Image.open(ctx/'protected_source_color.png'))>0;confidence[protected]=0
            gain=np.clip(light[food]/np.maximum(oldlight,1e-5),.5,1.6);target=light.copy();target[food]=np.where(observed[food,None],linear(original[food])*gain,target[food]);target_rgb=srgb(np.clip(target,0,1));body=base.astype(float).copy();body[food]=(base*(1-confidence[...,None])+target_rgb*confidence[...,None])[food];body[fresh]=target_rgb[fresh]
            detail=np.zeros(food.shape,float)
            for i,box in enumerate(tr['boxes']):
                x,y,x1,y1=box;part=raw[:,i*384:(i+1)*384]@LUMA;high=part-gaussian_filter(part,3)
                small=np.asarray(Image.fromarray(high.astype(np.float32)).resize((x1-x,y1-y),Image.Resampling.LANCZOS));m=food if i==0 else fresh;dest=np.zeros(food.shape,float);dest[y:y1,x:x1]=small;detail[m]=dest[m]
            ref=rgb(ctx/'reference.png')@LUMA;refdetail=ref-gaussian_filter(ref,3);bound=float(np.clip(np.quantile(np.abs(refdetail),.95),1,6));detail=np.clip(detail,-bound,bound)
            strength=confidence.copy();strength[fresh]=1;strength[food& (distance_transform_edt(food)<=1)]=0;strength[fresh&(distance_transform_edt(fresh)<=1)]=0;strength[protected]=0
            for arm in plan['arms']:
                candidate=body.copy()
                if arm=='paired_neural_grain':
                    lum=candidate@LUMA;ratio=np.maximum(lum+detail*strength,0)/np.maximum(lum,1);eligible=food|fresh;candidate[eligible]*=ratio[eligible,None]
                pre=np.uint8(np.clip(np.rint(candidate),0,255));action=food|fresh;assert np.array_equal(pre[~action],base[~action]);assert np.array_equal(pre[protected],base[protected])
                native=Image.fromarray(pre).crop(rect).resize(tuple(c['source_size']),Image.Resampling.LANCZOS).resize((w,h),Image.Resampling.LANCZOS);grid=source.copy();grid[t:t+h,l:l+w]=np.asarray(native);blend=np.clip(distance_transform_edt(edit)/5,0,1)[...,None];final=np.uint8(np.clip(np.rint(source*(1-blend)+grid*blend),0,255));assert np.array_equal(final[~edit],source[~edit]);d=g/arm/(cid+'__'+str(seed));d.mkdir(parents=True);Image.fromarray(pre).save(d/'pre_sampling.png');Image.fromarray(final).save(d/'composited.png');Image.fromarray(final).crop(rect).save(d/'view.png');np.save(d/'material_confidence.npy',confidence)
                row=dict(case_id=cid,seed=seed,arm=arm,new_raw_calls=0,parent_sha256=sha(parent),paired_raw_sha256=sha(rawpath),output_sha256=sha(d/'composited.png'),detail_luminance_bound=bound,outside_food_and_fresh_cut_parent_exact_before_sampling=True,protected_approximate_source_color_parent_exact_before_sampling=True,outside_geometry_source_exact=True,scope=plan['scope']);write(d/'result.json',row);rows.append(row)
            print('POSITIVE_LIGHT_BAND',cid,seed,flush=True)
        board=Image.new('RGB',(2048,1170),'white');dr=ImageDraw.Draw(board)
        for ri,arm in enumerate(['prior']+plan['arms']):
            paths=[('source',geo/'source.png')]+[(arm+' '+str(s),ROOT/'gate_v46/source_detail'/(cid+'__'+str(s))/'composited.png' if arm=='prior' else g/arm/(cid+'__'+str(s))/'composited.png') for s in [41,163,907]]
            for col,(title,p) in enumerate(paths):
                im=Image.open(p).convert('RGB').crop(rect);im.thumbnail((504,354));board.paste(im,(col*512+(504-im.width)//2,ri*390+28));dr.text((col*512+5,ri*390+5),title,fill='black')
        board.save(g/(cid+'_review.jpg'),quality=95)
    write(g/'execution.json',dict(status='complete_unreviewed',raw_calls=0,compositions=len(rows),rows=rows,unix=time.time()))
    with zipfile.ZipFile(ROOT/'gate_v52_complete.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in g.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(ROOT))
    print('POSITIVE_LIGHT_NEURAL_BAND_COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
