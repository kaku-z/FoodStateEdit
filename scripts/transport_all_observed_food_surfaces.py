"""Transfer all source-visible inherited surfaces, using an observation-aware pose.

Source-only camera/geometry/appearance correspondences remain approximate. This
does not reconstruct hidden texture, garnish relief or calibrated illumination.
"""
import json,hashlib,zipfile,time
from pathlib import Path
import cv2,numpy as np,trimesh
from PIL import Image,ImageDraw
from scipy.spatial import ConvexHull
from scipy.ndimage import distance_transform_edt,binary_erosion
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def build_fields():
    out=ROOT/'observed_food_surface_fields_v1';out.mkdir(exist_ok=False)
    (out/'executed_script.py').write_bytes(Path(__file__).read_bytes());rows=[]
    for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases']:
        cid=c['case_id'];geo=ROOT/'geometry_spoon_observed_surface_v1'/cid
        data=np.load(ROOT/'spoon_observed_surface_channels_v1'/cid/'geometry_channels.npz')
        g=json.loads((geo/'geometry_report.json').read_text());R=np.asarray(g['fit']['axes_camera_columns']);Q=np.asarray(g['cut_and_support']['rotation_food_frame']);origin=np.asarray(g['cut_and_support']['source_center']);destination=np.asarray(g['cut_and_support']['destination_center']);hi=np.asarray(g['fit']['high']);L=max(hi-np.asarray(g['fit']['low']))
        maps=(Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929/geometry')/cid/'maps.npz') if cid.startswith('new_') else ROOT/'geometry_v3'/cid/'maps.npz';K=np.load(maps)['intrinsics'].copy();K[0]*=640;K[1]*=480
        food=data['labels']==2;yy,xx=np.where(food);rays=np.c_[xx+.5,yy+.5,np.ones(len(xx))]@np.linalg.inv(K).T;cam=rays*data['scene_depth'][food][:,None];back=(cam@R-destination)@Q+origin;normal=data['scene_normals'][food]@R@Q
        full=trimesh.load(geo/'full.ply',process=False);planes=ConvexHull(np.asarray(full.vertices)@R).equations
        near=np.abs(back@planes[:,:3].T+planes[:,3])<L*1e-4;aligned=normal@planes[:,:3].T>.999;matches=near&aligned;inherited=matches.any(axis=1);face=np.argmax(matches,axis=1)
        original=back@R.T;original_normal=normal@R.T;visible=np.einsum('ij,ij->i',original_normal,-original)>0
        top=(np.abs(back[:,2]-hi[2])<L*1e-4)&(normal[:,2]>.99)
        projected=original@K.T;uv=projected[:,:2]/projected[:,2:]
        correspondence=json.loads((ROOT/'source_top_landmarks_v1'/(cid+'.json')).read_text());H=np.asarray(correspondence['old_to_source_top_homography']);corrected=np.c_[uv[top],np.ones(top.sum())]@H.T;uv[top]=corrected[:,:2]/corrected[:,2:]
        in_frame=(uv[:,0]>=0)&(uv[:,0]<639)&(uv[:,1]>=0)&(uv[:,1]<479)
        source=np.asarray(Image.open(geo/'source.png').convert('RGB'),float);source_mask=binary_erosion(np.asarray(Image.open(geo/'food_mask.png'))>0,iterations=1)
        mx=uv[:,0].astype(np.float32).reshape(-1,1);my=uv[:,1].astype(np.float32).reshape(-1,1)
        sampled=cv2.remap(source.astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).reshape(-1,3)
        seen=cv2.remap(np.uint8(source_mask),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT).reshape(-1)>0
        valid=inherited&visible&in_frame&seen
        coef=np.asarray(json.loads((geo/'appearance_calibration.json').read_text())['coefficient_rgb'])
        before=np.maximum(np.c_[np.ones(len(normal)),original_normal]@coef,.05)*255
        d=out/cid;d.mkdir();np.savez_compressed(d/'field.npz',xx=xx,yy=yy,top=top,valid=valid,face=face,sampled=sampled,before=before,food=food)
        mask=np.zeros(food.shape,np.uint8);mask[yy[valid],xx[valid]]=255;Image.fromarray(mask).save(d/'observed_surface_mask.png')
        top_mask=np.zeros(food.shape,np.uint8);top_mask[yy[valid&top],xx[valid&top]]=255;Image.fromarray(top_mask).save(d/'observed_top_mask.png')
        proxy=source.copy();proxy[yy[valid],xx[valid]]=sampled[valid];Image.fromarray(np.uint8(proxy)).save(d/'source_transport_diagnostic.png')
        row={'case_id':cid,'visible_food_pixels':int(food.sum()),'transportable_observed_surface_pixels':int(valid.sum()),'transportable_observed_top_pixels':int((valid&top).sum()),'transportable_observed_side_pixels':int((valid&~top).sum()),'observed_visible_food_fraction':float(valid.mean()),'source_only_geometry_report_sha256':sha(geo/'geometry_report.json'),'pose_yaw_degrees':g['cut_and_support']['yaw_degrees'],'top_correspondence':correspondence,'side_correspondence':'Original monocular camera/geometry projection; approximate, no manually fitted side landmarks','scope':'Source-visible inherited surfaces under a static supported pose, with source foreground membership. Hidden surfaces and garnish height are not recovered.'};(d/'audit.json').write_text(json.dumps(row,indent=2));rows.append(row);print('FIELD',cid,round(valid.mean(),4),int((valid&~top).sum()),flush=True)
    (out/'manifest.json').write_text(json.dumps({'status':'complete','fields_frozen_before_transport':True,'cases':rows,'script_sha256':sha(Path(__file__))},indent=2))

def transport():
    gate=ROOT/'gate_v35';cfg=json.loads((gate/'config.json').read_text());cases={c['case_id']:c for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases']};out=gate/'all_observed_surface_v1';out.mkdir(exist_ok=False);rows=[]
    (out/'frozen_plan.json').write_text(json.dumps({'created_unix':time.time(),'status':'frozen_before_transport','additional_neural_calls':0,'rules':['source_rgb_unrelit','source_rgb_scalar_neural_light'],'all_cells_same_rules':True,'limits':'Approximate monocular camera and flat surface appearance; indirect lighting and garnish geometry remain unknown.'},indent=2))
    for j in cfg['jobs']:
        cid=j['case_id'];fields=np.load(ROOT/'observed_food_surface_fields_v1'/cid/'field.npz');xx=fields['xx'];yy=fields['yy'];valid=fields['valid'];face=fields['face'];top=fields['top'];sampled=fields['sampled'];before=fields['before'];food=fields['food']
        geo=ROOT/'geometry_spoon_observed_surface_v1'/cid;source=np.asarray(Image.open(geo/'source.png').convert('RGB'));parent=gate/'collection_complete'/j['id']/'composited.png';candidate=np.asarray(Image.open(parent).convert('RGB'),float)
        reference=json.loads((ROOT/'gate_v13_run03'/cid/'reference_selection.json').read_text());a,b,c,d=reference['material_box_canvas'];white=np.median(source[b:d,a:c].reshape(-1,3),axis=0);white_chroma=white/white.sum();target=candidate[yy,xx];luminance=target@np.array([.2126,.7152,.0722]);chroma=target/np.maximum(target.sum(1,keepdims=True),1)
        gain=np.ones(len(xx));fit_rows=[]
        for plane in np.unique(face[valid]):
            region=valid&(face==plane);plain=region&(np.linalg.norm(chroma-white_chroma,axis=1)<.045);median=float(np.median(luminance[region]));plain&=(luminance>.85*median)&(luminance<min(254.,1.15*median))
            if plain.sum()<20:
                predicted=np.full(len(xx),float(np.quantile(luminance[region],.7)));mode='70th percentile fallback'
            else:
                u=(xx-xx[region].mean())/max(1,np.ptp(xx[region]));v=(yy-yy[region].mean())/max(1,np.ptp(yy[region]));X=np.c_[np.ones(len(xx)),u,v,u*u,u*v,v*v];A=X[plain];y=luminance[plain];weights=np.ones(len(y));penalty=np.diag([.001,2,2,4,4,4])
                for _ in range(8):
                    beta=np.linalg.solve(A.T@(weights[:,None]*A)+penalty,A.T@(weights*y));res=A@beta-y;scale=max(1.,1.4826*np.median(np.abs(res-np.median(res))));weights=np.minimum(1.,1.5*scale/np.maximum(np.abs(res),1e-6))
                predicted=np.clip(X@beta,.8*median,min(255.,1.2*median));mode='fixed quadratic IRLS'
            original_luminance=before@np.array([.2126,.7152,.0722]);gain[region]=np.clip(predicted[region]/np.maximum(original_luminance[region],1),.75,1.25)
            fit_rows.append({'plane':int(plane),'plain_pixels':int(plain.sum()),'mode':mode,'median_gain':float(np.median(gain[region]))})
        c=cases[cid];l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h);edit=np.asarray(Image.open(geo/'edit_mask.png'))>0
        for rule in ['source_rgb_unrelit','source_rgb_scalar_neural_light']:
            colors=sampled if rule=='source_rgb_unrelit' else sampled*gain[:,None]
            mask=np.zeros(food.shape,bool);mask[yy[valid],xx[valid]]=True;modified=candidate.copy();modified[yy[valid],xx[valid]]=colors[valid];alpha=(mask*np.clip(distance_transform_edt(food)/.75,0,1))[...,None]
            projected=np.uint8(np.clip(np.rint(candidate*(1-alpha)+modified*alpha),0,255));assert np.array_equal(projected[~mask],candidate.astype(np.uint8)[~mask])
            native=Image.fromarray(projected).crop(rect).resize(tuple(c['source_size']),Image.Resampling.LANCZOS).resize((w,h),Image.Resampling.LANCZOS);grid=source.copy();grid[t:t+h,l:l+w]=np.asarray(native);blend=np.clip(distance_transform_edt(edit)/5,0,1)[...,None];final=np.uint8(np.clip(np.rint(source*(1-blend)+grid*blend),0,255));assert np.array_equal(final[~edit],source[~edit])
            dest=out/rule/j['id'];dest.mkdir(parents=True);Image.fromarray(projected).save(dest/'transported.png');Image.fromarray(final).save(dest/'composited.png');Image.fromarray(final).crop(rect).save(dest/'view.png')
            row={'id':j['id'],'rule':rule,'seed':j['seed'],'raw_generation':False,'additional_neural_calls':0,'parent_sha256':sha(parent),'compositor_outside_observed_surface_parent_exact':True,'sampling_outside_geometry_source_exact':True,'observed_food_fraction':float(valid.mean()),'observed_top_pixels':int((valid&top).sum()),'observed_side_pixels':int((valid&~top).sum()),'lighting_prior_fits':fit_rows if rule!='source_rgb_unrelit' else [],'scope':'Observed source pixels on top and side surfaces. Flat garnish geometry and hidden material remain inferred; scalar neural illumination is an uncalibrated prior.'};(dest/'result.json').write_text(json.dumps(row,indent=2));rows.append(row)
    for cid,c in cases.items():
        l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h)
        for rule in ['source_rgb_unrelit','source_rgb_scalar_neural_light']:
            paths=[('source',ROOT/'geometry_spoon_observed_surface_v1'/cid/'source.png')]+[(rule+' seed'+str(seed),out/rule/(cid+'__observed_surface_food_intrinsic__'+str(seed))/'composited.png') for seed in [41,163,907]];board=Image.new('RGB',(2560,h+28),'white');draw=ImageDraw.Draw(board)
            for i,(title,p) in enumerate(paths):board.paste(Image.open(p).convert('RGB').crop(rect).resize((640,h),Image.Resampling.LANCZOS),(640*i,28));draw.text((640*i+5,6),title,fill='black')
            board.save(out/(cid+'_'+rule+'_three_seed.jpg'),quality=95)
    (out/'manifest.json').write_text(json.dumps({'status':'complete_unreviewed','derived_composites':len(rows),'rows':rows,'all_cases_same_rules':True,'script_sha256':sha(Path(__file__))},indent=2))
    with zipfile.ZipFile(ROOT/'all_observed_surface_transport_v1.zip','w',zipfile.ZIP_DEFLATED) as z:
        for folder in [out,ROOT/'observed_food_surface_fields_v1']:
            for p in folder.rglob('*'):
                if p.is_file():z.write(p,p.relative_to(ROOT))
    print('TRANSPORTED',len(rows),flush=True)
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--fields-only',action='store_true');a=ap.parse_args()
    if not (ROOT/'observed_food_surface_fields_v1').exists():build_fields()
    if not a.fields_only:transport()
