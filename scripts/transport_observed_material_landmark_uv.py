"""Transport only the observed source top surface through the actual rigid 3-D food pose."""
import argparse,json,hashlib
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gate',required=True);args=ap.parse_args();g=ROOT/args.gate;cfg=json.loads((g/'config.json').read_text());out=g/'observed_material_landmark_uv_v1';out.mkdir(exist_ok=False);rows=[]
 for j in cfg['jobs']:
  if args.gate=='gate_v20' and j['method']!='silken_food_only_canny':continue
  cid=j['case_id'];geometry=ROOT/'geometry_spoon_open_corner_v1'/cid;channel=ROOT/'spoon_open_corner_channels_v1'/cid
  p=g/'collection_complete'/j['id']/'composited.png';candidate=np.asarray(Image.open(p).convert('RGB'),float);source=np.asarray(Image.open(geometry/'source.png').convert('RGB'),float);source_mask=np.asarray(Image.open(geometry/'food_mask.png'))>0
  maps=Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929/geometry')/cid/'maps.npz' if cid.startswith('new_') else ROOT/'geometry_v3'/cid/'maps.npz';K=np.load(maps)['intrinsics'].copy();K[0]*=640;K[1]*=480
  report=json.loads((geometry/'geometry_report.json').read_text());fit=report['fit'];action=report['cut_and_support'];R=np.asarray(fit['axes_camera_columns']);Q=np.asarray(action['rotation_food_frame']);origin=np.asarray(action['source_center']);destination=np.asarray(action['destination_center']);hi=np.asarray(fit['high']);L=max(hi-np.asarray(fit['low']))
  data=np.load(channel/'geometry_channels.npz');bite=data['labels']==2;yy,xx=np.where(bite);rays=np.c_[xx+.5,yy+.5,np.ones(len(xx))]@np.linalg.inv(K).T;cam=rays*data['scene_depth'][bite][:,None];back=(cam@R-destination)@Q+origin;normals=data['scene_normals'][bite]@R@Q
  top=(np.abs(back[:,2]-hi[2])<L*1e-4)&(normals[:,2]>.99)
  original_camera=back@R.T;projected=original_camera@K.T;uv=projected[:,:2]/projected[:,2:]
  correspondence=json.loads((ROOT/'source_top_landmarks_v1'/(cid+'.json')).read_text());H=np.asarray(correspondence['old_to_source_top_homography']);corrected=np.c_[uv,np.ones(len(uv))]@H.T;uv=corrected[:,:2]/corrected[:,2:];observed=(uv[:,0]>=0)&(uv[:,0]<639)&(uv[:,1]>=0)&(uv[:,1]<479)
  mx=uv[:,0].astype(np.float32).reshape(-1,1);my=uv[:,1].astype(np.float32).reshape(-1,1)
  sampled=cv2.remap(source.astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).reshape(-1,3)
  seen=cv2.remap(np.uint8(source_mask),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT).reshape(-1)>0
  selected=top&observed&seen
  # Separate observed local appearance from inferred top-plane illumination.
  # Robust scalar shading rejects the model's generic colored/brown marker.
  calibration=json.loads((geometry/'appearance_calibration.json').read_text());coef=np.asarray(calibration['coefficient_rgb']);before=np.maximum(np.r_[1.,R[:,2]]@coef,.05)*255
  ref=json.loads((ROOT/'gate_v13_run03'/cid/'reference_selection.json').read_text());sx0,sy0,sx1,sy1=ref['material_box_canvas'];white=np.median(source[sy0:sy1,sx0:sx1].reshape(-1,3),axis=0);white_chroma=white/white.sum()
  target_rgb=candidate[yy,xx];chroma=target_rgb/np.maximum(target_rgb.sum(1,keepdims=True),1);plain=top&(np.linalg.norm(chroma-white_chroma,axis=1)<.045)
  luminance=target_rgb@np.array([.2126,.7152,.0722]);median_top=float(np.median(luminance[top]));plain&=(luminance>median_top*.85)&(luminance<min(254.,median_top*1.15))
  u=(xx-xx[top].mean())/max(1,np.ptp(xx[top]));v=(yy-yy[top].mean())/max(1,np.ptp(yy[top]));X=np.c_[np.ones(len(xx)),u,v,u*u,u*v,v*v]
  if plain.sum()<20:
   fitted=np.full(len(xx),float(np.quantile(luminance[top],.7)));light_fit='Global rule: 70th percentile top luminance when fewer than 20 unseasoned top pixels.'
  else:
   A=X[plain];b=luminance[plain];weights=np.ones(len(b));penalty=np.diag([.001,2.,2.,4.,4.,4.])
   for _ in range(8):
    beta=np.linalg.solve(A.T@(weights[:,None]*A)+penalty,A.T@(weights*b));residual=A@beta-b;scale=max(1.,1.4826*np.median(np.abs(residual-np.median(residual))));weights=np.minimum(1.,1.5*scale/np.maximum(np.abs(residual),1e-6))
   fitted=np.clip(X@beta,median_top*.8,min(255.,median_top*1.2));light_fit='Fixed quadratic IRLS on unseasoned target top pixels; scalar inferred lighting, not calibrated relighting.'
  target_chroma=np.median(chroma[plain],axis=0) if plain.sum()>=20 else white_chroma;target_chroma/=target_chroma.sum()
  target_white=fitted[:,None]*target_chroma[None,:]/max(float(target_chroma@np.array([.2126,.7152,.0722])),1e-6)
  ratio=np.median(target_white[top],axis=0)/before;sampled=np.clip(sampled/before[None,:]*target_white,0,255)
  mask=np.zeros((480,640),bool);mask[yy[selected],xx[selected]]=True;transport=candidate.copy();transport[yy[selected],xx[selected]]=sampled[selected];alpha=(mask.astype(float)*np.clip(distance_transform_edt(bite)/.75,0,1))[...,None]
  result=np.uint8(np.clip(np.rint(candidate*(1-alpha)+transport*alpha),0,255));assert np.array_equal(result[~mask],candidate.astype(np.uint8)[~mask]);d=out/j['id'];d.mkdir();Image.fromarray(result).save(d/'composited.png');Image.fromarray(np.uint8(mask)*255).save(d/'transport_mask.png')
  case=next(c for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases'] if c['case_id']==cid);l,t=case['preprocessing']['pad_left_top'];w,h=case['preprocessing']['resized'];rect=(l,t,l+w,t+h);Image.fromarray(result).crop(rect).save(d/'view.png')
  board=Image.new('RGB',(1280,h+28),'white');draw=ImageDraw.Draw(board)
  for i,(title,im) in enumerate([('food-only neural composite',Image.fromarray(np.uint8(candidate))),('observed source top transported in 3D',Image.fromarray(result))]):board.paste(im.crop(rect).resize((640,h),Image.Resampling.LANCZOS),(i*640,28));draw.text((i*640+5,6),title,fill='black')
  board.save(d/'comparison.jpg',quality=95);row={'id':j['id'],'raw_generation':False,'parent_composite_sha256':sha(p),'transported_observed_top_pixels':int(mask.sum()),'expected_top_pixels':int(top.sum()),'eligible_source_observation_fraction':float(selected.sum()/max(1,top.sum())),'inferred_top_rgb_tone_ratio':ratio.tolist(),'unseasoned_neural_top_fit_pixels':int(plain.sum()),'lighting_fit':light_fit,'source_correspondence':correspondence,'boundary_rule':'No inward feather at internal top-to-side boundary; only food external silhouette antialiasing.','outside_transport_mask_parent_exact':True,'scope':'Source-observed flat top appearance via approximate manual source-only quadrilateral correspondence; divided by a source-normal RGB lighting prior, multiplied by robust inferred neural top illumination. Garnish height and hidden material remain inferred; relighting is not calibrated.'};(d/'result.json').write_text(json.dumps(row,indent=2));rows.append(row)
 (out/'manifest.json').write_text(json.dumps({'status':'complete_unreviewed','rows':rows,'all_cells_same_rule':True,'additional_raw_calls':0,'script_sha256':sha(Path(__file__))},indent=2));print('TRANSPORTED',len(rows))
if __name__=='__main__':main()


