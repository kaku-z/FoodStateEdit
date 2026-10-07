"""Source-only seven-landmark frustum fit on development photos; no measured 3D."""
import json,numpy as np,cv2,trimesh,manifold3d as manifold
from pathlib import Path
from scipy.optimize import least_squares,linear_sum_assignment
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
import source_frustum_geometry as frustum
import structured_geometry_helper as geo
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
CASE=None
TOP={
'new_01_7442':[[130,223],[463,292],[563,187],[302,127]],
'new_02_7496':[[165,221],[364,337],[474,191],[306,92]],
'new_03_7443':[[175,164],[215,301],[513,253],[432,132]],
'new_04_7459':[[222,169],[308,323],[490,273],[369,126]],
'prospective_01_7473':[[178,241],[349,297],[425,157],[258,108]],
'prospective_02_7441':[[237.7143,274.2857],[466.2857,278.8571],[416,187.4286],[272,171.4286]],
'prospective_03_11160':[[83,281],[287,365],[402,97],[229,70]],
'prospective_04_7498':[[165,316],[402,357],[445,270],[247,236]]}
BOTTOM={
'new_01_7442':[[160,354],[432,428],[552,320]],
'new_02_7496':[[168,257],[359,381],[477,214]],
'new_03_7443':[[197,272],[223,398],[493,338]],
'new_04_7459':[[241,286],[341,385],[468,341]],
'prospective_01_7473':[[201,301],[369,319],[426,248]],
'prospective_02_7441':[[265.1429,365.7143],[427.4286,377.1429],[443,282]],
'prospective_03_11160':[[131,352],[296,401],[395,183]],
'prospective_04_7498':[[163,337],[394,390],[444,294]]}
def fit(maps,mask):
 global CASE
 if CASE=='prospective_02_7441':
  result=frustum.fit(maps,mask);result[4]['side_observability_rule']='Only front bottom edge is observable; use the established two-bottom-landmark source fit rather than annotate an occluded side.';return result
 f=json.loads((ROOT/'geometry_spoon_observed_surface_v1'/CASE/'geometry_report.json').read_text())['fit'];R0=np.asarray(f['axes_camera_columns']);lo=np.asarray(f['low']);hi=np.asarray(f['high']);center=(lo+hi)/2;extent=hi-lo;camera_center=center@R0.T
 K=maps['intrinsics'].copy();K[0]*=640;K[1]*=480;quad=np.asarray(TOP[CASE],float)
 for angle in [0,90,180,270]:
  rot=Rotation.from_euler('z',angle,degrees=True).as_matrix();R=R0@rot;e=extent.copy()
  if angle in [90,270]:e[:2]=e[[1,0]]
  c=camera_center@R;vv=frustum.vertices(c-e/2,c+e/2,[1,1]);uv=geo.project(vv[:4]@R.T,K);_,indices=linear_sum_assignment(np.linalg.norm(uv[:,None]-quad[None],axis=2))
  if indices[1]==1:break
 else:raise RuntimeError('Front corner registration failed')
 R0=R;center=c;extent=e;order=indices.copy();uv_target=np.r_[quad[order],np.asarray(BOTTOM[CASE])[order[:3]]];which=np.r_[np.arange(4),np.arange(4,7)]
 # The missing bottom rear corner is not annotated or treated as observed.
 assert np.all(order[:3]<3),order
 p0=np.r_[Rotation.from_matrix(R0).as_rotvec(),center,np.log(extent),np.log([.95,.95])];weights=np.array([1,1,1,.35,.7,1,.7]);depth=float(camera_center[2]);L=max(extent)
 def decode(p):
  R=Rotation.from_rotvec(p[:3]).as_matrix();e=np.exp(p[6:9]);c=p[3:6];return R,c-e/2,c+e/2,np.exp(p[9:11])
 def residual(p):
  R,lo,hi,t=decode(p);v=frustum.vertices(lo,hi,t);cam=v@R.T;pred=geo.project(cam,K)[which]
  return np.r_[((pred-uv_target)*weights[:,None]/640).ravel(),.005*(R-R0).ravel(),.015*(cam[:,2].mean()-depth)/depth,.01*np.log(t)]
 lower=np.r_[p0[:3]-.6,center-.7*L,np.log(extent*.35),np.log([.55,.55])];upper=np.r_[p0[:3]+.6,center+.7*L,np.log(extent*2.8),np.log([1.3,1.3])]
 sol=least_squares(residual,p0,bounds=(lower,upper),max_nfev=1200,ftol=1e-11,gtol=1e-11,xtol=1e-11);R,lo,hi,t=decode(sol.x);frustum.TAPER=t;frustum.PLANES=ConvexHull(frustum.vertices(lo,hi,t)).equations
 uv=geo.project(frustum.vertices(lo,hi,t)@R.T,K);pred=np.zeros_like(mask,np.uint8);cv2.fillConvexPoly(pred,np.round(uv[ConvexHull(uv).vertices]).astype(np.int32),1);body=np.zeros_like(mask,np.uint8);cv2.fillConvexPoly(body,np.round(uv_target[ConvexHull(uv_target).vertices]).astype(np.int32),1)
 report={'prior':'Closed tapered rectangular frustum registered to manually approximated source-only top and visible bottom landmarks','source_only':True,'top_landmarks_canvas':quad.tolist(),'visible_bottom_landmarks_canvas':BOTTOM[CASE],'target_landmarks':uv_target.tolist(),'landmark_order':order.tolist(),'projected_landmarks':uv[which].tolist(),'landmark_reprojection_pixels':np.linalg.norm(uv[which]-uv_target,axis=1).tolist(),'landmark_weights':weights.tolist(),'rear_landmarks_occluded_inferred':True,'source_silhouette_iou':float(np.sum((pred>0)&mask)/np.sum((pred>0)|mask)),'source_body_annotation_iou':float(np.sum((pred>0)&(body>0))/np.sum((pred>0)|(body>0))),'axes_camera_columns':R.tolist(),'low':lo.tolist(),'high':hi.tolist(),'bottom_taper_xy':t.tolist(),'normal_agreement_degrees':[float(np.rad2deg(Rotation.from_matrix(R@R0.T).magnitude()))],'solver_success':bool(sol.success),'evaluations':sol.nfev,'scale':'Arbitrary monocular scale','evaluation':'In-sample source landmark fit, not independent accuracy. Rounded real tofu corners and garnish relief are approximated.'}
 return R,lo,hi,K,report,pred
solid=frustum.solid
texture=frustum.texture
