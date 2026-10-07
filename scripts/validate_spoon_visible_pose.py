"""Audit the actual selected pose, including yaw and camera centre, independently of planner approximations."""
import json,hashlib
from pathlib import Path
import numpy as np
import trimesh
import manifold3d as manifold
import supported_spoon_visibility_geometry as planner
ROOT=planner.ROOT
rows=[]
for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases']:
 cid=c['case_id'];g=ROOT/'geometry_spoon_visibility_v1'/cid
 j=json.loads((g/'geometry_report.json').read_text());p=j['cut_and_support'];R=np.asarray(j['fit']['axes_camera_columns']);lo=np.asarray(j['fit']['low']);hi=np.asarray(j['fit']['high'])
 Q=np.asarray(p['rotation_food_frame']);center=np.asarray(p['source_cutter_center']);r,_,lower=p['cutter_radii'];upper=p['upper_vertical_radius_over_bowl_radius']*r
 egg=planner.geo.mesh_of(manifold.Manifold.sphere(1.,64));v=egg.vertices.copy();v[:,:2]*=r;v[:,2]*=np.where(v[:,2]>=0,upper,lower)
 cutter=manifold.Manifold(manifold.Mesh64(v,np.asarray(egg.faces,np.uint64))).translate(tuple(center))
 source=trimesh.load(g/'bite_source.ply',process=False);bite=manifold.Manifold(manifold.Mesh64(np.asarray(source.vertices)@R,np.asarray(source.faces,np.uint64)))
 force=planner.equilibrium(bite,cutter,max(hi-lo),orientation=Q)
 up=Q[:,2]@R.T;dest=np.asarray(p['destination_center'])@R.T;view=-dest/np.linalg.norm(dest)
 actual=float(up@view)
 row={'case_id':cid,'actual_joint_yaw_tilt_static_contact':force,'actual_food_centroid_spoon_top_view_cosine':actual,'planned_cosine':p['planned_spoon_top_view_cosine'],'view_threshold_prior':.54,'passes_view_threshold':actual>=.54,'source_bite_ply_sha256':hashlib.sha256((g/'bite_source.ply').read_bytes()).hexdigest()}
 rows.append(row);print(cid,actual,force['coulomb_friction_coefficient_assumed'],flush=True)
(ROOT/'visibility_pose_independent_audit.json').write_text(json.dumps({'status':'complete','rows':rows,'all_view_pass':all(r['passes_view_threshold'] for r in rows),'scope':'Actual selected rigid pose and static assumed-friction cone, not measured physical truth.','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
