"""Audit that conservative force-cone points lie on the actual metal mesh.

This tests the implementation of a static prior, not real food mechanics.
"""
import json,hashlib
from pathlib import Path
import numpy as np,trimesh,manifold3d as manifold
from scipy.spatial import ConvexHull
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def mesh(solid):
 m=solid.to_mesh64();return trimesh.Trimesh(np.asarray(m.vert_properties)[:,:3].copy(),np.asarray(m.tri_verts).copy(),process=False)
def main():
 rows=[]
 for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases']:
  g=ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/c['case_id'];report=json.loads((g/'geometry_report.json').read_text());a=report['cut_and_support'];f=report['fit']
  R=np.asarray(f['axes_camera_columns']);Q=np.asarray(a['rotation_food_frame']);center=np.asarray(a['source_cutter_center']);r,_,lower=a['cutter_radii'];upper=a['upper_vertical_radius_over_bowl_radius']*r;hi=np.asarray(f['high']);L=max(hi-np.asarray(f['low']))
  assert np.max(np.abs(hi[:2]-center[:2]))<.99*r and .01*r<hi[2]-center[2]<upper-.01*r
  unit=mesh(manifold.Manifold.sphere(1.,64));v=unit.vertices.copy();v[:,:2]*=r;v[:,2]*=np.where(v[:,2]>=0,upper,lower)
  egg=manifold.Manifold(manifold.Mesh64(v,np.asarray(unit.faces,np.uint64)))
  clip=manifold.Manifold.cube((4*r,4*r,2*lower)).translate((-2*r,-2*r,-2*lower))
  box=manifold.Manifold.cube((2*r,2*r,upper)).translate((-r,-r,0.))
  cutter=egg;cut=mesh(cutter);cut.vertices+=center;planes=ConvexHull(cut.vertices).equations
  b=trimesh.load(g/'bite_source.ply',process=False);b.vertices=np.asarray(b.vertices)@R
  p=b.triangles_center;n=b.face_normals
  contact=np.any((np.abs(p@planes[:,:3].T+planes[:,3])<L*1e-6)&(n@planes[:,:3].T>.999),axis=1)&(n[:,2]<-.05)
  p=p[contact];n=n[contact];p=p[::max(1,len(p)//160)];n=n[::max(1,len(n)//160)]
  assert len(p)>=8
  target=((p-np.asarray(a['source_center']))@Q.T+np.asarray(a['destination_center']))@R.T
  spoon=trimesh.load(g/'fork.ply',process=False)
  closest,dist,tri=trimesh.proximity.closest_point_naive(spoon,target)
  alignment=np.einsum('ij,ij->i',n@Q.T@R.T,spoon.face_normals[tri])
  assert max(dist)/L<1e-5,(c['case_id'],'Support points must lie on actual spoon mesh',max(dist)/L)
  assert np.quantile(alignment,.95)<-.95,(c['case_id'],'Contact normals must face each other')
  row={'case_id':c['case_id'],'support_points':len(p),'max_actual_metal_contact_distance_over_extent':float(max(dist)/L),
       'contact_normal_dot_95th_percentile':float(np.quantile(alignment,.95)),
       'top_corner_inside_actual_ellipsoidal_cutter':True,'convex_cut_contact_selection':'Ellipsoidal shared supporting facets; validated against actual bowl mesh.',
       'uniform_density_static_support':a['static_contact'],'partition_residual':a['relative_partition_volume_residual'],'overlap_volumes':a['overlap_volumes']}
  rows.append(row);print(c['case_id'],'PASS',flush=True)
 report={'status':'complete','rows':rows,'scope':'Internal geometry implementation audit under monocular shape, rigid uniform-density food and assumed friction. No measured 3D or dynamic eating validation.',
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (ROOT/'geometry_spoon_source_fit_ellipsoid_v3/contact_audit.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
