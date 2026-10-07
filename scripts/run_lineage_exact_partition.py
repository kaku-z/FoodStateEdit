"""Correct centroid-only volume allocation using actual per-region mesh cuts.

This supplementary state computation never overwrites frozen v1 budgets or RGB.
It is exact relative to the supplied proxy meshes and Boolean kernel, not real
food mass or hidden 3D truth. Source cells are coarse regions, not point particles.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')

def main(root):
 sys.path.insert(0,str(root))
 from foodstateedit.material_lineage.exact_partition import partition_convex_mesh
 cfg=json.loads((root/'config.json').read_text());out=root/'mass_partition_v3';out.mkdir(exist_ok=False)
 write(out/'recipe.json',{'status':'frozen_before_computation','cases':cfg['cases'],'source_config_sha256':sha(root/'config.json'),
  'script_sha256':sha(Path(__file__)),'exact_partition_code_sha256':sha(root/'foodstateedit/material_lineage/exact_partition.py'),
  'refinement':2,'density':1,'relative_volume_tolerance':1e-6,
  'scope':'192 persistent coarse material regions per supplied full convex proxy; actual Boolean intersection fractions. No real physical mass or exact particle position claim.',
  'rgb_and_v1_budget_unchanged':True,'created_unix':time.time()})
 rows=[]
 for cid in cfg['cases']:
  started=time.time();info=json.loads((root/cid/'source_inputs.json').read_text());R=np.asarray(info['R_food_to_camera'])
  inputs={Path(x['path']).name:Path(x['path']) for x in info['inputs']}
  full=trimesh.load(inputs['full.ply'],process=False);bite=trimesh.load(inputs['bite_source.ply'],process=False)
  vertices=np.asarray(full.vertices)@R;hull=ConvexHull(vertices);fullmesh=trimesh.Trimesh(vertices=vertices,faces=hull.simplices,process=True);fullmesh.fix_normals()
  result=partition_convex_mesh(fullmesh.vertices,fullmesh.faces,np.asarray(bite.vertices)@R,bite.faces,cid+':exact-material-regions',refinement=2)
  d=out/cid;d.mkdir();result.save_npz(d/'allocation.npz')
  row={'case_id':cid,'audit':result.audit,'input_mesh_sha256':{n:sha(inputs[n]) for n in ['full.ply','bite_source.ply']},'seconds':time.time()-started}
  write(d/'audit.json',row);rows.append(row);write(out/'manifest.json',{'status':'running','cases':rows});print('EXACT_PARTITION',cid,result.audit['relative_bite_volume_error_over_full'],flush=True)
 write(out/'manifest.json',{'status':'complete_verified_proxy_volume','cases':rows,'max_relative_bite_volume_error_over_full':max(r['audit']['relative_bite_volume_error_over_full'] for r in rows),
  'scope':'Actual supplied mesh volumes and per-region budgets only; photos, geometry/contact accuracy, density and real mass are not established.'})

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/host/space0/guo-z/tf-ufi/material_lineage_pilot_20261003'));a=ap.parse_args();main(a.root)
