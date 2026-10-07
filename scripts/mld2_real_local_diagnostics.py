"""Local-scale diagnostic for the second, explicitly exploratory real round."""
import argparse
import json
from pathlib import Path

import numpy as np


def footprint_at_depth(points):
    flat=points[:,:2]/points[:,2:3]*np.median(points[:,2])
    centered=flat-np.median(flat,axis=0)
    _,_,axes=np.linalg.svd(centered,full_matrices=False)
    coordinates=centered@axes.T
    widths=np.quantile(coordinates,.95,axis=0)-np.quantile(coordinates,.05,axis=0)
    return float(widths.max()),float(widths.min())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    rows=[]
    for folder in sorted(args.source.glob('real_*')):
        if not folder.is_dir():continue
        metadata=json.loads((folder/'geometry_raw.json').read_text())
        closure=np.load(folder/'closure_raw.npz')
        local=closure['source_local_xyz'];camera=closure['source_camera_xyz'];extent=float(closure['extent'])
        floor=closure['floor_local'];plane=np.asarray(metadata['plane_center'])@closure['world_axes']
        major,minor=footprint_at_depth(camera)
        top_height=local[:,2]-plane[2];thickness=local[:,2]-floor
        relief_span=float(np.quantile(local[:,2],.95)-np.quantile(local[:,2],.05))
        camera_depth_span=float(np.quantile(camera[:,2],.95)-np.quantile(camera[:,2],.05))
        row=dict(case_id=folder.name,source_cut_center=metadata['source_bite_center'],source_cut_radius=metadata['source_bite_radius'],
                 food_global_extent=extent,patch_camera_footprint_major=major,patch_camera_footprint_minor=minor,
                 patch_top_height_above_plate_quantiles=np.quantile(top_height,[0,.05,.5,.95,1]).tolist(),
                 patch_top_height_over_minor_quantiles=np.quantile(top_height/minor,[0,.05,.5,.95,1]).tolist(),
                 patch_top_normal_relief_p95_p05=relief_span,patch_camera_depth_relief_p95_p05=camera_depth_span,
                 patch_top_normal_relief_over_minor=relief_span/minor,patch_camera_depth_relief_over_minor=camera_depth_span/minor,
                 closure_thickness_quantiles=np.quantile(thickness,[0,.05,.5,.95,1]).tolist(),
                 closure_thickness_over_minor_quantiles=np.quantile(thickness/minor,[0,.05,.5,.95,1]).tolist(),
                 global_minimum_closure_over_minor=.03*extent/minor,
                 global_uniform_closure_over_minor=.055*extent/minor,
                 whole_food_extent_over_patch_minor=extent/minor,
                 interpretation='Inferred single-image camera scale, not measured geometry; source UV footprint at median depth isolates ray-depth relief from apparent source extent.')
        rows.append(row)
    report=dict(round='Second exploratory round; first 16 outcomes have been inspected, no blind holdout claim.',source_first_round=str(args.source),cases=rows,
                proposed_rule='Bound inferred underside thickness and source ray relief by local transported patch camera footprint minor. Preserve source UV, cut, translation and shared spoon. No per-food or per-case branch.')
    (args.output/'local_scale_diagnostics.json').write_text(json.dumps(report,indent=2))
    for row in rows:
        print(row['case_id'],json.dumps({k:row[k] for k in ['food_global_extent','patch_camera_footprint_minor','patch_top_normal_relief_over_minor','patch_camera_depth_relief_over_minor','closure_thickness_over_minor_quantiles','global_minimum_closure_over_minor','whole_food_extent_over_patch_minor']}))


if __name__=='__main__':main()
