"""Small deterministic sensitivity audit; no claim of held-out 3-D accuracy."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial.transform import Rotation
import trimesh

from repair_food3d_first_bite import fit_block, plan_bite


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--baseline-root',type=Path,required=True);args=p.parse_args()
    root=args.root;out=root/'results/sensitivity_v1';out.mkdir(exist_ok=False)
    (out/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    maps=dict(np.load(root/'results/depth_v1/maps.npz'))
    mask=np.asarray(Image.open(root/'inputs/food_mask.png'))>0
    rows=[]; reference=None
    for case in ['nominal','units_x0.5','units_x2','mask_erode_3px','mask_dilate_3px','focal_x0.9','focal_x1.1','normal_pitch_plus3deg']:
        start=time.monotonic();m={k:v.copy() for k,v in maps.items()};target=mask.copy()
        if case.startswith('units_'):
            factor=float(case.split('x')[1]);m['points']*=factor;m['depth']*=factor
        if case=='mask_erode_3px':target=binary_erosion(mask,iterations=3)
        if case=='mask_dilate_3px':target=binary_dilation(mask,iterations=3)
        if case.startswith('focal_'):
            factor=float(case.split('x')[1]);m['intrinsics'][0,0]*=factor;m['intrinsics'][1,1]*=factor
            m['points'][...,:2]/=factor
            m['normal'][...,:2]*=factor;m['normal']/=np.linalg.norm(m['normal'],axis=-1,keepdims=True)
        if case=='normal_pitch_plus3deg':m['normal']=m['normal']@Rotation.from_euler('x',3,degrees=True).as_matrix().T
        try:
            R,lo,hi,K,fit,_=fit_block(m,target)
            _,_,_,_,_,_,cut=plan_bite(R,lo,hi,K)
            if reference is None:reference=(R,hi-lo)
            delta=np.rad2deg(Rotation.from_matrix(R@reference[0].T).magnitude())
            relative_dims=(hi-lo)/max(hi-lo);refdims=reference[1]/max(reference[1])
            dimension_delta=float(max(np.abs(relative_dims-refdims)))
            row={'case':case,'status':'passed','seconds':time.monotonic()-start,
                'axes_change_deg':float(delta),'relative_dimension_max_change':dimension_delta,
                'silhouette_iou_to_perturbed_mask':fit['source_silhouette_iou'],
                'plane_error_max_over_extent':max(fit['plane_depth_residual_median_over_food_extent']),
                'cut':cut}
            if case.startswith('units_'):
                assert delta<.1 and dimension_delta<.005,'Geometry depends on arbitrary unit choice'
        except Exception as exc:
            row={'case':case,'status':'failed','error':repr(exc)}
        rows.append(row);print(json.dumps(row),flush=True)
        (out/'sensitivity.json').write_text(json.dumps({'cases':rows,'scope':'Development-only perturbation checks; not a generalization test or physical ground truth.'},indent=2)+'\n')
    baseline=[]
    for seed in [281,913]:
        path=args.baseline_root/f'results/geometry_review_v2/seed_{seed}_bite_at_source.ply'
        mesh=trimesh.load(path,force='mesh',process=False)
        extents=mesh.convex_hull.bounding_box_oriented.primitive.extents
        baseline.append({'seed':seed,'oriented_bounding_box_dimensions':extents.tolist(),
            'bite_aspect_ratio':float(max(extents)/min(extents)),
            'bite_volume_over_oriented_box_volume':float(abs(mesh.volume)/np.prod(extents)),
            'utensil_support':'Absent in original baseline'})
    result={'cases':rows,'baseline_bite_diagnostics':baseline,
        'scope':'Same-image development sensitivity, not a new dataset, model benchmark or blinded evaluation.',
        'acceptance':'All cases must complete partition, closed-solid and final static-support checks; arbitrary-unit cases must preserve geometry.',
        'limitations':['The cuboid shape and cube bite are explicit priors. Good compactness is imposed, not a learned ability.',
                       'Normal/focal perturbations are sensitivity probes, not statistically estimated uncertainty intervals.']}
    (out/'sensitivity.json').write_text(json.dumps(result,indent=2)+'\n')
    assert all(row['status']=='passed' for row in rows)
    print('AUDIT_PASSED',len(rows),flush=True)


if __name__=='__main__':main()
