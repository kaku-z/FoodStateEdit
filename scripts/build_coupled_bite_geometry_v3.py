"""Source-content-aware re-planning, preserving all version-one results."""
import json
from pathlib import Path
import time
import traceback
import numpy as np
from PIL import Image
import build_coupled_bite_geometry as base
import coupled_bite_geometry_helper_v3 as geo


def main():
    root=base.ROOT
    deadline=time.time()+3600
    while True:
        mp=root/'geometry/manifest.json'
        if mp.exists():
            try:
                old=json.loads(mp.read_text())
                if old['status']=='complete':break
            except OSError as e:
                if e.errno!=116:raise
        if time.time()>deadline:raise TimeoutError('Original geometry did not finish')
        time.sleep(10)
    base.geo=geo
    out=root/'geometry_v3';out.mkdir(exist_ok=False)
    inputs=json.loads((root/'inputs/manifest.json').read_text())
    manifest={'status':'processing','cases':[],'scope':'Updated pose uses source bounds and source occupancy only; no generated outputs are used.',
        'script_sha256':base.sha(Path(__file__)),'helper_sha256':base.sha(Path(geo.__file__)),
        'prior_source_manifest_sha256':base.sha(root/'inputs/manifest.json')}
    for case in inputs['cases']:
        folder=out/case['case_id'];folder.mkdir()
        try:
            maps=dict(np.load(root/'geometry'/case['case_id']/'maps.npz'))
            prep=case['preprocessing'];geo.VALID_IMAGE_RECT=tuple(prep['pad_left_top']+prep['resized'])
            geo.FOOD_MASK_FOR_PLANNING=np.asarray(Image.open(root/'inputs'/case['case_id']/'food_mask.png'))>0
            row=base.geometry(case,folder,maps)
            # Finite-only visualization, still not an input to the RGB generator.
            depth=maps['depth'];valid=maps['mask']&np.isfinite(depth)
            lo,hi=np.quantile(depth[valid],[.01,.99]);v=np.zeros_like(depth)
            v[valid]=np.clip((hi-depth[valid])/max(hi-lo,1e-8),0,1)
            Image.fromarray(np.uint8(v*255)).save(folder/'source_depth_finite.png')
        except Exception as exc:
            (folder/'FAILED.txt').write_text(traceback.format_exc())
            row={'case_id':case['case_id'],'status':'geometry_failed','error':repr(exc)}
        manifest['cases'].append(row)
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('GEOMETRY_V3',case['case_id'],row['status'],flush=True)
    manifest['status']='complete'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
