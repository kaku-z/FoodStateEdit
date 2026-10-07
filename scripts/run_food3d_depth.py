"""Estimate visible geometry from the untouched real photo; no image generation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    root=args.root
    sys.path[:0]=[str(root/'moge'),str(root/'utils3d_moge')]
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['HF_HOME']=str(root/'hf_cache')
    import torch
    import numpy as np
    from PIL import Image
    from moge.model.v2 import MoGeModel
    upstream=json.loads((root/'upstream_manifest.json').read_text())
    weight=root/'models/model.pt'
    h=hashlib.sha256()
    with weight.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    assert h.hexdigest()==upstream['model']['sha256']
    source=root/'inputs/source.jpg'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='cd92b2477620cba4688a4cd73bd32d155a357ca77e32effe138bb1b8a665a7e2'
    out=root/'results/depth_v1';out.mkdir(exist_ok=False)
    t=time.monotonic()
    checkpoint=torch.load(weight,map_location='cpu',weights_only=True)
    model=MoGeModel(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model'],strict=True)
    del checkpoint
    model=model.eval().to('cuda')
    arr=np.asarray(Image.open(source).convert('RGB'))
    input_image=torch.as_tensor(arr.copy(),device='cuda',dtype=torch.float32).permute(2,0,1)/255
    torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    with torch.inference_mode():result=model.infer(input_image,resolution_level=9,use_fp16=True)
    torch.cuda.synchronize()
    seconds=time.monotonic()-start
    maps={k:v.detach().cpu().numpy() for k,v in result.items()}
    np.savez_compressed(out/'maps.npz',**maps)
    if 'normal' in maps:
        Image.fromarray(np.uint8(np.clip(maps['normal']*.5+.5,0,1)*255)).save(out/'normal.png')
    valid=maps['mask']&np.isfinite(maps['depth'])
    lo,hi=np.quantile(maps['depth'][valid],[.02,.98])
    d=np.clip((maps['depth']-lo)/(hi-lo),0,1);d[~valid]=0
    Image.fromarray(np.uint8(d*255)).save(out/'depth.png')
    record={'model':upstream['model'],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'strict_state_dict':True,'resolution_level':9,'inference_seconds':seconds,'load_and_run_seconds':time.monotonic()-t,
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'intrinsics':maps['intrinsics'].tolist(),
        'valid_fraction':float(valid.mean()),'coordinate_system':'OpenCV: x right, y down, z forward',
        'scale_status':'Monocular model estimate; not measured metric ground truth.',
        'files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file()}}
    (out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
