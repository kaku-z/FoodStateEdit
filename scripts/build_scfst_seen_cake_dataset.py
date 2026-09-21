"""Freeze a one-sample cake pseudo-target dataset for SCFST capacity training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controls',type=Path,required=True)
    p.add_argument('--target-frames',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--prompt',required=True)
    a=p.parse_args(); out=a.output_root.resolve()
    if out.exists(): raise FileExistsError(out)
    frames=sorted(a.target_frames.glob('*.png'))
    if len(frames)!=21: raise ValueError(f'expected 21 target frames, found {len(frames)}')
    with np.load(a.controls/'controls.npz',allow_pickle=False) as z:
        condition=z['condition'].copy()
    if condition.shape[0]!=21 or condition.ndim!=4 or condition.shape[-1]!=3:
        raise ValueError(f'bad VACE condition shape: {condition.shape}')
    target=[np.asarray(Image.open(path).convert('RGB')) for path in frames]
    if any(item.shape!=target[0].shape for item in target) or condition.shape[1:3]!=target[0].shape[:2]:
        raise ValueError('target/control spatial mismatch')
    (out/'video').mkdir(parents=True)
    (out/'vace_video').mkdir()
    (out/'vace_reference_image').mkdir()
    target_path=out/'video/cake_e7_soft_pseudotarget.mp4'
    control_path=out/'vace_video/cake_e7_control.mp4'
    reference_path=out/'vace_reference_image/cake_reference.png'
    imageio.mimsave(target_path,target,fps=8,codec='libx264')
    imageio.mimsave(control_path,list(condition),fps=8,codec='libx264')
    shutil.copy2(a.controls/'reference.png',reference_path)
    metadata_path=out/'metadata.csv'
    with metadata_path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=['video','vace_video','vace_reference_image','prompt'])
        writer.writeheader(); writer.writerow({
            'video':'video/'+target_path.name,
            'vace_video':'vace_video/'+control_path.name,
            'vace_reference_image':'vace_reference_image/'+reference_path.name,
            'prompt':a.prompt,
        })
    manifest={
        'schema_version':'foodstateedit.scfst_seen_cake_dataset.v1',
        'status':'seen_synthetic_pseudotarget_capacity_diagnostic',
        'sample_count':1,'frame_count':21,'fps':8,
        'target_source':str(a.target_frames.resolve()),
        'controls_source':str(a.controls.resolve()),
        'builder_sha256':sha256(Path(__file__)),
        'limitations':[
            'target is the existing E7 soft output and contains known visual artifacts',
            'positive training loss behavior cannot establish visual improvement or generalization',
        ],
        'files':{},
    }
    for path in (target_path,control_path,reference_path,metadata_path):
        manifest['files'][path.relative_to(out).as_posix()]={'sha256':sha256(path),'size_bytes':path.stat().st_size}
    (out/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest))


if __name__=='__main__': main()
