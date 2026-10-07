"""Use transported observed RGB as material; estimate only bounded achromatic relighting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, distance_transform_edt


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def linear(a):
    a=a.astype(np.float32)/255
    return np.where(a<=.04045,a/12.92,((a+.055)/1.055)**2.4)


def srgb(a):
    a=np.maximum(a,0)
    return np.where(a<=.0031308,12.92*a,1.055*a**(1/2.4)-.055)


def transport(observed, generated, known, food):
    """The neural candidate supplies only a low-frequency scalar light ratio."""
    t,g=linear(observed),linear(generated)
    weight=np.clip(known,0,1)*np.clip(distance_transform_edt(food)/2,0,1)
    ty=np.sum(t*np.array([.2126,.7152,.0722]),axis=2)
    gy=np.sum(g*np.array([.2126,.7152,.0722]),axis=2)
    support=weight>0
    den=gaussian_filter(support.astype(float),3)
    target_mean=gaussian_filter(ty*support,3)/np.maximum(den,1e-6)
    generated_mean=gaussian_filter(gy*support,3)/np.maximum(den,1e-6)
    gain=np.clip((generated_mean+.01)/(target_mean+.01),.72,1.4)
    material=srgb(t*gain[...,None]).clip(0,1)*255
    out=np.rint(material*weight[...,None]+generated*(1-weight[...,None])).clip(0,255).astype(np.uint8)
    out[~support]=generated[~support]
    return out,gain,weight


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--base',default='factorized_auto_depth');p.add_argument('--indices',nargs='+',type=int,default=[2,13,14]);a=p.parse_args();rows=[]
    for f in sorted((a.root/'guides_factorized_depth').glob('real_*')):
        if int(f.name.split('_')[1]) not in a.indices:continue
        z=np.load(f/'full_guide_channels.npz')
        mf=a.root/'guides'/f.name/'material_texture.png'
        observed=np.asarray(Image.open(mf).convert('RGB'))
        base=a.root/'real'/f.name/'candidates'/a.base/'unprojected.png'
        generated=np.asarray(Image.open(base).convert('RGB'))
        output,gain,weight=transport(observed,generated,z['material_confidence'],z['target_food'])
        dest=a.root/'real'/f.name/'candidates/material_locked_v1';dest.mkdir(exist_ok=False)
        Image.fromarray(output).save(dest/'unprojected.png')
        np.savez_compressed(dest/'relighting.npz',gain=gain,weight=weight)
        sm=json.loads((f/'head/stage_manifest.json').read_text());x0,y0,x1,y1=sm['bbox_xyxy']
        Image.fromarray(output[y0:y1,x0:x1]).resize((4*(x1-x0),4*(y1-y0)),Image.Resampling.NEAREST).save(dest/'head_detail.png')
        row=dict(case_id=f.name,base_sha256=sha(base),source_rgb_material_sha256=sha(mf),
            method='Transported observed source RGB with bounded scalar low-frequency relighting; neural appearance only outside known material',
            known_pixels=int((weight>0).sum()),fully_locked_pixels=int((weight==1).sum()),
            gain_range=[.72,1.4],changed_outside_known_pixels=int(np.any(output!=generated,2)[weight==0].sum()),
            runner_sha256=sha(__file__),limitations='Source photograph lighting remains entangled with reflectance. Unknown surfaces and boundary shape are not identified from one view. This is a material preservation constraint, not a claim of perfect photorealism.')
        (dest/'generation.json').write_text(json.dumps(row,indent=2));rows.append(row)
    (a.root/'material_locked_v1_summary.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
