"""Controlled source-camera input scale probe on four development sources."""
import argparse
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
from PIL import Image
import torch
from mld2_real_geometry import field_context,prior_queries,canonical_observation


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True);a=ap.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
    from foodstateedit.material_lineage.mld2 import MLD2Model,MLD2Config
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=MLD2Model(MLD2Config(**ck['model_config'])).cuda().eval();model.load_state_dict(ck['state_dict'])
    rows=[]
    with torch.inference_mode():
        for name in ['real_02_16784','real_04_11680','real_07_10704','real_09_3901']:
            f=a.root/name;source=np.asarray(Image.open(f/'source.png').convert('RGB'))
            mask=np.asarray(Image.open(f/'food_mask.png'))>0;maps=dict(np.load(f/'depth.npz'))
            valid=mask & maps['mask'] & np.isfinite(maps['points']).all(-1)
            yy,xx=np.where(valid);keep=np.linspace(0,len(xx)-1,min(1024,len(xx))).astype(int);yy,xx=yy[keep],xx[keep]
            points=maps['points'][yy,xx];pix=np.c_[xx,yy];food=maps['points'][valid]
            extent=np.max(np.quantile(food,.98,axis=0)-np.quantile(food,.02,axis=0))
            for edge in [40,64]:
                cache,roi,im=field_context(model,source,mask,torch,max_edge=edge);im.save(f/('roi_scale_'+str(edge)+'.png'))
                xyz,view=canonical_observation(points,pix,roi,extent,np.median(food[:,2]))
                pp=prior_queries(model,cache,xyz,pix,roi,torch)
                z=np.linspace(-.85,.85,65);ray=np.repeat(xyz,65,axis=0)+np.tile(z,len(xyz))[:,None]*view
                q=prior_queries(model,cache,ray,np.repeat(pix,65,axis=0),roi,torch)
                inside=q['sdf'].reshape(-1,65)<0
                lengths=[]
                for values in inside:
                    d=np.diff(np.r_[False,values,False].astype(int));starts=np.where(d==1)[0];ends=np.where(d==-1)[0]
                    lengths.append(float(np.max(ends-starts))*(z[1]-z[0]) if len(starts) else 0.)
                row=dict(case_id=name,input_edge=edge,query_count=len(xyz),
                    source_queries_outside_canonical_fraction=float(np.any(np.abs(xyz)>1,axis=-1).mean()),
                    occupied_source_queries_fraction=float((pp['sdf']<0).mean()),
                    source_ray_nonempty_fraction=float(inside.any(-1).mean()),
                    continuous_inside_length_mean=float(np.mean(lengths)),continuous_inside_length_std=float(np.std(lengths)),
                    ray_fully_occupied_fraction=float(inside.all(-1).mean()),
                    geometry_accuracy_claim=False)
                rows.append(row);print(json.dumps(row),flush=True)
    receipt=dict(scope='Unpaired source-ray sensitivity; not measured real 3D accuracy',
        optimizer_steps=int(ck['optimizer_steps']),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        formal_scale_choice=40,choice_reason='Match fixed synthetic orthographic camera body footprint; not edited-image selection',cases=rows)
    (a.root/'input_scale_probe.json').write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
