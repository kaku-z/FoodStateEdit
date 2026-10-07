"""Disclosed geometry-owned appearance projection; preserves unprojected outputs.

This is an experimental renderer/compositor, not a claim that diffusion obeyed
the geometry. Material observations are generated; hidden surfaces are inferred.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation,distance_transform_edt,gaussian_filter

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rgb(p):return np.asarray(Image.open(p).convert('RGB'),dtype=float)
def mask(p):return np.asarray(Image.open(p).convert('L'))>0
def matte(m,feather=2.):return np.clip(distance_transform_edt(m)/feather,0,1)[...,None]
def blend(a,b,m,feather=2.):
    w=matte(m,feather);return a*(1-w)+b*w
def smooth(a,sigma):return gaussian_filter(a,(sigma,sigma,0))

def compose(src,proxy,hraw,traw,geo,prep,mode):
    hole=mask(prep/'hole_edit.png');target=mask(prep/'target_composite_mask.png')
    unconstrained=blend(blend(src,traw,target,5),hraw,hole,5)
    if mode=='soft':return unconstrained,hole|target
    bite=mask(geo/'bite_mask.png');fork=mask(geo/'visible_fork_mask.png')
    sourcebite=mask(geo/'source_bite_mask.png')
    # Fork topology belongs to the renderer; generated texture supplies only
    # bounded local reflectance detail. No generated replacement silhouette.
    fork_rgb=proxy+np.clip(traw-smooth(traw,2),-18,18)*.3
    if mode=='boundary':
        # Interior appearance is unrestricted, with a two-pixel geometric rim.
        inner=np.clip((distance_transform_edt(bite)-2)/4,0,1)[...,None]
        bite_rgb=proxy*(1-inner)+traw*inner
        out=blend(src,bite_rgb,bite,1.5);out=blend(out,fork_rgb,fork,1.25)
        out=blend(out,hraw,hole,5)
        return out,hole|bite|fork
    assert mode=='material'
    # The image-space residual is bounded so texture cannot erase the cut's
    # coarse directional shading. This is not a physical material estimate.
    lowdelta=np.clip(smooth(traw,6)-smooth(proxy,6),-14,14)
    bite_rgb=proxy+.30*lowdelta+np.clip(traw-smooth(traw,2.2),-14,14)*.65
    out=blend(src,bite_rgb,bite,1.5);out=blend(out,fork_rgb,fork,1.25)
    floorwhite=np.array([233,230,216.])
    food=mask(geo/'food_mask.png');lum=src.mean(-1)
    observed=src[food&(lum>=np.quantile(lum[food],.75))]
    white=np.median(observed,axis=0)
    cut_rgb=proxy.copy()
    # The neutral cut shader has three known face tones. A source-relative
    # material color and limited contrast preserve their depth ordering.
    delta=proxy-floorwhite
    cut_rgb=white+delta*.45
    cut_rgb+=np.clip(hraw-smooth(hraw,2.2),-10,10)*.65
    out=blend(out,cut_rgb,sourcebite,1.5)
    return out,bite|fork|sourcebite

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--gate',default='gate_v2');ap.add_argument('--geometry',default='geometry_v3');a=ap.parse_args()
    r=a.root;g=r/a.gate;cfg=json.loads((g/'config.json').read_text());out=g/'compositions';out.mkdir(exist_ok=True)
    found={p.parent.name:p for p in g.glob('worker_*/*/raw.png')};rows=[]
    for cid in sorted({j['case_id'] for j in cfg['jobs']}):
        for seed in sorted({j['seed'] for j in cfg['jobs'] if j['case_id']==cid}):
            jobs=[j for j in cfg['jobs'] if j['case_id']==cid and j['seed']==seed]
            hs=[j for j in jobs if j['method'].startswith('hole')];ts=[j for j in jobs if j['method'].startswith('target')]
            if len(hs)!=1 or len(ts)!=1 or hs[0]['id'] not in found or ts[0]['id'] not in found:continue
            prep=g/'prepared'/cid;geo=r/a.geometry/cid
            src=rgb(prep/'source.png');proxy=rgb(prep/'proxy.png');tr=json.loads((prep/'transforms.json').read_text())
            raws={}
            for kind,j in [('hole',hs[0]),('target',ts[0])]:
                x0,y0,x1,y1=tr[kind]['box_xyxy'];b=src.copy()
                b[y0:y1,x0:x1]=np.asarray(Image.open(found[j['id']]).convert('RGB').resize((x1-x0,y1-y0),Image.Resampling.LANCZOS))
                raws[kind]=b
            for mode in ['soft','boundary','material']:
                result,union=compose(src,proxy,raws['hole'],raws['target'],geo,prep,mode)
                result=np.uint8(np.clip(np.round(result),0,255));assert np.array_equal(result[~union],src[~union])
                path=out/f'{cid}__{mode}__{seed}.png';Image.fromarray(result).save(path)
                rows.append({'case_id':cid,'seed':seed,'mode':mode,'path':str(path),'sha256':sha(path),
                    'raw_components':{kind:sha(found[j['id']]) for kind,j in [('hole',hs[0]),('target',ts[0])]},
                    'outside_exact':True,'generated_raw':False,'source_geometry_not_measured_truth':True})
    (out/'manifest.json').write_text(json.dumps({'script_sha256':sha(Path(__file__)),'outputs':rows},indent=2)+'\n')
    print({'compositions':len(rows)})

if __name__=='__main__':main()
