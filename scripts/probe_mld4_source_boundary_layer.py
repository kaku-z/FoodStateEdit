"""Frozen CPU probe of a locally visible layer beside a removed food region.

No global donor search and no semantics/generation model. Absent or distant
boundary support is explicitly unknown. Candidate acceptance is a subset of R.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, label
from scipy.spatial import cKDTree

CONFIG = dict(indices=[0,2,13,14], neighborhood=8, donor_radius_at640=24., fill_radius_at640=24.,
    front_depth_tolerance_relative=.02, nearest_neighbors=4, soft_distance_temperature_at640=1.5,
    candidate_rule='Outside F, valid positive source depth, within local R radius, not clearly in front of nearest R; connected to a direct R neighbor.',
    confidence_rule='exp(-nearest_donor_distance/fill_radius); a geometric support indicator, not calibrated correctness.',
    acceptance_rule='R only, with all four donor distances <= fill radius. Otherwise retain baseline and mark unknown.',
    negative_control='14 is a prespecified interior-noodle negative control. No per-image parameter changes.',
    hypothesis='An adjacent visible exterior component may continue the revealed support. F complement alone is not verified nonfood.')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    Path(p).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def rgb(p):
    return np.asarray(Image.open(p).convert('RGB')).copy()


def mask(p):
    return np.asarray(Image.open(p).convert('L')) > 127


def save(p, a):
    Image.fromarray(a.astype(np.uint8)*255 if a.dtype==bool else a).save(p)


def localize(raw, root):
    value = str(raw).replace('\\','/')
    key='material_lineage_reform_20261004/'
    if key in value:
        return root/value.split(key,1)[1]
    key='material_lineage_deformable_20261004/'
    return root.parent/'material_lineage_deformable_20261004'/value.split(key,1)[1]


def prepare(case, root):
    cid=case['case_id']
    guide=localize(case['production_guides']['boundary'],root)
    geometry=localize(case['resolved_geometry_case'],root)
    production=root/'production_head_only_20261005'
    baseline=production/'real'/cid/'final.png'
    if not baseline.exists():
        baseline=production/'evaluation/assets'/cid/'final.png'
    inputs=dict(source=guide/'source.png', food=geometry/'source_food_mask.png',
        removed=geometry/'source_removed_mask.png', depth=guide/'guide_channels.npz',baseline=baseline)
    return dict(case_id=cid,index=case['index'],inputs={k:str(v) for k,v in inputs.items()},
        hashes={k:sha(v) for k,v in inputs.items()})


def probe(case, output):
    inputs={k:Path(v) for k,v in case['inputs'].items()}
    source=rgb(inputs['source']);base=rgb(inputs['baseline'])
    F=mask(inputs['food']);R=mask(inputs['removed'])
    assert not (R&~F).any()
    with np.load(inputs['depth']) as z:
        depth=z['source_depth'].copy()
        valid=z['source_depth_valid'].copy() & np.isfinite(depth) & (depth>0)
    h,w=R.shape;scale=max(h,w)/640.
    radius=CONFIG['donor_radius_at640']*scale;fill_radius=CONFIG['fill_radius_at640']*scale
    distance,nearest=distance_transform_edt(~R,return_indices=True)
    nearest_depth=depth[nearest[0],nearest[1]]
    tolerance=CONFIG['front_depth_tolerance_relative']*float(np.median(depth[R&valid]))
    potential=(~F)&valid&(distance<=radius)
    depth_allowed=potential&(depth>=nearest_depth-tolerance)
    adjacent=binary_dilation(R,structure=np.ones((3,3),bool))&~F
    seed=adjacent&depth_allowed
    components,n=label(depth_allowed,np.ones((3,3),int))
    ids=np.unique(components[seed]);ids=ids[ids!=0]
    donors=np.isin(components,ids)&depth_allowed
    dy,dx=np.where(donors);ry,rx=np.where(R)
    confidence=np.zeros((h,w),np.float32);covered=np.zeros((h,w),bool)
    donor_uv=np.full((h,w,2),-1,np.int32)
    donor_uv_k=np.full((len(ry),CONFIG['nearest_neighbors'],2),-1,np.int32)
    donor_weights=np.zeros((len(ry),CONFIG['nearest_neighbors']),np.float32)
    proposal=base.copy();proposal[R]=127
    nearest_distance=np.full((h,w),np.nan,np.float32)
    if len(dx)>=CONFIG['nearest_neighbors']:
        tree=cKDTree(np.column_stack([dx,dy]))
        distances,idx=tree.query(np.column_stack([rx,ry]),k=CONFIG['nearest_neighbors'])
        supported=distances[:,-1]<=fill_radius
        uv=np.stack([dx[idx],dy[idx]],axis=-1)
        weights=np.exp(-(distances-distances[:,0:1])/(CONFIG['soft_distance_temperature_at640']*scale))
        weights/=weights.sum(axis=1,keepdims=True)
        values=(source[uv[...,1],uv[...,0]].astype(float)*weights[...,None]).sum(axis=1)
        covered[ry[supported],rx[supported]]=True
        proposal[ry[supported],rx[supported]]=np.rint(values[supported]).clip(0,255).astype(np.uint8)
        confidence[ry[supported],rx[supported]]=np.exp(-distances[supported,0]/fill_radius)
        donor_uv[ry[supported],rx[supported]]=uv[supported,0]
        donor_uv_k[supported]=uv[supported];donor_weights[supported]=weights[supported]
        nearest_distance[ry,rx]=distances[:,0]
    unknown=R&~covered
    final=base.copy();final[covered]=proposal[covered]
    overlay=source.astype(float)
    overlay[donors]=.5*overlay[donors]+.5*np.array([0,240,255])
    overlay[R]=.7*overlay[R]+.3*np.array([255,0,100])
    overlay[F&~binary_erosion(F)]=[30,255,70]
    overlay[seed]=[255,230,0]
    coverage=source.astype(float)
    coverage[covered]=.5*coverage[covered]+.5*np.array([30,230,70])
    coverage[unknown]=.4*coverage[unknown]+.6*np.array([230,30,230])
    out=output/case['case_id'];out.mkdir()
    for name,a in dict(source=source,baseline=base,food_mask=F,removed_mask=R,
        potential_donor_mask=potential,depth_allowed_mask=depth_allowed,seed_mask=seed,donor_mask=donors,
        acceptance_mask=covered,unknown_mask=unknown,proposal_with_unknown_gray=proposal,final=final,
        donor_overlay=np.rint(overlay).astype(np.uint8),coverage_overlay=np.rint(coverage).astype(np.uint8),
        confidence=np.rint(confidence*255).astype(np.uint8)).items():save(out/(name+'.png'),a)
    np.savez_compressed(out/'donor_provenance.npz',donor_uv=donor_uv,donor_uv_k=donor_uv_k,
        donor_weights=donor_weights,removed_yx=np.column_stack([ry,rx]),confidence=confidence,
        nearest_distance=nearest_distance,source_depth=depth,source_depth_valid=valid,
        depth_margin=depth-nearest_depth)
    changed=np.any(final!=base,axis=2)
    row=dict(**case,source_shape=[h,w],scale=scale,donor_radius=radius,fill_radius=fill_radius,
        depth_tolerance=tolerance,adjacent_exterior_pixels=int(adjacent.sum()),seed_pixels=int(seed.sum()),
        potential_donor_pixels=int(potential.sum()),depth_rejected_pixels=int((potential&~depth_allowed).sum()),
        donors=int(donors.sum()),seeded_components=len(ids),removed_pixels=int(R.sum()),
        coverage_pixels=int(covered.sum()),coverage_fraction=float(covered.sum()/R.sum()),unknown_pixels=int(unknown.sum()),
        donor_outside_F=bool(not (donors&F).any()),changed_outside_R=int(changed[~R].sum()),
        changed_unknown=int(changed[unknown].sum()),input_hashes_unchanged=all(sha(v)==case['hashes'][k] for k,v in inputs.items()),
        mean_accepted_rgb=source[donors].mean(axis=0).tolist() if donors.any() else None,
        status='partial_support' if covered.any() else 'unknown_no_local_support',
        interpretation='A visible-local-layer hypothesis only. Unknown pixels retain A and must not be counted as reconstructed support; local donors require visual audit.')
    assert row['changed_outside_R']==row['changed_unknown']==0 and row['input_hashes_unchanged']
    write(out/'result.json',row)
    yy,xx=np.where(R);pad=round(30*scale)
    bounds=(max(0,int(xx.min())-pad),max(0,int(yy.min())-pad),min(w,int(xx.max())+pad+1),min(h,int(yy.max())+pad+1))
    panel=Image.new('RGB',(1500,350),'white');draw=ImageDraw.Draw(panel)
    for i,(name,title) in enumerate([('source','Original'),('donor_overlay','Cyan donors; yellow seed'),('coverage_overlay','Green accepted; purple unknown'),('baseline','A source crop'),('final','Local donor partial result')]):
        im=Image.open(out/(name+'.png')).crop(bounds)
        ratio=min(294/im.width,305/im.height)
        im=im.resize((round(im.width*ratio),round(im.height*ratio)),Image.Resampling.BICUBIC)
        panel.paste(im,(i*300+(294-im.width)//2,30+(305-im.height)//2))
        draw.text((i*300+3,5),case['case_id']+' '+title,fill='black')
    panel.save(out/'comparison.png')
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    route=json.loads((args.root/'routing_resolved_manifest.json').read_text())
    cases=[prepare(c,args.root) for c in route['cases'] if c['index'] in CONFIG['indices']]
    write(args.output/'frozen.json',dict(config=CONFIG,cases=cases,runner_sha256=sha(__file__)))
    shutil.copy2(__file__,args.output/Path(__file__).name)
    results=[]
    for case in cases:
        row=probe(case,args.output);results.append(row)
        print(json.dumps({k:row[k] for k in ['case_id','seed_pixels','donors','coverage_fraction','unknown_pixels','changed_outside_R']}),flush=True)
    write(args.output/'manifest.json',dict(status='complete',cases=results,config=CONFIG))


if __name__=='__main__':main()
