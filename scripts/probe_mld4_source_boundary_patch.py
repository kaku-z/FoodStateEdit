"""One frozen guard-band/coherent-translation correction to the boundary probe."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve

CONFIG=dict(indices=[0,2,13,14],guard_native_min=2.,guard_at640=3.,
    local_region='Exactly the v1 seed-reachable exterior connected domain, radius 24*maxdim/640, unchanged depth rule.',
    max_translation_at640=48.,context_radius_at640=8.,minimum_context_pixels=8,
    cost='0.25*offset_length/max_translation + mean absolute observed-context RGB difference/255',
    matching_context='Both original and translated context pixels must belong to the guarded observed exterior domain; never match original R food RGB.',
    completion='One integer translation supplies the whole R. No blend/nearest-neighbor warp; no partial patch. No complete valid patch -> unknown/no-op.',
    model_usage='None, CPU only',parameter_selection='Uniform prespecified four cases; no output-selected tuning.')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def rgb(p):return np.asarray(Image.open(p).convert('RGB')).copy()
def mask(p):return np.asarray(Image.open(p).convert('L'))>127
def save(p,a):Image.fromarray(a.astype(np.uint8)*255 if a.dtype==bool else a).save(p)


def run_case(cid,prior,output):
    old=prior/cid;out=output/cid;out.mkdir()
    original=rgb(old/'source.png');base=rgb(old/'baseline.png');F=mask(old/'food_mask.png');R=mask(old/'removed_mask.png')
    connected=mask(old/'donor_mask.png');h,w=R.shape;scale=max(h,w)/640.
    guard=max(CONFIG['guard_native_min'],CONFIG['guard_at640']*scale)
    donor=connected&(distance_transform_edt(~F)>guard)
    context=donor&(distance_transform_edt(~R)<=CONFIG['context_radius_at640']*scale)
    ry,rx=np.where(R);cy,cx=np.where(context)
    radius=CONFIG['max_translation_at640']*scale
    # Cross-correlation counts exact translation support of the whole R stencil.
    counts=fftconvolve(donor.astype(float),R[::-1,::-1].astype(float),mode='full')
    ys,xs=np.where(counts>=R.sum()-1e-4)
    oy,ox=ys-(h-1),xs-(w-1)
    close=ox*ox+oy*oy<=radius*radius
    candidates=list(zip(ox[close].tolist(),oy[close].tolist()))
    scored=[]
    for dx,dy in candidates:
        qx,qy=rx+dx,ry+dy
        if not ((qx>=0)&(qx<w)&(qy>=0)&(qy<h)).all() or not donor[qy,qx].all():continue
        tx,ty=cx+dx,cy+dy
        valid=(tx>=0)&(tx<w)&(ty>=0)&(ty<h)
        take=np.where(valid)[0]
        take=take[donor[ty[take],tx[take]]]
        if len(take)<CONFIG['minimum_context_pixels']:continue
        error=float(np.abs(original[cy[take],cx[take]].astype(float)-original[ty[take],tx[take]]).mean()/255.)
        distance=float(np.hypot(dx,dy))
        cost=.25*distance/radius+error
        scored.append((cost,distance,dy,dx,error,len(take)))
    scored.sort()
    final=base.copy();used=np.zeros(R.shape,bool);accepted=np.zeros(R.shape,bool)
    donor_uv=np.full((h,w,2),-1,np.int32);confidence=np.zeros(R.shape,np.float32)
    selected=None
    if scored:
        cost,distance,dy,dx,error,count=scored[0]
        final[ry,rx]=original[ry+dy,rx+dx]
        used[ry+dy,rx+dx]=True;accepted=R.copy()
        donor_uv[ry,rx]=np.column_stack([rx+dx,ry+dy])
        confidence[R]=1./(1.+cost)
        selected=dict(dx=dx,dy=dy,cost=cost,distance=distance,context_rgb_mae_normalized=error,context_pixels=count)
    unknown=R&~accepted
    overlay=original.astype(float);overlay[donor]=.5*overlay[donor]+.5*np.array([0,240,255])
    overlay[connected&~donor]=[255,170,20]
    overlay[used]=.3*overlay[used]+.7*np.array([50,255,80])
    overlay[R]=.6*overlay[R]+.4*np.array([255,0,200])
    for name,a in dict(source=original,baseline=base,food_mask=F,removed_mask=R,guarded_donor_mask=donor,
        rejected_guard_mask=connected&~donor,observed_context_mask=context,actual_used_donor_mask=used,
        acceptance_mask=accepted,unknown_mask=unknown,final=final,donor_overlay=np.rint(overlay).astype(np.uint8),
        confidence=np.rint(confidence*255).astype(np.uint8)).items():save(out/(name+'.png'),a)
    np.savez_compressed(out/'donor_provenance.npz',donor_uv=donor_uv,confidence=confidence)
    changed=np.any(final!=base,axis=2)
    row=dict(case_id=cid,guard_pixels=guard,local_donor_radius=24.*scale,max_translation=radius,
        prior_connected_pixels=int(connected.sum()),guarded_donor_pixels=int(donor.sum()),
        removed_pixels=int(R.sum()),enough_donor_area_for_full_patch=bool(donor.sum()>=R.sum()),
        full_stencil_valid_translations=len(candidates),context_scored_translations=len(scored),selected=selected,
        coverage_pixels=int(accepted.sum()),unknown_pixels=int(unknown.sum()),
        changed_outside_R=int(changed[~R].sum()),changed_retained_food=int(changed[F&~R].sum()),
        unknown_changed=int(changed[unknown].sum()),
        outcome='full_patch_proposal' if selected else 'unknown_noop',
        reason=None if selected else ('Not enough guarded donor area for any complete R patch' if donor.sum()<R.sum() else 'No complete translation with observed-layer context evidence'),
        input_hashes={n:sha(old/n) for n in ['source.png','baseline.png','food_mask.png','removed_mask.png','donor_mask.png','result.json']},
        final_sha256=sha(out/'final.png'))
    assert row['changed_outside_R']==row['changed_retained_food']==row['unknown_changed']==0
    write(out/'result.json',row)
    yy,xx=np.where(R);pad=round(30*scale)
    bounds=(max(0,int(xx.min())-pad),max(0,int(yy.min())-pad),min(w,int(xx.max())+pad+1),min(h,int(yy.max())+pad+1))
    sheet=Image.new('RGB',(1200,350),'white');draw=ImageDraw.Draw(sheet)
    for i,(name,title) in enumerate([('source','Original'),('donor_overlay','Cyan guarded; orange rejected'),('baseline','A'),('final',row['outcome'])]):
        im=Image.open(out/(name+'.png')).crop(bounds);zoom=min(294/im.width,305/im.height)
        im=im.resize((round(im.width*zoom),round(im.height*zoom)),Image.Resampling.BICUBIC)
        sheet.paste(im,(i*300+(294-im.width)//2,30+(305-im.height)//2));draw.text((i*300+3,5),cid+' '+title,fill='black')
    sheet.save(out/'comparison.png')
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    manifest=json.loads((args.prior/'manifest.json').read_text());cases=[c['case_id'] for c in manifest['cases']]
    assert [int(c.split('_')[1]) for c in cases]==CONFIG['indices']
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'frozen.json',dict(config=CONFIG,prior=str(args.prior),prior_frozen_sha256=sha(args.prior/'frozen.json'),
        cases=cases,runner_sha256=sha(__file__),inputs={c:{n:sha(args.prior/c/n) for n in ['source.png','baseline.png','food_mask.png','removed_mask.png','donor_mask.png','result.json']} for c in cases}))
    shutil.copy2(__file__,args.output/Path(__file__).name)
    rows=[]
    for cid in cases:
        row=run_case(cid,args.prior,args.output);rows.append(row)
        print(json.dumps({k:row[k] for k in ['case_id','guarded_donor_pixels','removed_pixels','full_stencil_valid_translations','outcome']}),flush=True)
    write(args.output/'manifest.json',dict(status='complete',config=CONFIG,cases=rows))


if __name__=='__main__':main()
