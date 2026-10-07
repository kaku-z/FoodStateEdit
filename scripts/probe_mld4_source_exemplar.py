"""Source-only support-texture completion control; hidden appearance is a hypothesis."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mask(path):
    return np.asarray(Image.open(path).convert('L')) > 127


def complete(source, removed, food, depth, valid):
    h,w=removed.shape
    y,x=np.where(removed)
    donor=~binary_dilation(food,iterations=1) & valid
    contact=binary_dilation(removed,iterations=2) & ~food & ~removed & valid
    cy,cx=np.where(contact)
    span=max(32,int(.35*max(h,w)))
    target_depth=float(np.median(depth[removed & valid]))
    best=None
    # An entire translated patch preserves actual source texture and coherence.
    for dy in range(-span,span+1,2):
        for dx in range(-span,span+1,2):
            if x.min()+dx<0 or x.max()+dx>=w or y.min()+dy<0 or y.max()+dy>=h:continue
            if not donor[y+dy,x+dx].all():continue
            donor_depth=float(np.median(depth[y+dy,x+dx]))
            if donor_depth<target_depth*.98:continue
            ok=(cx+dx>=0)&(cx+dx<w)&(cy+dy>=0)&(cy+dy<h)
            if ok.any():
                observed=source[cy[ok],cx[ok]].astype(float)
                transferred=source[cy[ok]+dy,cx[ok]+dx].astype(float)
                seam=float(np.mean(np.abs(observed-transferred))/255)
            else:seam=0.
            score=seam+.12*np.hypot(dx,dy)/max(h,w)
            if best is None or score<best[0]:best=(score,dy,dx,seam,donor_depth)
    if best is None:
        raise ValueError('No wholly observed background patch supports this cavity shape')
    score,dy,dx,seam,donor_depth=best
    filled=source.copy();filled[y,x]=source[y+dy,x+dx]
    return filled,dict(translation_xy=[dx,dy],score=score,contact_rgb_error=seam,
        observed_contact_pixels=int(contact.sum()),donor_pixels=int(donor.sum()),
        source_selected_depth_median=target_depth,donor_depth_median=donor_depth,
        transferred_pixels=int(removed.sum()),hidden_surface_verified=False),np.stack([x+dx,y+dy],1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True);p.add_argument('--indices',type=int,nargs='+',default=[2,13,14])
    p.add_argument('--variant',default='source_exemplar_v1');a=p.parse_args();rows=[]
    for f in sorted((a.root/'guides_factorized_depth').glob('real_*')):
        if int(f.name.split('_')[1]) not in a.indices:continue
        original=np.asarray(Image.open(f/'full_source.png').convert('RGB'))
        z=np.load(f/'full_guide_channels.npz');removed=z['source_removed']
        food=mask(a.geometry/'real'/f.name/'source_food_mask.png')
        dest=a.root/'real'/f.name/'candidates'/a.variant;dest.mkdir(parents=True,exist_ok=False)
        try:
            completed,audit,uv=complete(original,removed,food,z['source_depth'],z['source_depth_valid'])
        except ValueError as e:
            (dest/'failure.json').write_text(json.dumps({'reason':str(e)},indent=2));rows.append({'case_id':f.name,'status':'unsupported'});continue
        base=a.root/'real'/f.name/'candidates/factorized_auto_depth/unprojected.png'
        output=np.asarray(Image.open(base).convert('RGB')).copy()
        region=mask(f/'source_stage_full_mask.png')
        output[region]=original[region];output[removed & region]=completed[removed & region]
        Image.fromarray(output).save(dest/'unprojected.png')
        Image.fromarray(completed).save(dest/'source_completion.png')
        np.savez_compressed(dest/'donor_correspondence.npz',source_uv=uv,target_yx=np.argwhere(removed))
        sm=json.loads((f/'source/stage_manifest.json').read_text());x0,y0,x1,y1=sm['bbox_xyxy']
        Image.fromarray(output[y0:y1,x0:x1]).resize((4*(x1-x0),4*(y1-y0)),Image.Resampling.NEAREST).save(dest/'source_detail.png')
        audit.update(case_id=f.name,status='complete',method='Coherent observed non-food texture translation, nearest seam and displacement score',
            runner_sha256=sha(__file__),base_sha256=sha(base),source_sha256=sha(f/'full_source.png'),
            food_mask_sha256=sha(a.geometry/'real'/f.name/'source_food_mask.png'),
            source_retained_changes=int(np.any(output!=original,2)[region & ~removed].sum()),
            generated_neural_pixels=0,limitation='No unseen cut wall is recovered. Donor mask/depth are estimates; interior cavities have no observed support evidence.')
        (dest/'generation.json').write_text(json.dumps(audit,indent=2));rows.append(audit)
    (a.root/(a.variant+'_summary.json')).write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
