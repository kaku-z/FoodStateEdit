"""Dense observed-material head conditioning without internal food edge cues."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mask(path):
    return np.asarray(Image.open(path).convert('L')) > 127


def rgb(path):
    return np.asarray(Image.open(path).convert('RGB')).copy()


def morphology(value, radius, expand):
    kernel=ImageFilter.MaxFilter(2*radius+1) if expand else ImageFilter.MinFilter(2*radius+1)
    return np.asarray(Image.fromarray(value.astype(np.uint8)*255).filter(kernel))>0


def fill_holes(value):
    image=Image.fromarray(np.pad(value.astype(np.uint8)*255,1)).copy()
    ImageDraw.floodfill(image,(0,0),127)
    return np.asarray(image)[1:-1,1:-1]!=127


def distance_at_least_two(value):
    # On an integer grid, all offsets with Euclidean distance < 2 are the
    # nine pixels of the 3x3 neighborhood. Equality at distance 2 is allowed.
    h,w=value.shape;pad=np.pad(value,1)
    return np.logical_and.reduce([pad[1+dy:1+dy+h,1+dx:1+dx+w]
        for dy in (-1,0,1) for dx in (-1,0,1)])


def prepare_case(parent,output):
    if parent.resolve()==output.resolve():
        raise ValueError('Dense guides require a separate output directory')
    shutil.copytree(parent,output,dirs_exist_ok=True)
    head=output/'head';oldhead=parent/'head'
    stage=json.loads((head/'stage_manifest.json').read_text())
    food=mask(head/'target_food_mask.png');edit=mask(head/'inpaint_mask.png')
    with np.load(head/'guide_channels.npz') as payload:
        channels={key:payload[key] for key in payload.files}
    confidence=channels['material_confidence']
    dense=food & distance_at_least_two(food) & (confidence>=.75) & edit
    condition=rgb(head/'coarse.png');texture=rgb(head/'material_texture.png')
    condition[dense]=texture[dense]
    oldedge=np.asarray(Image.open(head/'edge.png').convert('L')).copy()
    if stage['mode']=='strand':
        # Spaces between distinct visible strands are part of their silhouette.
        envelope=food.copy()
        envelope_scope='Exact transported strand support; gaps between strands remain.'
    else:
        # Cohesive top/side raster cracks are not photographed food creases.
        closed=morphology(morphology(food,2,True),2,False)
        envelope=fill_holes(closed) & channels['target_tolerance']
        envelope_scope='Two-pixel closing and hole filling of cohesive support, clipped to the existing silhouette-tolerance mask. This is only a control envelope; geometry, source ownership, and final edit masks remain unchanged.'
    suppress=morphology(envelope,1,True)
    edge=oldedge.copy();edge[suppress]=0
    outline=morphology(envelope,1,True)^morphology(envelope,1,False)
    edge[outline]=220
    conditionmask=edit & ~dense
    Image.fromarray(condition).save(head/'conditioning_image.png')
    Image.fromarray(conditionmask.astype(np.uint8)*255).save(head/'conditioning_mask.png')
    Image.fromarray(dense.astype(np.uint8)*255).save(head/'anchor_mask.png')
    Image.fromarray(envelope.astype(np.uint8)*255).save(head/'food_control_envelope.png')
    Image.fromarray(edge).save(head/'edge.png')
    for name in ('conditioning_mask','anchor_mask'):
        Image.open(head/(name+'.png')).resize(stage['generation_size'],Image.Resampling.NEAREST).save(head/(name+'_1024.png'))
    channels.update(stage_conditioning_mask=conditionmask,stage_anchor_mask=dense,
        dense_material_core=dense,food_control_envelope=envelope)
    np.savez_compressed(head/'guide_channels.npz',**channels)
    y,x=np.where(dense);native=np.column_stack([x,y]);x0,y0=stage['bbox_xyxy'][:2]
    size=np.asarray(stage['generation_size']);native_size=np.asarray(stage['native_size'])
    np.savez_compressed(head/'stage_anchors.npz',target_pixel_xy=native,
        target_pixel_full_xy=native+[x0,y0],
        target_pixel_generation_xy=(native+.5)*size/native_size-.5,
        source_uv=channels['target_source_uv'][dense],source_sample_rgb=texture[dense],
        material_confidence=confidence[dense])
    record=dict(case_id=parent.name,variant='dense_observed_material_head_v1',
        parent=str(parent.resolve()),parent_stage_manifest_sha256=sha(oldhead/'stage_manifest.json'),
        known_core_pixels=int(dense.sum()),food_pixels=int(food.sum()),
        known_core_food_fraction=float(dense.sum()/max(1,food.sum())),
        confidence_min=float(confidence[dense].min()) if dense.any() else None,
        core_rule='All target pixels with material confidence >=0.75 and Euclidean distance to background >=2 native pixels. No sparse point sampling; disconnected observed strand regions remain separate.',
        envelope_scope=envelope_scope,
        suppressed_old_edge_pixels=int(np.count_nonzero(oldedge[suppress])),
        new_interior_edge_pixels=int(np.count_nonzero(edge[distance_at_least_two(envelope)])),
        exact_material_texture_at_core=bool(np.array_equal(condition[dense],texture[dense])),
        source_stage_unchanged=all(sha(p)==sha(output/'source'/p.name) for p in (parent/'source').iterdir() if p.is_file()),
        full_geometry_and_masks_unchanged=all(sha(parent/name)==sha(output/name) for name in
            ['full_guide_channels.npz','full_inpaint_mask.png','head_stage_full_mask.png','source_stage_full_mask.png','handle_passthrough_mask.png']),
        final_head_mask_unchanged=sha(oldhead/'inpaint_mask.png')==sha(head/'inpaint_mask.png'),
        limitation='Known-core image conditioning is not a guaranteed final pixel constraint. The generator can still reinterpret material; evaluate the actual generated core separately.',
        builder_sha256=sha(Path(__file__)))
    stage.update(dense_conditioning=record,anchor_pixels=int(dense.sum()))
    stage['images']['food_control_envelope']='food_control_envelope.png'
    stage['file_hashes']={p.name:sha(p) for p in head.iterdir() if p.is_file() and p.name not in ['stage_manifest.json','guide_manifest.json']}
    for name in ('stage_manifest.json','guide_manifest.json'):
        (head/name).write_text(json.dumps(stage,indent=2)+'\n')
    factor=json.loads((output/'factorization.json').read_text())
    factor.update(variant='factorized_dense_observed_head_v1',dense_conditioning=record)
    factor['stages']['head']=stage
    (output/'factorization.json').write_text(json.dumps(factor,indent=2)+'\n')
    (output/'dense_conditioning.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--indices',type=int,nargs='+')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for parent in sorted(args.input_root.glob('real_*')):
        if args.indices is not None and int(parent.name.split('_')[1]) not in args.indices:continue
        if not (parent/'factorization.json').exists():continue
        record=prepare_case(parent,args.output/parent.name);rows.append(record)
        print(json.dumps({k:record[k] for k in ['case_id','known_core_pixels','known_core_food_fraction','new_interior_edge_pixels']}),flush=True)
    (args.output/'dense_manifest.json').write_text(json.dumps(dict(cases=rows,generated=False),indent=2)+'\n')


if __name__=='__main__':main()
