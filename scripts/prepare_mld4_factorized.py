"""Independent high-resolution head/source crops for an MLD4 ablation.

The runner must start from a same-protocol full-frame candidate. Head and source
stages edit disjoint frozen masks; the remaining handle stays in that candidate.
No new food selection, geometry, ingredient classifier, or GPU model is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageFilter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rgb(path):
    return np.asarray(Image.open(path).convert('RGB')).copy()


def mask(path):
    return np.asarray(Image.open(path).convert('L')) > 127


def save(path, array):
    array = np.asarray(array)
    Image.fromarray(array.astype(np.uint8)*255 if array.dtype == bool else
                    np.clip(array, 0, 255).astype(np.uint8)).save(path)


def dilate(value, radius):
    return np.asarray(Image.fromarray(value.astype(np.uint8)*255).filter(
        ImageFilter.MaxFilter(2*radius+1))) > 0


def bbox(value):
    y, x = np.where(value)
    if not len(x):
        raise ValueError('Cannot crop an empty stage mask')
    return [int(x.min()), int(y.min()), int(x.max())+1, int(y.max())+1]


def expand(box, margin, shape):
    h, w = shape
    return [max(0, box[0]-margin), max(0, box[1]-margin),
            min(w, box[2]+margin), min(h, box[3]+margin)]


def rectangle(box, shape):
    result = np.zeros(shape, bool)
    result[box[1]:box[3], box[0]:box[2]] = True
    return result


def crop(array, box):
    return array[box[1]:box[3], box[0]:box[2]].copy()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def build_stage(name, box, stage_full, full, parent, parent_manifest, channels,
                arrays, anchor_full, owned_reference, reference_mask, args):
    output = full/name
    output.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = box
    h, w = y1-y0, x1-x0
    stage_mask = crop(stage_full, box)
    anchors = crop(anchor_full & stage_full, box) if name == 'head' else np.zeros((h,w), bool)
    condition = crop(arrays['coarse'], box)
    texture = crop(arrays['material_texture'], box)
    condition[anchors] = texture[anchors]
    conditioning_mask = stage_mask & ~anchors
    factor = args.resolution/max(w,h)
    generation_size = [max(64, int(round(w*factor/32)*32)),
                       max(64, int(round(h*factor/32)*32))]
    sx, sy = generation_size[0]/w, generation_size[1]/h
    full_to_crop = np.array([[1,0,-x0],[0,1,-y0],[0,0,1]], dtype=float)
    # Pixel-index maps match the half-pixel convention used by image resizing.
    crop_to_generation = np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1]], dtype=float)
    inverse = np.linalg.inv(full_to_crop)
    image_files = dict(conditioning_image=condition, conditioning_mask=conditioning_mask,
        inpaint_mask=stage_mask, anchor_mask=anchors, source=crop(arrays['source'],box),
        coarse=crop(arrays['coarse'],box), material_texture=texture,
        material_confidence=crop(channels['material_confidence'],box)*255,
        edge=crop(arrays['edge'],box), depth=crop(arrays['depth'],box),
        target_food_mask=crop(arrays['target_food_mask'],box),
        spoon_mask=crop(arrays['spoon_mask'],box),
        source_removal_mask=crop(arrays['source_removal_mask'],box),
        source_inpaint_mask=crop(arrays['source_inpaint_mask'],box),
        source_reference=owned_reference, source_reference_mask=reference_mask)
    for key, value in image_files.items():
        save(output/(key+'.png'), value)
    for key in ('conditioning_mask','inpaint_mask','anchor_mask'):
        Image.open(output/(key+'.png')).resize(generation_size, Image.Resampling.NEAREST).save(
            output/(key+'_1024.png'))
    cropped_channels = {}
    frame_shape = arrays['source'].shape[:2]
    for key, value in channels.items():
        cropped_channels[key] = crop(value,box) if value.ndim >= 2 and value.shape[:2] == frame_shape else value.copy()
    intrinsic = channels['camera_intrinsics'].copy()
    intrinsic[0,2] -= x0
    intrinsic[1,2] -= y0
    cropped_channels.update(camera_intrinsics=intrinsic,
        camera_intrinsics_full=channels['camera_intrinsics'],
        crop_bbox_xyxy=np.asarray(box), full_to_crop=full_to_crop,
        crop_to_full=inverse, crop_to_generation=crop_to_generation,
        stage_inpaint=stage_mask, stage_conditioning_mask=conditioning_mask,
        stage_anchor_mask=anchors)
    np.savez_compressed(output/'guide_channels.npz', **cropped_channels)
    ay, ax = np.where(anchors)
    anchor_local = np.column_stack([ax,ay])
    anchor_full_xy = anchor_local + [x0,y0]
    anchor_generation = (anchor_local+.5)*[sx,sy]-.5
    np.savez_compressed(output/'stage_anchors.npz', target_pixel_xy=anchor_local,
        target_pixel_full_xy=anchor_full_xy, target_pixel_generation_xy=anchor_generation,
        source_uv=channels['target_source_uv'][anchor_full],
        source_sample_rgb=texture[anchors],
        material_confidence=crop(channels['material_confidence'],box)[anchors]) if name == 'head' else np.savez_compressed(
            output/'stage_anchors.npz', target_pixel_xy=anchor_local,
            target_pixel_full_xy=anchor_full_xy, target_pixel_generation_xy=anchor_generation,
            source_uv=np.empty((0,2)), source_sample_rgb=np.empty((0,3),np.uint8),
            material_confidence=np.empty(0,np.float32))
    stage = dict(case_id=parent.name, stage=name, bbox_xyxy=box,
        native_size=[w,h], generation_size=generation_size, requested_long_edge=args.resolution,
        source_full_size=[frame_shape[1],frame_shape[0]],
        full_to_crop=full_to_crop.tolist(), crop_to_full=inverse.tolist(),
        crop_to_generation_pixel_indices=crop_to_generation.tolist(),
        generation_to_full_pixel_indices=(inverse@np.linalg.inv(crop_to_generation)).tolist(),
        source_uv_frame='Original full source photograph, never crop-local or generation-local.',
        full_source_path='../full_source.png', camera_intrinsics_crop=intrinsic.tolist(),
        mask_semantics='White edits; black preserves. Final paste uses stage inpaint_mask, not conditioning_mask.',
        full_stage_mask='../'+name+'_stage_full_mask.png',
        edit_pixels=int(stage_mask.sum()), anchor_pixels=int(anchors.sum()),
        source_unknown_pixels_in_crop=int(crop(arrays['source_inpaint_mask'],box).sum()),
        target_food_pixels_in_crop=int(crop(arrays['target_food_mask'],box).sum()),
        same_protocol_fullframe_base_required=True,
        base_initialization='Crop the completed same-protocol full-frame candidate at bbox_xyxy. For head, overwrite only anchor_mask pixels with material_texture before generation. Supplied conditioning_image is a geometry-guide initialization preview, not the required completed candidate.',
        reference='source_reference.png is the exact selected source crop with non-owned pixels replaced by RGB 127. Its independent source_reference_mask is supplied.',
        reference_recommended=(name == 'head'),
        parent_guide_revision=parent_manifest['guide_revision'],
        parent_manifest_sha256=sha(parent/'guide_manifest.json'),
        prompt=parent_manifest.get('prompt','food'), mode=parent_manifest['mode'],
        images={key:key+'.png' for key in image_files},
        file_hashes={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name not in ['stage_manifest.json','guide_manifest.json']})
    write_json(output/'stage_manifest.json',stage)
    write_json(output/'guide_manifest.json',stage)
    return stage


def prepare_case(parent, output, args):
    output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((parent/'guide_manifest.json').read_text(encoding='utf-8'))
    with np.load(parent/'guide_channels.npz') as payload:
        channels={key:payload[key] for key in payload.files}
    arrays={key:rgb(parent/(key+'.png')) for key in ['source','coarse','material_texture']}
    arrays.update({key:np.asarray(Image.open(parent/(key+'.png')).convert('L')).copy() for key in ['edge','depth']})
    arrays.update({key:mask(parent/(key+'.png')) for key in
        ['target_food_mask','spoon_mask','source_removal_mask','source_inpaint_mask','inpaint_mask']})
    shape=arrays['source'].shape[:2]
    food,spoon=arrays['target_food_mask'],arrays['spoon_mask']
    full_edit=arrays['inpaint_mask']
    scale=max(shape)/640
    target_ownership=channels['target_tolerance'] | dilate(spoon,max(2,round(3*scale)))
    source_full=full_edit & arrays['source_inpaint_mask'] & ~target_ownership
    target_full=full_edit & ~source_full
    food_box=bbox(food)
    food_long=max(food_box[2]-food_box[0],food_box[3]-food_box[1])
    head_window=rectangle(expand(food_box,int(np.ceil(.5*food_long)),shape),shape)
    head_support=food | (spoon & head_window)
    head_box=expand(bbox(head_support),max(12,int(np.ceil(.35*food_long))),shape)
    head_rectangle=rectangle(head_box,shape)
    head_full=target_full & head_rectangle
    handle_full=target_full & ~head_rectangle
    source_box=bbox(source_full)
    source_long=max(source_box[2]-source_box[0],source_box[3]-source_box[1])
    source_box=expand(source_box,max(12,int(np.ceil(.55*source_long))),shape)
    anchor_full=mask(parent/'anchor_mask.png') if (parent/'anchor_mask.png').exists() else np.zeros(shape,bool)
    if np.any(anchor_full & ~head_full):
        raise ValueError('Head crop must include all transported material anchors')
    if np.any(food & ~head_full):
        raise ValueError('Head stage must include the entire transported food')
    if np.any(head_full & source_full) or np.any(handle_full & source_full):
        raise ValueError('Factorized stages must be disjoint')
    if not np.array_equal(head_full | source_full | handle_full, full_edit):
        raise ValueError('Stage masks must exactly partition the original edit footprint')
    for name,array in dict(full_inpaint_mask=full_edit,head_stage_full_mask=head_full,
        source_stage_full_mask=source_full,handle_passthrough_mask=handle_full,
        full_source=arrays['source'],full_coarse=arrays['coarse']).items():
        save(output/(name+'.png'),array)
    shutil.copyfile(parent/'guide_channels.npz',output/'full_guide_channels.npz')
    reference=rgb(parent/'source_reference.png')
    reference_mask=mask(parent/'source_reference_mask.png')
    reference[~reference_mask]=127
    stages={}
    for name,box,stage_mask in [('head',head_box,head_full),('source',source_box,source_full)]:
        stages[name]=build_stage(name,box,stage_mask,output,parent,manifest,channels,
            arrays,anchor_full,reference,reference_mask,args)
    record=dict(case_id=parent.name,variant='factorized_head_source_v1',parent=str(parent.resolve()),
        parent_manifest_sha256=sha(parent/'guide_manifest.json'),
        same_protocol_fullframe_base_required=True,
        head_bbox_xyxy=head_box,source_bbox_xyxy=source_box,
        full_edit_pixels=int(full_edit.sum()),head_edit_pixels=int(head_full.sum()),
        source_edit_pixels=int(source_full.sum()),handle_passthrough_pixels=int(handle_full.sum()),
        original_full_edit_mask_sha256=sha(parent/'inpaint_mask.png'),
        copied_full_edit_mask_sha256=sha(output/'full_inpaint_mask.png'),
        original_geometry_channels_sha256=sha(parent/'guide_channels.npz'),
        copied_geometry_channels_sha256=sha(output/'full_guide_channels.npz'),
        head_omits_source_removal_region=not bool((head_rectangle & source_full).any()),
        stages=stages,
        selection='Head window uses transported food bbox expanded by half its long edge to select the local spoon head, then adds 35% context. Source crop adds 55% context around source-edit support. No image-specific coordinates or generated-output ranking.',
        composition='Start from the completed same-protocol full-frame candidate. Independently generate source/head crop stages and paste each only through its frozen full-stage mask. Keep handle_passthrough_mask from the full-frame candidate. Enforce original source outside full_inpaint_mask at final projection.',
        limitations='Source and head depth/material hypotheses are unchanged. Cropping increases effective target resolution and separates conditioning contexts; it cannot prove ingredient identity, source mass removal, or hidden-surface correctness.',
        generated=False,builder_sha256=sha(Path(__file__)))
    write_json(output/'factorization.json',record)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--indices',type=int,nargs='+')
    parser.add_argument('--resolution',type=int,default=1024)
    args=parser.parse_args()
    if args.input_root.resolve()==args.output.resolve():
        parser.error('Factorized guides require a separate output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for parent in sorted(args.input_root.glob('real_*')):
        if args.indices is not None and int(parent.name.split('_')[1]) not in args.indices:
            continue
        if not (parent/'guide_manifest.json').exists():
            continue
        record=prepare_case(parent,args.output/parent.name,args)
        rows.append(record)
        print(json.dumps({key:record[key] for key in ['case_id','head_bbox_xyxy','source_bbox_xyxy','head_omits_source_removal_region','handle_passthrough_pixels']}),flush=True)
    write_json(args.output/'factorized_manifest.json',dict(cases=rows,generated=False))
    (args.output/'README.md').write_text(
        '# Factorized MLD4 guide protocol\n\n'
        'Requires a completed **same-protocol full-frame candidate** as base, including the full spoon handle.\n\n'
        'Each case contains `head/` and `source/`. Read `stage_manifest.json` for the exact half-open `bbox_xyxy`, native size, and generation size (1024 long edge, dimensions rounded to 32). Crop the base at that box. For the head stage, impose `material_texture.png` only at `anchor_mask.png` before generation. Use `conditioning_mask.png` as the model mask, `edge.png` as structural control, and the owned-only `source_reference.png` if required. The source stage uses no material anchors and does not require a material reference.\n\n'
        'Final crop projection uses stage `inpaint_mask.png`, which retains anchor pixels in its editable domain. Paste using the untouched full-frame `head_stage_full_mask.png` or `source_stage_full_mask.png`; these are disjoint. Preserve `handle_passthrough_mask.png` from the full-frame candidate. Then preserve the original source outside `full_inpaint_mask.png`. Never use a conditioning mask as the final projection mask.\n\n'
        '`guide_channels.npz` contains crop-local depth and masks, crop camera intrinsics, unchanged material IDs, and `target_source_uv` in the **original full source image** coordinates. `stage_anchors.npz` maps native, full, and generation pixel positions. Nearest-neighbor 1024 conditioning/inpaint/anchor mask previews are supplied. `full_guide_channels.npz` and `full_inpaint_mask.png` retain the original full-frame state exactly.\n',encoding='utf-8')


if __name__=='__main__':
    main()
