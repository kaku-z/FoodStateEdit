"""Location-bound source-material anchors for a controlled MLD4 ablation.

Copy an existing guide bundle, retaining its full final inpaint mask. A separate
conditioning mask fixes sparse high-confidence transported source samples while
the generator reconstructs the surrounding food. No model or geometry is run.
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


def erode(value, radius=1):
    return np.asarray(Image.fromarray(value.astype(np.uint8)*255).filter(
        ImageFilter.MinFilter(2*radius+1))) > 0


def disk(radius):
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
    return np.column_stack([yy[yy*yy+xx*xx <= radius*radius],
                            xx[yy*yy+xx*xx <= radius*radius]])


def distributed_anchors(food, confidence, fraction=.20, threshold=.75):
    """Greedy farthest-point coverage, using only observed food interior."""
    h, w = food.shape
    radius = int(np.clip(round(max(h, w)*3/640), 2, 3))
    known = food & (confidence >= threshold)
    eligible = known & erode(food)
    offsets = disk(radius)
    padded = np.pad(eligible.astype(np.int16), radius)
    counts = sum(padded[radius+dy:radius+dy+h, radius+dx:radius+dx+w]
                 for dy, dx in offsets)
    # Clipping a disc to known material is allowed, but single-pixel pins are not.
    candidate = eligible & (counts >= max(3, int(.35*len(offsets))))
    coordinates = np.column_stack(np.where(candidate))
    anchor = np.zeros_like(food)
    centers, patch_counts = [], []
    desired = int(round(fraction*known.sum()))
    maximum = max(desired, int(np.floor(.25*known.sum())))
    if len(coordinates) and desired:
        center = coordinates.mean(axis=0)
        first = np.argmin(np.sum((coordinates-center)**2, axis=1))
        distances = np.full(len(coordinates), np.inf)
        selected = np.zeros(len(coordinates), bool)
        choice = int(first)
        while int(anchor.sum()) < desired:
            y, x = coordinates[choice]
            location = coordinates[choice]
            py, px = (location + offsets).T
            valid = (py>=0)&(py<h)&(px>=0)&(px<w)
            py, px = py[valid], px[valid]
            take = eligible[py, px] & ~anchor[py, px]
            py, px = py[take], px[take]
            selected[choice] = True
            if len(py) >= 3 and int(anchor.sum())+len(py) <= maximum:
                anchor[py, px] = True
                centers.append([int(x), int(y)])
                patch_counts.append(len(py))
                distances = np.minimum(distances, np.sum((coordinates-location)**2, axis=1))
            allowed = ~selected & (distances >= (2*radius+1)**2)
            if not allowed.any():
                break
            # Distinct locations take priority; patch support and confidence
            # break geometric ties without consulting any generated image.
            weights = counts[coordinates[:,0], coordinates[:,1]] / len(offsets)
            weights *= confidence[coordinates[:,0], coordinates[:,1]]
            score = distances*(.8+.2*weights)
            score[~allowed] = -np.inf
            choice = int(np.argmax(score))
    return anchor, dict(radius_original_pixels=radius, requested_known_food_fraction=fraction,
        minimum_material_confidence=threshold, known_food_pixels=int(known.sum()),
        eligible_interior_pixels=int(eligible.sum()), requested_anchor_pixels=desired,
        anchor_pixels=int(anchor.sum()),
        coverage_known_food=float(anchor.sum()/max(1, known.sum())),
        coverage_all_target_food=float(anchor.sum()/max(1, food.sum())),
        coverage_eligible_interior=float(anchor.sum()/max(1, eligible.sum())),
        centers_xy=centers, pixels_per_anchor=patch_counts,
        spacing_between_centers_pixels=2*radius+1,
        target_coverage_reached=bool(.15 <= anchor.sum()/max(1, known.sum()) <= .25),
        selection='Deterministic farthest-point locations, initialized nearest the eligible-material centroid. Discs are clipped to confidence>=threshold and one-pixel-eroded target food. No generated output is inspected.')


def sample_source(source, uv, pixel_centers):
    uv = uv-.5 if pixel_centers else uv
    h, w = source.shape[:2]
    x, y = np.clip(uv[:,0],0,w-1), np.clip(uv[:,1],0,h-1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0+1,w-1), np.minimum(y0+1,h-1)
    fx, fy = (x-x0)[:,None], (y-y0)[:,None]
    value = ((1-fx)*(1-fy)*source[y0,x0]+fx*(1-fy)*source[y0,x1]
             +(1-fx)*fy*source[y1,x0]+fx*fy*source[y1,x1])
    return np.clip(value,0,255).astype(np.uint8)


def prepare_case(folder, output, args):
    if folder.resolve() == output.resolve():
        raise ValueError('Anchor ablation requires a separate output directory')
    original_manifest = json.loads((folder/'guide_manifest.json').read_text(encoding='utf-8'))
    full_edit = mask(folder/'inpaint_mask.png')
    food = mask(folder/'target_food_mask.png')
    coarse, texture, source = rgb(folder/'coarse.png'), rgb(folder/'material_texture.png'), rgb(folder/'source.png')
    with np.load(folder/'guide_channels.npz') as payload:
        confidence = payload['material_confidence'].copy()
        source_uv = payload['target_source_uv'].copy()
        material_id = payload['target_material_id'].copy()
    anchors, record = distributed_anchors(food, confidence, args.fraction, args.confidence)
    if np.any(anchors & ~full_edit):
        raise ValueError('Transported material anchors must lie inside the full edit footprint')
    conditioning = coarse.copy()
    conditioning[anchors] = texture[anchors]
    conditioning_mask = full_edit & ~anchors
    direct_sample = sample_source(source, source_uv[anchors], original_manifest['mode'] != 'strand')
    sample_delta = np.abs(direct_sample.astype(int)-texture[anchors].astype(int))
    exact = not bool(sample_delta.any())
    if not exact:
        raise ValueError('Material texture anchors no longer equal direct source-UV samples')
    shutil.copytree(folder, output, dirs_exist_ok=True)
    Image.fromarray(conditioning).save(output/'conditioning_image.png')
    Image.fromarray(conditioning_mask.astype(np.uint8)*255).save(output/'conditioning_mask.png')
    Image.fromarray(anchors.astype(np.uint8)*255).save(output/'anchor_mask.png')
    ys, xs = np.where(anchors)
    np.savez_compressed(output/'anchors.npz', anchor_mask=anchors,
        target_pixel_xy=np.column_stack([xs,ys]), source_uv=source_uv[anchors],
        material_id=material_id[anchors], material_confidence=confidence[anchors],
        source_sample_rgb=texture[anchors], anchor_center_xy=np.asarray(record['centers_xy']).reshape(-1,2))
    files = ['conditioning_image.png','conditioning_mask.png','anchor_mask.png','anchors.npz']
    preserved = ['source.png','coarse.png','edge.png','depth.png','inpaint_mask.png',
                 'source_removal_mask.png','target_food_mask.png','spoon_mask.png','guide_channels.npz','material_texture.png']
    record.update(case_id=folder.name, variant='sparse_owned_material_anchors_v1',
        input_folder=str(folder.resolve()), input_manifest_sha256=sha(folder/'guide_manifest.json'),
        input_revision=original_manifest['guide_revision'],
        confidence_min_at_anchors=float(confidence[anchors].min()) if anchors.any() else None,
        confidence_mean_at_anchors=float(confidence[anchors].mean()) if anchors.any() else None,
        estimated_disc_radius_at_1024=1024*record['radius_original_pixels']/max(food.shape),
        exact_source_uv_sample_proof=dict(equal=True, max_absolute_uint8_difference=int(sample_delta.max()) if sample_delta.size else 0,
            anchored_source_sample_sha256=hashlib.sha256(direct_sample.tobytes()).hexdigest(),
            conditioning_anchor_sha256=hashlib.sha256(conditioning[anchors].tobytes()).hexdigest()),
        preserved_files_sha256={name:sha(output/name) for name in preserved},
        preserved_files_equal=all(sha(folder/name)==sha(output/name) for name in preserved),
        file_hashes={name:sha(output/name) for name in files},
        full_edit_pixels=int(full_edit.sum()), conditioning_edit_pixels=int(conditioning_mask.sum()),
        generator_contract='Use conditioning_image.png and conditioning_mask.png for model inputs. Keep inpaint_mask.png as the complete allowed edit region for final projection. Anchor preservation during generation is a hypothesis to verify on output, not guaranteed by this preprocessing.',
        limitations='Sparse source samples constrain local food appearance and location. They do not establish hidden surfaces, ingredient identity, conservation of physical mass, or recovery of missing source material.',
        builder_sha256=sha(Path(__file__)))
    (output/'anchors.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    manifest = dict(original_manifest)
    manifest.update(parent_guide_revision=original_manifest['guide_revision'],
        guide_revision='source_material_anchors_v1', conditioning_image='conditioning_image.png',
        conditioning_mask='conditioning_mask.png', final_projection_mask='inpaint_mask.png',
        material_anchor_record='anchors.json', material_anchors=record)
    manifest['images'] = dict(manifest['images'], conditioning_image='conditioning_image.png',
        conditioning_mask='conditioning_mask.png', anchor_mask='anchor_mask.png')
    manifest['file_hashes'] = dict(manifest['file_hashes'], **{name:sha(output/name) for name in files if name.endswith('.png')})
    (output/'guide_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--indices',type=int,nargs='+')
    parser.add_argument('--fraction',type=float,default=.20)
    parser.add_argument('--confidence',type=float,default=.75)
    args = parser.parse_args()
    if not .15 <= args.fraction <= .25:
        parser.error('--fraction must remain within the specified 15-25 percent range')
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for folder in sorted(args.input_root.glob('real_*')):
        if args.indices is not None and int(folder.name.split('_')[1]) not in args.indices:
            continue
        if not (folder/'guide_manifest.json').exists():
            continue
        record=prepare_case(folder,args.output/folder.name,args)
        rows.append(record)
        print(json.dumps({key:record[key] for key in ['case_id','anchor_pixels','coverage_known_food','coverage_all_target_food','target_coverage_reached']}),flush=True)
    (args.output/'anchors_manifest.json').write_text(json.dumps(dict(cases=rows,generated=False),indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
