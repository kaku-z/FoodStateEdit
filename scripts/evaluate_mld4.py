"""CPU audits of actual MLD4 images; none of these metrics certify realism.

The frozen editing envelope is a contract. Guide masks describe intentions,
not the generated image. Optional final observer masks require a matching input
hash. Identity descriptors provide fallible evidence, never material ground
truth. Visual reviews must be authored after viewing the hashed images.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

try:
    import cv2
except ImportError:
    cv2 = None


CRITERIA = {
    'source_identity': 'Same dish, ingredient, garnish and scene; compare source and carried food, not just broad color.',
    'pre_mouth_elevation': 'A meaningful bite is visibly lifted clear of the source and has not reached a mouth.',
    'food_geometry': 'Cohesive food has plausible cut/hidden surfaces; strands have smooth, credible lengths and continuity rather than tiny polygonal chips.',
    'utensil_and_contact': 'One coherent new utensil supports the bite with plausible occlusion/contact, perspective and thickness.',
    'source_removal': 'Removal is commensurate with the lifted bite, with plausible exposed food/container and no outlined, pasted or repeated-texture cavity.',
    'appearance_and_seams': 'Lighting, shadows, focus, edges and texture form a coherent photograph at native scale and detail view.',
    'scene_preservation': 'No extra food, humans, mouth, duplicated utensils, moved dish or unexplained changes in the permitted region.',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path, default=None):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else (default or {})


def average(values):
    return float(np.mean(values)) if np.size(values) else None


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def rgb(path):
    return np.asarray(Image.open(path).convert('RGB'))


def mask(path, shape):
    if not path.exists():
        return None
    arr = np.asarray(Image.open(path).convert('L')) > 0
    if arr.shape != shape:
        raise ValueError(f'Mask dimensions {arr.shape} != image {shape}: {path}')
    return arr


def boundary(m):
    return m & ~ndi.binary_erosion(m)


def mask_metrics(m):
    if m is None:
        return {'status': 'not_available'}
    labels, count = ndi.label(m, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())[1:]
    y, x = np.where(m)
    if not len(x):
        return {'status': 'empty', 'pixels': 0, 'components_8_connected': 0}
    h, w = m.shape
    return {'status': 'measured', 'pixels': int(m.sum()), 'image_fraction': average(m),
            'components_8_connected': int(count), 'components_at_least_8_pixels': int(np.sum(sizes >= 8)),
            'largest_component_fraction': ratio(sizes.max(), m.sum()),
            'holes_pixels': int((ndi.binary_fill_holes(m) & ~m).sum()),
            'bbox_xyxy': [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1],
            'centroid_xy': [float(x.mean()), float(y.mean())],
            'minimum_frame_margin_pixels': int(min(x.min(), y.min(), w - 1 - x.max(), h - 1 - y.max())),
            'boundary_pixels': int(boundary(m).sum())}


def js_distance(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a = a / max(a.sum(), 1e-12); b = b / max(b.sum(), 1e-12)
    mid = (a + b) * .5
    def kl(p):
        keep = p > 0
        return np.sum(p[keep] * np.log2(p[keep] / np.maximum(mid[keep], 1e-12)))
    return float(np.sqrt(max(0, (kl(a) + kl(b)) * .5)))


def descriptors(image, region):
    # Interior reduces silhouette/background contamination. Small bites retain
    # their full mask, explicitly reported, rather than inventing a descriptor.
    interior = ndi.binary_erosion(region, iterations=2)
    if interior.sum() < 32:
        interior = region
    pixels = image[interior].astype(float) / 255
    if len(pixels) < 8:
        return None
    chroma = pixels / np.maximum(pixels.sum(1, keepdims=True), .03)
    hist = np.histogram2d(chroma[:, 0], chroma[:, 1], bins=16, range=((0, 1), (0, 1)))[0]
    gray = image.astype(float) @ np.array([.299, .587, .114])
    padded = np.pad(gray, 1, mode='edge')
    lbp = np.zeros(gray.shape, np.uint8)
    for bit, (dy, dx) in enumerate(((-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1))):
        lbp |= ((padded[1+dy:1+dy+gray.shape[0], 1+dx:1+dx+gray.shape[1]] >= gray).astype(np.uint8) << bit)
    lbp_hist = np.bincount(lbp[interior], minlength=256)
    return {'pixels': int(len(pixels)), 'used_eroded_interior': bool(np.any(interior != region)),
            'chroma_hist': hist.ravel(), 'lbp_hist': lbp_hist,
            'rgb_median_255': np.median(pixels * 255, axis=0).tolist()}


def identity_metrics(source, final, source_region, target_region):
    result = {'scope': 'Appearance evidence within declared regions; affected by view, lighting, generated surfaces and mask error. Not semantic identity certification.'}
    if source_region is None or target_region is None:
        return {**result, 'status': 'missing_regions'}
    a, b = descriptors(source, source_region), descriptors(final, target_region)
    if a is None or b is None:
        return {**result, 'status': 'insufficient_region_pixels'}
    result.update(status='measured', source_descriptor_pixels=a['pixels'], target_descriptor_pixels=b['pixels'],
                  chromaticity_histogram_js_distance=js_distance(a['chroma_hist'], b['chroma_hist']),
                  local_binary_pattern_js_distance=js_distance(a['lbp_hist'], b['lbp_hist']),
                  source_rgb_median_255=a['rgb_median_255'], target_rgb_median_255=b['rgb_median_255'])
    if cv2 is None:
        result['local_feature_matching'] = {'status': 'cv2_not_installed'}
        return result
    cv2.setRNGSeed(417)
    sift = cv2.SIFT_create(nfeatures=1800, contrastThreshold=.018)
    ka, da = sift.detectAndCompute(cv2.cvtColor(source, cv2.COLOR_RGB2GRAY), source_region.astype('uint8') * 255)
    kb, db = sift.detectAndCompute(cv2.cvtColor(final, cv2.COLOR_RGB2GRAY), target_region.astype('uint8') * 255)
    raw_counts = [len(ka), len(kb)]
    def interior_keypoints(keys, desc, region):
        # A center-only mask can match unchanged background just outside food.
        # Require a conservative support margin and report the attrition.
        distance = ndi.distance_transform_edt(region)
        keep = [i for i,k in enumerate(keys) if distance[min(region.shape[0]-1,round(k.pt[1])), min(region.shape[1]-1,round(k.pt[0]))] >= 3*k.size]
        return [keys[i] for i in keep], desc[keep] if desc is not None and keep else None
    ka, da = interior_keypoints(ka, da, source_region)
    kb, db = interior_keypoints(kb, db, target_region)
    matches = {'status': 'insufficient_keypoints', 'source_keypoints': len(ka), 'target_keypoints': len(kb),
               'source_keypoints_before_support_filter': raw_counts[0], 'target_keypoints_before_support_filter': raw_counts[1],
               'required_region_interior_margin_keypoint_size_multiplier': 3,
               'ratio_threshold': .75, 'mutual_ratio_matches': 0, 'affine_inliers': None, 'pairs': [],
               'scope': 'Mutual Lowe-ratio SIFT matches; RANSAC permits affine change. Few matches are inconclusive and repeated food textures can cause false matches.'}
    if da is not None and db is not None and min(len(da), len(db)) >= 2:
        bf = cv2.BFMatcher(cv2.NORM_L2)
        ab = [m for pair in bf.knnMatch(da, db, k=2) if len(pair) == 2 for m, n in [pair] if m.distance < .75 * n.distance]
        ba = {(m.trainIdx, m.queryIdx) for pair in bf.knnMatch(db, da, k=2) if len(pair) == 2 for m, n in [pair] if m.distance < .75 * n.distance}
        mutual = [m for m in ab if (m.queryIdx, m.trainIdx) in ba]
        points_a = np.array([ka[m.queryIdx].pt for m in mutual], dtype=np.float32)
        points_b = np.array([kb[m.trainIdx].pt for m in mutual], dtype=np.float32)
        inliers = None
        if len(mutual) >= 3:
            _, inliers = cv2.estimateAffine2D(points_a, points_b, method=cv2.RANSAC,
                                            ransacReprojThreshold=3, maxIters=2000, confidence=.99)
        matches.update(status='measured', mutual_ratio_matches=len(mutual),
                       affine_inliers=int(inliers.sum()) if inliers is not None else None,
                       pairs=[{'source_xy': p.tolist(), 'target_xy': q.tolist(),
                               'affine_inlier': bool(inliers[i, 0]) if inliers is not None else None}
                              for i, (p, q) in enumerate(zip(points_a, points_b))])
    result['local_feature_matching'] = matches
    return result


def boundary_diagnostics(source, final, edit, guide_masks):
    gray = final.astype(float) @ np.array([.299, .587, .114])
    grad = np.hypot(ndi.sobel(gray, 0), ndi.sobel(gray, 1)) / 8
    source_gray = source.astype(float) @ np.array([.299, .587, .114])
    source_grad = np.hypot(ndi.sobel(source_gray, 0), ndi.sobel(source_gray, 1)) / 8
    ring = ndi.binary_dilation(edit, iterations=2) & ~ndi.binary_erosion(edit, iterations=2)
    result = {'scope': 'Seam and edge diagnostics, not artifact detectors or acceptance scores.',
              'edit_boundary_ring_pixels': int(ring.sum()),
              'edit_boundary_final_gradient_mean': average(grad[ring]),
              'edit_boundary_source_gradient_mean': average(source_grad[ring]),
              'edit_boundary_positive_gradient_excess_mean': average(np.maximum(0, grad-source_grad)[ring])}
    if cv2 is None:
        return result
    edges = cv2.Canny(final, 60, 150) > 0
    distance = ndi.distance_transform_edt(~edges)
    for name, region in guide_masks.items():
        if region is not None and region.any():
            d = distance[boundary(region)]
            result[name + '_guide_boundary_to_final_edge_mean_px'] = average(d)
            result[name + '_guide_boundary_to_final_edge_p95_px'] = float(np.quantile(d, .95))
    return result


def relative_geometry(food, utensil, removed):
    result = {'scope': '2D silhouette evidence only. Screen separation is not measured 3D height; mask components do not prove object topology.'}
    if food is not None and food.any():
        if removed is not None and removed.any():
            fy, fx = np.where(food); sy, sx = np.where(removed)
            result.update(food_removed_overlap_pixels=int((food & removed).sum()),
                          food_removed_minimum_distance_px=float(ndi.distance_transform_edt(~removed)[food].min()),
                          source_to_food_centroid_upward_px=float(sy.mean()-fy.mean()),
                          source_to_food_centroid_displacement_px=float(np.hypot(sy.mean()-fy.mean(), sx.mean()-fx.mean())),
                          food_to_removed_projected_area_ratio=ratio(food.sum(), removed.sum()))
        if utensil is not None and utensil.any():
            distance = ndi.distance_transform_edt(~utensil)
            result.update(food_utensil_overlap_pixels=int((food & utensil).sum()),
                          food_to_utensil_minimum_distance_px=float(distance[food].min()),
                          food_boundary_within_3px_of_utensil_fraction=average(distance[boundary(food)] <= 3))
    return result


def review_status(review, case_id, hashes):
    entry = review.get('cases', {}).get(case_id, {})
    if not entry:
        return {'status': 'not_reviewed', 'criteria': {k: 'unreviewed' for k in CRITERIA}}
    supplied = entry.get('artifact_sha256', {})
    mismatch = [name for name, value in hashes.items() if supplied.get(name) != value]
    criteria = entry.get('criteria', {})
    def value(name):
        item = criteria.get(name, 'unreviewed')
        return item.get('status', 'unreviewed') if isinstance(item, dict) else item
    states = {name: value(name) for name in CRITERIA}
    observer = entry.get('observer', review.get('observer'))
    missing_notes = [name for name in CRITERIA if not isinstance(criteria.get(name),dict)
                     or not str(criteria[name].get('note','')).strip()]
    complete = (all(s in ('pass', 'fail', 'uncertain') for s in states.values()) and not missing_notes
                and bool(observer) and not str(observer).startswith('REQUIRED:'))
    if mismatch:
        status = 'stale_or_unbound_review'
    elif not complete:
        status = 'incomplete_review'
    elif 'fail' in states.values():
        status = 'reviewed_with_failures'
    elif 'uncertain' in states.values():
        status = 'reviewed_with_uncertainty'
    else:
        status = 'reviewer_all_criteria_pass'
    return {'status': status, 'mismatched_or_missing_hashes': mismatch, 'criteria': criteria,
            'missing_criterion_evidence_notes': missing_notes,
            'observer': observer, 'notes': entry.get('notes', ''),
            'scope': 'Named visual observer judgment, not an automated or blinded realism rating.'}


def evaluate_case(folder, review, baseline_root=None):
    source, final = rgb(folder/'source.png'), rgb(folder/'final.png')
    if source.shape != final.shape:
        raise ValueError(f'Source/final dimensions mismatch: {folder}')
    shape = source.shape[:2]
    edit = mask(folder/'edit_mask.png', shape)
    if edit is None:
        raise ValueError(f'Missing frozen edit_mask.png: {folder}')
    state = read_json(folder/'reform_state.json')
    removed = mask(folder/'source_removed_mask.png', shape)
    food = mask(folder/'moved_food_mask.png', shape)
    spoon = mask(folder/'spoon_mask.png', shape)
    guide_path = folder/'guide.png'
    if not guide_path.exists():
        guide_path = folder/'deformed.png'
    artifact_paths = {name: folder/name for name in ('source.png','final.png','edit_mask.png','moved_food_mask.png','spoon_mask.png','source_removed_mask.png') if (folder/name).exists()}
    if guide_path.exists():
        artifact_paths[guide_path.name] = guide_path
    hashes = {name: sha256(path) for name, path in artifact_paths.items()}
    delta = np.abs(final.astype(float)-source.astype(float))
    changed = np.any(delta > 0, axis=-1)
    outside = ~edit
    image = {'width': shape[1], 'height': shape[0], 'declared_edit_fraction': average(edit),
             'actual_changed_fraction': average(changed), 'changed_pixels': int(changed.sum()),
             'outside_edit_changed_pixels': int((changed & outside).sum()),
             'outside_edit_mae_255': average(delta[outside]),
             'outside_edit_max_255': float(delta[outside].max()) if outside.any() else None,
             'inside_edit_changed_fraction': average(changed[edit]),
             'changed_region': mask_metrics(changed)}
    for name, region in (('guide_food',food), ('guide_utensil',spoon), ('source_removal',removed)):
        if region is not None:
            image[name+'_source_to_final_mae_255'] = average(delta[region])
    if guide_path.exists():
        guide = rgb(guide_path)
        if guide.shape != source.shape:
            raise ValueError(f'Guide dimensions mismatch: {guide_path}')
        appearance_delta = np.abs(final.astype(float)-guide.astype(float))
        image['final_vs_guide_changed_pixels'] = int(np.any(appearance_delta>0,axis=-1).sum())
        image['final_vs_guide_mae_255'] = average(appearance_delta)
        image['guide_food_changed_pixels_from_guide'] = int((np.any(appearance_delta>0,axis=-1)&food).sum()) if food is not None else None
    frozen_hash = state.get('frozen_edit_mask_sha256')
    checks = {'outside_frozen_edit_unchanged': bool(not np.any(changed & outside)),
              'source_and_final_dimensions_match': True,
              'frozen_edit_hash_matches_record': hashes['edit_mask.png'] == frozen_hash if frozen_hash else None,
              'source_hash_matches_record': hashes['source.png'] == state['source_sha256'] if state.get('source_sha256') else None,
              'edit_does_not_cover_entire_frame': bool(outside.any())}
    baseline_path = Path(state['baseline_final_path']) if state.get('baseline_final_path') else (baseline_root/folder.name/'final.png' if baseline_root else None)
    baseline_source = baseline_path.parent/'source.png' if baseline_path else None
    if baseline_source and baseline_source.exists():
        checks['source_matches_baseline_pixels'] = bool(np.array_equal(source,rgb(baseline_source)))
    observer_meta = read_json(folder/'observation_metadata.json')
    observer_valid = observer_meta.get('input_final_sha256') == hashes['final.png']
    observed_food = mask(folder/'observed_final_food_mask.png', shape)
    observed_spoon = mask(folder/'observed_final_utensil_mask.png', shape)
    observed = {'status': 'hash_bound_observer_masks' if observer_valid else 'missing_or_unverified_final_observer',
                'metadata': observer_meta,
                'scope': 'Independent image segmentation remains fallible; it is not measured physical topology.'}
    target_for_identity = observed_food if observer_valid and observed_food is not None else food
    if observer_valid:
        observed.update(food=mask_metrics(observed_food), utensil=mask_metrics(observed_spoon),
                        geometry=relative_geometry(observed_food,observed_spoon,removed))
        for name, region, guide_mask in (('food',observed_food,food), ('utensil',observed_spoon,spoon)):
            if region is not None:
                observed[name+'_outside_edit_pixels'] = int((region & ~edit).sum())
                if guide_mask is not None:
                    observed[name+'_guide_iou'] = ratio((region & guide_mask).sum(), (region | guide_mask).sum())
    identity = identity_metrics(source,final,removed,target_for_identity)
    identity['target_region_provenance'] = 'hash_bound_final_observer' if observer_valid and observed_food is not None else 'intended_guide_region_not_final_segmentation'
    candidates = []
    for path in sorted((folder/'candidates').glob('*/projected.png')):
        candidate = rgb(path)
        if candidate.shape == source.shape:
            cd = np.abs(candidate.astype(float)-source.astype(float))
            candidates.append({'variant': path.parent.name, 'path': str(path.resolve()), 'sha256': sha256(path),
                               'outside_edit_changed_pixels': int(np.any(cd>0,axis=-1)[outside].sum()),
                               'inside_edit_mae_255': average(cd[edit])})
    return {'case_id': folder.name, 'folder': str(folder.resolve()), 'state': state,
            'baseline_final_path': str(baseline_path.resolve()) if baseline_path and baseline_path.exists() else None,
            'artifact_sha256': hashes, 'contract_checks': checks,
            'measured_contract_status': 'fail' if False in checks.values() else ('incomplete_provenance' if None in checks.values() else 'pass'),
            'image': image, 'identity_diagnostics': identity,
            'guide_masks': {'food': mask_metrics(food),'utensil':mask_metrics(spoon),'removed':mask_metrics(removed),
                            'scope':'Generator intentions only; not final image topology.'},
            'guide_geometry': relative_geometry(food,spoon,removed), 'observed_final': observed,
            'boundary_diagnostics': boundary_diagnostics(source,final,edit,{'food':food,'utensil':spoon,'removed':removed}),
            'visual_review': review_status(review,folder.name,hashes), 'candidates': candidates}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--case-root',type=Path)
    parser.add_argument('--baseline-root',type=Path, help='MLD3 real/ directory, for provenance and comparison only')
    parser.add_argument('--review',type=Path)
    parser.add_argument('--label',default='MLD4 reconstruction',help='Honest image-phase label, e.g. E3 diagnostic reference for validation')
    args=parser.parse_args()
    root=args.root.resolve(); case_root=(args.case_root or root/'real').resolve()
    out=root/'evaluation'; out.mkdir(parents=True,exist_ok=True)
    review_path=args.review or out/'visual_review.json'
    review=read_json(review_path)
    rows=[]; pending=[]; errors=[]
    for folder in sorted(p for p in case_root.glob('real_*') if p.is_dir()):
        missing=[name for name in ('source.png','final.png','edit_mask.png') if not (folder/name).exists()]
        if missing:
            pending.append({'case_id':folder.name,'missing':missing}); continue
        try:
            rows.append(evaluate_case(folder,review,args.baseline_root))
        except Exception as exc:
            errors.append({'case_id':folder.name,'error':str(exc)})
    summary={'status':'evaluated_actual_artifacts' if rows else 'no_complete_outputs_yet',
             'case_count':len(rows),'input_case_root':str(case_root),'phase_label':args.label,'pending':pending,'errors':errors,
             'scope':{'hard_contract':'Source and frozen edit locality; zero outside edits is required, but not evidence of realism.',
                      'identity':'Descriptor distances and sparse feature matches are diagnostics with view/lighting/texture ambiguity; no identity pass threshold is asserted.',
                      'geometry':'Guide masks cannot certify final food/utensil topology. Hash-bound independent final masks are reported separately.',
                      'visual':'Native image review is necessary. No invented ratings, no aggregate realism score, no perfection claim.',
                      'data':'Previously inspected development images are exploratory, not a blind holdout. Hidden surfaces have no measured truth.'},
             'review_path':str(review_path.resolve()),'criteria':CRITERIA,
             'counts':{'locality_pass':sum(r['contract_checks']['outside_frozen_edit_unchanged'] for r in rows),
                       'contract_pass':sum(r['measured_contract_status']=='pass' for r in rows),
                       'contract_fail':sum(r['measured_contract_status']=='fail' for r in rows),
                       'hash_bound_final_observer':sum(r['observed_final']['status']=='hash_bound_observer_masks' for r in rows),
                       'fresh_complete_visual_review':sum(r['visual_review']['status'].startswith('reviewed_') or r['visual_review']['status']=='reviewer_all_criteria_pass' for r in rows)},
             'cases':rows}
    (out/'reform_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'reform_cases.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n',encoding='utf-8')
    flat=[{'case_id':r['case_id'],'contract_status':r['measured_contract_status'],
           'visual_review_status':r['visual_review']['status'],
           **{k:v for k,v in r['image'].items() if not isinstance(v,dict)}} for r in rows]
    if flat:
        names=list(dict.fromkeys(k for row in flat for k in row))
        with (out/'reform_cases.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=names); writer.writeheader(); writer.writerows(flat)
    template={'observer':'REQUIRED: name and review method','blind':False,
              'instructions':'View actual native images and gallery detail panels. Replace unreviewed with pass/fail/uncertain and explain evidence. Do not copy template hashes after an image changes without reviewing it again.',
              'cases':{r['case_id']:{'artifact_sha256':r['artifact_sha256'],
                                   'criteria':{name:{'status':'unreviewed','note':''} for name in CRITERIA},
                                   'notes':''} for r in rows}}
    (out/'visual_review_template.json').write_text(json.dumps(template,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'cases':len(rows),'pending':len(pending),'errors':errors,'summary':str(out/'reform_summary.json')},ensure_ascii=False))


if __name__=='__main__':
    main()
