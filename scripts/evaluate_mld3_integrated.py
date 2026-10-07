"""Audit the executed MLD3 transport and ablations, without inventing physical GT.

Inputs are actual per-case images, transport.npz and state.json emitted by the
integrated pipeline. ID/UV/color consistency is a construction property; mesh
volume, support and strain are model-space proxies. A visual review is separate.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


VARIANTS = {'rigid': 'rigid_xyz', 'deformed': 'target_xyz',
            'no_contact': 'no_contact_xyz', 'no_gravity': 'no_gravity_xyz'}


def mean(value):
    return float(np.mean(value)) if np.size(value) else None


def bilinear_ownership(mask, sampling_uv):
    """Independently inspect the nonzero-weight texels of each stored UV."""
    base = np.floor(sampling_uv).astype(int)
    fraction = sampling_uv-base
    owned = np.ones(len(base),dtype=bool)
    h,w = mask.shape
    for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
        pixels = base+[dx,dy]
        weight = (fraction[:,0] if dx else 1-fraction[:,0])*(fraction[:,1] if dy else 1-fraction[:,1])
        active = weight!=0
        valid = (pixels[:,0]>=0)&(pixels[:,0]<w)&(pixels[:,1]>=0)&(pixels[:,1]<h)
        safe_x,safe_y = np.clip(pixels[:,0],0,w-1),np.clip(pixels[:,1],0,h-1)
        owned[active] &= valid[active]&mask[safe_y[active],safe_x[active]]
    return owned


def mesh_measure(vertices, faces):
    """Unsigned oriented volume only has its usual meaning for a closed mesh."""
    triangles = vertices[faces] - vertices.mean(0)
    a, b, c = triangles.transpose(1, 0, 2)
    volume = abs(float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6))
    area = float(np.linalg.norm(np.cross(b-a, c-a), axis=-1).sum() / 2)
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True, return_inverse=True)
    orientation_balance = np.bincount(inverse, weights=np.sign(directed[:, 1]-directed[:, 0]))
    triangle_areas = np.linalg.norm(np.cross(b-a, c-a), axis=-1)*.5
    return {'oriented_mesh_volume_abs_proxy': volume, 'surface_area_proxy': area,
            'unreferenced_vertex_count': int(len(vertices)-len(np.unique(faces))),
            'face_referenced_vertex_fraction': len(np.unique(faces))/len(vertices),
            'edge_manifold_closed': bool(np.all(counts == 2)),
            'edge_winding_consistent': bool(np.all(orientation_balance == 0)),
            'boundary_edge_count': int(np.sum(counts == 1)),
            'nonmanifold_edge_count': int(np.sum(counts > 2)),
            'triangle_degenerate_fraction': mean(triangle_areas < max(area, 1e-12)*1e-12)}


def read_spoon_ply(path):
    """Read the actual triangular binary PLY emitted by the pipeline's exporter."""
    types = {'float': '<f4', 'double': '<f8', 'uchar': 'u1', 'int': '<i4'}
    with path.open('rb') as f:
        lines = []
        while True:
            line = f.readline().decode('ascii').strip(); lines.append(line)
            if line == 'end_header':
                break
        assert 'format binary_little_endian 1.0' in lines
        vertex_count = int(next(x for x in lines if x.startswith('element vertex ')).split()[-1])
        face_count = int(next(x for x in lines if x.startswith('element face ')).split()[-1])
        v_start = next(i for i,x in enumerate(lines) if x.startswith('element vertex '))+1
        v_end = next(i for i,x in enumerate(lines) if x.startswith('element face '))
        properties = [(p.split()[2], types[p.split()[1]]) for p in lines[v_start:v_end] if p.startswith('property ')]
        vertices = np.fromfile(f, dtype=np.dtype(properties), count=vertex_count)
        faces = np.fromfile(f, dtype=np.dtype([('count', 'u1'), ('indices', '<i4', (3,))]), count=face_count)
        assert np.all(faces['count']==3)
    return np.column_stack([vertices[n] for n in ('x','y','z')]).astype(float), faces['indices']


def spoon_surface_data(folder, state):
    vertices, faces = read_spoon_ply(folder/'spoon.ply')
    axes = np.asarray(state['spoon']['spoon_axes_camera'])
    local = vertices @ axes
    triangles = local[faces]
    normals = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
    # Keep the real mesh's upper surface, including bowl/neck/handle blend.
    top = normals[:,2] > .5*np.linalg.norm(normals, axis=-1)
    triangles = triangles[top]
    return {'axes': axes, 'triangles': triangles,
            'tree': cKDTree(triangles[:,:,:2].mean(1)), 'topology': mesh_measure(vertices, faces)}


def mesh_support_metrics(vertices, extent, support):
    local = vertices @ support['axes']; tri = support['triangles']
    _, nearest = support['tree'].query(local[:,:2], k=min(32,len(tri)))
    candidates = tri[nearest]
    a, b, c = candidates.transpose(2,0,1,3)
    d1, d2 = b[:,:,:2]-a[:,:,:2], c[:,:,:2]-a[:,:,:2]
    p = local[:,None,:2]-a[:,:,:2]
    det = d1[:,:,0]*d2[:,:,1]-d1[:,:,1]*d2[:,:,0]
    u = (p[:,:,0]*d2[:,:,1]-p[:,:,1]*d2[:,:,0])/det
    v = (d1[:,:,0]*p[:,:,1]-d1[:,:,1]*p[:,:,0])/det
    inside_tri = (u>=-1e-6)&(v>=-1e-6)&(u+v<=1+1e-6)
    z = a[:,:,2]+u*(b[:,:,2]-a[:,:,2])+v*(c[:,:,2]-a[:,:,2])
    z[~inside_tri] = -np.inf
    height = z.max(1); inside = np.isfinite(height)
    gap = local[inside,2]-height[inside]
    return {'actual_spoon_mesh_domain_vertex_fraction': mean(inside),
        'actual_spoon_mesh_min_gap_bite_extent': float(gap.min())/extent if len(gap) else None,
        'actual_spoon_mesh_max_penetration_bite_extent': float(np.maximum(0,-gap).max())/extent if len(gap) else None,
        'actual_spoon_mesh_contact_fraction_tolerance_001_bite_extent': mean(np.abs(gap)<.001*extent)}


def tube_centerline_metrics(rest, target, edges, sides):
    """Recover the executed ring centers from actual stable tube mesh vertices."""
    assert len(rest)%sides==0
    source_rings, target_rings = rest.reshape(-1,sides,3), target.reshape(-1,sides,3)
    source_nodes, target_nodes = source_rings.mean(1), target_rings.mean(1)
    node_edges = np.unique(np.sort(edges//sides, axis=1), axis=0)
    node_edges = node_edges[node_edges[:,0]!=node_edges[:,1]]
    s_radius = np.linalg.norm(source_rings-source_nodes[:,None], axis=-1).mean(1)
    t_radius = np.linalg.norm(target_rings-target_nodes[:,None], axis=-1).mean(1)
    s_length = np.linalg.norm(source_nodes[node_edges[:,0]]-source_nodes[node_edges[:,1]], axis=-1)
    t_length = np.linalg.norm(target_nodes[node_edges[:,0]]-target_nodes[node_edges[:,1]], axis=-1)
    strain = np.abs(t_length/s_length-1)
    def volume(radius, length):
        r1, r2 = radius[node_edges[:,0]], radius[node_edges[:,1]]
        return float(np.sum(np.pi/3*length*(r1*r1+r1*r2+r2*r2)))
    source_volume, target_volume = volume(s_radius,s_length), volume(t_radius,t_length)
    return {'tube_ring_sides': sides, 'centerline_node_count': int(len(source_nodes)),
        'centerline_edge_count': int(len(node_edges)),
        'centerline_edge_strain_abs_mean': mean(strain),
        'centerline_edge_strain_abs_p95': float(np.quantile(strain,.95)),
        'centerline_edge_strain_abs_max': float(strain.max()),
        'ring_radius_relative_change_abs_mean': mean(np.abs(t_radius/s_radius-1)),
        'centerline_frustum_volume_proxy_relative_drift': abs(target_volume-source_volume)/max(source_volume,1e-12)}


def geometry_metrics(rest, rigid, vertices, edges, faces, arrays, prefix):
    rest_length = np.linalg.norm(rest[edges[:, 0]]-rest[edges[:, 1]], axis=-1)
    length = np.linalg.norm(vertices[edges[:, 0]]-vertices[edges[:, 1]], axis=-1)
    keep = rest_length > 1e-12
    strain = length[keep]/rest_length[keep]-1
    extent = float(np.linalg.norm(np.ptp(rest, axis=0)))
    delta = vertices-rigid
    centroid_delta = delta.mean(0)
    displacement = np.linalg.norm(delta, axis=-1)
    shape_delta = np.linalg.norm(delta-centroid_delta, axis=-1)
    rest_measure = mesh_measure(rest, faces)
    measure = mesh_measure(vertices, faces)
    metrics = {**measure, 'vertex_count': int(len(vertices)),
        'edge_strain_abs_mean': mean(np.abs(strain)),
        'edge_strain_abs_p95': float(np.quantile(np.abs(strain), .95)),
        'edge_strain_abs_max': float(np.max(np.abs(strain))),
        'target_vs_rigid_displacement_mean_bite_extent': mean(displacement)/extent,
        'target_vs_rigid_displacement_max_bite_extent': float(displacement.max())/extent,
        'centroid_settling_displacement_bite_extent': float(np.linalg.norm(centroid_delta))/extent,
        'shape_deformation_mean_bite_extent': mean(shape_delta)/extent,
        'shape_deformation_rms_bite_extent': float(np.sqrt(np.mean(shape_delta**2)))/extent,
        'shape_deformation_max_bite_extent': float(shape_delta.max())/extent,
        'mesh_volume_relative_drift': abs(measure['oriented_mesh_volume_abs_proxy']-
            rest_measure['oriented_mesh_volume_abs_proxy'])/max(rest_measure['oriented_mesh_volume_abs_proxy'], 1e-12),
        'area_relative_drift': abs(measure['surface_area_proxy']-
            rest_measure['surface_area_proxy'])/max(rest_measure['surface_area_proxy'], 1e-12)}
    gap_key, inside_key = prefix+'_support_gap', prefix+'_support_inside'
    if gap_key in arrays:
        inside = arrays[inside_key].astype(bool)
        gap = arrays[gap_key][inside]
        metrics['support_domain_vertex_count'] = int(inside.sum())
        metrics['support_domain_vertex_fraction'] = mean(inside)
        metrics['support_min_gap_bite_extent'] = float(gap.min())/extent if len(gap) else None
        metrics['support_penetration_mean_bite_extent'] = mean(np.maximum(0, -gap))/extent if len(gap) else None
        metrics['support_penetration_max_bite_extent'] = float(np.maximum(0, -gap).max())/extent if len(gap) else None
        metrics['support_contact_fraction_tolerance_001_bite_extent'] = mean(np.abs(gap)<.001*extent)
    return metrics


def evaluate_case(folder):
    state = json.loads((folder/'state.json').read_text(encoding='utf-8'))
    arrays = dict(np.load(folder/'transport.npz', allow_pickle=False))
    source = np.asarray(Image.open(folder/'source.png').convert('RGB')).astype(float)
    edit = np.asarray(Image.open(folder/'edit_mask.png').convert('L')) > 0
    source_ids, carried, remaining = (arrays[k].astype(np.int64) for k in
                                    ('source_id', 'carried_ids', 'remaining_ids'))
    target_ids = arrays['target_ids'].astype(np.int64)
    source_set, carry_set, remain_set, target_set = map(set,
        (source_ids.tolist(), carried.tolist(), remaining.tolist(), target_ids.tolist()))
    allocation = {'source_unique_ids': len(source_set), 'carried_unique_ids': len(carry_set),
        'remaining_unique_ids': len(remain_set), 'overlap_ids': len(carry_set & remain_set),
        'missing_source_ids': len(source_set-(carry_set | remain_set)),
        'added_partition_ids': len((carry_set | remain_set)-source_set),
        'target_missing_carried_ids': len(carry_set-target_set),
        'target_added_ids': len(target_set-carry_set),
        'source_duplicate_ids': len(source_ids)-len(source_set),
        'carried_duplicate_ids': len(carried)-len(carry_set),
        'remaining_duplicate_ids': len(remaining)-len(remain_set),
        'source_uv_identity_max_difference': float(np.max(np.abs(arrays['source_uv']-arrays['target_source_uv'])))}
    if 'material_mass' in arrays:
        weight = dict(zip(source_ids.tolist(), arrays['material_mass'].tolist()))
        source_mass = sum(weight.values())
        part_mass = sum(weight.get(i, 0) for i in carry_set)+sum(weight.get(i, 0) for i in remain_set)
        allocation['proxy_source_mass'] = source_mass
        allocation['proxy_partition_mass_relative_error'] = abs(part_mass-source_mass)/max(abs(source_mass), 1e-12)
        allocation['proxy_carried_mass_fraction'] = sum(weight.get(i, 0) for i in carry_set)/max(source_mass, 1e-12)
    if 'source_material_rgb' in arrays:
        allocation['carried_material_rgb_max_difference'] = float(np.max(np.abs(
            arrays['source_material_rgb'].astype(float)-arrays['target_material_rgb'].astype(float))))
    removed = np.asarray(Image.open(folder/'source_removed_mask.png').convert('L'))>0
    sampling_uv = arrays['source_uv']-(0 if state['branch']=='strand' else .5)
    allocation['source_texture_bilinear_owned_removed_fraction'] = mean(bilinear_ownership(removed,sampling_uv))
    if (folder/'source_food_mask.png').exists():
        observed_food = np.asarray(Image.open(folder/'source_food_mask.png').convert('L'))>0
        allocation['source_texture_bilinear_owned_observed_food_fraction'] = mean(bilinear_ownership(observed_food,sampling_uv))
        xy = (np.rint(arrays['source_uv']) if state['branch']=='strand' else np.floor(arrays['source_uv'])).astype(int)
        in_frame = (xy[:, 0]>=0)&(xy[:, 0]<source.shape[1])&(xy[:, 1]>=0)&(xy[:, 1]<source.shape[0])
        xy_valid = xy[in_frame]
        allocation['carried_uv_in_frame_fraction'] = mean(in_frame)
        allocation['carried_uv_inside_observed_food_fraction'] = mean(observed_food[xy_valid[:, 1],xy_valid[:, 0]])
        allocation['carried_uv_inside_removed_footprint_fraction'] = mean(removed[xy_valid[:, 1],xy_valid[:, 0]])
        allocation['removed_pixels_outside_observed_food'] = int(np.count_nonzero(removed&~observed_food))
    if (folder/'source_ingredient_mask.png').exists():
        ingredient = np.asarray(Image.open(folder/'source_ingredient_mask.png').convert('L'))>0
        allocation['source_texture_bilinear_owned_ingredient_fraction'] = mean(bilinear_ownership(ingredient,sampling_uv))
        allocation['removed_pixels_outside_source_ingredient'] = int(np.count_nonzero(removed&~ingredient))
    if 'strand_node_source_uv' in arrays:
        geometry_uv = arrays['strand_node_source_uv']
        exemplar_uv = arrays['strand_node_texture_exemplar_uv']
        offset = np.linalg.norm(geometry_uv-exemplar_uv,axis=-1)
        allocation['texture_exemplar_vs_geometry_uv_offset_mean_pixels'] = mean(offset)
        allocation['texture_exemplar_vs_geometry_uv_offset_max_pixels'] = float(offset.max())
        allocation['texture_exemplar_offset_nonzero_node_fraction'] = mean(offset>0)
        node_pixels = np.rint(geometry_uv).astype(int)
        allocation['geometry_node_uv_inside_removed_fraction'] = mean(removed[node_pixels[:,1],node_pixels[:,0]])
    if 'remaining_source_uv' in arrays:
        xy = np.floor(arrays['remaining_source_uv']).astype(int)
        allocation['remaining_uv_inside_removed_footprint_count'] = int(np.count_nonzero(removed[xy[:, 1],xy[:, 0]]))
    rest, rigid = arrays['rest_xyz'], arrays['rigid_xyz']
    edges, faces = arrays['edges'].astype(int), arrays['faces'].astype(int)
    support = spoon_surface_data(folder, state)
    component_count, labels = connected_components(csr_matrix((np.ones(2*len(edges)),
        (np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])), shape=(len(rest),len(rest))), directed=False)
    referenced = np.unique(faces)
    topology = {'source_vertex_graph_component_count': int(component_count),
        'source_face_component_count': int(len(np.unique(labels[referenced]))),
        'source_component_vertex_sizes_descending': sorted(np.bincount(labels).tolist(), reverse=True),
        'source_unreferenced_vertex_count': int(len(rest)-len(referenced)),
        'actual_spoon_mesh_topology': support['topology']}
    extent = float(np.linalg.norm(np.ptp(rest, axis=0)))
    variants = {}
    for variant, key in VARIANTS.items():
        image = np.asarray(Image.open(folder/(variant+'.png')).convert('RGB')).astype(float)
        delta = np.abs(image-source)
        variant_mask_path = folder/(variant+'_edit_mask.png')
        variant_edit = np.asarray(Image.open(variant_mask_path).convert('L'))>0 if variant_mask_path.exists() else edit
        prefix = 'target' if variant == 'deformed' else variant
        variants[variant] = {**geometry_metrics(rest, rigid, arrays[key], edges, faces, arrays, prefix),
            **mesh_support_metrics(arrays[key], extent, support),
            'outside_edit_mae_255': mean(delta[~variant_edit]),
            'outside_edit_max_255': float(delta[~variant_edit].max()) if np.any(~variant_edit) else None,
            'outside_edit_changed_pixels': int(np.count_nonzero(np.any(delta[~variant_edit] > 0, axis=-1))),
            'declared_variant_edit_fraction': mean(variant_edit),
            'image_mae_vs_source_255': mean(delta),
            'image_mae_vs_rigid_255': mean(np.abs(image-np.asarray(Image.open(folder/'rigid.png').convert('RGB'))))}
        if state['branch']=='strand':
            variants[variant].update(tube_centerline_metrics(rest,arrays[key],edges,state.get('tube_sides',10)))
        up = np.asarray(state['action']['up_axis'])
        lift = (arrays[key]-rest) @ up / state['action']['source_extent']
        variants[variant].update(actual_material_lift_mean_source_extent=mean(lift),
            actual_material_lift_min_source_extent=float(lift.min()),
            material_vertices_with_negative_source_relative_lift_fraction=mean(lift<0))
    final = np.asarray(Image.open(folder/'final.png').convert('RGB')).astype(float)
    final_delta = np.abs(final-source)
    deformed_image = np.asarray(Image.open(folder/'deformed.png').convert('RGB')).astype(float)
    appearance_delta = np.abs(final-deformed_image)
    appearance_changed = np.any(appearance_delta>0,axis=-1)
    image_metrics = {'declared_edit_fraction': mean(edit),
        'source_removed_pixel_count': int(removed.sum()),
        'actual_final_changed_fraction': mean(np.any(final_delta>0, axis=-1)),
        'final_outside_edit_mae_255': mean(final_delta[~edit]),
        'final_outside_edit_max_255': float(final_delta[~edit].max()) if np.any(~edit) else None,
        'final_outside_edit_changed_pixels': int(np.count_nonzero(np.any(final_delta[~edit]>0, axis=-1))),
        'final_vs_deformed_changed_pixels': int(appearance_changed.sum()),
        'final_vs_deformed_mae_255': mean(appearance_delta)}
    if (folder/'spoon_mask.png').exists():
        spoon = np.asarray(Image.open(folder/'spoon_mask.png').convert('L'))>0
        image_metrics['appearance_changed_outside_visible_spoon_pixels'] = int(np.count_nonzero(appearance_changed&~spoon))
        image_metrics['appearance_outside_visible_spoon_max_255'] = float(appearance_delta[~spoon].max())
    accepted_path = folder/'metal_candidate'/'accepted_mask.png'
    if accepted_path.exists():
        accepted = np.asarray(Image.open(accepted_path).convert('L'))>0
        image_metrics['accepted_metal_pixels_measured'] = int(accepted.sum())
        image_metrics['appearance_changed_outside_accepted_metal_pixels'] = int(np.count_nonzero(appearance_changed&~accepted))
        image_metrics['accepted_metal_outside_visible_spoon_pixels'] = int(np.count_nonzero(accepted&~spoon))
    if (folder/'moved_food_mask.png').exists():
        food = np.asarray(Image.open(folder/'moved_food_mask.png').convert('L')) > 0
        image_metrics['final_food_vs_deformed_mae_255'] = mean(appearance_delta[food])
        image_metrics['final_food_vs_deformed_max_255'] = float(appearance_delta[food].max())
        image_metrics['appearance_changed_food_pixels'] = int(np.count_nonzero(appearance_changed&food))
        image_metrics['visible_carried_food_pixels'] = int(food.sum())
        image_metrics['actual_source_removed_target_food_overlap_pixels'] = int(np.count_nonzero(removed&food))
    extent = float(np.linalg.norm(np.ptp(rest, axis=0)))
    ablations = {'full_vs_no_gravity_vertex_mean_bite_extent':
        mean(np.linalg.norm(arrays['target_xyz']-arrays['no_gravity_xyz'], axis=-1))/extent,
        'full_vs_no_contact_vertex_mean_bite_extent':
        mean(np.linalg.norm(arrays['target_xyz']-arrays['no_contact_xyz'], axis=-1))/extent}
    for name in ('no_gravity', 'no_contact'):
        delta = arrays['target_xyz']-arrays[name+'_xyz']
        ablations['full_vs_'+name+'_shape_difference_mean_bite_extent'] = mean(
            np.linalg.norm(delta-delta.mean(0), axis=-1))/extent
    if 'camera_intrinsics' in arrays:
        p = arrays['target_xyz'] @ arrays['camera_intrinsics'].T
        target_uv = p[:, :2]/p[:, 2:]
        source_p = rest @ arrays['camera_intrinsics'].T
        source_projected_uv = source_p[:,:2]/source_p[:,2:]
        image_metrics['projected_model_source_to_target_mean_displacement_pixels'] = mean(
            np.linalg.norm(target_uv-source_projected_uv, axis=-1))
    return {'case_id': folder.name, 'branch': state['branch'], 'state': state, 'topology': topology,
            'allocation': allocation, 'variants': variants, 'image': image_metrics, 'ablations': ablations}


def numeric_macro(rows, key):
    names = sorted({name for row in rows for name, value in row[key].items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)})
    return {name: {'mean': mean([row[key][name] for row in rows if row[key].get(name) is not None]),
                   'n': sum(row[key].get(name) is not None for row in rows)} for name in names}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--case-root', type=Path, help='Read an archived real-case phase without moving its artifacts')
    args = parser.parse_args()
    case_root = args.case_root or args.root/'real'
    folders = sorted(p for p in case_root.glob('real_*') if p.is_dir())
    rows = [evaluate_case(p) for p in folders]
    assert rows, 'No executed real cases found'
    out = args.root/'evaluation'; out.mkdir(parents=True, exist_ok=True)
    variants = {variant: numeric_macro([{'metrics': r['variants'][variant]} for r in rows], 'metrics')
                for variant in VARIANTS}
    branch_variant = {branch: {variant: numeric_macro([{'metrics': r['variants'][variant]} for r in rows if r['branch']==branch], 'metrics')
        for variant in VARIANTS} for branch in sorted({r['branch'] for r in rows})}
    summary = {'status': 'evaluated_executed_artifacts', 'case_count': len(rows),
        'input_case_root': str(case_root.resolve()),
        'scope': {'allocation_uv_material': 'Construction and source provenance; not measured conservation of actual food mass.',
                  'geometry': 'Monocular model-space hypotheses. Volume is oriented mesh proxy and requires closed edges; no measured physical ground truth.',
                  'support': 'Gap to the modeled spoon surface, in bite-extent units; no measured force/contact ground truth.',
                  'appearance': 'Image changes and pixel provenance only; these do not certify photographic realism.',
                  'data': 'Source/ground-truth/blind status is recorded in each case. The current 16-source development set was previously inspected; tuning on it is exploratory.',
                  'rod_measurements': 'Centerline and radii recovered independently from stable tube mesh rings; centerline frustum proxy excludes rounded endcaps and is distinct from oriented surface mesh volume.'},
        'branches': {branch: sum(r['branch']==branch for r in rows) for branch in sorted({r['branch'] for r in rows})},
        'allocation_macro': numeric_macro(rows, 'allocation'), 'image_macro': numeric_macro(rows, 'image'),
        'ablation_macro': numeric_macro(rows, 'ablations'), 'variant_macro': variants,
        'topology_counts': {'food_edge_closed': sum(r['variants']['deformed']['edge_manifold_closed'] for r in rows),
            'food_winding_consistent': sum(r['variants']['deformed']['edge_winding_consistent'] for r in rows),
            'food_no_unreferenced_vertices': sum(r['topology']['source_unreferenced_vertex_count']==0 for r in rows),
            'spoon_edge_closed': sum(r['topology']['actual_spoon_mesh_topology']['edge_manifold_closed'] for r in rows),
            'spoon_winding_consistent': sum(r['topology']['actual_spoon_mesh_topology']['edge_winding_consistent'] for r in rows)},
        'branch_variant_macro': branch_variant, 'cases': rows}
    (out/'integrated_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'integrated_cases.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows)+'\n', encoding='utf-8')
    flat = []
    for row in rows:
        for variant, values in row['variants'].items():
            flat.append({'case_id': row['case_id'], 'branch': row['branch'], 'variant': variant,
                         **row['allocation'], **row['image'], **row['ablations'], **values})
    names = list(dict.fromkeys(k for r in flat for k in r))
    with (out/'integrated_cases.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=names); writer.writeheader(); writer.writerows(flat)
    print(json.dumps({'cases': len(rows), 'summary': str(out/'integrated_summary.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
