"""Source photograph -> persistent material -> contact-aware deformable first bite.

Run on gp40 with the existing SAM3, MoGe2 and frozen MLD2 installations.
Existing inferred observations can be reused with --input-root; --image performs
the complete perception and reconstruction path for a new photograph.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, '/host/space0/guo-z/tf-ufi/food3d_repair_20260928/compat_packages')
import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import binary_dilation, gaussian_filter
import trimesh
from OpenGL.arrays.ctypesparameters import CtypesParameterHandler
CtypesParameterHandler().register(CtypesParameterHandler.HANDLED_TYPES)

from foodstateedit.material_lineage.photographic_state import PhotographicMaterialState
from foodstateedit.material_lineage.deformable import solve_deformation
from foodstateedit.material_lineage.strands import reconstruct_strands, reanchor_strands, deform_strands, tube_geometry
from foodstateedit.material_lineage.strand_selection import select_visible_strand_bite
from foodstateedit.material_lineage.topology import coherent_closed_patch
from foodstateedit.material_lineage.spoon_surface import surface_height, surface_domain
from mld2_real_geometry_v2 import closed_patch, Render, texture_top, project
from mld2_real_spoon_v3 import utensil, metal_pass

DEFAULT_CHECKPOINT = '/host/space0/guo-z/tf-ufi/material_lineage_full_20261004/runs_continuous/shared_s41/inference_checkpoint.pt'
SAM_ROOT = Path('/host/space0/guo-z/Evol-SAM3')
MOGE_ROOT = Path('/host/space0/guo-z/tf-ufi/food3d_repair_20260928')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def prepare_image(args):
    import torch
    folder = args.output / 'observations' / 'real_00_input'
    folder.mkdir(parents=True, exist_ok=True)
    im = ImageOps.exif_transpose(Image.open(args.image)).convert('RGB')
    im.thumbnail((640, 640), Image.Resampling.LANCZOS)
    im.save(folder / 'source.png')
    sys.path.insert(0, str(SAM_ROOT))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model(bpe_path=str(SAM_ROOT / 'assets/bpe_simple_vocab_16e6.txt.gz'),
        checkpoint_path=str(SAM_ROOT / 'sam3/sam3.pt'), load_from_HF=False, device='cuda', eval_mode=True)
    processor = Sam3Processor(model, confidence_threshold=.18)
    image_state = processor.set_image(im)
    candidates = []
    for prompt in [args.food_prompt, 'food']:
        processor.reset_all_prompts(image_state)
        result = processor.set_text_prompt(state=image_state, prompt=prompt)
        masks = result['masks'].detach().cpu().numpy().astype(bool).reshape(-1, im.height, im.width)
        scores = result['scores'].detach().cpu().numpy().reshape(-1)
        candidates.extend((float(s), prompt, m) for s, m in zip(scores, masks)
                          if .006 < m.mean() < .88)
    specific = [x for x in candidates if x[1] == args.food_prompt]
    score, prompt, mask = max(specific or candidates, key=lambda x: x[0])
    Image.fromarray(mask.astype(np.uint8) * 255).save(folder / 'food_mask.png')
    np.savez_compressed(folder / 'segmentation.npz', masks=np.stack([x[2] for x in candidates]),
        scores=[x[0] for x in candidates], prompts=[x[1] for x in candidates])
    write(folder / 'segmentation.json', dict(selected_prompt=prompt, confidence=score))
    if choose_mode(prompt, args.material_mode) == 'strand':
        from prepare_mld3_ingredients import prepare_folder
        prepare_folder(folder, processor)
    del processor, model, image_state, result
    torch.cuda.empty_cache()
    sys.path[:0] = [str(MOGE_ROOT / 'moge'), str(MOGE_ROOT / 'utils3d_moge')]
    from moge.model.v2 import MoGeModel
    weights = torch.load(MOGE_ROOT / 'models/model.pt', map_location='cpu', weights_only=True)
    model = MoGeModel(**weights['model_config'])
    model.load_state_dict(weights['model'])
    model = model.eval().cuda()
    del weights
    tensor = torch.from_numpy(np.asarray(im).copy()).cuda().float().permute(2, 0, 1) / 255
    with torch.inference_mode():
        prediction = model.infer(tensor, resolution_level=7, use_fp16=True)
    np.savez_compressed(folder / 'depth.npz', **{k: v.detach().cpu().numpy() for k, v in prediction.items()})
    del model, tensor, prediction
    torch.cuda.empty_cache()
    from foodstateedit.material_lineage.mld2 import MLD2Model, MLD2Config
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = MLD2Model(MLD2Config(**checkpoint['model_config'])).eval().cuda()
    model.load_state_dict(checkpoint['state_dict'])
    from mld2_real_geometry import build_case as initialize
    from mld2_real_geometry_v2 import build_case as local_closure
    with torch.inference_mode():
        initialize(folder, model, torch)
        (folder / 'first_round_geometry.json').write_bytes((folder / 'geometry_raw.json').read_bytes())
        (folder / 'first_round_source_bite_mask.png').write_bytes((folder / 'source_bite_mask.png').read_bytes())
        (folder / 'first_round_frozen_edit_mask.png').write_bytes((folder / 'edit_mask.png').read_bytes())
        local_closure(folder, model, torch)
    del model, checkpoint
    torch.cuda.empty_cache()
    return folder.parent


def prepare_selected_sources(source_root, args):
    """Reconstruct the new source-selected bite with the same frozen learned field."""
    import torch
    torch.set_num_threads(4)
    from foodstateedit.material_lineage.mld2 import MLD2Model, MLD2Config
    from mld2_real_geometry_v2 import build_case, lift_plan
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = MLD2Model(MLD2Config(**checkpoint['model_config'])).eval().cuda()
    model.load_state_dict(checkpoint['state_dict'])
    prepared = args.output / 'observations'; prepared.mkdir(exist_ok=True)
    ingredient_processor = None
    for folder in sorted(x for x in source_root.glob('real_*') if x.is_dir()):
        index = int(folder.name.split('_')[1])
        if args.indices is not None and index not in args.indices:
            continue
        out = prepared / folder.name; out.mkdir(exist_ok=True)
        for name in ['source.png', 'depth.npz', 'food_mask.png', 'segmentation.npz', 'segmentation.json',
                     'closure_raw.npz', 'geometry_raw.json', 'source_bite_mask.png',
                     'noodle_mask.png', 'excluded_ingredients.png', 'ingredients.json', 'ingredient_candidates.npz', 'selected_food_mask.png']:
            if not (folder / name).exists():
                continue
            if (out / name).resolve() != (folder / name).resolve():
                (out / name).write_bytes((folder / name).read_bytes())
        segmentation = json.loads((out / 'segmentation.json').read_text())
        if choose_mode(segmentation['selected_prompt'], args.material_mode) != 'strand':
            continue
        if not (out / 'noodle_mask.png').exists():
            from prepare_mld3_ingredients import build_processor, prepare_folder
            if ingredient_processor is None:
                ingredient_processor = build_processor()
            prepare_folder(out, ingredient_processor)
        source = np.asarray(Image.open(out / 'source.png').convert('RGB'))
        mask = np.asarray(Image.open(out / 'noodle_mask.png')) > 0
        maps = dict(np.load(out / 'depth.npz')); closure = dict(np.load(out / 'closure_raw.npz'))
        mask &= maps['mask'] & np.isfinite(maps['points']).all(axis=-1)
        previous = json.loads((out / 'geometry_raw.json').read_text())
        seed, center, radius, selection = select_visible_strand_bite(source, mask,
            maps['points'], closure['world_axes'])
        if center is None:
            raise ValueError(f'{folder.name}: no observed source ingredient strands')
        bundle = reconstruct_strands(source, mask, maps['points'], seed, closure['world_axes'])
        patch = bundle.removal_mask
        Image.fromarray(seed.astype(np.uint8) * 255).save(out / 'source_strand_seed_mask.png')
        selection['ingredient_segmentation'] = json.loads((out / 'ingredients.json').read_text())
        selection['width_aware_ownership'] = bundle.metrics
        h, w = source.shape[:2]; K = maps['intrinsics'].copy(); K[0] *= w; K[1] *= h
        previous['target_translation'] = lift_plan(maps['points'][patch], closure['world_axes'],
            float(closure['extent']), K, w, h).tolist()
        previous['source_bite_center'] = np.rint(center).astype(int).tolist()
        previous['source_bite_radius'] = radius
        write(out / 'first_round_geometry.json', previous)
        Image.fromarray(patch.astype(np.uint8) * 255).save(out / 'first_round_source_bite_mask.png')
        old_edit = np.asarray(Image.open(folder / 'frozen_edit_mask.png')) > 0
        Image.fromarray((old_edit | binary_dilation(patch, iterations=8)).astype(np.uint8) * 255).save(out / 'first_round_frozen_edit_mask.png')
        with torch.inference_mode():
            build_case(out, model, torch)
        write(out / 'source_selection.json', selection)
        print('source_selection', folder.name, json.dumps(selection), flush=True)
    del model, checkpoint, ingredient_processor
    torch.cuda.empty_cache()
    return prepared


def support_metadata(bite, axes, translation, info):
    normal = np.asarray(info['bowl_normal_camera'])
    right = np.array([1., 0., 0.]); right -= normal * (right @ normal)
    right /= np.linalg.norm(right)
    spoon_axes = np.stack([right, np.cross(normal, right), normal], axis=1)
    lower = (bite['low'] + translation) @ spoon_axes
    center = lower[:, :2].mean(0)
    a, b = info['radius_ab']
    radial = ((lower[:, 0] - center[0]) / a)**2 + ((lower[:, 1] - center[1]) / b)**2
    original_floor = float(np.min(lower[radial < .90, 2] - info['bowl_curvature_height'] * radial[radial < .90]))
    support = dict(info, spoon_axes_camera=spoon_axes.tolist(), center_local=center.tolist(),
                   floor_local=0., bowl_handle_blend=True)
    inside = surface_domain(lower[:, :2], support)
    floor = float(np.min(lower[inside, 2] - surface_height(lower[inside, :2], support)))
    return dict(support, floor_local=floor, original_constructor_floor=original_floor,
                inferred_contact_min_gap=0., inferred_penetration=0.,
                initial_support_adjustment_camera=floor - original_floor)


def support_gaps(vertices, info):
    local = vertices @ np.asarray(info['spoon_axes_camera'])
    return local[:, 2] - surface_height(local[:, :2], info), surface_domain(local[:, :2], info)


def choose_mode(prompt, requested):
    if requested != 'auto':
        return requested
    if any(x in prompt.lower() for x in ['noodle', 'spaghetti', 'udon', 'ramen']):
        return 'strand'
    return 'soft' if any(x in prompt.lower() for x in ['egg', 'omelet', 'sashimi', 'pancake']) else 'cohesive'


def transported_parts(bite, vertices):
    count = len(bite['points'])
    result = []
    for key in ['top', 'side', 'bottom']:
        mesh = bite[key].copy()
        mesh.vertices = vertices[:count] if key == 'top' else vertices[count:] if key == 'bottom' else vertices
        result.append(mesh)
    return result


def render_food(parts, K, w, h):
    double = K.copy(); double[:2] *= 2
    renderer = Render(double, w * 2, h * 2)
    for part in parts:
        renderer.add(part)
    rgb, depth = renderer.image(); renderer.close()
    rgb = np.asarray(Image.fromarray(rgb).resize((w, h), Image.Resampling.LANCZOS))
    depth = np.asarray(Image.fromarray(depth).resize((w, h), Image.Resampling.NEAREST))
    return rgb, depth


def projected_shadow(vertices, faces, axes, plane_center, K, w, h, radius):
    normal = axes[:, 2]
    light = .85 * normal - .20 * axes[:, 0] + .35 * axes[:, 1]
    height = (vertices - plane_center) @ normal
    cast = vertices - height[:, None] / (light @ normal) * light
    pixels = project(cast, K)
    silhouette = np.zeros((h, w), dtype=np.uint8)
    faces = np.asarray(faces)
    face_height = height[faces]
    visible = (cast[faces, 2] > .001).all(axis=1)
    supported_casters = visible & (face_height >= 0).all(axis=1)
    for triangle in pixels[faces[supported_casters]]:
        if triangle[:, 0].max() >= 0 and triangle[:, 0].min() < w and triangle[:, 1].max() >= 0 and triangle[:, 1].min() < h:
            cv2.fillConvexPoly(silhouette, np.rint(triangle).astype(np.int32), 255)
    # Clip only triangles crossing the receiver plane, before projecting light.
    # Food that falls underneath the dish contributes no above-dish shadow.
    for face in faces[visible & (face_height.max(axis=1) > 0) & (face_height.min(axis=1) < 0)]:
        polygon = []
        for a, b in zip(face, np.roll(face, -1)):
            if height[a] >= 0:
                polygon.append(vertices[a])
            if (height[a] >= 0) != (height[b] >= 0):
                polygon.append(vertices[a] + height[a] / (height[a] - height[b]) * (vertices[b] - vertices[a]))
        polygon = np.asarray(polygon)
        cast_polygon = polygon - ((polygon - plane_center) @ normal)[:, None] / (light @ normal) * light
        projected = project(cast_polygon, K)
        cv2.fillConvexPoly(silhouette, np.rint(projected).astype(np.int32), 255)
    return gaussian_filter(silhouette.astype(float) / 255, max(1.5, radius * .20)) * .14


def compose(source, removed, cavity, cavity_depth, food_rgb, food_depth,
            metal_rgb, metal_depth, shadow, food_mask):
    background = source.copy()
    background[removed] = cavity[removed]
    wall = (cavity_depth > 0) & binary_dilation(removed, iterations=2)
    background[wall] = cavity[wall]
    shadow = shadow * binary_dilation(food_mask, iterations=16)
    background = np.clip(background * (1 - shadow[..., None]), 0, 255).astype(np.uint8)
    spoon_visible = (metal_depth > 0) & ((food_depth == 0) | (metal_depth < food_depth))
    food_visible = (food_depth > 0) & ((metal_depth == 0) | (food_depth <= metal_depth))
    metal = metal_rgb.copy(); accepted = np.zeros(removed.shape, bool)
    result = background.copy(); result[spoon_visible] = metal[spoon_visible]
    result[food_visible] = food_rgb[food_visible]
    edit = binary_dilation(removed | wall | food_visible | spoon_visible | (shadow > .003), iterations=3)
    result[~edit] = source[~edit]
    return result, edit, food_visible, spoon_visible, accepted


def run_case(folder, output, args, index):
    output.mkdir(parents=True, exist_ok=True)
    source = np.asarray(Image.open(folder / 'source.png').convert('RGB')); h, w = source.shape[:2]
    observation = dict(np.load(folder / 'depth.npz'))
    closure = dict(np.load(folder / 'closure_raw.npz'))
    geometry = json.loads((folder / 'geometry_raw.json').read_text())
    segmentation = json.loads((folder / 'segmentation.json').read_text())
    prompt = segmentation['selected_prompt']; mode = choose_mode(prompt, args.material_mode)
    axes = closure['world_axes']; translation = np.asarray(geometry['target_translation'])
    translation += (args.lift_multiplier - 1) * (translation @ axes[:, 2]) * axes[:, 2]
    patch = np.asarray(Image.open(folder / 'source_bite_mask.png')) > 0
    food_mask = np.asarray(Image.open(folder / 'food_mask.png')) > 0
    topology = None
    if mode != 'strand':
        cleaned = coherent_closed_patch(patch, closure)
        patch, closure, topology = cleaned['patch'], cleaned['closure'], cleaned['metrics']
        np.savez_compressed(output / 'source_topology.npz', keep_pixel_indices=cleaned['keep_pixel_indices'],
            original_to_kept=cleaned['original_to_kept'], discarded_mask=cleaned['discarded_mask'])
    points = observation['points'].copy(); points[patch] = closure['source_camera_xyz']
    K = observation['intrinsics'].copy(); K[0] *= w; K[1] *= h
    bite = closed_patch(points, patch, closure['floor_local'], axes,
                        closure['interior_rgb'], source, closure['bottom_rgb'])
    spoon, info = utensil(bite, axes, translation, source)
    info = support_metadata(bite, axes, translation, info)
    spoon.apply_translation(info['initial_support_adjustment_camera'] * np.asarray(info['bowl_normal_camera']))
    bundle = None
    if mode == 'strand':
        ingredient = np.asarray(Image.open(folder / 'noodle_mask.png')) > 0
        ingredient &= observation['mask'] & np.isfinite(observation['points']).all(axis=-1)
        seed = np.asarray(Image.open(folder / 'source_strand_seed_mask.png')) > 0
        bundle = reconstruct_strands(source, ingredient, observation['points'], seed, axes)
        bundle = reanchor_strands(bundle, points)
        rest = tube_geometry(bundle, source)
        vertices, faces = rest['vertices'], rest['faces']
        uv, material_rgb = rest['uv_pixels'], rest['source_rgb']
        removed = np.asarray(bundle.removal_mask, bool)
        Image.fromarray(ingredient.astype(np.uint8) * 255).save(output / 'source_ingredient_mask.png')
        Image.fromarray(seed.astype(np.uint8) * 255).save(output / 'source_strand_seed_mask.png')
        solver_metrics = {}
        variants = {'rigid': vertices + translation}
        for name, contact, gravity in [('full', True, 1.), ('no_contact', False, 1.), ('no_gravity', True, 0.)]:
            moved_bundle, metrics = deform_strands(bundle, translation, info, axes,
                iterations=args.iterations, contact=contact, gravity_strength=gravity)
            variants[name] = tube_geometry(moved_bundle, source)['vertices']
            solver_metrics[name] = metrics
        parts = lambda xyz: [texture_top(xyz, faces, uv + .5, source)]
        Image.fromarray(np.asarray(bundle.ridge_mask, dtype=np.uint8) * 255).save(output / 'source_ridges.png')
    else:
        vertices, faces = np.asarray(bite['complete'].vertices), np.asarray(bite['complete'].faces)
        uv = np.tile(bite['pixels'] + .5, (2, 1))
        material_rgb = np.r_[source[patch], closure['bottom_rgb']]
        removed = patch
        compliance = args.compliance if args.compliance is not None else .35 if mode == 'soft' else .08
        count = len(bite['points'])
        solid_pairs = np.c_[np.arange(count), np.arange(count) + count]
        variants = {'rigid': vertices + translation}; solver_metrics = {}
        for name, contact, gravity in [('full', True, .09), ('no_contact', False, .09), ('no_gravity', True, 0.)]:
            variants[name], solver_metrics[name] = solve_deformation(vertices, faces, axes, translation,
                info, compliance=compliance, iterations=args.iterations, contact=contact, gravity_strength=gravity,
                solid_pairs=solid_pairs)
        parts = lambda xyz: transported_parts(bite, xyz)
    if getattr(args, 'edge_connected_handle', False):
        from foodstateedit.material_lineage.utensil_placement import connect_handle_to_frame
        carried_x = (variants['full'] @ np.asarray(info['spoon_axes_camera']))[:, 0] - info['center_local'][0]
        if carried_x.max() >= info['radius_ab'][0]:
            raise ValueError('Handle extension requires the carried food to remain in the fixed bowl domain')
        spoon.vertices, info['frame_connection'] = connect_handle_to_frame(spoon.vertices, info, K, w)
    renderer = Render(K, w, h)
    renderer.add(bite['cavityfloor']); renderer.add(bite['cavityside'])
    cavity, cavity_depth = renderer.image(); renderer.close()
    metal, metal_depth = metal_pass(spoon, info, source, K, w, h)
    outputs = {}; records = {}
    for name, xyz in variants.items():
        food_rgb, depth = render_food(parts(xyz), K, w, h)
        plane_center = np.asarray(geometry['plane_center'])
        caster_faces = np.r_[faces, np.asarray(spoon.faces) + len(xyz)]
        shadow = projected_shadow(np.r_[xyz, spoon.vertices], caster_faces, axes, plane_center, K, w, h, geometry['source_bite_radius'])
        image, edit, fv, sv, accepted = compose(source, removed, cavity, cavity_depth,
            food_rgb, depth, metal, metal_depth, shadow, food_mask)
        outputs[name] = (image, edit, fv, sv, accepted)
        records[name] = dict(changed_pixels=int(np.any(image != source, axis=-1).sum()),
            food_visible_pixels=int(fv.sum()), spoon_visible_pixels=int(sv.sum()),
            accepted_cached_metal_pixels=int(accepted.sum()),
            source_target_overlap=int((removed & fv).sum()))
        Image.fromarray(image).save(output / ('deformed.png' if name == 'full' else name + '.png'))
        Image.fromarray(edit.astype(np.uint8) * 255).save(output / ('deformed_edit_mask.png' if name == 'full' else name + '_edit_mask.png'))
    final, edit, moved_mask, spoon_mask, accepted = outputs['full']
    for name, array in [('source', source), ('final', final), ('edit_mask', edit),
                        ('source_removed_mask', removed), ('source_food_mask', food_mask),
                        ('moved_food_mask', moved_mask), ('spoon_mask', spoon_mask)]:
        Image.fromarray(array.astype(np.uint8) * 255 if array.dtype == bool else array).save(output / (name + '.png'))
    state = PhotographicMaterialState.from_geometry(vertices, faces, uv, material_rgb, food_mask, removed)
    transport = state.transport_record(variants['full'], variants['rigid'], variants['no_contact'], variants['no_gravity'])
    yy, xx = np.where(food_mask & ~removed)
    transport['remaining_source_uv'] = np.c_[xx, yy] + .5
    transport['camera_intrinsics'] = K
    if bundle is not None:
        transport['strand_node_source_uv'] = np.concatenate([c['source_uv'] for c in bundle.curves])
        transport['strand_node_texture_exemplar_uv'] = np.concatenate([c['texture_center_uv'] for c in bundle.curves])
        transport['strand_curve_offsets'] = np.cumsum([0] + [len(c['xyz']) for c in bundle.curves])
    for name, xyz in variants.items():
        key = 'target' if name == 'full' else name
        gaps, inside = support_gaps(xyz, info)
        transport[key + '_support_gap'] = gaps; transport[key + '_support_inside'] = inside
    np.savez_compressed(output / 'transport.npz', **transport)
    trimesh.Trimesh(vertices, faces, process=False).export(output / 'source_material.ply')
    trimesh.Trimesh(variants['full'], faces, process=False).export(output / 'deformed_material.ply')
    spoon.export(output / 'spoon.ply')
    row = dict(case_id=folder.name, index=index, status='rendered', prompt=prompt, mode=mode, branch=mode,
        camera_intrinsics=K.tolist(),
        action=dict(translation=translation.tolist(), lift_multiplier=args.lift_multiplier,
                    up_axis=axes[:, 2].tolist(), source_extent=float(closure['extent'])),
        spoon=info, solver=solver_metrics, render=records,
        source_observation=str(folder), strand_reconstruction=None if bundle is None else bundle.metrics,
        source_topology=topology, tube_sides=10 if mode == 'strand' else None,
        shadow_model='Prescribed directional illumination, source-plane clipped material/spoon triangle silhouettes with soft blur; source light is not calibrated.',
        source_selection=json.loads((folder / 'source_selection.json').read_text()) if (folder / 'source_selection.json').exists() else {'rule': 'Original source bite selection retained'},
        source_state_mass_scope='Unit-density carried reconstruction plus inferred equal-column remaining source material. Fixed lumped weights; no measured food mass or density.',
        remaining_material_scope='Unmoved source-visible food columns; hidden remaining volume is a proxy.',
        deformation_scope='Prescribed material compliance/rod stiffness in inferred source scale; not identified from physical observations.',
        geometry_uncertainty=geometry.get('local_scale_diagnostics', {}),
        learned_prior=geometry.get('hidden_prior_diagnostics', {}),
        models={'SAM3': 'source food and ingredient masks', 'MoGe2': 'inferred source points and camera',
                'MLD2': 'frozen65k learned source-conditioned hidden geometry/material closure',
                'Qwen-Image-2.1 + Fun ControlNet-Union': 'optional metal appearance inside unchanged visible spoon; no generated food geometry or RGB'},
        generated_food_pixels=0, paired_real_ground_truth=False, blind_holdout=False,
        source_uv_convention='Strand source_uv uses pixel-index sampling coordinates; mesh source_uv uses pixel centers; strand renderer adds .5 for the texture atlas.',
        final_appearance=dict(accepted_cached_metal_pixels=int(accepted.sum()),
            visible_spoon_pixels=int(spoon_mask.sum()), generated_food_pixels=0),
        source_hash=hashlib.sha256((folder / 'source.png').read_bytes()).hexdigest(),
        script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(output / 'state.json', row)
    print(json.dumps(dict(case_id=folder.name, mode=mode, solver=solver_metrics), ensure_ascii=True), flush=True)
    return row


def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--input-root', type=Path); group.add_argument('--image', type=Path)
    p.add_argument('--food-prompt', default='food'); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=Path(DEFAULT_CHECKPOINT))
    p.add_argument('--indices', type=int, nargs='+')
    p.add_argument('--material-mode', choices=['auto', 'strand', 'soft', 'cohesive'], default='auto')
    p.add_argument('--iterations', type=int, default=180); p.add_argument('--compliance', type=float)
    p.add_argument('--lift-multiplier', type=float, default=1.)
    p.add_argument('--edge-connected-handle', action='store_true', help='Keep the food-contact bowl fixed and extend the distal shaft beyond the frame.')
    p.add_argument('--appearance', action='store_true', help='Run optional Qwen metal appearance and exact silhouette projection.')
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    source_root = prepare_image(args) if args.image else args.input_root
    source_root = prepare_selected_sources(source_root, args)
    rows = []
    for folder in sorted(x for x in source_root.glob('real_*') if x.is_dir()):
        index = int(folder.name.split('_')[1])
        if args.indices is not None and index not in args.indices:
            continue
        rows.append(run_case(folder, args.output / 'real' / folder.name, args, index))
        write(args.output / 'run_manifest.json', dict(status='executing', cases=rows))
    if args.appearance:
        from refine_mld3_metal import refine
        refine(args.output, indices=args.indices)
        rows = [json.loads((args.output / 'real' / row['case_id'] / 'state.json').read_text(encoding='utf-8')) for row in rows]
    write(args.output / 'run_manifest.json', dict(status='complete', cases=rows,
          checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
          seconds=time.time() - started, source_only=True, fresh_perception=bool(args.image),
          photographic_ground_truth=False, residual_sampling=False, optional_metal_appearance=args.appearance))


if __name__ == '__main__':
    main()
