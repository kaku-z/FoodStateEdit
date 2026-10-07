"""Ray trace spoon appearance for the material-lineage pilot without retraining.

Only CPU ownership label 3 is replaced in copies of the v1 photographs. The
original mesh positions, topology, food RGB, and all other pixels stay fixed.
One metal render per case/pose is shared across all seeds and variants. Lighting
and reflection materials are source-derived priors, not calibrated estimates.
This script does not modify v1 files or train a model.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image


DEFAULT_ROOT = Path('/host/space0/guo-z/tf-ufi/material_lineage_pilot_20261003')
MITSUBA_VENDOR = '/mnt/tmp/guo-z_first_bite_structure_20260930_runtime/mitsuba_vendor'
SCOPE = ('Physically ray-traced aluminum spoon appearance on existing inferred '
         'joint geometry. Source-derived broad area light, constant environment, '
         'and white-food reflection/occlusion proxies are uncalibrated priors. '
         'No new food generation, geometry, contact, or realism claim.')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    # The experiment filesystem may transiently report ESTALE.
    for attempt in range(10):
        try:
            return json.loads(Path(path).read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            if attempt == 9:
                raise
            time.sleep(.3)


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def linear(rgb):
    rgb = np.asarray(rgb, dtype=np.float64) / 255.
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)


def srgb(rgb):
    rgb = np.maximum(np.asarray(rgb), 0.)
    encoded = np.where(rgb <= .0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - .055)
    return np.uint8(np.clip(np.rint(encoded * 255), 0, 255))


def image_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))


def freeze(root):
    cfg = read_json(root / 'config.json')
    cases, seeds, variants = cfg['cases'], cfg['seeds'], cfg['variants']
    poses = list(range(int(cfg['poses'])))
    count = len(cases) * len(seeds) * len(variants) * len(poses)
    assert count == int(cfg['expected_images']) == 144, 'Unexpected v1 experiment inventory'
    inputs = []
    parents = []
    for cid in cases:
        case_dir = root / cid
        inputs.append(dict(path=str(case_dir / 'source_inputs.json'), sha256=sha(case_dir / 'source_inputs.json')))
        for pose in poses:
            pdir = case_dir / ('pose' + str(pose))
            for name in ['remaining.ply', 'carried.ply', 'spoon.ply', 'geometry_channels.npz', 'pose.json']:
                inputs.append(dict(path=str(pdir / name), sha256=sha(pdir / name)))
            for variant in variants:
                for seed in seeds:
                    d = pdir / ('%s_s%d' % (variant, seed))
                    parents.append(dict(case_id=cid, pose=pose, seed=seed, variant=variant,
                                        image_sha256=sha(d / 'composited.png'), record_sha256=sha(d / 'record.json')))
    out = root / 'metal_v2'
    out.mkdir(exist_ok=False)
    schema = root / 'runroot_schema.json'
    recipe = dict(
        status='frozen_before_ray_tracing', created_unix=time.time(), root=str(root),
        cases=cases, poses=poses, seeds=seeds, variants=variants, expected_images=count,
        source_config_sha256=sha(root / 'config.json'),
        runroot_schema_sha256=sha(schema) if schema.exists() else None,
        schema_source='Actual v1 config.json, source_inputs.json, and pose directories',
        script_sha256=sha(Path(__file__)), scope=SCOPE,
        metal=dict(bsdf='roughconductor', material='Al', distribution='ggx', alpha=.18,
                   normals='Trimesh smooth weighted-average vertex normals; positions/faces unchanged'),
        integrator=dict(type='path', max_depth=12, rr_depth=5, spp=512, spp_batch=32, seed=41),
        camera=dict(type='perspective', transform_diagonal=[-1, -1, 1],
                    principal_point_required=[320, 240], equal_focal_lengths_required=True,
                    film_size=[640, 480], primary_samples='Exact CPU pixel centers; no primary jitter or antialiasing'),
        lighting=dict(
            source_direction='0.35 normalized source affine RGB directional coefficients + 0.65 [-0.4,-0.6,-0.7] prior; x<=-0.15,y<=-0.25,z<=-0.45, then normalize',
            area_distance_over_extent=8., area_width_and_height_over_extent=3.,
            area_radiance=8., ambient_radiance=.25,
            source_exposure='clip(source bare-patch linear luminance / 0.65, 0.4, 1.25)',
            tint='clip(source bare-patch linear median / its RGB mean, 0.75, 1.25)',
            limitations='Single frozen source-only recipe, no output-specific fitting or measured lighting'),
        food_reflection_proxy=dict(bsdf='diffuse', reflectance=.80, geometry='Actual remaining and carried meshes'),
        ownership='Replace only saved CPU labels == 3; all non-metal pixels copied byte-exact from each v1 parent',
        invariants=['No geometry or identity changes', 'No training or food RGB changes',
                    'Same metal RGB for all six parents of each case/pose', 'Original v1 assets preserved'],
        inputs=inputs, parents=parents,
    )
    write_json(out / 'recipe.json', recipe)
    print('LINEAGE_METAL_V2_FROZEN', count, flush=True)


def source_light(root, cid, source_info, recipe):
    paths = [Path(row['path']) for row in source_info['inputs']]
    calibration_path = next(p for p in paths if p.name == 'appearance_calibration.json')
    source_path = next(p for p in paths if p.name == 'source.png')
    cal = read_json(calibration_path)
    patch = cal['source_material_box_canvas']
    x0, y0, x1, y1 = map(int, patch)
    source = image_rgb(source_path)
    bare = linear(source[y0:y1, x0:x1])
    if bare.size == 0:
        raise ValueError('Empty source material patch')
    base = np.median(bare, axis=(0, 1))
    exposure = float(np.clip(base @ [.2126, .7152, .0722] / .65, .4, 1.25))
    tint = np.clip(base / max(float(base.mean()), 1e-8), .75, 1.25)
    prior = np.array([-.4, -.6, -.7])
    prior /= np.linalg.norm(prior)
    fitted = np.asarray(cal['coefficient_rgb'], dtype=float)[1:].mean(1)
    if np.linalg.norm(fitted) < 1e-8:
        fitted = prior.copy()
    else:
        fitted /= np.linalg.norm(fitted)
    direction = .35 * fitted + .65 * prior
    direction = np.minimum(direction, [-.15, -.25, -.45])
    direction /= np.linalg.norm(direction)
    return direction, tint, exposure, dict(
        calibration_path=str(calibration_path), calibration_sha256=sha(calibration_path),
        source_path=str(source_path), source_sha256=sha(source_path), source_patch_canvas=patch,
        source_bare_linear_median=base.tolist(), direction_camera=direction.tolist(),
        tint=tint.tolist(), exposure=exposure, source_directional_fit_underconstrained=True,
        scope='Source-only appearance/light prior; source albedo and illumination remain confounded')


def make_scene(pdir, K, extent, direction, tint, exposure, recipe, mi):
    import trimesh
    meshes = {name: trimesh.load(pdir / (name + '.ply'), process=False)
              for name in ['remaining', 'carried', 'spoon']}
    center = np.asarray(meshes['spoon'].vertices).mean(0)
    lights = recipe['lighting']
    light_center = center + direction * extent * lights['area_distance_over_extent']
    light_pose = mi.ScalarTransform4f().look_at(
        origin=light_center.tolist(), target=center.tolist(), up=[0., 1., 0.]
    ) @ mi.ScalarTransform4f().scale([
        extent * lights['area_width_and_height_over_extent'],
        extent * lights['area_width_and_height_over_extent'], 1.])
    camera = dict(type='perspective', fov=float(np.degrees(2 * np.arctan(320 / K[0, 0]))),
                  fov_axis='x', to_world=mi.ScalarTransform4f().scale([-1., -1., 1.]),
                  near_clip=.001 * extent, far_clip=100 * extent,
                  sampler=dict(type='independent', sample_count=recipe['integrator']['spp']),
                  film=dict(type='hdrfilm', width=640, height=480, pixel_format='rgb', rfilter=dict(type='box')))
    config = dict(type='scene', integrator={k: recipe['integrator'][k] for k in ['type', 'max_depth', 'rr_depth']},
                  sensor=camera,
                  environment=dict(type='constant', radiance=dict(type='rgb', value=(tint * exposure * lights['ambient_radiance']).tolist())),
                  key_light=dict(type='rectangle', to_world=light_pose,
                                 emitter=dict(type='area', radiance=dict(type='rgb', value=(tint * exposure * lights['area_radiance']).tolist()))))
    for name in ['remaining', 'carried']:
        config[name] = dict(type='ply', filename=str(pdir / (name + '.ply')), face_normals=True,
                            bsdf=dict(type='diffuse', reflectance=recipe['food_reflection_proxy']['reflectance']))
    config['spoon'] = dict(type='ply', filename=str(pdir / 'spoon.ply'), face_normals=False,
                           bsdf={k: recipe['metal'][k] for k in ['material', 'distribution', 'alpha']})
    config['spoon']['bsdf']['type'] = recipe['metal']['bsdf']
    scene = mi.load_dict(config)
    params = mi.traverse(scene)
    # Explicit normals, rather than relying on the PLY loader's smooth default.
    positions = np.asarray(params['spoon.vertex_positions']).reshape(-1, 3)
    faces = np.asarray(params['spoon.faces']).reshape(-1, 3)
    assert np.array_equal(positions, np.asarray(meshes['spoon'].vertices, np.float32))
    assert np.array_equal(faces, np.asarray(meshes['spoon'].faces, np.uint32))
    normals = np.asarray(meshes['spoon'].vertex_normals, dtype=np.float32)
    assert np.isfinite(normals).all() and np.all(np.linalg.norm(normals, axis=1) > .9)
    normal_array_type = type(params['spoon.vertex_normals'])
    params['spoon.vertex_normals'] = normal_array_type(normals.reshape(-1))
    params.update()
    assert np.array_equal(positions, np.asarray(params['spoon.vertex_positions']).reshape(-1, 3))
    return scene, dict(light_center_camera=light_center.tolist(), target_camera=center.tolist(),
                       spoon_vertex_count=len(positions), spoon_face_count=len(faces),
                       spoon_smooth_normals_sha256=hashlib.sha256(normals.tobytes()).hexdigest(),
                       mesh_positions_and_faces_unchanged=True)


def trace_metal(scene, K, mask, cpu_depth, extent, recipe, mi):
    """Path trace only ownership-metal pixel centers, avoiding RGB boundary mix."""
    yy, xx = np.where(mask)
    if len(xx) < 1:
        raise ValueError('No CPU-owned metal pixels')
    film = np.c_[(xx + .5) / 640., (yy + .5) / 480.]
    ray, _ = scene.sensors()[0].sample_ray(0., .5, mi.Point2f(film.T), mi.Point2f(.5, .5))
    direction = np.asarray(ray.d).T
    expected = np.c_[xx + .5, yy + .5, np.ones(len(xx))] @ np.linalg.inv(K).T
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    ray_error = float(np.linalg.norm(direction - expected, axis=1).max())
    assert ray_error < 2e-6, ('Camera ray mismatch', ray_error)
    intersection = scene.ray_intersect(ray)
    valid = np.asarray(intersection.is_valid())
    hit_depth = np.asarray(intersection.p).T[:, 2]
    depth_error = np.abs(hit_depth - cpu_depth[yy, xx]) / extent
    spoon = next(shape for shape in scene.shapes() if shape.id() == 'spoon')
    spoon_hits = np.asarray(intersection.shape == spoon)
    ownership_agreement = float(spoon_hits.mean())
    assert valid.all() and depth_error.max() < 5e-4, ('Primary depth mismatch', float(depth_error.max()))
    # Tiny coincident-contact discrepancies are reported, never hidden. A
    # substantial ownership mismatch invalidates this render instead of moving
    # the joint geometry or copying a previous generated appearance.
    assert ownership_agreement >= .995, ('Metal primary ownership mismatch', ownership_agreement)
    total = np.zeros((len(xx), 3), dtype=np.float64)
    spp, batch = recipe['integrator']['spp'], recipe['integrator']['spp_batch']
    for start in range(0, spp, batch):
        count = min(batch, spp - start)
        sample_film = np.tile(film, (count, 1))
        sampler = mi.load_dict(dict(type='independent', sample_count=count))
        sampler.seed(int(recipe['integrator']['seed'] + start), wavefront_size=len(sample_film))
        primary, _ = scene.sensors()[0].sample_ray(0., .5, mi.Point2f(sample_film.T), mi.Point2f(.5, .5))
        spectrum, _, _ = scene.integrator().sample(scene, sampler, mi.RayDifferential3f(primary))
        values = np.asarray(spectrum).T.reshape(count, len(xx), 3)
        if not np.isfinite(values).all() or values.min() < -1e-6:
            raise RuntimeError('Path tracer produced invalid radiance')
        total += values.sum(axis=0, dtype=np.float64)
    metal = np.zeros((480, 640, 3), dtype=np.float64)
    metal[yy, xx] = np.maximum(total / spp, 0)
    return metal, dict(
        metal_pixels=len(xx), actual_spp=spp, primary_camera_ray_max_error=ray_error,
        primary_depth_max_error_over_extent=float(depth_error.max()),
        primary_cpu_mitsuba_spoon_ownership_agreement=ownership_agreement,
        coincident_primary_ownership_discrepancies=int(np.count_nonzero(~spoon_hits)),
        scene_includes_food_occlusion_and_reflection=True,
        smooth_vertex_normals=True, primary_sample_positions='Fixed saved CPU pixel centers',
        raw_metal_linear_rgb_min=float(metal[mask].min()), raw_metal_linear_rgb_max=float(metal[mask].max()),
        rgb_boundary_mixing='No primary-ray pixel jitter; replace only CPU ownership 3',
        scope=SCOPE)


def process_case(root, cid, recipe, mi):
    source_info_path = root / cid / 'source_inputs.json'
    frozen_sources = {row['path']: row['sha256'] for row in recipe['inputs']}
    assert sha(source_info_path) == frozen_sources[str(source_info_path)]
    info = read_json(source_info_path)
    assert all(sha(Path(row['path'])) == row['sha256'] for row in info['inputs']), 'Source inputs changed'
    K = np.asarray(info['intrinsics_canvas'], dtype=float)
    assert K.shape == (3, 3) and np.isfinite(K).all()
    assert abs(K[0, 0] - K[1, 1]) < 1e-3 and K[0, 0] > 0
    assert abs(K[0, 2] - 320) < 1e-3 and abs(K[1, 2] - 240) < 1e-3
    assert abs(K[0, 1]) < 1e-8 and abs(K[1, 0]) < 1e-8 and np.allclose(K[2], [0, 0, 1])
    extent = float(max(np.asarray(info['high']) - np.asarray(info['low'])))
    direction, tint, exposure, light_audit = source_light(root, cid, info, recipe)
    cdir = root / 'metal_v2' / cid
    cdir.mkdir(exist_ok=True)
    rows = []
    for pose in recipe['poses']:
        started = time.time()
        pdir = root / cid / ('pose' + str(pose))
        out = cdir / ('pose' + str(pose))
        out.mkdir(exist_ok=True)
        inputs = {name: sha(pdir / name) for name in ['remaining.ply', 'carried.ply', 'spoon.ply', 'geometry_channels.npz']}
        frozen_inputs = {row['path']: row['sha256'] for row in recipe['inputs']}
        assert all(frozen_inputs[str(pdir / name)] == digest for name, digest in inputs.items())
        with np.load(pdir / 'geometry_channels.npz', allow_pickle=False) as data:
            labels = data['labels']
            mask = labels == 3
            depth = data['depth']
        assert mask.shape == depth.shape == (480, 640)
        scene, scene_audit = make_scene(pdir, K, extent, direction, tint, exposure, recipe, mi)
        rendered, trace_audit = trace_metal(scene, K, mask, depth, extent, recipe, mi)
        metal_rgb = srgb(rendered)
        Image.fromarray(metal_rgb).save(out / 'metal_rgb.png')
        np.save(out / 'metal_linear_rgb.npy', rendered.astype(np.float32))
        Image.fromarray(np.uint8(mask) * 255).save(out / 'metal_ownership_mask.png')
        parents = {(row['variant'], row['seed']): row for row in recipe['parents']
                   if row['case_id'] == cid and row['pose'] == pose}
        for variant in recipe['variants']:
            for seed in recipe['seeds']:
                parent_dir = pdir / ('%s_s%d' % (variant, seed))
                parent = parent_dir / 'composited.png'
                expected_parent = parents[(variant, seed)]
                assert sha(parent) == expected_parent['image_sha256']
                record = read_json(parent_dir / 'record.json')
                assert sha(parent_dir / 'record.json') == expected_parent['record_sha256']
                original = image_rgb(parent)
                assert original.shape == metal_rgb.shape
                final = original.copy()
                final[mask] = metal_rgb[mask]
                assert np.array_equal(final[~mask], original[~mask])
                result = out / ('%s_s%d' % (variant, seed))
                result.mkdir(exist_ok=True)
                Image.fromarray(final).save(result / 'composited.png')
                Image.fromarray(final).crop(tuple(record['native_rect'])).save(result / 'view.png')
                row = dict(case_id=cid, pose=pose, seed=seed, variant=variant,
                           parent_path=str(parent), parent_sha256=sha(parent),
                           final_sha256=sha(result / 'composited.png'),
                           metal_rgb_sha256=sha(out / 'metal_rgb.png'),
                           nonmetal_pixels_exact=True, nonmetal_changed_pixels=0,
                           changed_metal_pixels=int(np.count_nonzero(np.any(final != original, axis=2))),
                           raw_generation=False, additional_neural_calls=0,
                           joint_geometry_changed=False, source_recipe_sha256=sha(root / 'metal_v2' / 'recipe.json'), scope=SCOPE)
                write_json(result / 'record.json', row)
                rows.append(row)
        assert all(sha(pdir / name) == digest for name, digest in inputs.items())
        write_json(out / 'render_receipt.json', dict(status='complete_unreviewed', case_id=cid, pose=pose,
                                                   mesh_and_channels_sha256=inputs, lighting=light_audit,
                                                   scene=scene_audit, trace=trace_audit, results=6,
                                                   mitsuba_version=str(mi.__version__), seconds=time.time() - started,
                                                   scope=SCOPE))
        print('LINEAGE_METAL_V2_POSE', cid, pose, 6, flush=True)
        del scene
    write_json(cdir / 'manifest.json', dict(status='complete_unreviewed', case_id=cid, results=rows, scope=SCOPE))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=6)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.root)
        return
    if not args.shards > 0 or not 0 <= args.shard < args.shards:
        raise ValueError('Invalid shard/shards')
    recipe = read_json(args.root / 'metal_v2' / 'recipe.json')
    assert recipe['script_sha256'] == sha(Path(__file__)), 'Script differs from frozen recipe'
    assert recipe['source_config_sha256'] == sha(args.root / 'config.json')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    sys.path.insert(0, MITSUBA_VENDOR)
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')
    rows = []
    status_path = args.root / 'metal_v2' / ('worker_%d.json' % args.shard)
    for index, cid in enumerate(recipe['cases']):
        if index % args.shards != args.shard:
            continue
        rows.extend(process_case(args.root, cid, recipe, mi))
        write_json(status_path, dict(status='running', shard=args.shard, shards=args.shards,
                                     gpu=args.gpu, final_images=len(rows), rows=rows, scope=SCOPE))
    write_json(status_path, dict(status='complete_unreviewed', shard=args.shard, shards=args.shards,
                                 gpu=args.gpu, final_images=len(rows), rows=rows, scope=SCOPE))
    print('LINEAGE_METAL_V2_WORKER_COMPLETE', args.shard, len(rows), flush=True)


if __name__ == '__main__':
    main()
