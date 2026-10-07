"""Fixed held-out topology probes: dense cached fields and shared-material lift."""
from pathlib import Path
import argparse
import json
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.ndimage import map_coordinates
from PIL import Image, ImageDraw, ImageFont
import torch
from foodstateedit.material_lineage.mld2 import MLD2Config, MLD2Model
from foodstateedit.material_lineage.mld2_depth import DepthSurfaceMLD2, DepthSurfaceDetachedMLD2
from train_mld2 import Data, make_actions
from mld2_scene import shape_sdf, material_rgb
from mld2_lowpass import attach_lowpass


def rotation_matrix(a):
    theta = np.linalg.norm(a)
    if theta < 1e-10:
        return np.eye(3)
    x, y, z = a/theta
    k = np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3)+math.sin(theta)*k+(1-math.cos(theta))*(k@k)


def view_basis(yaw=30, elevation=35):
    yaw, elev = np.deg2rad([yaw, elevation])
    return np.asarray([[-np.sin(yaw), np.cos(yaw), 0],
                       [-np.sin(elev)*np.cos(yaw), -np.sin(elev)*np.sin(yaw), np.cos(elev)],
                       [np.cos(elev)*np.cos(yaw), np.cos(elev)*np.sin(yaw), np.sin(elev)]])


def linear_to_srgb(rgb):
    rgb = np.clip(rgb, 0, 1)
    return np.where(rgb <= .0031308, 12.92*rgb, 1.055*rgb**(1/2.4)-.055)


def grid_sdf(field, xyz):
    n = len(field)
    coord = ((xyz+1)*n/2-.5).T
    return map_coordinates(field, coord, order=1, mode='constant', cval=1)*.5


def branch_sdf(field, xyz, action=None, carried=False):
    canonical = xyz
    if carried:
        canonical = (xyz-action[4:7])@rotation_matrix(action[7:10])
    value = grid_sdf(field, canonical)
    if action is not None:
        plane = canonical@action[:3]-action[3]
        value = np.maximum(value, -plane if carried else plane)
    return value, canonical


@torch.inference_mode()
def render(model, cache, field, device, yaw=30, action=None, size=160, true_parameters=None):
    basis = view_basis(yaw)
    scale = 1.5 if action is None else 1.75
    p = (np.arange(size)+.5)/size*2-1
    u, v = np.meshgrid(p, -p)
    origin = (u[..., None]*scale*basis[0]+v[..., None]*scale*basis[1]+2.8*basis[2]).reshape(-1, 3)
    direction = -basis[2]
    distance = np.zeros(len(origin))
    active = np.ones(len(origin), dtype=bool)
    hit = np.zeros(len(origin), dtype=bool)
    branch = np.zeros(len(origin), dtype=np.uint8)
    for _ in range(170):
        selected = np.flatnonzero(active)
        if not len(selected):
            break
        xyz = origin[selected]+distance[selected, None]*direction
        sd, _ = branch_sdf(field, xyz, action)
        if action is not None:
            lifted, _ = branch_sdf(field, xyz, action, True)
            use_lifted = lifted < sd
            sd = np.minimum(sd, lifted)
        else:
            use_lifted = np.zeros(len(selected), dtype=bool)
        arrived = sd < .008
        hit[selected[arrived]] = True
        branch[selected[arrived]] = use_lifted[arrived]
        active[selected[arrived]] = False
        marching = selected[~arrived]
        distance[marching] += np.clip(sd[~arrived]*.6, .004, .12)
        active[marching] &= distance[marching] < 5.6
    rgb = np.full((len(origin), 3), .91)
    indices = np.flatnonzero(hit)
    if len(indices):
        xyz = origin[indices]+distance[indices, None]*direction
        carried = branch[indices].astype(bool)
        canonical = xyz.copy()
        if action is not None:
            canonical[carried] = (xyz[carried]-action[4:7])@rotation_matrix(action[7:10])
        if true_parameters is None:
            color_parts = []
            for start in range(0, len(canonical), 8192):
                point = torch.from_numpy(canonical[start:start+8192].astype(np.float32))[None].to(device)
                color_parts.append(((model.query(cache, point)['rgb'][0]+1)/2).cpu().numpy())
            color = np.concatenate(color_parts)
        else:
            color = material_rgb(canonical, true_parameters)
        normal = np.zeros_like(xyz)
        for axis in range(3):
            offset = np.eye(3)[axis]*.014
            plus, _ = branch_sdf(field, xyz+offset, action)
            minus, _ = branch_sdf(field, xyz-offset, action)
            if action is not None:
                a, _ = branch_sdf(field, xyz+offset, action, True)
                b, _ = branch_sdf(field, xyz-offset, action, True)
                plus, minus = np.minimum(plus, a), np.minimum(minus, b)
            normal[:, axis] = plus-minus
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
        light = np.asarray([-.2, -.5, 1.])
        light /= np.linalg.norm(light)
        shade = .62+.35*np.maximum(normal@light, 0)
        rgb[indices] = color*shade[:, None]
    return Image.fromarray((linear_to_srgb(rgb.reshape(size, size, 3))*255).round().astype(np.uint8))


def main(args):
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model_class = DepthSurfaceMLD2 if checkpoint.get('model_variant') == 'depth_surface' else MLD2Model
    if checkpoint.get('observation_maps_detached'):
        model_class = DepthSurfaceDetachedMLD2
    model = model_class(MLD2Config(**checkpoint['model_config'])).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'])
    if 'coordinate_lowpass_max_frequency' in checkpoint:
        attach_lowpass(model, checkpoint['coordinate_lowpass_max_frequency'])
    data = Data(args.dataset_root)
    selected = data.groups[1 if args.split == 'validation' else 2][:8]
    centers = (np.arange(args.grid_size)+.5)*2/args.grid_size-1
    points = np.stack(np.meshgrid(centers, centers, centers, indexing='ij'), -1).reshape(-1, 3).astype(np.float32)
    columns = 6 if args.with_truth else 4
    canvas = Image.new('RGB', (columns*240, len(selected)*265+35), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    except OSError:
        font = ImageFont.load_default()
    labels = ['Observed synthetic image', 'Predicted source / front', 'Predicted source / side', 'Shared state / cut + lift']
    if args.with_truth:
        labels = ['Observed synthetic image', 'GT source / front', 'Predicted / front', 'GT source / side', 'Predicted / side', 'Shared state / cut + lift']
    for column, label in enumerate(labels):
        draw.text((column*240+8, 10), label, fill='black', font=font)
    records = []
    with torch.inference_mode():
        for row, index in enumerate(selected):
            image = np.asarray(data.a['source_rgb'][index], dtype=np.float32).transpose(2, 0, 1)/255
            cache = model.encode_image(torch.from_numpy(image)[None].to(device))
            field_parts = []
            for start in range(0, len(points), 16384):
                point = torch.from_numpy(points[start:start+16384])[None].to(device)
                field_parts.append(model.query(cache, point)['sdf'][0, :, 0].cpu().numpy())
            field = np.concatenate(field_parts).reshape((args.grid_size,)*3)
            record = data.records[int(index)]
            action = make_actions([record['scene_id']])[0]
            views = [Image.fromarray(np.asarray(data.a['source_rgb'][index])),
                     render(model, cache, field, device, size=args.image_size),
                     render(model, cache, field, device, yaw=-70, size=args.image_size),
                     render(model, cache, field, device, action=action, size=args.image_size)]
            if args.with_truth:
                parameters = record['target_generator_parameters']
                true_field = (np.clip(shape_sdf(points, parameters), -.5, .5)/.5).reshape(field.shape)
                views = [views[0], render(model, cache, true_field, device, size=args.image_size, true_parameters=parameters),
                         views[1], render(model, cache, true_field, device, yaw=-70, size=args.image_size, true_parameters=parameters),
                         views[2], views[3]]
            for column, view in enumerate(views):
                view.save(output/f'{index}_{column}.png')
                canvas.paste(view.resize((230, 230), Image.Resampling.BICUBIC), (column*240+5, row*265+35))
            draw.text((8, row*265+268), f'{index} / {record["shape_family"]}', fill='black', font=font)
            records.append({'scene_index': int(index), 'scene_id': record['scene_id'],
                            'shape_family': record['shape_family'], 'action': action.tolist(),
                            'checkpoint_step': checkpoint['optimizer_steps']})
            print(json.dumps(records[-1]), flush=True)
    canvas.save(output/'synthetic_fields_and_lift.png')
    (output/'manifest.json').write_text(json.dumps({'scope': 'Fixed first eight scenes of selected split; first eight test scenes were used to diagnose alias and are development diagnostics, not blind test. Front/side use one cached continuous field; both action branches reuse canonical material. No photograph truth or spoon/contact claim.', 'split': args.split,
        'checkpoint': str(args.checkpoint), 'grid_size': args.grid_size, 'records': records}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--grid-size', type=int, default=64)
    p.add_argument('--image-size', type=int, default=160)
    p.add_argument('--split', choices=['validation', 'test'], default='test')
    p.add_argument('--with-truth', action='store_true')
    main(p.parse_args())
