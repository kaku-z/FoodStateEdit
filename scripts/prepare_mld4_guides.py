"""CPU-only reconstruction guides from an executed MLD3 material transport.

The cut is an unknown region, never a rendered cavity. Geometry and material
correspondence constrain a subsequent image reconstruction; they are not evidence
that the hidden food surface, source exposure, or strand continuation is known.
Only numpy and Pillow are required. No perception or generation models run here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def read_rgb(path):
    return np.asarray(Image.open(path).convert('RGB')).copy()


def read_mask(path):
    return np.asarray(Image.open(path).convert('L')) > 127


def save_image(path, array):
    array = np.asarray(array)
    Image.fromarray(array.astype(np.uint8) * 255 if array.dtype == bool else
                    np.clip(array, 0, 255).astype(np.uint8)).save(path)


def dilate(mask, radius):
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(
        ImageFilter.MaxFilter(2 * int(radius) + 1))) > 0 if radius else mask.copy()


def erode(mask, radius):
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(
        ImageFilter.MinFilter(2 * int(radius) + 1))) > 0 if radius else mask.copy()


def blur(array, radius):
    return np.asarray(Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius))).astype(np.float32)


def fill_mask_holes(mask):
    """Flood from the padded exterior so internal ingredient holes are not rims."""
    padded = np.pad(mask.astype(np.uint8)*255, 1)
    image = Image.fromarray(padded).copy()
    ImageDraw.floodfill(image, (0, 0), 127)
    return np.asarray(image)[1:-1, 1:-1] != 127


def mask_components(mask):
    active = set(zip(*np.where(mask)))
    while active:
        seed = active.pop()
        component, queue = [seed], [seed]
        while queue:
            y, x = queue.pop()
            for dy, dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
                neighbor = (y+dy, x+dx)
                if neighbor in active:
                    active.remove(neighbor)
                    component.append(neighbor)
                    queue.append(neighbor)
        yield np.asarray(component, int)


def source_boundary_control(food, removed, unknown, excluded, scale):
    """An open external silhouette hypothesis, never a closed cavity outline."""
    before = fill_mask_holes(food)
    after = fill_mask_holes(before & ~removed)
    exposed = before & ~after
    notch = bool(exposed.any())
    control = np.zeros(food.shape, np.float32)
    components_kept = 0
    sigma = max(1., 2.2*scale)
    if notch:
        smooth = fill_mask_holes(blur(after*255, sigma) >= 127.5)
        silhouette = smooth & ~erode(smooth, 1)
        domain = unknown & ~excluded
        rim = domain & ~erode(domain, 2)
        candidate = silhouette & domain
        accepted = np.zeros_like(candidate)
        for component in mask_components(candidate):
            yy, xx = component.T
            contacts = component[rim[yy, xx]]
            # Enclosed components would reintroduce a closed cavity outline.
            if len(contacts) < 2 or np.linalg.norm(np.ptp(contacts, axis=0)) < max(3., 4*scale):
                continue
            accepted[yy, xx] = True
            components_kept += 1
        control = blur(accepted*255, max(.55, .8*scale))
        if control.max() > 0:
            control *= 160 / control.max()
        control[~domain] = 0
        control[control < 2] = 0
    return control, dict(enabled=True, removal_reaches_exterior=notch,
        exterior_connected_removal_pixels=int(exposed.sum()),
        interior_removed_pixels_left_unknown=int((removed & after).sum()),
        open_boundary_components=components_kept, boundary_pixels=int(np.count_nonzero(control)),
        maximum_amplitude=160, smoothing_sigma_pixels=sigma,
        rule='Fill original mask holes; subtract removed footprint; fill holes again. Only exterior-connected removal can alter the external silhouette. Smooth the remaining support, clip its contour to the unknown domain, and reject enclosed components.',
        assumptions='A local remaining-food silhouette hypothesis, not an observed cut surface, exposed-layer label, or mass measurement. Internal removals stay blank; no cavity floor or closed cavity perimeter is supplied.')


def project(xyz, intrinsic):
    xyh = np.asarray(xyz) @ intrinsic.T
    return xyh[:, :2] / np.maximum(xyh[:, 2:], 1e-8)


def fill_unknown(source, unknown):
    """Masked normalized convolution initialization, explicitly not a prediction."""
    known = ~unknown
    numerator = source.astype(np.float32) * known[..., None]
    filled = source.astype(np.float32).copy()
    pending = unknown.copy()
    for radius in (3, 6, 12, 24, 48, 96):
        support = blur(known * 255, radius) / 255
        average = blur(numerator, radius) / np.maximum(support[..., None], 1 / 255)
        take = pending & (support > .12)
        filled[take] = average[take]
        pending[take] = False
        if not pending.any():
            break
    if pending.any():
        filled[pending] = np.median(source[known], axis=0)
    return np.clip(filled, 0, 255).astype(np.uint8)


def raster_material(transport, intrinsic, shape, mode, sides):
    """Perspective-correct UV/depth/confidence and nearest-vertex material ID."""
    h, w = shape
    xyz, faces = transport['target_xyz'], transport['faces']
    uv = transport['target_source_uv']
    screen = project(xyz, intrinsic)
    confidence = np.zeros(len(xyz), np.float32)
    if mode == 'strand':
        # Only the camera-facing half of the original cylinder has observed RGB.
        confidence = np.maximum(0, np.cos(np.arange(len(xyz)) % sides * 2 * np.pi / sides)).astype(np.float32)
    else:
        # MLD3 stores the photographed top followed by the inferred lower closure.
        confidence[:len(xyz) // 2] = 1
    packed = np.c_[uv, transport['target_material_rgb'], confidence]
    depth = np.full((h, w), np.inf, np.float32)
    attrs = np.zeros((h, w, packed.shape[1]), np.float32)
    ids = np.full((h, w), -1, np.int64)
    material_ids = transport['carried_ids']
    for face in faces:
        p = screen[face]
        z = xyz[face, 2]
        if (z <= 1e-6).any() or not np.isfinite(p).all():
            continue
        x0, y0 = np.maximum(0, np.floor(p.min(axis=0)).astype(int))
        x1, y1 = np.minimum([w - 1, h - 1], np.ceil(p.max(axis=0)).astype(int))
        if x0 > x1 or y0 > y1:
            continue
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        xx, yy = xx + .5, yy + .5
        a, b, c = p
        denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denom) < 1e-10:
            continue
        u = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / denom
        v = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / denom
        bary = np.stack([u, v, 1 - u - v], axis=-1)
        inside = np.min(bary, axis=-1) >= -1e-6
        weights = bary / z
        invz = weights.sum(axis=-1)
        here = np.divide(1., invz, out=np.full_like(invz, np.inf), where=invz > 0)
        sub = depth[y0:y1 + 1, x0:x1 + 1]
        take = inside & (here < sub)
        if not take.any():
            continue
        sub[take] = here[take]
        weights /= np.maximum(invz[..., None], 1e-10)
        interpolated = weights @ packed[face]
        hidden_face = (np.all(face // sides == face[0] // sides) if mode == 'strand'
                       else not np.all(face < len(xyz)//2))
        if hidden_face:
            interpolated[..., 5] = 0  # End caps / closure sides were never photographed.
        attrs[y0:y1 + 1, x0:x1 + 1][take] = interpolated[take]
        ids[y0:y1 + 1, x0:x1 + 1][take] = material_ids[face[np.argmax(weights, axis=-1)]][take]
    return depth, attrs, ids


def smooth_curve(values, factor=5):
    """Interpolating Hermite curve; node anchors and endpoints are retained."""
    if len(values) < 3:
        t = np.linspace(0, 1, factor + 1)[:, None]
        return values[:1] * (1 - t) + values[-1:] * t
    tangent = np.gradient(values, axis=0)
    result = []
    t = np.linspace(0, 1, factor, endpoint=False)[:, None]
    for i in range(len(values) - 1):
        segment = ((2*t**3-3*t**2+1)*values[i] + (t**3-2*t**2+t)*tangent[i]
                   + (-2*t**3+3*t**2)*values[i+1] + (t**3-t**2)*tangent[i+1])
        # Limit tiny spline excursions near a terminal or sharp projected bend.
        pad = .15 * np.abs(values[i+1] - values[i])
        segment = np.clip(segment, np.minimum(values[i], values[i+1])-pad,
                          np.maximum(values[i], values[i+1])+pad)
        result.append(segment)
    return np.r_[np.concatenate(result), values[-1:]]


def sample_source(source, uv, pixel_centers=False):
    """Bilinear observed texture lookup, separate from the smoothed RGB guide."""
    h, w = source.shape[:2]
    uv = uv - .5 if pixel_centers else uv
    x, y = np.clip(uv[:, 0], 0, w-1), np.clip(uv[:, 1], 0, h-1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
    fx, fy = (x-x0)[:, None], (y-y0)[:, None]
    return ((1-fx)*(1-fy)*source[y0, x0] + fx*(1-fy)*source[y0, x1]
            + (1-fx)*fy*source[y1, x0] + fx*fy*source[y1, x1])


def strand_appearance(transport, intrinsic, source, sides, target_mask):
    """Smooth tube guide along existing curves, without joining unknown strands."""
    h, w = target_mask.shape
    rings = transport['target_xyz'].reshape(-1, sides, 3)
    nodes = rings.mean(axis=1)
    offsets = transport['strand_curve_offsets'].astype(int)
    node_uv = transport['strand_node_texture_exemplar_uv']
    pixels = np.rint(node_uv).astype(int)
    colors = source[np.clip(pixels[:, 1], 0, h-1), np.clip(pixels[:, 0], 0, w-1)].astype(float)
    radii_world = np.linalg.norm(rings - nodes[:, None], axis=2).mean(axis=1)
    points = project(nodes, intrinsic)
    radii = radii_world * np.sqrt(intrinsic[0, 0] * intrinsic[1, 1]) / nodes[:, 2]
    rgb = np.zeros_like(source, np.float32)
    zbuffer = np.full((h, w), np.inf)
    lines = Image.new('L', (w*3, h*3))
    draw = ImageDraw.Draw(lines)
    for a, b in zip(offsets[:-1], offsets[1:]):
        curve = smooth_curve(np.c_[points[a:b], nodes[a:b, 2], radii[a:b], radii_world[a:b], colors[a:b]])
        draw.line([tuple(p * 3) for p in curve[:, :2]], fill=220, width=3)
        # Dense capsule sampling avoids polygonal sides and visibly flat caps.
        for first, second in zip(curve[:-1], curve[1:]):
            count = max(2, int(np.linalg.norm(second[:2]-first[:2]) / .4) + 1)
            for t in np.linspace(0, 1, count, endpoint=False):
                x, y, z, radius, world_radius, *color = first*(1-t)+second*t
                radius = max(.6, radius)
                x0, y0 = np.maximum(0, np.floor([x-radius-1, y-radius-1]).astype(int))
                x1, y1 = np.minimum([w-1, h-1], np.ceil([x+radius+1, y+radius+1]).astype(int))
                if x0 > x1 or y0 > y1:
                    continue
                yy, xx = np.mgrid[y0:y1+1, x0:x1+1]
                dx, dy = (xx+.5-x)/radius, (yy+.5-y)/radius
                r2 = dx*dx+dy*dy
                normal_z = np.sqrt(np.maximum(0, 1-r2))
                depth = z-world_radius*normal_z
                take = (r2 <= 1) & (depth < zbuffer[y0:y1+1, x0:x1+1])
                zbuffer[y0:y1+1, x0:x1+1][take] = depth[take]
                # This deliberately weak shading is a synthesis cue, not relighting evidence.
                shade = np.clip(.81 + .19*normal_z-.055*dx-.085*dy, .65, 1.12)
                value = np.asarray(color)[None, None] * shade[..., None]
                rgb[y0:y1+1, x0:x1+1][take] = value[take]
    line = np.asarray(lines.resize((w, h), Image.Resampling.LANCZOS)).copy()
    line[~dilate(target_mask, 2)] = 0
    return rgb, np.isfinite(zbuffer) & dilate(target_mask, 2), line


def spoon_depth(path, intrinsic, shape, mask):
    """Read the actual MLD3 PLY vertices and interpolate their projected depth."""
    data = path.read_bytes()
    end = data.index(b'end_header\n') + len(b'end_header\n')
    header = data[:end].decode('ascii').splitlines()
    fields, count, active = [], 0, False
    types = {'float': '<f4', 'double': '<f8', 'uchar': 'u1', 'int': '<i4'}
    for line in header:
        if line.startswith('element vertex '):
            count, active = int(line.split()[-1]), True
        elif line.startswith('element '):
            active = False
        elif active and line.startswith('property '):
            _, kind, name = line.split()
            fields.append((name, types[kind]))
    if 'format binary_little_endian 1.0' not in header:
        raise ValueError('Expected the binary little-endian MLD3 spoon PLY')
    vertices = np.frombuffer(data, dtype=np.dtype(fields), count=count, offset=end)
    xyz = np.column_stack([vertices[k] for k in ('x', 'y', 'z')])
    uv = np.rint(project(xyz, intrinsic)-.5).astype(int)
    h, w = shape
    valid = (uv[:, 0]>=0)&(uv[:, 0]<w)&(uv[:, 1]>=0)&(uv[:, 1]<h)&(xyz[:, 2]>0)
    result = np.full(h*w, np.inf, np.float32)
    np.minimum.at(result, uv[valid, 1]*w+uv[valid, 0], xyz[valid, 2])
    result = result.reshape(h, w)
    # Dense projected vertices leave small raster gaps; average only known neighbors.
    for _ in range(12):
        missing = mask & ~np.isfinite(result)
        if not missing.any():
            break
        finite = np.isfinite(result)
        values = np.pad(np.where(finite, result, 0), 1)
        weights = np.pad(finite.astype(float), 1)
        total = sum(values[1+dy:1+dy+h, 1+dx:1+dx+w] for dy, dx in ((0,1),(0,-1),(1,0),(-1,0)))
        weight = sum(weights[1+dy:1+dy+h, 1+dx:1+dx+w] for dy, dx in ((0,1),(0,-1),(1,0),(-1,0)))
        fill = missing & (weight > 0)
        result[fill] = total[fill] / weight[fill]
    return result


def prepare_case(folder, input_root, output, args):
    output.mkdir(parents=True, exist_ok=True)
    state = json.loads((folder / 'state.json').read_text(encoding='utf-8'))
    source = read_rgb(folder / 'source.png')
    final = read_rgb(folder / 'final.png')
    removed = read_mask(folder / 'source_removed_mask.png')
    target = read_mask(folder / 'moved_food_mask.png')
    spoon = read_mask(folder / 'spoon_mask.png')
    with np.load(folder / 'transport.npz') as payload:
        transport = {key: payload[key] for key in payload.files}
    intrinsic = transport['camera_intrinsics']
    h, w = removed.shape
    observations = input_root / 'observations' / folder.name
    if not observations.exists():
        observations = Path(state['source_observation'])
    with np.load(observations / 'depth.npz') as payload:
        source_depth = payload['depth'].copy().astype(np.float32)
        source_depth_valid = payload['mask'].copy() & np.isfinite(source_depth) & (source_depth > 0)
    protected = read_mask(observations / 'excluded_ingredients.png') if (observations / 'excluded_ingredients.png').exists() else np.zeros_like(removed)
    scale = max(h, w) / 640
    radius = max(3, round(args.source_margin * scale))
    source_unknown = dilate(removed, radius) & ~protected
    source_unknown |= removed
    target_margin = max(2, round(args.target_margin * scale))
    target_slack = dilate(target, target_margin)
    spoon_slack = dilate(spoon, max(2, round(3*scale)))
    inpaint = source_unknown | target_slack | spoon_slack
    context = fill_unknown(source, source_unknown)
    coarse = context.copy()
    coarse[spoon] = final[spoon]
    coarse[target] = final[target]
    sides = int(state.get('tube_sides') or 10)
    depth, attrs, ids = raster_material(transport, intrinsic, (h, w), state['mode'], sides)
    centerline = np.zeros((h, w), np.uint8)
    if state['mode'] == 'strand':
        smooth_rgb, smooth_mask, centerline = strand_appearance(transport, intrinsic, source, sides, target)
        # Preserve the exact observed-transport footprint; softness belongs to the guide.
        take = smooth_mask & target
        coarse[take] = np.clip(smooth_rgb[take], 0, 255).astype(np.uint8)
    elif np.isfinite(depth).any():
        observed = target & np.isfinite(depth) & (attrs[..., 5] > .65)
        coarse[observed] = np.clip(attrs[..., 2:5][observed], 0, 255).astype(np.uint8)
    source_context = source.copy()
    source_context[source_unknown] = context[source_unknown]
    gray = blur(source @ np.array([.299, .587, .114]), 1.0)
    gy, gx = np.gradient(gray)
    edge = np.clip(np.hypot(gx, gy)*5, 0, 145)
    edge[source_unknown | dilate(target | spoon, 2)] = 0
    silhouette = dilate(target, 1) ^ erode(target, 1)
    spoon_edge = dilate(spoon, 1) ^ erode(spoon, 1)
    edge[silhouette] = 220
    edge[spoon_edge & ~target] = 210
    edge = np.maximum(edge, centerline*.70)
    # Do not draw either cut topology or inferred cavity contours into ControlNet.
    edge[source_unknown & ~target_slack & ~spoon_slack] = 0
    source_boundary = np.zeros((h, w), np.float32)
    source_shape = dict(enabled=False, removal_reaches_exterior=None, boundary_pixels=0)
    if args.source_shape_control:
        food = read_mask(folder / 'source_food_mask.png')
        source_boundary, source_shape = source_boundary_control(food, removed, source_unknown,
            target_slack | spoon_slack, scale)
        edge = np.maximum(edge, source_boundary)
    metric_depth = source_depth.copy()
    depth_valid = source_depth_valid.copy()
    metric_depth[source_unknown] = np.nan
    depth_valid[source_unknown] = False
    sd = spoon_depth(folder / 'spoon.ply', intrinsic, (h, w), spoon)
    take = spoon & np.isfinite(sd)
    metric_depth[take], depth_valid[take] = sd[take], True
    take = target & np.isfinite(depth)
    metric_depth[take], depth_valid[take] = depth[take], True
    valid_values = metric_depth[depth_valid]
    low, high = np.quantile(valid_values, [.02, .98])
    depth_image = np.zeros((h, w), np.float32)
    depth_image[depth_valid] = 255*(1-np.clip((metric_depth[depth_valid]-low)/max(high-low, 1e-7), 0, 1))
    # 127 is neutral/unknown for the visualization; validity is explicit in NPZ.
    depth_image[~depth_valid] = 127
    target_valid = target & np.isfinite(depth)
    source_uv = np.full((h, w, 2), np.nan, np.float32)
    source_uv[target_valid] = attrs[..., :2][target_valid]
    appearance_confidence = np.zeros((h, w), np.float32)
    appearance_confidence[target_valid] = attrs[..., 5][target_valid]
    material_texture = coarse.copy()
    material_texture[target_valid] = np.clip(sample_source(source, source_uv[target_valid],
        pixel_centers=state['mode'] != 'strand'), 0, 255).astype(np.uint8)
    material_sigma = max(1.5, 3*scale)
    material_weight = blur(appearance_confidence*255, material_sigma)/255
    material_average = blur(material_texture*appearance_confidence[..., None], material_sigma)
    material_average /= np.maximum(material_weight[..., None], 1/255)
    lowfrequency_valid = target & (appearance_confidence > 0) & (material_weight > .02)
    material_lowfrequency = material_texture.copy()
    material_lowfrequency[lowfrequency_valid] = np.clip(material_average[lowfrequency_valid], 0, 255).astype(np.uint8)
    lowfrequency_confidence = np.where(lowfrequency_valid, appearance_confidence, 0).astype(np.float32)
    ids[~target_valid] = -1
    region = np.zeros((h, w), np.uint8)
    region[source_unknown] = 1
    region[spoon] = 2
    region[target] = 3
    y, x = np.where(removed)
    padding = max(5, round(.22 * max(np.ptp(x)+1, np.ptp(y)+1)))
    bbox = [max(0, int(x.min())-padding), max(0, int(y.min())-padding),
            min(w, int(x.max())+padding+1), min(h, int(y.max())+padding+1)]
    reference = source[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    images = dict(source=source, coarse=coarse, source_context=source_context,
        inpaint_mask=inpaint, source_removal_mask=removed, source_inpaint_mask=source_unknown,
        target_food_mask=target, target_tolerance_mask=target_slack, spoon_mask=spoon,
        edge=edge, depth=depth_image, depth_valid_mask=depth_valid,
        material_confidence=appearance_confidence*255, material_texture=material_texture, source_reference=reference,
        material_lowfrequency=material_lowfrequency, material_lowfrequency_confidence=lowfrequency_confidence*255,
        source_reference_mask=removed[bbox[1]:bbox[3], bbox[0]:bbox[2]], strand_centerline=centerline,
        source_boundary_control=source_boundary)
    for name, array in images.items():
        save_image(output / (name+'.png'), array)
    np.savez_compressed(output / 'guide_channels.npz',
        depth=metric_depth, depth_valid=depth_valid, source_depth=source_depth,
        source_depth_valid=source_depth_valid, target_depth=depth,
        target_source_uv=source_uv, target_material_id=ids,
        observed_material_confidence=appearance_confidence, material_confidence=appearance_confidence,
        material_lowfrequency_rgb=material_lowfrequency.astype(np.float32)/255,
        material_lowfrequency_confidence=lowfrequency_confidence, source_boundary_control=source_boundary/255,
        source_removed=removed, source_unknown=source_unknown, target_food=target,
        target_tolerance=target_slack, spoon=spoon, inpaint=inpaint, region=region,
        camera_intrinsics=intrinsic, source_reference_bbox=np.asarray(bbox),
        carried_ids=transport['carried_ids'], target_ids=transport['target_ids'])
    manifest = dict(case_id=folder.name, index=state['index'], mode=state['mode'],
        guide_revision='open_source_boundary_v1' if args.source_shape_control else 'hidden_surface_confidence_v2',
        source_prompt=state['prompt'], prompt=state['prompt'], food_prompt=state['prompt'], source_case=str(folder.resolve()),
        source_sha256=hashlib.sha256((folder/'source.png').read_bytes()).hexdigest(),
        size=[w, h], source_reference_bbox=bbox, source_margin_pixels=radius,
        target_tolerance_pixels=target_margin,
        source_removed_pixels=int(removed.sum()), source_unknown_pixels=int(source_unknown.sum()),
        target_food_pixels=int(target.sum()), inpaint_pixels=int(inpaint.sum()),
        target_correspondence_coverage=float(target_valid.sum()/max(1, target.sum())),
        protected_ingredient_edit_pixels=int((protected & source_unknown & ~removed).sum()),
        source_cavity_edge_pixels=int(np.count_nonzero(edge[source_unknown & ~target_slack & ~spoon_slack & (source_boundary == 0)])),
        source_unknown_edge_pixels=int(np.count_nonzero(edge[source_unknown & ~target_slack & ~spoon_slack])),
        source_shape_control=source_shape,
        geometry_identifiers_preserved=bool(np.array_equal(transport['carried_ids'], transport['target_ids'])),
        region_labels={'0':'preserved photographic context','1':'unknown source exposure','2':'spoon','3':'transported food'},
        depth_convention='Camera z in the inferred MLD3 scale; NPZ validity is authoritative. PNG near=white, far=black; invalid=127.',
        source_uv_convention=state.get('source_uv_convention', 'as stored by MLD3'),
        source_cut='Dilated uncertainty domain initialized only from surrounding unremoved pixels. No cavity mesh RGB or cavity edges are used.',
        material_confidence_scope='Original visible top or camera-facing strand half only. Correspondence is an appearance exemplar and geometry identity, not evidence for hidden texture.',
        material_texture_scope='Perspective-correct bilinear source UV samples. Use material_texture.png, weighted by material_confidence, for source-detail projection; coarse.png is the softer generative initialization.',
        material_lowfrequency_scope='Confidence-normalized Gaussian average of observed material only, restricted to the same target mask and observed-material confidence; never incorporates background or hidden/cap colors. Apply only with material_lowfrequency_confidence.',
        material_lowfrequency_sigma_pixels=material_sigma,
        strand_guide='Existing separate transported curve anchors, interpolated smooth capsules. No missing continuation, new strand connection, or physical topology is asserted.',
        reconstruction_contract='Freeze pixels outside inpaint_mask after synthesis. Use source_reference as material exemplar, not target layout. Keep target_food_mask within target_tolerance_mask; reconstruct source_unknown from context. Do not project source exposure onto generated hidden food.',
        images={name: name+'.png' for name in images}, generated=False,
        file_hashes={name+'.png': hashlib.sha256((output/(name+'.png')).read_bytes()).hexdigest() for name in images},
        channels_sha256=hashlib.sha256((output/'guide_channels.npz').read_bytes()).hexdigest(),
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'guide_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--indices', type=int, nargs='+')
    parser.add_argument('--source-margin', type=int, default=9, help='Uncertainty dilation at 640 px, including old cut edges and shadows.')
    parser.add_argument('--target-margin', type=int, default=5, help='Allowed synthesis silhouette tolerance at 640 px.')
    parser.add_argument('--source-shape-control', action='store_true', help='Add a smooth open source boundary only for exterior-connected removal; use a separate output directory for this ablation.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_root = args.input_root/'real' if (args.input_root/'real').exists() else args.input_root
    rows = []
    for folder in sorted(source_root.glob('real_*')):
        if not (folder/'state.json').exists():
            continue
        index = int(folder.name.split('_')[1])
        if args.indices is not None and index not in args.indices:
            continue
        rows.append(prepare_case(folder, args.input_root, args.output/folder.name, args))
        print(json.dumps({key: rows[-1][key] for key in ('case_id','mode','source_unknown_pixels','target_correspondence_coverage')}), flush=True)
    (args.output/'guides_manifest.json').write_text(json.dumps(dict(cases=rows, source_only=True, generated=False), indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
