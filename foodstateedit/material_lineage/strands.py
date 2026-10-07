"""Observed-ridge material strands and length-constrained utensil transport.

The graph describes visible image curves, not the unobserved topology of a meal.
Every node and texture sample keeps a source-image identity; no class-conditioned
noodle shape, colour, or hidden continuation is introduced.
"""
from dataclasses import dataclass, field
import copy

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, binary_dilation
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from .spoon_surface import surface_height_gradient, surface_domain


@dataclass
class StrandBundle:
    curves: list
    ridge_response: np.ndarray
    skeleton: np.ndarray
    removal_mask: np.ndarray
    metrics: dict = field(default_factory=dict)

    @property
    def ridge_mask(self):
        return self.skeleton


def _sample(array, uv):
    coordinates = np.asarray(uv).T[::-1]
    if array.ndim == 2:
        return map_coordinates(array, coordinates, order=1, mode='nearest')
    return np.stack([map_coordinates(array[..., j], coordinates, order=1,
        mode='nearest') for j in range(array.shape[-1])], axis=-1)


def _source_curve_geometry(points,uv,normal_uv,width_pixels):
    surface=_sample(np.asarray(points,float),uv)
    across=_sample(np.asarray(points,float),uv+normal_uv*width_pixels[:,None])
    radius=np.linalg.norm(across/across[:,2:]-surface/surface[:,2:],axis=1)*surface[:,2]
    radius=np.clip(radius,np.quantile(radius,.2)*.65,np.quantile(radius,.7)*1.35)
    ray=surface/np.linalg.norm(surface,axis=1)[:,None]
    return surface,surface+radius[:,None]*ray,radius


def bilinear_footprint_owned(mask,uv):
    """All nonzero-weight source texels, including ingredient-hole boundaries."""
    uv=np.asarray(uv);base=np.floor(uv).astype(int);fraction=uv-base
    owned=np.ones(len(uv),bool);h,w=mask.shape
    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
        pixel=base+[dx,dy]
        weight=(fraction[:,0] if dx else 1-fraction[:,0])*(fraction[:,1] if dy else 1-fraction[:,1])
        valid=(pixel[:,0]>=0)&(pixel[:,0]<w)&(pixel[:,1]>=0)&(pixel[:,1]<h)
        value=mask[np.clip(pixel[:,1],0,h-1),np.clip(pixel[:,0],0,w-1)]
        owned &= (weight==0)|(valid&value)
    return owned


def _ingredient_exemplar_uv(mask,geometry_uv):
    exemplar=geometry_uv.copy();mixed=~bilinear_footprint_owned(mask,geometry_uv)
    # Geometry remains at its observed fractional camera ray. For a texture
    # touching excluded ingredient texels, use the nearest observed valid
    # integer ingredient texel and declare that exemplar separately.
    for i in np.flatnonzero(mixed):
        base=np.floor(geometry_uv[i]).astype(int)
        corners=base+np.array([[0,0],[1,0],[0,1],[1,1]])
        valid=(corners[:,0]>=0)&(corners[:,0]<mask.shape[1])&(corners[:,1]>=0)&(corners[:,1]<mask.shape[0])
        corners=corners[valid];corners=corners[mask[corners[:,1],corners[:,0]]]
        exemplar[i]=corners[np.argmin(np.linalg.norm(corners-geometry_uv[i],axis=1))]
    return exemplar


def _tube_texture_uv(curve,sides):
    theta=np.arange(sides)*2*np.pi/sides
    cross=curve['radius_pixels'][:,None]*np.sin(theta)[None]
    cross=np.clip(cross,-curve.get('cross_section_negative_pixels',curve['radius_pixels'])[:,None],curve.get('cross_section_positive_pixels',curve['radius_pixels'])[:,None])
    return curve.get('texture_center_uv',curve['source_uv'])[:,None]+curve['source_normal_uv'][:,None]*cross[...,None]


def _ridge_image(source, food_mask):
    image = np.asarray(source, float)/255
    # A luminance ridge follows an observed cylindrical highlight; complementary
    # colour channels keep dark noodles visible against bright sauce as well.
    channels = [image @ np.array([.2126, .7152, .0722]), *image.transpose(2, 0, 1)]
    response = np.zeros(food_mask.shape)
    scales = np.zeros(food_mask.shape)
    direction = np.zeros((*food_mask.shape, 2))
    for sigma in [1., 1.5, 2.2, 3.2, 4.5]:
        for channel in channels:
            xx = gaussian_filter(channel, sigma, order=(0, 2))*sigma*sigma
            xy = gaussian_filter(channel, sigma, order=(1, 1))*sigma*sigma
            yy = gaussian_filter(channel, sigma, order=(2, 0))*sigma*sigma
            delta = np.sqrt((xx-yy)**2+4*xy**2)
            first, second = (xx+yy-delta)/2, (xx+yy+delta)/2
            swap = np.abs(first) < np.abs(second)
            normal = np.where(swap, second, first)
            along = np.where(swap, first, second)
            # Follow convex source highlights, not the dark spaces between two
            # noodles. Equal two-direction curvature is a blob, not a strand.
            score = np.abs(normal)*np.exp(-2*(along/(np.abs(normal)+1e-8))**2)
            score *= 1-np.exp(-(normal*normal+along*along)/.0008)
            score *= normal < 0
            score[~food_mask] = 0
            update = score > response
            angle = .5*np.arctan2(2*xy, xx-yy)
            nx, ny = np.cos(angle), np.sin(angle)
            # Eigenvector of the selected normal; sign is immaterial.
            nx, ny = np.where(swap, nx, -ny), np.where(swap, ny, nx)
            direction[update, 0] = -ny[update]
            direction[update, 1] = nx[update]
            response[update] = score[update]
            scales[update] = sigma
    # Suppress parallel shoulders: retain the maximum across each ridge normal.
    yy, xx = np.indices(food_mask.shape)
    normal = np.stack([-direction[..., 1], direction[..., 0]], axis=-1)
    offset = np.maximum(.85, scales*.65)
    plus = map_coordinates(response, [yy+normal[..., 1]*offset, xx+normal[..., 0]*offset], order=1)
    minus = map_coordinates(response, [yy-normal[..., 1]*offset, xx-normal[..., 0]*offset], order=1)
    ridge = (response >= plus) & (response >= minus)
    positive = response[food_mask & (response > 0)]
    threshold = max(.003, float(np.quantile(positive, .62))) if positive.size else .003
    high = response > threshold
    low = (response > .50*threshold) & ridge & food_mask
    from scipy.ndimage import binary_propagation
    line = binary_propagation(high & ridge, mask=low)
    return response, scales, skeletonize(line)


def _trace_graph(skeleton):
    pixels = list(zip(*np.where(skeleton)))
    active = set(pixels)
    graph = {p: [] for p in pixels}
    for y, x in pixels:
        for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            q = (y+dy, x+dx)
            if q not in active:
                continue
            # Orthogonal neighbours already provide a connection; the diagonal
            # would create artificial triangles around a one-pixel junction.
            if dy and dx and ((y, x+dx) in active or (y+dy, x) in active):
                continue
            graph[(y, x)].append(q)
    visited, paths = set(), []
    seeds = sorted(p for p in pixels if len(graph[p]) != 2)
    for start in seeds+sorted(pixels):
        for q in graph[start]:
            edge = tuple(sorted((start, q)))
            if edge in visited:
                continue
            path = [start]
            previous, current = start, q
            visited.add(edge)
            while True:
                path.append(current)
                if len(graph[current]) != 2 or current == start:
                    break
                nxt = next(p for p in graph[current] if p != previous)
                edge = tuple(sorted((current, nxt)))
                if edge in visited:
                    break
                visited.add(edge)
                previous, current = current, nxt
            if len(path) > 3:
                paths.append(np.asarray(path, float)[:, ::-1])
    # At an observed junction, continue the two most nearly tangent incident
    # paths. No edge or hidden span is introduced. Crossings remain ambiguous.
    while True:
        endpoints = {}
        for i, path in enumerate(paths):
            for side in [0, -1]:
                p = tuple(path[side])
                tangent = path[1]-path[0] if side == 0 else path[-2]-path[-1]
                tangent /= np.linalg.norm(tangent)
                endpoints.setdefault(p, []).append((i, side, tangent))
        best = None
        for entries in endpoints.values():
            for a in range(len(entries)):
                for b in range(a+1, len(entries)):
                    i, si, ti = entries[a];j, sj, tj = entries[b]
                    score = -ti@tj
                    if i != j and score > .72 and (best is None or score > best[0]):
                        best = (score, i, si, j, sj)
        if best is None:
            break
        _, i, si, j, sj = best
        left = paths[i][::-1] if si == 0 else paths[i]
        right = paths[j] if sj == 0 else paths[j][::-1]
        joined = np.concatenate([left, right[1:]])
        paths = [p for k, p in enumerate(paths) if k not in [i, j]]+[joined]
    return paths


def reconstruct_strands(source, food_mask, points, patch_mask=None, axes=None, *,
                        max_curves=64, min_length_pixels=7., sample_step=1.7):
    source = np.asarray(source, np.uint8)
    food_mask = np.asarray(food_mask, bool) & np.isfinite(points).all(axis=-1)
    patch = food_mask if patch_mask is None else food_mask & np.asarray(patch_mask, bool)
    response, scales, skeleton = _ridge_image(source, food_mask)
    paths = _trace_graph(skeleton)
    smooth_points = np.asarray(points, float)
    records = []
    removal = np.zeros(food_mask.shape, bool)
    for graph_id, uv in enumerate(paths):
        selected = _sample(patch.astype(float), uv) > .5
        transitions = np.flatnonzero(np.diff(np.r_[False, selected, False]))
        for start, stop in zip(transitions[::2], transitions[1::2]):
            segment = uv[start:stop]
            if len(segment) < 3:
                continue
            distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(segment, axis=0), axis=1))]
            if distance[-1] < min_length_pixels:
                continue
            samples = np.linspace(0, distance[-1], max(4, int(distance[-1]/sample_step)+1))
            uv2 = np.stack([np.interp(samples, distance, segment[:, j]) for j in range(2)], axis=-1)
            tangent_uv = np.gradient(uv2, axis=0)
            tangent_uv /= np.maximum(1e-8, np.linalg.norm(tangent_uv, axis=1))[:, None]
            normal_uv = np.stack([-tangent_uv[:, 1], tangent_uv[:, 0]], axis=-1)
            width_pixels = np.clip(1.25*_sample(scales, uv2), 1.15, 5.5)
            texture_center_uv=_ingredient_exemplar_uv(food_mask,uv2)
            cross_positive, cross_negative = np.zeros(len(uv2)), np.zeros(len(uv2))
            positive_contiguous=np.ones(len(uv2),bool);negative_contiguous=np.ones(len(uv2),bool)
            for fraction in np.linspace(0, 1, int(np.ceil(width_pixels.max()*4))+1):
                width = width_pixels*fraction
                positive_inside = bilinear_footprint_owned(food_mask,texture_center_uv+normal_uv*width[:,None])
                negative_inside = bilinear_footprint_owned(food_mask,texture_center_uv-normal_uv*width[:,None])
                positive_contiguous &= positive_inside;negative_contiguous &= negative_inside
                cross_positive[positive_contiguous] = width[positive_contiguous]
                cross_negative[negative_contiguous] = width[negative_contiguous]
            xyz_surface,xyz,radius=_source_curve_geometry(smooth_points,uv2,normal_uv,width_pixels)
            p = np.rint(texture_center_uv).astype(int)
            material_ids = p[:, 1]*source.shape[1]+p[:, 0]
            records.append(dict(curve_id=len(records), graph_id=graph_id, xyz=xyz,
                rest_xyz=xyz.copy(), source_surface_xyz=xyz_surface, source_uv=uv2,
                texture_center_uv=texture_center_uv,
                source_normal_uv=normal_uv, radius=radius, radius_pixels=width_pixels,
                cross_section_positive_pixels=cross_positive, cross_section_negative_pixels=cross_negative,
                source_rgb=_sample(source.astype(float), texture_center_uv), material_ids=material_ids,
                source_path_length_pixels=float(distance[-1]),
                cut_at_patch_boundary=bool(start > 0 or stop < len(uv))))
    # Rank by measured length * ridge contrast, not a semantic class label.
    records.sort(key=lambda c: c['source_path_length_pixels']*float(_sample(response, c['source_uv']).mean()), reverse=True)
    records = records[:max_curves]
    for i, c in enumerate(records):
        c['curve_id'] = i
        for pos, radius_px in zip(np.rint(c['source_uv']).astype(int), c['radius_pixels']):
            x, y = pos
            r = int(np.ceil(radius_px))
            ys, xs = np.ogrid[max(0, y-r):min(source.shape[0], y+r+1), max(0, x-r):min(source.shape[1], x+r+1)]
            removal[max(0, y-r):min(source.shape[0], y+r+1), max(0, x-r):min(source.shape[1], x+r+1)] |= (xs-x)**2+(ys-y)**2 <= radius_px**2
        # Every bilinear texture footprint is an explicit observed material
        # exemplar belonging to the carried source ingredient. The ownership
        # cut expands beyond the seed circle to include these width samples.
        texture_uv=_tube_texture_uv(c,10).reshape(-1,2)
        low=np.floor(texture_uv).astype(int);high=np.ceil(texture_uv).astype(int)
        for pixels in [low,high,np.c_[low[:,0],high[:,1]],np.c_[high[:,0],low[:,1]]]:
            pixels[:,0]=np.clip(pixels[:,0],0,source.shape[1]-1)
            pixels[:,1]=np.clip(pixels[:,1],0,source.shape[0]-1)
            removal[pixels[:,1],pixels[:,0]]=True
    removal &= food_mask
    ownership_fraction=float(np.mean(np.concatenate([bilinear_footprint_owned(removal,_tube_texture_uv(c,10).reshape(-1,2)) for c in records]))) if records else 0.
    metrics = dict(curve_count=len(records), node_count=sum(len(c['xyz']) for c in records),
        graph_path_count=len(paths), selected_source_pixels=int(patch.sum()),
        ridge_covered_source_pixels=int(removal.sum()),
        ownership_expanded_beyond_seed_pixels=int((removal&~patch).sum()),
        tube_source_uv_owned_fraction=ownership_fraction,
        ownership_outside_ingredient_pixels=int((removal&~food_mask).sum()),
        ownership_scope='Width-aware observed source cross-section exemplars; hidden circumference repeats these exemplars rather than claiming unseen material observations.',
        ingredient_cross_section_rule='Source normal texture cross-section terminates at first invalid nonzero-weight bilinear ingredient footprint (quarterpixel sampling); cannot jump holes or borrow excluded garnish.',
        texture_exemplar_node_count=int(sum(np.linalg.norm(c['texture_center_uv']-c['source_uv'],axis=1).astype(bool).sum() for c in records)),
        texture_exemplar_max_offset_pixels=float(max((np.linalg.norm(c['texture_center_uv']-c['source_uv'],axis=1).max() for c in records),default=0.)),
        source_geometry_uv_changed=False,
        texture_exemplar_scope='Geometry nodes retain observed source ray UV. Camera-boundary mixed ingredient samples use separately declared nearest valid source ingredient texel exemplars; hidden ring repeats these observed exemplars.',
        source_provenance='Visible RGB ridge graph sampled through source-estimated camera points; unseen connectivity is not reconstructed.',
        hidden_surface='Observed cross-section texture repeated around an inferred cylinder; no generated noodle colours.',
        original_food_mask_used=True, patch_boundary_cuts=True,
        remaining_food_anchors_used=False, semantic_label_used=False)
    return StrandBundle(records, response, skeleton, removal, metrics)


def reanchor_strands(bundle,points):
    """Update source depth after frozen-field closure without retracing ownership."""
    result=copy.deepcopy(bundle)
    for c in result.curves:
        surface,xyz,radius=_source_curve_geometry(points,c['source_uv'],c['source_normal_uv'],c['radius_pixels'])
        c['source_surface_xyz']=surface;c['xyz']=xyz;c['rest_xyz']=xyz.copy();c['radius']=radius
        for key in ['_material_phase','_source_first_tangent','_source_first_normal']:
            c.pop(key,None)
    result.metrics['reanchored_geometry']=True
    result.metrics['source_uv_topology_ownership_frozen']=True
    return result


def flatten_bundle(bundle):
    offsets = np.cumsum([0]+[len(c['xyz']) for c in bundle.curves])
    if not bundle.curves:
        return dict(xyz=np.empty((0, 3)), edges=np.empty((0, 2), int), radii=np.empty(0),
                    source_uv=np.empty((0, 2)), material_ids=np.empty(0, np.int64), rest_xyz=np.empty((0, 3)))
    result = {key: np.concatenate([c[key] for c in bundle.curves]) for key in ['xyz', 'rest_xyz', 'source_uv', 'material_ids']}
    result['radii'] = np.concatenate([c['radius'] for c in bundle.curves])
    result['edges'] = np.concatenate([np.c_[np.arange(offsets[i], offsets[i+1]-1), np.arange(offsets[i]+1, offsets[i+1])]
        for i in range(len(bundle.curves))])
    result['curve_offsets'] = offsets
    return result


def _support(xyz, radii, spoon):
    axes = np.asarray(spoon.get('axes', spoon.get('spoon_axes_camera')))
    center = np.asarray(spoon['center_local']) if 'center_local' in spoon else np.asarray(spoon['center_camera']) @ axes
    local = xyz @ axes
    ab = np.asarray(spoon['radius_ab'])
    radial = np.sum(((local[:, :2]-center[:2])/ab)**2, axis=1)
    curvature = float(spoon['bowl_curvature_height'])
    surface, slope = surface_height_gradient(local[:, :2], spoon)
    clearance = radii*np.sqrt(1+(slope*slope).sum(axis=1))+curvature*radii*radii*np.max(1/(ab*ab))
    gap = local[:, 2]-surface-clearance
    inside = surface_domain(local[:, :2], spoon)
    if not spoon.get('bowl_handle_blend',False):
        inside |= radial <= (1+radii/min(ab))**2
    return axes, local, radial, gap, inside


def deform_strands(bundle, translation, spoon_info, gravity_axes, iterations=180, *,
                   contact=True, gravity_strength=1., self_collision=True, bending_stiffness=.13):
    result = copy.deepcopy(bundle)
    # Support dimensions determine the action/time scale in every ablation;
    # disabling collisions must not silently change the gravity acceleration.
    support_scale = float(min(spoon_info['radius_ab'])) if spoon_info is not None else None
    if not contact:
        spoon_info = None
    flat = flatten_bundle(bundle)
    rest = flat['xyz']
    if len(rest) == 0:
        return result, dict(node_count=0, curve_count=0)
    rigid = rest+np.asarray(translation)
    xyz, previous = rigid.copy(), rigid.copy()
    edges, radii = flat['edges'], flat['radii']
    lengths = np.linalg.norm(rest[edges[:, 1]]-rest[edges[:, 0]], axis=1)
    bends = np.concatenate([np.c_[np.arange(a, b-2), np.arange(a+2, b)] for a, b in zip(flat['curve_offsets'][:-1], flat['curve_offsets'][1:])])
    bend_lengths = np.linalg.norm(rest[bends[:, 1]]-rest[bends[:, 0]], axis=1)
    axes = np.asarray(gravity_axes)
    gravity = -axes[:, 2] if axes.shape == (3, 3) else axes
    gravity = gravity/np.linalg.norm(gravity)
    scale = float(np.median(lengths))*5
    if support_scale is not None:
        scale = support_scale
    acceleration = gravity*scale*.0035*gravity_strength
    all_edge_pairs = set(map(tuple, edges))
    node_curve = np.concatenate([np.full(b-a,i) for i,(a,b) in enumerate(zip(flat['curve_offsets'][:-1],flat['curve_offsets'][1:]))])
    arclength = np.concatenate([np.r_[0,np.cumsum(np.linalg.norm(np.diff(rest[a:b],axis=0),axis=1))]
        for a,b in zip(flat['curve_offsets'][:-1],flat['curve_offsets'][1:])])
    source_ambiguous_pairs = set()
    for i,j in cKDTree(rest).query_pairs(2*np.max(radii)):
        local_neighbour = node_curve[i] == node_curve[j] and abs(arclength[i]-arclength[j]) < 1.8*(radii[i]+radii[j])
        if (i,j) not in all_edge_pairs and not local_neighbour and np.linalg.norm(rest[j]-rest[i]) < radii[i]+radii[j]:
            source_ambiguous_pairs.add((i,j))

    def constraints(pairs, target, strength):
        # Even/odd segment colouring avoids in-place repeated-index ambiguity.
        for parity in [0, 1]:
            selected = np.arange(len(pairs)) % 2 == parity
            e = pairs[selected]
            delta = xyz[e[:, 1]]-xyz[e[:, 0]]
            length = np.linalg.norm(delta, axis=1)
            shift = .5*strength*(1-target[selected]/np.maximum(1e-10, length))[:, None]*delta
            np.add.at(xyz, e[:, 0], shift)
            np.add.at(xyz, e[:, 1], -shift)

    def contact():
        if spoon_info is None:
            return np.zeros(len(xyz), bool)
        support_axes, _, _, gap, inside = _support(xyz, radii, spoon_info)
        hit = inside & (gap < 0)
        xyz[hit] += (-gap[hit])[:, None]*support_axes[:, 2]
        if spoon_info.get('bowl_handle_blend',False):
            # The neck ridge is not a parabola. Evaluate the exact rendered
            # cross-sections and lift whole centreline nodes, then re-solve edge
            # lengths. Tube radii/UV and individual surface vertices stay fixed.
            rings = np.concatenate([_tube_vertices(c,10,xyz[a:b]) for c,a,b in zip(result.curves,flat['curve_offsets'][:-1],flat['curve_offsets'][1:])])
            local = rings.reshape(-1,3) @ support_axes
            surface,_ = surface_height_gradient(local[:,:2],spoon_info)
            domain = surface_domain(local[:,:2],spoon_info)
            deficits = np.where(domain,surface-local[:,2],0.).reshape(-1,10).max(axis=1)
            surface_hit = deficits > 0
            xyz[surface_hit] += deficits[surface_hit,None]*support_axes[:,2]
            hit |= surface_hit
        return hit

    def collisions():
        pairs = np.array(list(cKDTree(xyz).query_pairs(2*np.max(radii))), int)
        if not len(pairs):
            return
        keep = np.array([tuple(e) not in all_edge_pairs for e in pairs])
        # The source graph may assign separate cylinders to two highlights of
        # one actual noodle, or to an unresolved crossing. Moving them apart at
        # t=0 would invent geometry and masquerade as gravity deformation.
        keep &= np.array([tuple(e) not in source_ambiguous_pairs for e in pairs])
        # Neighbouring tube cross-sections overlap by design; do not repel their
        # centres unless their distance along the curve exceeds both radii.
        local_neighbour = (node_curve[pairs[:,0]] == node_curve[pairs[:,1]]) & (np.abs(arclength[pairs[:,0]]-arclength[pairs[:,1]]) < 1.8*(radii[pairs[:,0]]+radii[pairs[:,1]]))
        keep &= ~local_neighbour
        pairs = pairs[keep]
        delta = xyz[pairs[:, 1]]-xyz[pairs[:, 0]]
        length = np.linalg.norm(delta, axis=1)
        penetration = radii[pairs[:, 0]]+radii[pairs[:, 1]]-length
        keep = penetration > 0
        pairs, delta, length, penetration = pairs[keep], delta[keep], length[keep], penetration[keep]
        displacement = .35*penetration[:, None]*delta/np.maximum(1e-8, length)[:, None]
        np.add.at(xyz, pairs[:, 0], -displacement)
        np.add.at(xyz, pairs[:, 1], displacement)

    steps = max(1, iterations//4)
    for step in range(steps):
        velocity = (xyz-previous)*.91
        previous = xyz.copy()
        xyz += velocity+acceleration
        for _ in range(4):
            constraints(edges, lengths, 1.)
            constraints(bends, bend_lengths, bending_stiffness)
            hit = contact()
        if self_collision and step % 4 == 0:
            collisions()
            contact()
        # Contact friction is local and does not invent a pinned noodle endpoint.
        previous[hit] = .85*xyz[hit]+.15*previous[hit]
    for _ in range(180):
        constraints(edges, lengths, 1.)
        contact()
    deformed_lengths = np.linalg.norm(xyz[edges[:, 1]]-xyz[edges[:, 0]], axis=1)
    strain = deformed_lengths/np.maximum(1e-10, lengths)-1
    signed_gap = np.full(len(xyz), np.inf)
    if spoon_info is not None:
        _, _, _, gap, inside = _support(xyz, radii, spoon_info)
        signed_gap[inside] = gap[inside]
    for i, c in enumerate(result.curves):
        a, b = flat['curve_offsets'][i:i+2]
        c['xyz'] = xyz[a:b]
        c['rigid_xyz'] = rigid[a:b]
        c['support_signed_gap'] = signed_gap[a:b]
    surface_min_gap = None
    surface_inside_count = 0
    if spoon_info is not None:
        surface_vertices = np.concatenate([_tube_vertices(c,10).reshape(-1,3) for c in result.curves])
        _, _, _, surface_gap, surface_inside = _support(surface_vertices,np.zeros(len(surface_vertices)),spoon_info)
        surface_inside_count = int(surface_inside.sum())
        if surface_inside_count:
            surface_min_gap = float(surface_gap[surface_inside].min())
    rest_volume = np.pi*np.mean(radii[edges], axis=1)**2*lengths
    carried_volume = np.pi*np.mean(radii[edges], axis=1)**2*deformed_lengths
    before_bend = np.linalg.norm(rest[bends[:, 1]]-rest[bends[:, 0]], axis=1)
    after_bend = np.linalg.norm(xyz[bends[:, 1]]-xyz[bends[:, 0]], axis=1)
    metrics = dict(curve_count=len(result.curves), node_count=len(xyz), solver_steps=steps,
        edge_strain_mean_abs=float(np.mean(np.abs(strain))), edge_strain_p95=float(np.quantile(np.abs(strain), .95)),
        edge_strain_max_abs=float(np.max(np.abs(strain))),
        nonrigid_displacement_mean=float(np.linalg.norm(xyz-rigid, axis=1).mean()),
        nonrigid_displacement_over_support_scale=float(np.linalg.norm(xyz-rigid, axis=1).mean()/scale),
        bending_chord_relative_change_mean=float(np.mean(np.abs(after_bend/np.maximum(1e-10, before_bend)-1))),
        support_min_gap=float(np.min(signed_gap)) if spoon_info is not None else None,
        support_contact_node_count=int((signed_gap < scale*.01).sum()),
        tube_surface_min_support_gap=surface_min_gap,
        tube_surface_vertices_inside_bowl=surface_inside_count,
        support_clearance='Shared actual spoon surface; slope/curvature sphere clearance and, for blendedneck, exact10-sided tube contact solved through centreline nodes. Actual renderedtube vertices evaluated.',
        gravity_acceleration_camera=acceleration.tolist(),
        capsule_volume_proxy_relative_change=float(abs(carried_volume.sum()/rest_volume.sum()-1)),
        material_ids_preserved=bool(np.array_equal(flat['material_ids'], flatten_bundle(result)['material_ids'])),
        source_uv_max_change=float(np.max(np.abs(flat['source_uv']-flatten_bundle(result)['source_uv']))),
        geometry_scope='Visible source ridge hypotheses; image-derived radii and hidden cylinder surfaces are not measured 3D.',
        gravity_strength=float(gravity_strength), contact_enabled=spoon_info is not None,
        self_collision=bool(self_collision), remaining_source_anchor_count=0)
    metrics['source_ambiguous_overlap_pair_count'] = len(source_ambiguous_pairs)
    metrics['source_ambiguous_overlap_treatment'] = 'Initially overlapping nonadjacent inferred cylinders are unresolved source topology; excluded from self-collision correction, not separated or counted as recovered geometry.'
    result.metrics.update(metrics)
    return result, metrics


def _curve_tangents(xyz):
    tangent=np.gradient(xyz,axis=0)
    norm=np.linalg.norm(tangent,axis=1)
    return tangent/np.maximum(1e-10,norm)[:,None]


def _rotate_normal(normal, before, after):
    v=np.cross(before,after);cosine=float(np.clip(before@after,-1,1))
    if cosine > -.999999:
        normal=normal+np.cross(v,normal)+np.cross(v,np.cross(v,normal))/(1+cosine)
    # For a half turn, keep the material normal as the rotation axis.
    normal-=after*(normal@after)
    return normal/max(1e-10,np.linalg.norm(normal))


def _transport_normals(tangent,first_normal):
    normal=np.empty_like(tangent);normal[0]=first_normal
    for i in range(1,len(tangent)):
        normal[i]=_rotate_normal(normal[i-1],tangent[i-1],tangent[i])
    return normal


def _material_frames(curve,xyz):
    rest=curve['rest_xyz']
    if '_material_phase' not in curve:
        original_tangent=_curve_tangents(rest)
        observed_normal=-curve['source_surface_xyz'].copy()
        observed_normal-=(observed_normal*original_tangent).sum(axis=1)[:,None]*original_tangent
        observed_normal/=np.maximum(1e-10,np.linalg.norm(observed_normal,axis=1))[:,None]
        bishop=_transport_normals(original_tangent,observed_normal[0])
        across=np.cross(original_tangent,bishop)
        curve['_material_phase']=np.arctan2((observed_normal*across).sum(axis=1),(observed_normal*bishop).sum(axis=1))
        curve['_source_first_tangent']=original_tangent[0].copy()
        curve['_source_first_normal']=observed_normal[0].copy()
    tangent=_curve_tangents(xyz)
    first=_rotate_normal(curve['_source_first_normal'].copy(),curve['_source_first_tangent'],tangent[0])
    bishop=_transport_normals(tangent,first);binormal=np.cross(tangent,bishop)
    phase=curve['_material_phase']
    normal=np.cos(phase)[:,None]*bishop+np.sin(phase)[:,None]*binormal
    return normal,np.cross(tangent,normal)


def _tube_vertices(curve, sides, xyz=None):
    xyz = curve['xyz'] if xyz is None else xyz
    view,across=_material_frames(curve,xyz)
    theta = np.arange(sides)*2*np.pi/sides
    return xyz[:, None]+curve['radius'][:, None, None]*(np.cos(theta)[None, :, None]*view[:, None]+np.sin(theta)[None, :, None]*across[:, None])


def tube_geometry(bundle, source, sides=10):
    vertices, faces, uv_all, ids_all = [], [], [], []
    offset = 0
    for curve in bundle.curves:
        xyz = curve['xyz']
        theta = np.arange(sides)*2*np.pi/sides
        rings = _tube_vertices(curve,sides)
        # Each circumference sample refers to the observed source cross-section;
        # camera-hidden half repeats it instead of borrowing unrelated food RGB.
        uv = _tube_texture_uv(curve,sides)
        vertices.append(rings.reshape(-1, 3));uv_all.append(uv.reshape(-1, 2))
        sample_pixels = np.rint(uv.reshape(-1,2)).astype(int)
        ids_all.append(sample_pixels[:,1]*source.shape[1]+sample_pixels[:,0])
        for k in range(len(xyz)-1):
            for j in range(sides):
                a, b = offset+k*sides+j, offset+k*sides+(j+1)%sides
                faces.extend([[a, b, a+sides], [b, b+sides, a+sides]])
        # Caps reuse the terminal source sample; the cut has no separate colour.
        for j in range(1, sides-1):
            faces.extend([[offset, offset+j+1, offset+j], [offset+(len(xyz)-1)*sides, offset+(len(xyz)-1)*sides+j, offset+(len(xyz)-1)*sides+j+1]])
        offset += len(xyz)*sides
    vertices = np.concatenate(vertices) if vertices else np.empty((0, 3))
    uv = np.asarray(np.concatenate(uv_all) if uv_all else np.empty((0, 2)))
    return dict(vertices=vertices, faces=np.asarray(faces, int).reshape(-1, 3),
        uv_pixels=uv, source_rgb=_sample(np.asarray(source, float), uv),
        material_ids=np.concatenate(ids_all) if ids_all else np.empty(0, np.int64))
