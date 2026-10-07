"""Material-coordinate elastic food transport with a unilateral spoon obstacle.

The mesh is a surface proxy, not a measured finite-element solid.  Source vertex
indices are material identities and are never changed.  Elastic graph springs,
a graph bending term, gravity, and coupled closed-volume pressure produce a relaxed
shape after the prescribed lift.  All parameters act in bite-normalized units.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized
from .spoon_surface import surface_height_gradient, surface_domain


def mesh_edges(faces):
    """Undirected material-neighbour edges; no spatial nearest-neighbour gluing."""
    f = np.asarray(faces, dtype=np.int64)
    return np.unique(np.sort(np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]])), axis=1), axis=0)


def signed_volume(vertices, faces):
    p = np.asarray(vertices, dtype=np.float64)
    p = p - p.mean(axis=0)
    t = p[np.asarray(faces, dtype=np.int64)]
    return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.)


def support_gaps(vertices, spoon_info):
    """Return raw source-camera-unit vertical bowl gaps and ellipse membership.

    Bowl axes are columns of spoon_axes_camera.  The surface exactly matches the
    shallow spoon in mld2_real_spoon_v3.utensil(), before its handle blend.
    """
    axes = np.asarray(spoon_info["spoon_axes_camera"], dtype=float)
    q = np.asarray(vertices, dtype=float) @ axes
    surface, _ = surface_height_gradient(q[:, :2], spoon_info)
    return q[:, 2] - surface, surface_domain(q[:, :2], spoon_info)


def _mesh_graph(rest, faces, solid_pairs):
    n = len(rest)
    tri = rest[faces]
    areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * .5
    masses = np.bincount(faces.ravel(), weights=np.repeat(areas / 3., 3), minlength=n)
    masses = np.maximum(masses, np.sum(areas) / n * .03)
    all_edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    edges, inverse, counts = np.unique(all_edges, axis=0, return_inverse=True, return_counts=True)
    edge_areas = np.bincount(inverse, weights=np.tile(areas / 3., 3))
    lengths = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    weights = edge_areas / np.maximum(lengths * lengths, 1.e-14)
    if len(solid_pairs):
        # Material columns and cross-column diagonals make the two observed /
        # inferred surfaces a deformable solid lattice instead of two loose skins.
        paired = np.full(n, -1, dtype=np.int64)
        paired[solid_pairs[:, 0]] = solid_pairs[:, 1]
        upper = np.zeros(n, dtype=bool)
        upper[solid_pairs[:, 0]] = True
        surface_edges = edges[upper[edges].all(axis=1)]
        braces = np.r_[solid_pairs, np.c_[surface_edges[:, 0], paired[surface_edges[:, 1]]],
                       np.c_[surface_edges[:, 1], paired[surface_edges[:, 0]]]]
        braces = np.unique(np.sort(braces, axis=1), axis=0)
        brace_lengths = np.linalg.norm(rest[braces[:, 0]] - rest[braces[:, 1]], axis=1)
        brace_weights = (masses[braces[:, 0]] + masses[braces[:, 1]]) / np.maximum(brace_lengths ** 2, 1.e-14)
        edges = np.r_[edges, braces]
        lengths = np.r_[lengths, brace_lengths]
        weights = np.r_[weights, brace_weights]
        edges, inverse = np.unique(edges, axis=0, return_inverse=True)
        weights = np.bincount(inverse, weights=weights)
        lengths = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    rows = np.repeat(np.arange(len(edges)), 2)
    incidence = sparse.csr_matrix((np.tile([1., -1.], len(edges)), (rows, edges.ravel())), shape=(len(edges), n))
    laplace = incidence.T @ sparse.diags(weights) @ incidence
    bending = laplace @ sparse.diags(1. / masses) @ laplace
    return edges, lengths, weights, incidence, masses, laplace.tocsc(), bending.tocsc(), bool(np.all(counts == 2))


def solve_deformation(vertices, faces, food_axes, translation, spoon_info,
                      compliance=.35, iterations=120, gravity_strength=.09,
                      contact=True, preserve_volume=True, bending_strength=None,
                      solid_pairs=None, tangential_pose_stiffness=.8):
    """Relax a lifted food surface while keeping its vertex/material lineage.

    Args:
        vertices/faces: source camera-space mesh, with consistent face winding.
        food_axes: camera-space food axes (column 2 points away from the plate).
        translation: prescribed rigid lift in camera coordinates.
        spoon_info: axes, center_local, radius_ab, floor_local, curvature height.
        compliance: 0 freezes the rigid lift; .10 is meat-like, .35 soft food,
            .8 flexible food.  These are prescribed proxy stiffnesses, not fits.
        iterations: maximum sparse preconditioned elastic energy descent steps.
        gravity_strength: normalized acceleration, zero for the gravity ablation.
        contact: enable the unilateral bowl constraint; no-contact falls freely.
        preserve_volume: closed-surface volume penalty; actual drift is reported.

    The elastic graph connects face neighbours and explicit material columns.
    Bending penalizes change in the rest graph curvature.  This linearized bending
    approximation is suitable for moderate soft-food deformation, not strands or
    large articulated rotations.  Contact is a hard vertical epigraph projection
    on the bowl's elliptical footprint. Tangential pose anchors condition the
    prescribed lift; they are declared priors rather than measured friction.
    Sparse pressure coupling avoids separately projecting scalar volume and
    injecting elastic energy. Vertex ordering and faces remain intact.
    A weak .005 rest-pose tether regularizes unsupported static translations;
    without contact the equilibrium is far below the bowl, not a time trajectory.
    """
    source = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    axes = np.asarray(spoon_info["spoon_axes_camera"], dtype=np.float64)
    food_axes = np.asarray(food_axes, dtype=np.float64)
    rigid = source + np.asarray(translation, dtype=np.float64)
    origin = rigid.mean(axis=0) @ axes
    scale = float(np.round(np.max(np.ptp(rigid @ axes, axis=0)), 10))
    # Normalize the solver frame below meaningful source precision.  This keeps
    # active-set and line-search ties independent of a camera-frame rotation.
    rest = np.round((rigid @ axes - origin) / scale, 10)
    x = rest.copy()
    if solid_pairs is None:
        # Existing closed_patch stores top IDs [0,N) and corresponding underside
        # IDs [N,2N).  Recognize exact columns in the prescribed food frame; other
        # geometries remain shells unless the caller supplies real material pairs.
        half = len(source) // 2
        delta_columns = (source[:half] - source[half:]) @ food_axes if len(source) % 2 == 0 else np.empty((0, 3))
        paired_layout = len(delta_columns) > 0 and np.max(np.abs(delta_columns[:, :2])) < 1.e-7 * scale and np.min(delta_columns[:, 2]) > 1.e-6 * scale
        solid_pairs = np.c_[np.arange(half), np.arange(half, 2 * half)] if paired_layout else np.empty((0, 2), dtype=np.int64)
    solid_pairs = np.asarray(solid_pairs, dtype=np.int64).reshape(-1, 2)
    edges, lengths, weights, incidence, masses, laplace, bend, closed = _mesh_graph(rest, f, solid_pairs)
    rest_volume = signed_volume(rest, f)
    center = np.round((np.asarray(spoon_info["center_local"], dtype=float)[:2] - origin[:2]) / scale, 10)
    radius = np.round(np.asarray(spoon_info["radius_ab"], dtype=float) / scale, 10)
    floor = float(np.round((float(spoon_info["floor_local"]) - origin[2]) / scale, 10))
    curvature = float(np.round(float(spoon_info["bowl_curvature_height"]) / scale, 10))
    gravity = np.round((-food_axes[:, 2]) @ axes * float(gravity_strength), 10)
    handle_blend = bool(spoon_info.get("bowl_handle_blend", False))
    normalized_spoon = dict(center_local=center, radius_ab=radius, floor_local=floor,
                            bowl_curvature_height=curvature, bowl_handle_blend=handle_blend)
    history = []
    velocities = np.zeros_like(x)
    steps_done = 0

    def bowl_project(q):
        height, _ = surface_height_gradient(q[:, :2], normalized_spoon)
        inside = surface_domain(q[:, :2], normalized_spoon)
        active = inside & (q[:, 2] < height)
        q[active, 2] = height[active]
        return active

    if compliance > 0:
        stiffness = .10 / float(compliance) ** 2
        bending = (2.e-5 * max(.08, 1. - .85 * float(compliance))
                   if bending_strength is None else float(bending_strength))
        anchor = .005
        contact_stiffness = 1200.
        volume_stiffness = 120.
        metric = (stiffness * laplace + bending * bend + sparse.diags(anchor * masses)).tocsc()
        solve = factorized(metric)
        solve_tangent = factorized((metric + sparse.diags(float(tangential_pose_stiffness) * masses)).tocsc())
        active_previous = None
        solve_z = solve

        def volume_gradient(q):
            p = q - q.mean(axis=0)
            a, b, c = (p[f[:, i]] for i in range(3))
            volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.
            gradient = np.zeros_like(q)
            for ids, values in zip(f.T, (np.cross(b, c), np.cross(c, a), np.cross(a, b))):
                np.add.at(gradient, ids, values / 6.)
            return volume, gradient

        def energy_gradient(q):
            displacement = q - rest
            delta = q[edges[:, 0]] - q[edges[:, 1]]
            current_lengths = np.maximum(np.linalg.norm(delta, axis=1), 1.e-12)
            elongation = current_lengths - lengths
            edge_force = stiffness * weights[:, None] * (elongation / current_lengths)[:, None] * delta
            gradient = incidence.T @ edge_force
            bend_force = bending * (bend @ displacement)
            gradient += bend_force + anchor * masses[:, None] * displacement - masses[:, None] * gravity
            gradient[:, :2] += float(tangential_pose_stiffness) * masses[:, None] * displacement[:, :2]
            energy = .5 * stiffness * np.sum(weights * elongation ** 2)
            energy += .5 * np.sum(displacement * bend_force) + .5 * anchor * np.sum(masses[:, None] * displacement ** 2)
            energy += .5 * float(tangential_pose_stiffness) * np.sum(masses[:, None] * displacement[:, :2] ** 2)
            energy -= np.sum(masses[:, None] * q * gravity)
            height, bowl_gradient = surface_height_gradient(q[:, :2], normalized_spoon)
            gap = q[:, 2] - height
            # Include touching vertices in the contact metric before they carry
            # load.  A zero-gap contact must not depend on camera roundoff.
            active = surface_domain(q[:, :2], normalized_spoon) & (gap <= 1.e-8) if contact else np.zeros(len(q), dtype=bool)
            negative_gap = np.minimum(gap[active], 0.)
            normal = np.c_[-bowl_gradient, np.ones(len(q))]
            gradient[active] += (contact_stiffness * masses[active] * negative_gap)[:, None] * normal[active]
            energy += .5 * contact_stiffness * np.sum(masses[active] * negative_gap ** 2)
            volume, gv = volume_gradient(q)
            volume_coefficient = volume_stiffness / rest_volume ** 2 if preserve_volume and closed and abs(rest_volume) > 1.e-12 else 0.
            gradient += volume_coefficient * (volume - rest_volume) * gv
            energy += .5 * volume_coefficient * (volume - rest_volume) ** 2
            return float(energy), np.asarray(gradient), active, gv, volume_coefficient

        for step in range(int(iterations)):
            previous = x.copy()
            energy, gradient, active, gv, volume_coefficient = energy_gradient(x)
            if np.linalg.norm(gradient) < 1.e-9:
                history.append(dict(step=step, energy=energy, step_length=0., update_rms=0.,
                                    volume_relative_error=float(abs(signed_volume(x, f) / rest_volume - 1.)) if abs(rest_volume) > 1.e-12 else None))
                break
            if active_previous is None or np.any(active != active_previous):
                solve_z = factorized((metric + sparse.diags(contact_stiffness * masses * active)).tocsc())
                active_previous = active.copy()
            direction = np.column_stack([solve_tangent(gradient[:, 0]), solve_tangent(gradient[:, 1]), solve_z(gradient[:, 2])])
            # A rank-one pressure metric enforces incompressibility without
            # injecting the energy of a separate Euclidean volume projection.
            pressure = np.column_stack([solve_tangent(gv[:, 0]), solve_tangent(gv[:, 1]), solve_z(gv[:, 2])])
            direction -= pressure * (volume_coefficient * np.sum(gv * direction) / (1. + volume_coefficient * np.sum(gv * pressure)))
            slope = float(np.sum(gradient * direction))
            step_length = 1.
            candidate = x - direction
            candidate_energy = energy_gradient(candidate)[0]
            while candidate_energy > energy - 1.e-4 * step_length * slope and step_length > 1.e-7:
                step_length *= .5
                candidate = x - step_length * direction
                candidate_energy = energy_gradient(candidate)[0]
            x = candidate
            velocities = x - previous
            steps_done = step + 1
            if step % 10 == 0 or step == int(iterations) - 1:
                history.append(dict(step=step + 1, energy=candidate_energy, step_length=step_length, update_rms=float(np.sqrt(np.mean(velocities ** 2))),
                                    volume_relative_error=float(abs(signed_volume(x, f) / rest_volume - 1.)) if abs(rest_volume) > 1.e-12 else None))
        if contact:
            bowl_project(x)

    result = (x * scale + origin) @ axes.T if compliance > 0 else rigid.copy()
    if contact and compliance > 0:
        raw_gaps, raw_inside = support_gaps(result, spoon_info)
        active = raw_inside & (raw_gaps < 0)
        result[active] -= raw_gaps[active, None] * axes[:, 2]
        x = (result @ axes - origin) / scale
    gaps, inside = support_gaps(result, spoon_info)
    deformed_lengths = np.linalg.norm(x[edges[:, 0]] - x[edges[:, 1]], axis=1)
    strain = deformed_lengths / np.maximum(lengths, 1.e-12) - 1.
    tri0, tri1 = rest[f], x[f]
    normals0 = np.cross(tri0[:, 1] - tri0[:, 0], tri0[:, 2] - tri0[:, 0])
    normals1 = np.cross(tri1[:, 1] - tri1[:, 0], tri1[:, 2] - tri1[:, 0])
    normal_alignment = np.einsum("ij,ij->i", normals0, normals1)
    displacement = result - rigid
    centered_displacement = displacement - displacement.mean(axis=0)
    metrics = dict(
        model="Quasi-static material graph elastic energy descent; spring strain + rest-curvature bending + gravity + unilateral elliptical bowl + coupled volume pressure + tangential action prior",
        units="Source camera proxy units, normalized by lifted bite diameter; stiffness and gravity are prescribed, not measured",
        compliance=float(compliance), iterations=steps_done, requested_iterations=int(iterations),
        gravity_strength=float(gravity_strength), contact_enabled=bool(contact),
        preserve_volume=bool(preserve_volume and closed), watertight_edge_topology=closed,
        vertex_count=len(source), face_count=len(f), edge_count=len(edges),
        internal_material_column_count=len(solid_pairs),
        tangential_pose_stiffness=float(tangential_pose_stiffness),
        tangential_pose_prior="Soft per-material-vertex anchor to the prescribed action in spoon tangent axes; source footprint conditioning, not measured static friction",
        material_vertex_ids_preserved=True, face_topology_preserved=True,
        characteristic_length=scale, displacement_rms=float(np.sqrt(np.mean(displacement ** 2))),
        nonrigid_displacement_rms=float(np.sqrt(np.mean(centered_displacement ** 2))),
        normalized_nonrigid_displacement_rms=float(np.sqrt(np.mean(centered_displacement ** 2))) / scale,
        centroid_displacement_camera=displacement.mean(axis=0).tolist(),
        edge_strain_rms=float(np.sqrt(np.mean(strain ** 2))), edge_strain_p95=float(np.quantile(np.abs(strain), .95)),
        edge_strain_max=float(np.max(np.abs(strain))),
        rest_signed_volume_camera=float(rest_volume * scale ** 3),
        deformed_signed_volume_camera=float(signed_volume(x, f) * scale ** 3),
        volume_relative_error=float(abs(signed_volume(x, f) / rest_volume - 1.)) if abs(rest_volume) > 1.e-12 else None,
        face_orientation_reversal_count=int(np.sum(normal_alignment < 0)),
        face_orientation_metric="Normal dot rest normal < 0; detects large flips as well as valid rotations, not tetrahedral inversion",
        degenerate_face_count=int(np.sum(np.linalg.norm(normals1, axis=1) < 1.e-10)),
        support_footprint_vertex_fraction=float(inside.mean()),
        supported_vertex_fraction=float(np.mean(inside & (np.abs(gaps) <= .005 * scale))),
        contact_tolerance_camera=.005 * scale,
        bowl_handle_blend=handle_blend,
        bowl_min_gap_camera=float(gaps[inside].min()) if inside.any() else None,
        penetration_depth_camera=float(max(0., -gaps[inside].min())) if inside.any() else 0.,
        penetration_depth_normalized=float(max(0., -gaps[inside].min()) / scale) if inside.any() else 0.,
        final_update_rms_normalized=float(np.sqrt(np.mean(velocities ** 2))),
        relaxation_history=history,
    )
    return result, metrics
