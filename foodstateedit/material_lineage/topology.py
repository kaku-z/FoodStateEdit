"""Source-only coherent material selection before closed-mesh construction.

The pixel mask can contain isolated points and tiny disconnected triangulated
islands.  Those are not a single cohesive bite.  Select the largest triangulated
source component before assigning a shared volume/contact solve to the bite.
Boundary pinches cannot be extruded into a manifold solid. Their ambiguous
source pixels are left on the dish, then the largest surviving triangulated
component is retained. No target deformation is consulted.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


PIXEL_CLOSURE_FIELDS = (
    "source_local_xyz", "source_camera_xyz", "floor_local",
    "uniform_floor_local", "interior_rgb", "bottom_rgb", "source_uv_pixels",
)


def patch_triangles(patch):
    """Exact top-face construction and pixel order of geometry_v2.closed_patch.

    Returns faces and XY pixels; faces index the row-major np.where(mask) order.
    A lower-right triangle is emitted only when the top-left anchor also exists,
    matching the existing closed-patch reconstruction without filling mask gaps.
    """
    mask = np.asarray(patch, dtype=bool)
    yy, xx = np.where(mask)
    lookup = np.full(mask.shape, -1, dtype=np.int64)
    lookup[yy, xx] = np.arange(len(xx))
    a, b = lookup[:-1, :-1], lookup[:-1, 1:]
    c, d = lookup[1:, :-1], lookup[1:, 1:]
    first = (a >= 0) & (b >= 0) & (c >= 0)
    second = first & (d >= 0)
    # Sort on anchor to preserve the original row-by-row face order as well.
    first_faces = np.c_[a[first], c[first], b[first]]
    second_faces = np.c_[b[second], c[second], d[second]]
    faces = np.r_[first_faces, second_faces]
    anchor = np.r_[a[first] * 2, a[second] * 2 + 1]
    faces = faces[np.argsort(anchor, kind="stable")]
    return faces, np.c_[xx, yy]


def _trim_boundary_pinches(mask):
    """Remove source samples where more than two boundary edges meet.

    A vertex pinch becomes a four-face vertical edge after column closure.
    Duplicating coincident vertex IDs would only hide that geometric ambiguity.
    Instead, leave the pinched source sample on the plate and reconstruct the
    largest remaining source component, without adding or moving observed UVs.
    """
    cleaned = mask.copy()
    history = []
    while True:
        faces, pixels = patch_triangles(cleaned)
        if not len(faces):
            raise ValueError("No triangulated material remains after boundary-pinch repair")
        all_edges = np.sort(np.r_[faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1)
        edges, counts = np.unique(all_edges, axis=0, return_counts=True)
        boundary = edges[counts == 1]
        degree = np.bincount(boundary.ravel(), minlength=len(pixels))
        pinched = (degree != 0) & (degree != 2)
        if not pinched.any():
            break
        bad_pixels = pixels[pinched]
        before = int(cleaned.sum())
        cleaned[bad_pixels[:, 1], bad_pixels[:, 0]] = False
        next_faces, next_pixels = patch_triangles(cleaned)
        if not len(next_faces):
            raise ValueError("No triangulated material remains after boundary-pinch repair")
        next_edges = np.r_[next_faces[:, [0, 1]], next_faces[:, [1, 2]], next_faces[:, [2, 0]]]
        graph = sparse.csr_matrix((np.ones(len(next_edges)), (next_edges[:, 0], next_edges[:, 1])),
                                  shape=(len(next_pixels), len(next_pixels)))
        _, labels = connected_components(graph, directed=False)
        referenced = np.zeros(len(next_pixels), bool)
        referenced[np.unique(next_faces)] = True
        triangle_counts = np.bincount(labels[next_faces[:, 0]])
        vertex_counts = np.bincount(labels[referenced], minlength=len(triangle_counts))
        selected = min(np.unique(labels[referenced]), key=lambda label:
                       (-int(triangle_counts[label]), -int(vertex_counts[label]), int(label)))
        retained = referenced & (labels == selected)
        cleaned[:] = False
        kept_pixels = next_pixels[retained]
        cleaned[kept_pixels[:, 1], kept_pixels[:, 0]] = True
        history.append(dict(pinched_source_xy=bad_pixels.tolist(),
                            boundary_degrees=degree[pinched].astype(int).tolist(),
                            direct_pinched_pixels=len(bad_pixels),
                            total_pixels_left_on_source=before-int(cleaned.sum())))
    return cleaned, history


def coherent_closed_patch(patch, closure=None):
    """Return a single source triangulated bite and persistent index mappings.

    Output keys:
        patch: source mask used for *both* removal/cavity and all target controls.
        closure: known per-pixel arrays subset in original source order; global
            world_axes/extent and unknown metadata are copied unchanged.
        keep_pixel_indices: old source-pixel IDs of the retained rows.
        original_to_kept: index map, -1 for pixels left on the original plate.
        discarded_mask: original selected pixels excluded from this cohesive bite.
        top_faces: faces of the filtered patch in its new compact pixel order.
        metrics: measured source-only topology counts and selection scope.

    Connectivity uses source triangles, not image nearest neighbours or target
    outliers. A 1-pixel bridge with no supporting triangle is not glued into the
    solid. Boundary pinches are trimmed before column closure; all discarded
    material remains on the source. The largest component is selected by top-triangle area (each regular
    source-grid triangle has area .5 pixel²), then source vertex count. Selecting
    one piece is a cohesive-food prior; loose pieces need independent material
    states and contact/volume solves rather than this default shared solid.
    """
    mask = np.asarray(patch, dtype=bool)
    faces, pixels = patch_triangles(mask)
    n = len(pixels)
    if not len(faces):
        raise ValueError("The source bite contains no triangulated material surface")
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    graph = sparse.csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    referenced = np.zeros(n, dtype=bool)
    referenced[np.unique(faces)] = True
    active_labels = np.unique(labels[referenced])
    face_labels = labels[faces[:, 0]]
    vertex_counts = np.bincount(labels[referenced])
    triangle_counts = np.bincount(face_labels, minlength=len(vertex_counts))
    ranked = sorted(active_labels.tolist(), key=lambda k: (-int(triangle_counts[k]), -int(vertex_counts[k]), k))
    selected_label = ranked[0]
    retained = referenced & (labels == selected_label)
    keep = np.flatnonzero(retained)
    cleaned = np.zeros_like(mask)
    cleaned[pixels[keep, 1], pixels[keep, 0]] = True
    initial_selected_pixels = int(cleaned.sum())
    cleaned, repair_history = _trim_boundary_pinches(cleaned)
    retained = cleaned[pixels[:, 1], pixels[:, 0]]
    keep = np.flatnonzero(retained)
    discarded = mask & ~cleaned
    original_to_kept = np.full(n, -1, dtype=np.int64)
    original_to_kept[keep] = np.arange(len(keep))
    clean_faces, _ = patch_triangles(cleaned)
    clean_edges, clean_counts = np.unique(np.sort(np.r_[clean_faces[:, [0, 1]],
        clean_faces[:, [1, 2]], clean_faces[:, [2, 0]]], axis=1), axis=0, return_counts=True)
    a, b = clean_edges[clean_counts == 1].T
    count = len(keep)
    sides = np.r_[np.c_[a, b, b+count], np.c_[a, b+count, a+count]]
    complete = np.r_[clean_faces, clean_faces[:, ::-1]+count, sides]
    _, closed_counts = np.unique(np.sort(np.r_[complete[:, [0, 1]], complete[:, [1, 2]],
        complete[:, [2, 0]]], axis=1), axis=0, return_counts=True)
    if not np.all(closed_counts == 2):
        raise ValueError("Source topology repair failed: column closure has nonmanifold edges")
    filtered_closure = None
    if closure is not None:
        filtered_closure = {k: np.asarray(v)[keep].copy() if k in PIXEL_CLOSURE_FIELDS else np.asarray(v).copy()
                            for k, v in closure.items()}
    metrics = dict(
        rule="Largest source triangulated material component before target solving; no target outlier clamp",
        connectivity="Triangles share material vertices; no proximity edges between separate islands",
        cohesive_single_piece_prior=True,
        original_selected_pixels=n, original_top_triangles=len(faces),
        source_triangulated_component_count=len(active_labels),
        triangulated_component_vertex_counts=[int(vertex_counts[k]) for k in ranked],
        triangulated_component_triangle_counts=[int(triangle_counts[k]) for k in ranked],
        source_unreferenced_pixels=int((~referenced).sum()),
        discarded_triangulated_island_pixels=int(np.sum(referenced & ~retained)),
        retained_source_pixels=len(keep), retained_top_triangles=len(clean_faces),
        discarded_source_pixels=int(discarded.sum()),
        retained_top_vertex_coverage=float(len(np.unique(clean_faces)) / len(keep)),
        retained_source_order_preserved=bool(np.all(np.diff(keep) > 0)),
        selected_top_area_pixels=float(len(clean_faces) * .5),
        discarded_pixels_remain_in_source=True,
        initial_largest_component_pixels=initial_selected_pixels,
        manifold_trimmed_source_pixels=initial_selected_pixels-len(keep),
        boundary_pinch_repair=repair_history,
        manifold_repair_scope="Source-only single-solid hypothesis: ambiguous boundary pinch samples and disconnected remnants stay on the source. No new source UVs, coincident vertex-ID splitting, or hidden observed material is introduced.",
        closed_column_edge_incidence_valid=True,
        closed_column_edge_count=int(len(closed_counts)),
    )
    return dict(patch=cleaned, closure=filtered_closure, keep_pixel_indices=keep,
                original_to_kept=original_to_kept, discarded_mask=discarded,
                top_faces=clean_faces, metrics=metrics)
