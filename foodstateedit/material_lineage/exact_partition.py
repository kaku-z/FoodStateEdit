"""Boolean volume fractions for coarse persistent material regions.

Every full-source tetrahedron is intersected with the supplied bite mesh. This
replaces centroid-inside classification, while retaining each coarse cell ID
and its original material-coordinate anchor. A partial cell is a *region*, not
a point particle at that anchor. Its actual child centroid is separate geometry
metadata. All masses are uniform-density proxy volumes in arbitrary source units.

Manifold3D is required at execution. There is no fake Boolean fallback. Solids
are normalized to unit extent before Mesh64 operations to improve conditioning.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .core import MaterialLedger, MaterialState, sample_convex_mesh


_TETRA_FACES = np.asarray([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]], dtype=np.uint64)


def _backend():
    try:
        import manifold3d
    except ImportError as error:
        raise ImportError("Exact material partition requires the actual manifold3d Boolean backend") from error
    return manifold3d


def _mesh(vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float64)
    raw_faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must be finite N x 3")
    if (raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or raw_faces.dtype.kind not in "iu"
            or len(raw_faces) == 0 or np.any(raw_faces < 0) or np.any(raw_faces >= len(vertices))):
        raise ValueError("faces must be in-range integer M x 3")
    faces = np.array(raw_faces, dtype=np.uint64, copy=True)
    triangle = vertices[faces]
    center = vertices.mean(axis=0)
    signed_volume = float(np.einsum("ij,ij->i", triangle[:, 0] - center,
                                   np.cross(triangle[:, 1] - center, triangle[:, 2] - center)).sum() / 6)
    if signed_volume < 0:
        faces = faces[:, [0, 2, 1]]
        signed_volume = -signed_volume
    if not np.isfinite(signed_volume) or signed_volume <= 0:
        raise ValueError("mesh must have positive enclosed volume")
    return np.ascontiguousarray(vertices), np.ascontiguousarray(faces), signed_volume


def _tetrahedra(vertices, faces, refinement):
    """Use exactly the ordering of the core convex quadrature and its IDs."""
    center = vertices.mean(axis=0)
    tetra = np.concatenate([np.broadcast_to(center, (len(faces), 1, 3)), vertices[faces]], axis=1)
    for _ in range(refinement):
        middle = tetra.mean(axis=1)
        tetra = np.concatenate([
            np.concatenate([middle[:, None], tetra[:, face]], axis=1)
            for face in ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))
        ], axis=0)
    return tetra


def _solid(module, vertices, faces):
    solid = module.Manifold(module.Mesh64(np.ascontiguousarray(vertices, dtype=np.float64),
                                         np.ascontiguousarray(faces, dtype=np.uint64)))
    if solid.status() != module.Error.NoError:
        raise ValueError(f"Manifold3D rejected mesh: {solid.status()}")
    return solid


def _centroid(solid, volume):
    if volume <= 0:
        return np.full(3, np.nan)
    mesh = solid.to_mesh64()
    vertices = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    faces = np.asarray(mesh.tri_verts, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("positive child volume has no boundary mesh")
    reference = vertices.mean(axis=0)
    triangle = vertices[faces]
    signed = np.einsum("ij,ij->i", triangle[:, 0] - reference,
                       np.cross(triangle[:, 1] - reference, triangle[:, 2] - reference)) / 6
    denominator = float(signed.sum())
    if denominator <= 0:
        raise ValueError("positive child solid has nonpositive signed boundary volume")
    return np.sum((triangle.sum(axis=1) + reference) * signed[:, None], axis=0) / (4 * denominator)


@dataclass(frozen=True)
class ExactPartitionResult:
    source: MaterialState
    remaining: MaterialState
    carried: MaterialState
    tetrahedra: np.ndarray
    remaining_fractions: np.ndarray
    carried_fractions: np.ndarray
    remaining_geometric_volumes: np.ndarray
    carried_geometric_volumes: np.ndarray
    remaining_geometric_centroids: np.ndarray
    carried_geometric_centroids: np.ndarray
    audit: dict

    def save_npz(self, path: str | Path):
        """Store source ledger plus child geometry, without executable pickles."""
        np.savez_compressed(
            path, material_ids=np.asarray(self.source.material_ids),
            coords=self.source.material_coordinates, root_mass=self.source.reference_masses,
            remaining_fraction=self.remaining_fractions, carried_fraction=self.carried_fractions,
            tetrahedra=self.tetrahedra,
            remaining_geometric_volumes=self.remaining_geometric_volumes,
            carried_geometric_volumes=self.carried_geometric_volumes,
            remaining_geometric_centroids=self.remaining_geometric_centroids,
            carried_geometric_centroids=self.carried_geometric_centroids,
            geometric_centroid_scope=np.asarray("Child region volume centroids, not persistent texture coordinates"),
            material_coordinate_scope=np.asarray("Original coarse-region anchors, not point particles or child centroids"),
        )


def partition_convex_mesh(full_vertices, full_faces, bite_vertices, bite_faces,
                          namespace, refinement=2, density=1.0,
                          relative_volume_tolerance=1e-6):
    """Partition full tetrahedral regions by actual Mesh64 Boolean volumes.

    Only the source must be convex for its tetrahedral fan (a caller may
    explicitly supply its convex hull and record that prior). The bite can be
    any valid closed oriented mesh accepted by Manifold3D, including nonconvex
    boundaries. It is never substituted with a hull. A tiny outside fraction
    caused by input serialization is reported; larger out-of-source cuts are
    rejected. No volume is rescaled to force the global bite amount to match.
    """
    if not np.isfinite(relative_volume_tolerance) or not 0 < relative_volume_tolerance <= 1e-3:
        raise ValueError("relative_volume_tolerance must be in (0, 1e-3]")
    full_vertices, full_faces, _ = _mesh(full_vertices, full_faces)
    bite_vertices, bite_faces, _ = _mesh(bite_vertices, bite_faces)
    source = sample_convex_mesh(full_vertices, full_faces, namespace, refinement, density)
    tetra = _tetrahedra(full_vertices, full_faces, refinement)
    if not np.array_equal(tetra.mean(axis=1), source.material_coordinates):
        raise ValueError("tetrahedron order differs from persistent source-cell anchors")
    scale = float(np.ptp(full_vertices, axis=0).max())
    origin = full_vertices.mean(axis=0)
    module = _backend()
    full = _solid(module, (full_vertices - origin) / scale, full_faces)
    bite = _solid(module, (bite_vertices - origin) / scale, bite_faces)
    full_volume_normalized = float(full.volume())
    bite_volume_normalized = float(bite.volume())
    if full_volume_normalized <= 0 or bite_volume_normalized <= 0:
        raise ValueError("full and bite must have positive actual Boolean volume")
    outside = bite - full
    if outside.status() != module.Error.NoError:
        raise ValueError(f"bite containment operation failed: {outside.status()}")
    outside_fraction = float(outside.volume()) / full_volume_normalized
    if outside_fraction > relative_volume_tolerance:
        raise ValueError(f"bite lies outside full source by {outside_fraction} of full volume")
    remaining_volume = np.zeros(len(tetra))
    carried_volume = np.zeros(len(tetra))
    remaining_centroid = np.full((len(tetra), 3), np.nan)
    carried_centroid = np.full((len(tetra), 3), np.nan)
    fractions = np.zeros(len(tetra))
    maximum_fraction_rounding = 0.0
    maximum_cell_volume_residual = 0.0
    for index, vertices in enumerate(tetra):
        normalized = (vertices - origin) / scale
        if np.linalg.det(normalized[1:] - normalized[:1]) < 0:
            normalized = normalized[[0, 1, 3, 2]]
        cell = _solid(module, normalized, _TETRA_FACES)
        volume = float(cell.volume())
        if volume <= 0:
            raise ValueError("positive source tetrahedron collapsed in Boolean backend")
        child, rest = cell ^ bite, cell - bite
        if child.status() != module.Error.NoError or rest.status() != module.Error.NoError:
            raise ValueError(f"tetrahedron {index} Boolean operation failed")
        cvolume, rvolume = float(child.volume()), float(rest.volume())
        raw_fraction = cvolume / volume
        rounding = max(0.0, -raw_fraction, raw_fraction - 1)
        if rounding > 5e-10:
            raise ValueError("Boolean intersection exceeds the original cell volume")
        maximum_fraction_rounding = max(maximum_fraction_rounding, rounding)
        fractions[index] = np.clip(raw_fraction, 0, 1)
        carried_volume[index], remaining_volume[index] = cvolume * scale**3, rvolume * scale**3
        maximum_cell_volume_residual = max(maximum_cell_volume_residual, abs(cvolume + rvolume - volume))
        if cvolume > 0:
            carried_centroid[index] = _centroid(child, cvolume) * scale + origin
        if rvolume > 0:
            remaining_centroid[index] = _centroid(rest, rvolume) * scale + origin
    parts = source.transfer(np.stack([1 - fractions, fractions]), ("remaining", "carried"))
    budget = MaterialLedger(source).validate(list(parts.values()))
    geometric_full_volume = full_volume_normalized * scale**3
    geometric_bite_volume = bite_volume_normalized * scale**3
    actual_carried_volume = float(carried_volume.sum())
    relative_bite_error = abs(actual_carried_volume - geometric_bite_volume) / geometric_full_volume
    relative_full_error = abs(source.total_mass / density - geometric_full_volume) / geometric_full_volume
    relative_cell_residual = maximum_cell_volume_residual / full_volume_normalized
    if max(relative_bite_error, relative_full_error, relative_cell_residual) > relative_volume_tolerance:
        raise ValueError(f"Exact partition geometry audit failed: bite={relative_bite_error}, "
                         f"full={relative_full_error}, cell={relative_cell_residual}")
    budget.update(
        method="Normalized Mesh64 Boolean intersection of every source tetrahedral region with actual bite",
        refinement=int(refinement), density_assumed=float(density), cells=len(tetra),
        backend="manifold3d", full_geometric_volume=geometric_full_volume,
        actual_bite_geometric_volume=geometric_bite_volume,
        summed_carried_child_volume=actual_carried_volume,
        actual_bite_fraction=geometric_bite_volume / geometric_full_volume,
        carried_mass_fraction=parts["carried"].total_mass / source.total_mass,
        relative_bite_volume_error_over_full=relative_bite_error,
        relative_full_volume_error=relative_full_error,
        max_child_volume_residual_over_full=relative_cell_residual,
        outside_bite_volume_over_full=outside_fraction,
        max_fraction_numerical_clipping=maximum_fraction_rounding,
        bite_boundary_scope="Actual supplied closed oriented mesh; convexity is not required or inferred",
        relative_volume_tolerance=float(relative_volume_tolerance),
        material_coordinate_scope="Original coarse-region centroid anchors; not physical point-particle positions",
        child_centroid_scope="Separate volume centroids of actual child regions; not reassigned persistent coordinates",
        physical_scope="Uniform density, supplied proxy meshes and arbitrary monocular scale; no real measured mass",
    )
    return ExactPartitionResult(source, parts["remaining"], parts["carried"], tetra,
                                1 - fractions, fractions, remaining_volume, carried_volume,
                                remaining_centroid, carried_centroid, budget)
