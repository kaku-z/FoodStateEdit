"""A conservative, executable material state, with paired new surfaces.

Coordinates and masses describe the chosen discretization. They are not measured
food geometry or mass. Noise is an immutable material-ID field, not a trained
diffusion model. The renderer must query this field using source/material IDs,
never screen coordinates. The mesh adapter accepts the source-aligned G37 full
mesh; it deliberately rejects nonconvex meshes instead of using an invalid fan.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np


def _array(value, shape, name, dtype=np.float64):
    out = np.array(value, dtype=dtype, copy=True)
    if out.shape != shape or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class RigidTransform:
    """Proper rotation plus translation; row-vector points use x @ R.T + t."""

    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self):
        rotation = _array(self.rotation, (3, 3), "rotation")
        translation = _array(self.translation, (3,), "translation")
        if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-9):
            raise ValueError("rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-9):
            raise ValueError("rotation must have determinant +1")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    @classmethod
    def identity(cls):
        return cls(np.eye(3), np.zeros(3))

    def apply(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("points must be a finite N x 3 array")
        return points @ self.rotation.T + self.translation

    def apply_vectors(self, vectors):
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[1] != 3 or not np.all(np.isfinite(vectors)):
            raise ValueError("vectors must be a finite N x 3 array")
        # Do not add then subtract translation: a large t would erase precision.
        return vectors @ self.rotation.T

    def inverse(self):
        return RigidTransform(self.rotation.T, -self.rotation.T @ self.translation)

    def then(self, following: RigidTransform):
        """Compose this transform followed by `following`."""
        return RigidTransform(
            following.rotation @ self.rotation,
            following.rotation @ self.translation + following.translation,
        )


def rigid_from_g37_report(report: Mapping):
    """G37: world_lifted = (source_point - src) @ Q.T + dest.

    `translation_food_frame` is dest-src, not the homogeneous transform's
    translation when Q is nonidentity. This adapter uses the actual convention.
    """
    rotation = np.asarray(report["rotation_food_frame"], dtype=np.float64)
    source = np.asarray(report["source_center"], dtype=np.float64)
    destination = np.asarray(report["destination_center"], dtype=np.float64)
    return RigidTransform(rotation, destination - rotation @ source)


@dataclass(frozen=True)
class MaterialState:
    """One branch of persistent source cells, with fractions of reference masses.

    A cell ID can occur in different branches after fractional splitting. Its
    reference mass and material coordinate must remain identical across branches;
    a MaterialLedger checks their combined budget. Arrays are copied and frozen.
    """

    material_ids: tuple[str, ...]
    material_coordinates: np.ndarray
    reference_masses: np.ndarray
    fractions: np.ndarray
    world_positions: np.ndarray
    branch: str = "source"

    def __post_init__(self):
        ids = tuple(self.material_ids)
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("material IDs must be nonempty and unique within a branch")
        if not isinstance(self.branch, str) or not self.branch:
            raise ValueError("branch must be a nonempty string")
        n = len(ids)
        object.__setattr__(self, "material_ids", ids)
        for name, shape in (
            ("material_coordinates", (n, 3)), ("reference_masses", (n,)),
            ("fractions", (n,)), ("world_positions", (n, 3)),
        ):
            object.__setattr__(self, name, _array(getattr(self, name), shape, name))
        if np.any(self.reference_masses <= 0):
            raise ValueError("reference masses must be strictly positive")
        if np.any(self.fractions <= 0) or np.any(self.fractions > 1):
            raise ValueError("retained fractions must lie in (0, 1]")

    @classmethod
    def create(cls, coordinates, masses, namespace="material"):
        coordinates = np.asarray(coordinates, dtype=np.float64)
        if (coordinates.ndim != 2 or coordinates.shape[1] != 3
                or not isinstance(namespace, str) or not namespace):
            raise ValueError("coordinates must be N x 3 and namespace must be nonempty")
        n = len(coordinates)
        return cls(tuple(f"{namespace}:{i}" for i in range(n)), coordinates,
                   masses, np.ones(n), coordinates)

    @property
    def masses(self):
        return self.reference_masses * self.fractions

    @property
    def total_mass(self):
        return float(np.sum(self.masses))

    def transformed(self, pose: RigidTransform):
        return MaterialState(self.material_ids, self.material_coordinates,
                             self.reference_masses, self.fractions,
                             pose.apply(self.world_positions), self.branch)

    def partition(self, membership, names=("remaining", "carried")):
        """Exclusive cell partition; True cells go to the second branch."""
        membership = np.asarray(membership)
        if membership.dtype != np.bool_ or membership.shape != (len(self.material_ids),):
            raise ValueError("membership must be a boolean array with one entry per cell")
        return self.transfer(np.stack([~membership, membership]).astype(float), names)

    def transfer(self, allocation, names: Sequence[str]):
        """Conservatively distribute each current cell across named branches.

        Columns must sum to one. Off-screen material needs an explicit branch;
        loss or duplication is rejected. The operator does not invent geometry
        inside a partly split cell; callers must refine their cells for that.
        """
        names = tuple(names)
        if (not names or any(not isinstance(i, str) or not i for i in names)
                or len(set(names)) != len(names)):
            raise ValueError("target branch names must be nonempty and distinct")
        p = _array(allocation, (len(names), len(self.material_ids)), "allocation")
        if np.any(p < 0) or np.any(p > 1):
            raise ValueError("allocation entries must lie in [0, 1]")
        if not np.allclose(p.sum(axis=0), 1, rtol=0, atol=1e-12):
            raise ValueError("allocation must conserve every source cell; copying is forbidden")
        results = {}
        for name, weights in zip(names, p):
            selected = weights > 0
            results[name] = MaterialState(
                tuple(i for i, take in zip(self.material_ids, selected) if take),
                self.material_coordinates[selected], self.reference_masses[selected],
                self.fractions[selected] * weights[selected],
                self.world_positions[selected], f"{self.branch}/{name}",
            )
        MaterialLedger.from_state(self).validate(tuple(results.values()))
        return results

    def noise(self, seed: int, channels=3, stream="appearance"):
        return material_noise(self.material_ids, seed, channels, stream)


@dataclass(frozen=True)
class MaterialLedger:
    """Reference budget that rejects forged mass, IDs, or material coordinates."""

    source: MaterialState

    @classmethod
    def from_state(cls, state: MaterialState):
        return cls(state)

    def validate(self, branches: Sequence[MaterialState], require_complete=True):
        lookup = {i: j for j, i in enumerate(self.source.material_ids)}
        allocated = np.zeros(len(lookup), dtype=np.float64)
        branch_rows = 0
        for branch in branches:
            branch_rows += len(branch.material_ids)
            for row, identity in enumerate(branch.material_ids):
                if identity not in lookup:
                    raise ValueError(f"unknown material ID: {identity}")
                root = lookup[identity]
                if branch.reference_masses[row] != self.source.reference_masses[root]:
                    raise ValueError(f"reference mass changed for {identity}")
                if not np.array_equal(branch.material_coordinates[row],
                                      self.source.material_coordinates[root]):
                    raise ValueError(f"material coordinates changed for {identity}")
                allocated[root] += branch.fractions[row]
        budget = self.source.fractions
        # A tiny fractional descendant has a tiny budget too. An absolute
        # tolerance would incorrectly permit copying it wholesale.
        tolerance = 1e-12 * budget
        if np.any(allocated > budget + tolerance):
            raise ValueError("material budget exceeded: duplicated source mass")
        if require_complete and np.any(np.abs(allocated - budget) > tolerance):
            raise ValueError("material budget incomplete: lost source mass")
        residual = (allocated - budget) * self.source.reference_masses
        return {
            "source_cells": len(lookup), "branch_rows": branch_rows,
            "source_mass": self.source.total_mass,
            "allocated_mass": float(allocated @ self.source.reference_masses),
            "max_cell_mass_residual": float(np.max(np.abs(residual), initial=0)),
            "relative_total_mass_residual": float(abs(residual.sum()) /
                                                 max(self.source.total_mass, 1e-300)),
            "scope": "Chosen discrete material budget, not measured physical mass",
        }


def material_noise(material_ids: Sequence[str], seed: int, channels=3, stream="appearance"):
    """SHA-256 + Box-Muller Gaussian field keyed only by ID, seed, and stream.

    No row index, branch, world position, view, or pose enters the key. This is
    exact order/partition invariance, including fractional descendants. An ID
    must encode the persistent material sample, not a per-frame screen pixel.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(channels, bool) or not isinstance(channels, (int, np.integer)) or channels < 1:
        raise ValueError("channels must be a positive integer")
    if not isinstance(stream, str) or not stream:
        raise ValueError("stream must be a nonempty string")
    out = np.empty((len(material_ids), channels), dtype=np.float64)
    for row, identity in enumerate(material_ids):
        if not isinstance(identity, str) or not identity:
            raise ValueError("material IDs must be nonempty strings")
        for pair in range((channels + 1) // 2):
            # Length prefixes prevent ambiguous concatenations or delimiters.
            key = b"material-lineage-noise-v1"
            for part in (str(int(seed)), stream, identity, str(pair)):
                encoded = part.encode("utf-8")
                key += len(encoded).to_bytes(8, "little") + encoded
            digest = hashlib.sha256(key).digest()
            u1 = ((int.from_bytes(digest[:8], "little") >> 11) + 0.5) / 2**53
            u2 = ((int.from_bytes(digest[8:16], "little") >> 11) + 0.5) / 2**53
            scale = math.sqrt(-2 * math.log(u1))
            out[row, 2 * pair] = scale * math.cos(2 * math.pi * u2)
            if 2 * pair + 1 < channels:
                out[row, 2 * pair + 1] = scale * math.sin(2 * math.pi * u2)
    return out


@dataclass(frozen=True)
class SeamSide:
    sample_ids: tuple[str, ...]
    material_coordinates: np.ndarray
    material_normals: np.ndarray
    pose: RigidTransform

    def __post_init__(self):
        ids = tuple(self.sample_ids)
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("seam sample IDs must be nonempty and distinct")
        object.__setattr__(self, "sample_ids", ids)
        coords = _array(self.material_coordinates, (len(ids), 3), "seam coordinates")
        normals = _array(self.material_normals, (len(ids), 3), "seam normals")
        if not np.allclose(np.linalg.norm(normals, axis=1), 1, rtol=0, atol=1e-9):
            raise ValueError("seam normals must be unit length")
        object.__setattr__(self, "material_coordinates", coords)
        object.__setattr__(self, "material_normals", normals)

    @property
    def world_positions(self):
        return self.pose.apply(self.material_coordinates)

    @property
    def world_normals(self):
        return self.pose.apply_vectors(self.material_normals)


@dataclass(frozen=True)
class PairedSeam:
    """One source interface, two material-coordinate matched surface sides."""

    interface_id: str
    remaining: SeamSide
    carried: SeamSide
    sample_areas: np.ndarray

    def __post_init__(self):
        if not isinstance(self.interface_id, str) or not self.interface_id:
            raise ValueError("interface ID must be nonempty")
        a, b = self.remaining, self.carried
        if a.sample_ids != b.sample_ids or not np.array_equal(a.material_coordinates, b.material_coordinates):
            raise ValueError("both seam sides must have exactly matched IDs and source coordinates")
        if not np.array_equal(a.material_normals, -b.material_normals):
            raise ValueError("source seam normals must be opposite")
        areas = _array(self.sample_areas, (len(a.sample_ids),), "sample areas")
        if np.any(areas <= 0):
            raise ValueError("seam areas must be strictly positive")
        object.__setattr__(self, "sample_areas", areas)

    def correspondence_residual(self):
        # Reconstruct both source positions after their independent movements.
        a = self.remaining.pose.inverse().apply(self.remaining.world_positions)
        b = self.carried.pose.inverse().apply(self.carried.world_positions)
        return float(np.max(np.linalg.norm(a - b, axis=1), initial=0))

    def noise(self, seed, channels=3):
        return material_noise(self.remaining.sample_ids, seed, channels, "cut-interface")


def _mesh_arrays(vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float64)
    raw_faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.all(np.isfinite(vertices)):
        raise ValueError("vertices must be finite N x 3")
    if raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or raw_faces.dtype.kind not in "iu":
        raise ValueError("faces must be an integer M x 3 array")
    faces = raw_faces.astype(np.int64)
    if len(faces) == 0 or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("mesh must have faces with in-range vertex indices")
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    if np.any(norm <= 0):
        raise ValueError("degenerate mesh triangle")
    return vertices, faces, tri, cross, norm


def paired_seam_from_interface_mesh(vertices, faces, interface_id,
                                    remaining_pose=None, carried_pose=None):
    """Sample a common source cut patch once, then derive both surface sides.

    Callers must pass the cut patch (not the whole cutter). This function cannot
    establish that a supplied patch belongs to actual bite/remaining meshes.
    Face winding supplies the remaining side's outward normal. Shared sample
    IDs bind any generated internal material to this source interface.
    """
    _, _, tri, cross, norm = _mesh_arrays(vertices, faces)
    coords = tri.mean(axis=1)
    normals = cross / norm[:, None]
    ids = tuple(f"{interface_id}:sample:{i}" for i in range(len(tri)))
    a = SeamSide(ids, coords, normals, remaining_pose or RigidTransform.identity())
    b = SeamSide(ids, coords, -normals, carried_pose or RigidTransform.identity())
    return PairedSeam(interface_id, a, b, norm / 2)


def sample_convex_mesh(vertices, faces, namespace, refinement=2, density=1.0):
    """Conservative tetrahedral centroid quadrature of a closed convex mesh.

    A centroid-to-face fan partitions the full volume. Each refinement replaces
    a tetrahedron with its four centroid-to-face tetrahedra. Weights sum to exact
    mesh volume up to floating-point error; a cutter classification at cell
    centroids remains an approximation and must be convergence-checked. Mass
    units are density times arbitrary source mesh volume, not grams.
    """
    vertices, faces, tri, cross, norm = _mesh_arrays(vertices, faces)
    if isinstance(refinement, bool) or not isinstance(refinement, (int, np.integer)) or not 0 <= refinement <= 7:
        raise ValueError("refinement must be an integer in [0, 7]")
    if not np.isfinite(density) or density <= 0:
        raise ValueError("density must be finite and positive")
    edges = {}
    for face in faces:
        for a, b in zip(face, np.roll(face, -1)):
            key = (min(int(a), int(b)), max(int(a), int(b)))
            edges.setdefault(key, []).append((int(a), int(b)))
    if any(len(e) != 2 or e[0] != e[1][::-1] for e in edges.values()):
        raise ValueError("mesh must be closed with consistently oriented edges")
    center = vertices.mean(axis=0)
    signs = np.einsum("ij,ij->i", tri[:, 0] - center, cross)
    if np.all(signs < 0):
        tri = tri[:, [0, 2, 1]]
        cross = -cross
        signs = -signs
    if np.any(signs <= 0):
        raise ValueError("mesh must be a nonzero convex solid with center inside")
    scale = max(float(np.ptp(vertices, axis=0).max()), 1e-300)
    normals = cross / norm[:, None]
    # Convexity is a genuine precondition for this positive tetrahedral fan.
    for start in range(0, len(tri), 128):
        outside = np.einsum("vfi,fi->vf", vertices[:, None] - tri[None, start:start+128, 0],
                            normals[start:start+128])
        if np.any(outside > scale * 1e-9):
            raise ValueError("nonconvex mesh cannot use convex tetrahedral quadrature")
    tetra = np.concatenate([np.broadcast_to(center, (len(tri), 1, 3)), tri], axis=1)
    for _ in range(refinement):
        middle = tetra.mean(axis=1)
        tetra = np.concatenate([
            np.concatenate([middle[:, None], tetra[:, face]], axis=1)
            for face in ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))
        ], axis=0)
    edges3 = tetra[:, 1:] - tetra[:, :1]
    volumes = np.abs(np.linalg.det(edges3)) / 6
    return MaterialState.create(tetra.mean(axis=1), volumes * density, namespace)
