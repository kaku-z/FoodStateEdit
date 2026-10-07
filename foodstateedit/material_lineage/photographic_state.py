"""Persistent source-conditioned material bookkeeping for photographic actions.

The weights are unit-density reconstruction proxies, not measured food masses.
Deformation moves material identities without re-estimating their weights.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class PhotographicMaterialState:
    ids: np.ndarray
    mass: np.ndarray
    carried_ids: np.ndarray
    remaining_ids: np.ndarray
    source_uv: np.ndarray
    rgb: np.ndarray
    rest_xyz: np.ndarray
    faces: np.ndarray

    @classmethod
    def from_geometry(cls, vertices, faces, uv, rgb, food_mask, removed_mask):
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        triangles = vertices[faces] - vertices.mean(axis=0)
        volume = abs(np.einsum('ij,ij->i', triangles[:, 0],
                     np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6)
        carried = np.arange(len(vertices), dtype=np.int64)
        remaining_count = int((food_mask & ~removed_mask).sum())
        remaining = np.arange(len(vertices), len(vertices) + remaining_count, dtype=np.int64)
        weights = np.r_[np.full(len(vertices), volume / len(vertices)),
                        np.full(remaining_count, volume / max(1, int(removed_mask.sum())))]
        return cls(np.r_[carried, remaining], weights, carried, remaining,
                   np.asarray(uv), np.asarray(rgb), vertices, faces)

    def transport_record(self, target, rigid, no_contact, no_gravity):
        edges = np.unique(np.sort(np.concatenate([self.faces[:, [0, 1]],
                 self.faces[:, [1, 2]], self.faces[:, [2, 0]]]), axis=1), axis=0)
        return dict(source_id=self.ids, carried_ids=self.carried_ids,
                    remaining_ids=self.remaining_ids, target_ids=self.carried_ids,
                    material_mass=self.mass, source_uv=self.source_uv,
                    target_source_uv=self.source_uv.copy(),
                    source_material_rgb=self.rgb, target_material_rgb=self.rgb.copy(),
                    rest_xyz=self.rest_xyz, target_xyz=np.asarray(target),
                    rigid_xyz=np.asarray(rigid), no_contact_xyz=np.asarray(no_contact),
                    no_gravity_xyz=np.asarray(no_gravity), faces=self.faces, edges=edges)
