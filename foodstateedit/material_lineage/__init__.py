"""Discrete material lineage prototype; no learned generator or physical truth."""

from .core import (
    MaterialLedger,
    MaterialState,
    PairedSeam,
    RigidTransform,
    SeamSide,
    material_noise,
    paired_seam_from_interface_mesh,
    rigid_from_g37_report,
    sample_convex_mesh,
)

__all__ = [
    "MaterialLedger", "MaterialState", "PairedSeam", "RigidTransform", "SeamSide",
    "material_noise", "paired_seam_from_interface_mesh", "rigid_from_g37_report",
    "sample_convex_mesh",
]
