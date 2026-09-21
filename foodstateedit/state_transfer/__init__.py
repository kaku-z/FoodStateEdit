"""Experimental source-consistent state transfer (no trained model bundled).

The geometry contract requires numpy only. Import ``routing`` explicitly for
the optional PyTorch adapter. Existing generation pipelines are unchanged.
"""

from .geometry import Camera, FoodState, Transition, apply_transition, from_mask, project

__all__ = ["Camera", "FoodState", "Transition", "apply_transition", "from_mask", "project"]
