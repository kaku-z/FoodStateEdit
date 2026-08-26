"""Material-aware state layer for the FoodStateEdit prototype."""

from .acceptance import evaluate_constraints
from .engine import SimulationError, simulate
from .model import FoodObject, Scene

__all__ = [
    "FoodObject",
    "Scene",
    "SimulationError",
    "evaluate_constraints",
    "simulate",
]

__version__ = "0.1.0"
