"""Small deterministic solvers for state-level cooking edits.

The engine deliberately stops before photorealistic rendering. It produces a
physically auditable target state that can later be converted into masks,
depth, proxy geometry and GeoEdit conditioning.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from .model import FoodObject, Scene


class SimulationError(ValueError):
    """Raised when an edit violates a material or conservation constraint."""


@dataclass
class SimulationResult:
    before: Scene
    after: Scene
    trace: list[dict[str, Any]]


def _positive_number(value: Any, label: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise SimulationError(f"{label} must be positive")
    return number


def _update_fill_level(item: FoodObject) -> None:
    if item.material != "liquid":
        return
    volume = float(item.state.get("volume_ml", 0.0))
    capacity = _positive_number(item.state.get("capacity_ml"), "capacity_ml")
    if volume < -1e-9 or volume > capacity + 1e-9:
        raise SimulationError(
            f"liquid {item.object_id} volume {volume} exceeds [0, {capacity}]"
        )
    item.state["volume_ml"] = volume
    item.state["fill_level"] = volume / capacity


def _transfer_liquid(scene: Scene, action: dict[str, Any]) -> dict[str, Any]:
    source = scene.get_object(str(action["source"]))
    target = scene.get_object(str(action["target"]))
    if source.material != "liquid" or target.material != "liquid":
        raise SimulationError("transfer_liquid requires two liquid objects")
    amount = _positive_number(action["amount_ml"], "amount_ml")
    source_before = float(source.state.get("volume_ml", 0.0))
    target_before = float(target.state.get("volume_ml", 0.0))
    if amount > source_before + 1e-9:
        raise SimulationError("liquid source does not contain the requested volume")
    source.state["volume_ml"] = source_before - amount
    target.state["volume_ml"] = target_before + amount
    _update_fill_level(source)
    _update_fill_level(target)
    return {
        "material_solver": "liquid_volume_v0.1",
        "amount_ml": amount,
        "total_before_ml": source_before + target_before,
        "total_after_ml": source.state["volume_ml"] + target.state["volume_ml"],
    }


def _component_sum(item: FoodObject) -> float:
    components = item.state.get("components_g", {})
    return float(sum(float(value) for value in components.values()))


def _scoop_granular(scene: Scene, action: dict[str, Any]) -> dict[str, Any]:
    source = scene.get_object(str(action["source"]))
    target = scene.get_object(str(action["target"]))
    if source.material != "granular" or target.material != "granular":
        raise SimulationError("scoop_granular requires two granular objects")
    amount = _positive_number(action["amount_g"], "amount_g")
    source_mass = _component_sum(source)
    target_mass = _component_sum(target)
    if amount > source_mass + 1e-9:
        raise SimulationError("granular source does not contain the requested mass")
    ratio = amount / source_mass
    source_components = source.state.setdefault("components_g", {})
    target_components = target.state.setdefault("components_g", {})
    transferred_state: dict[str, float] = {}
    for field in ("mix_uniformity", "browning", "cookedness", "temperature_c"):
        if field not in source.state:
            continue
        source_value = float(source.state[field])
        if target_mass > 0.0 and field in target.state:
            target_value = (
                target_mass * float(target.state[field]) + amount * source_value
            ) / (target_mass + amount)
        else:
            target_value = source_value
        target.state[field] = target_value
        transferred_state[field] = target_value
    moved: dict[str, float] = {}
    for name, raw_mass in list(source_components.items()):
        component_mass = float(raw_mass)
        moved_mass = component_mass * ratio
        source_components[name] = component_mass - moved_mass
        target_components[name] = float(target_components.get(name, 0.0)) + moved_mass
        moved[name] = moved_mass
    source.state["mass_g"] = _component_sum(source)
    target.state["mass_g"] = _component_sum(target)
    return {
        "material_solver": "granular_proportional_scoop_v0.1",
        "amount_g": amount,
        "components_moved_g": moved,
        "payload_intensive_state": transferred_state,
        "total_before_g": source_mass + target_mass,
        "total_after_g": source.state["mass_g"] + target.state["mass_g"],
    }


def _mix_granular(scene: Scene, action: dict[str, Any]) -> dict[str, Any]:
    target = scene.get_object(str(action["target"]))
    if target.material != "granular":
        raise SimulationError("mix_granular requires a granular target")
    target_uniformity = float(action["target_uniformity"])
    current = float(target.state.get("mix_uniformity", 0.0))
    if not 0.0 <= target_uniformity <= 1.0:
        raise SimulationError("target_uniformity must be in [0, 1]")
    if target_uniformity + 1e-9 < current:
        raise SimulationError("mixing cannot reduce uniformity")
    components_before = dict(target.state.get("components_g", {}))
    target.state["mix_uniformity"] = target_uniformity
    return {
        "material_solver": "granular_mix_state_v0.1",
        "uniformity_before": current,
        "uniformity_after": target_uniformity,
        "components_unchanged": components_before == target.state.get("components_g", {}),
    }


def _set_appearance_state(scene: Scene, action: dict[str, Any]) -> dict[str, Any]:
    target = scene.get_object(str(action["target"]))
    field = str(action["field"])
    value = float(action["value"])
    if not 0.0 <= value <= 1.0:
        raise SimulationError(f"appearance field {field} must be in [0, 1]")
    before = target.state.get(field)
    target.state[field] = value
    return {
        "material_solver": "appearance_state_v0.1",
        "field": field,
        "before": before,
        "after": value,
    }


HANDLERS: dict[str, Callable[[Scene, dict[str, Any]], dict[str, Any]]] = {
    "transfer_liquid": _transfer_liquid,
    "scoop_granular": _scoop_granular,
    "mix_granular": _mix_granular,
    "set_appearance_state": _set_appearance_state,
}


def simulate(scene: Scene) -> SimulationResult:
    before = Scene.from_dict(deepcopy(scene.to_dict()))
    after = Scene.from_dict(deepcopy(scene.to_dict()))
    trace: list[dict[str, Any]] = []
    for index, action in enumerate(after.actions):
        action_type = str(action.get("type", ""))
        if action_type not in HANDLERS:
            raise SimulationError(f"unsupported action type: {action_type}")
        details = HANDLERS[action_type](after, action)
        trace.append(
            {
                "index": index,
                "action_id": action.get("id", f"action_{index}"),
                "type": action_type,
                "details": details,
            }
        )
    after.validate()
    return SimulationResult(before=before, after=after, trace=trace)
