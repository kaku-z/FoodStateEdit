"""Constraint evaluation shared by soup, rice and noodle tasks."""

from __future__ import annotations

from math import isclose
from typing import Any

from .model import Scene


def _state_value(scene: Scene, object_id: str, field: str) -> Any:
    return scene.get_object(object_id).state[field]


def _check_constraint(
    constraint: dict[str, Any], before: Scene, after: Scene
) -> tuple[bool, Any, Any]:
    kind = str(constraint["type"])
    if kind == "equals":
        actual = _state_value(after, str(constraint["object"]), str(constraint["field"]))
        expected = constraint["value"]
        tolerance = float(constraint.get("tolerance", 1e-6))
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            return isclose(float(actual), float(expected), abs_tol=tolerance), actual, expected
        return actual == expected, actual, expected
    if kind == "range":
        actual = float(
            _state_value(after, str(constraint["object"]), str(constraint["field"]))
        )
        minimum = float(constraint.get("min", float("-inf")))
        maximum = float(constraint.get("max", float("inf")))
        return minimum <= actual <= maximum, actual, [minimum, maximum]
    if kind == "conserve_total":
        object_ids = [str(value) for value in constraint["objects"]]
        field = str(constraint["field"])
        before_total = sum(float(_state_value(before, item, field)) for item in object_ids)
        after_total = sum(float(_state_value(after, item, field)) for item in object_ids)
        tolerance = float(constraint.get("tolerance", 1e-6))
        return isclose(before_total, after_total, abs_tol=tolerance), after_total, before_total
    if kind == "conserve_components":
        object_ids = [str(value) for value in constraint["objects"]]
        component_names = set()
        for scene in (before, after):
            for object_id in object_ids:
                component_names.update(
                    scene.get_object(object_id).state.get("components_g", {}).keys()
                )
        before_total = {
            name: sum(
                float(
                    before.get_object(item)
                    .state.get("components_g", {})
                    .get(name, 0.0)
                )
                for item in object_ids
            )
            for name in sorted(component_names)
        }
        after_total = {
            name: sum(
                float(
                    after.get_object(item)
                    .state.get("components_g", {})
                    .get(name, 0.0)
                )
                for item in object_ids
            )
            for name in sorted(component_names)
        }
        tolerance = float(constraint.get("tolerance", 1e-6))
        passed = all(
            isclose(before_total[name], after_total[name], abs_tol=tolerance)
            for name in component_names
        )
        return passed, after_total, before_total
    if kind == "relation_exists":
        expected = {
            "type": constraint["relation"],
            "subject": constraint["subject"],
            "object": constraint["object"],
        }
        passed = any(
            all(relation.get(key) == value for key, value in expected.items())
            for relation in after.relations
        )
        return passed, passed, True
    raise ValueError(f"unsupported constraint type: {kind}")


def evaluate_constraints(before: Scene, after: Scene) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for index, constraint in enumerate(after.constraints):
        passed, actual, expected = _check_constraint(constraint, before, after)
        checks.append(
            {
                "id": constraint.get("id", f"constraint_{index}"),
                "type": constraint["type"],
                "passed": bool(passed),
                "actual": actual,
                "expected": expected,
            }
        )
    state_success = all(item["passed"] for item in checks)
    return {
        "state_success": state_success,
        "render_success": None,
        "strict_end_to_end_success": False,
        "status": "state_only_pass" if state_success else "state_failed",
        "checks": checks,
        "note": "Photorealistic rendering has not been evaluated for this state-only case.",
    }
