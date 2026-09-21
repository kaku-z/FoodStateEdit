import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from state_projection_control import (plan_visible_hold_offset,
                                      validate_state_projection_schedule,
                                      visible_source_fraction)


def test_visibility_planner_finds_shortest_feasible_offset():
    source = np.zeros((40, 60), bool); source[15:30, 20:40] = True
    payload = source.copy()
    plan = plan_visible_hold_offset(
        source,
        payload,
        [(0, 0), (5, 0), (10, 0), (15, 0), (20, 0)],
        minimum_visible_fraction=.75,
    )
    assert plan.offset_xy == (15, 0)
    assert plan.visible_fraction_before == 0
    assert plan.visible_fraction_after == .75


def test_visibility_metric_rejects_empty_source():
    try:
        visible_source_fraction(np.zeros((5, 5), bool), np.zeros((5, 5), bool))
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty source must fail")


def test_projection_schedule_exposes_exact_active_steps():
    schedule = validate_state_projection_schedule(20, 0, 0, 8)
    assert schedule["cavity_projection_steps"] == list(range(8))
    assert schedule["rigid_end"] == 0


def test_projection_schedule_rejects_endpoint_after_run():
    try:
        validate_state_projection_schedule(20, 0, 0, 21)
    except ValueError as exc:
        assert "cavity_end" in str(exc)
    else:
        raise AssertionError("out-of-range endpoint must fail")
