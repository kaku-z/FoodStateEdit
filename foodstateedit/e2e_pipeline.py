"""Contracts and deterministic planning for the FoodStateEdit E2E v1 study.

This module deliberately does not import or load SAM, VACE, Qwen, or ChordEdit.  It
validates inputs, freezes the A--F comparison matrix, creates collision-proof
run identifiers, and implements the shared and event-aware frame selectors.
"""
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


METHODS = ("A_qwen_direct", "B_vace_text", "C_proxy_qwen", "D_independent_vace", "E_coupled_vace")
DERIVED_METHOD = "F_event_selection"
REQUIRED_CASE_KEYS = (
    "case_id", "food_type", "original_image", "food_mask", "selected_bite_mask",
    "container_mask", "source_point", "target_point", "utensil_type", "allowed_edit_region",
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"v1 standardized inputs must be PNG files: {path}")
    return struct.unpack(">II", header[16:24])


def validate_normalized_point(name: str, point: Any) -> tuple[float, float]:
    if not isinstance(point, list) or len(point) != 2:
        raise ValueError(f"{name} must contain exactly two coordinates")
    result = (float(point[0]), float(point[1]))
    if not all(0.0 <= value <= 1.0 for value in result):
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def validate_case(case: dict[str, Any], root: Path, *, require_files: bool = True) -> dict[str, Any]:
    missing = [key for key in REQUIRED_CASE_KEYS if key not in case]
    if missing:
        raise ValueError(f"case is missing required keys: {missing}")
    if case["food_type"] not in {"dumpling", "fried_rice", "ramen"}:
        raise ValueError("food_type must be dumpling, fried_rice, or ramen")
    expected_utensil = "spoon" if case["food_type"] == "fried_rice" else "chopsticks"
    if case["utensil_type"] != expected_utensil:
        raise ValueError(f"{case['food_type']} requires {expected_utensil} in v1")
    source = validate_normalized_point("source_point", case["source_point"])
    target = validate_normalized_point("target_point", case["target_point"])
    if source == target:
        raise ValueError("source_point and target_point must differ")
    records: dict[str, Any] = {}
    dimensions = set()
    for key in ("original_image", "food_mask", "selected_bite_mask", "container_mask", "allowed_edit_region"):
        path = (root / case[key]).resolve()
        record = {"path": case[key], "exists": path.is_file()}
        if require_files and not path.is_file():
            raise FileNotFoundError(path)
        if path.is_file():
            size = png_size(path)
            dimensions.add(size)
            record.update({"bytes": path.stat().st_size, "sha256": file_sha256(path), "size_wh": list(size)})
        records[key] = record
    if len(dimensions) > 1:
        raise ValueError(f"case images and masks must share one resolution, found {sorted(dimensions)}")
    return {"case_id": case["case_id"], "food_type": case["food_type"], "files": records}


def phase_for_frame(index: int, frame_count: int = 81) -> str:
    if frame_count != 81 or not 0 <= index < frame_count:
        raise ValueError("v1 uses exactly 81 frames indexed 0..80")
    if index <= 15:
        return "approach"
    if index <= 28:
        return "contact"
    if index <= 60:
        return "lift"
    return "hold"


def run_id(method: str, case_id: str, seed: int, frozen_config: dict[str, Any]) -> str:
    if method not in METHODS:
        raise ValueError(f"{method} is not a generating method")
    digest = canonical_sha256(frozen_config)[:12]
    return f"{method}__{case_id}__seed_{int(seed)}__{digest}"


@dataclass(frozen=True)
class FrameMetric:
    index: int
    arrival: float
    sharpness: float
    background_error: float
    contact: float | None = None
    following: float | None = None
    event_order_valid: bool | None = None

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "FrameMetric":
        return cls(**{name: item.get(name) for name in cls.__dataclass_fields__})


def _unit_interval(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def select_frame(metrics: Iterable[FrameMetric], weights: dict[str, float], *, require_events: bool = False) -> dict[str, Any]:
    rows = list(metrics)
    if not rows:
        raise ValueError("no frame metrics supplied")
    if set(weights) != {"arrival", "sharpness", "background_preservation"}:
        raise ValueError("selector weights are incomplete")
    if abs(sum(float(v) for v in weights.values()) - 1.0) > 1e-8:
        raise ValueError("selector weights must sum to one")
    scored = []
    for row in rows:
        if not 1 <= row.index <= 80:
            raise ValueError("selection candidates must be in frames 1..80")
        arrival = _unit_interval(row.arrival, "arrival")
        sharpness = _unit_interval(row.sharpness, "sharpness")
        background = _unit_interval(row.background_error, "background_error")
        eligible = True
        event_reason = "not_requested"
        if require_events:
            event_values = (row.contact, row.following, row.event_order_valid)
            eligible = None not in event_values and bool(row.event_order_valid) and float(row.contact) >= 0.5 and float(row.following) >= 0.5
            event_reason = "passed" if eligible else "unconfirmed_or_failed"
        score = (
            weights["arrival"] * arrival
            + weights["sharpness"] * sharpness
            + weights["background_preservation"] * (1.0 - background)
        )
        scored.append({"index": row.index, "score": score, "eligible": eligible, "event_reason": event_reason})
    eligible_rows = [row for row in scored if row["eligible"]]
    if not eligible_rows:
        return {"status": "unable_to_confirm", "selected_index": None, "scores": scored}
    selected = max(eligible_rows, key=lambda row: (row["score"], -row["index"]))
    return {"status": "selected", "selected_index": selected["index"], "selected_score": selected["score"], "scores": scored}


def build_plan(config: dict[str, Any], cases: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    if config.get("schema_version") != "foodstateedit.e2e_experiment.v1":
        raise ValueError("unexpected experiment schema")
    if config["inference"]["vace"]["num_frames"] != 81:
        raise ValueError("v1 VACE frame count must be 81")
    if (config["inference"]["vace"]["num_frames"] - 1) % 4:
        raise ValueError("VACE frame count must satisfy 4n+1")
    refinement = config["inference"].get("appearance_refinement")
    if refinement:
        if refinement.get("backend") != "ChordEdit_SD_Turbo":
            raise ValueError("unexpected appearance-refinement backend")
        if not refinement.get("single_candidate") or not refinement.get("exact_outside_support"):
            raise ValueError("appearance refinement must be single-candidate and support-locked")
        if not refinement.get("rollback_on_failed_or_uncertain_structure_or_semantics"):
            raise ValueError("appearance refinement must roll back on a failed or uncertain gate")
    audits = [validate_case(case, root) for case in cases]
    jobs = []
    for case in cases:
        for method in METHODS:
            for seed in config["seeds"]:
                jobs.append({"method": method, "case_id": case["case_id"], "seed": seed, "run_id": run_id(method, case["case_id"], seed, config)})
        jobs.append({"method": DERIVED_METHOD, "case_id": case["case_id"], "source_method": "E_coupled_vace", "new_generation": False})
    return {
        "schema_version": "foodstateedit.e2e_plan.v1",
        "config_sha256": canonical_sha256(config),
        "case_audits": audits,
        "jobs": jobs,
        "generation_job_count": sum(job.get("new_generation", True) for job in jobs),
        "post_selection_refinement": refinement,
        "claim_limit": config["claim_limit"],
    }
