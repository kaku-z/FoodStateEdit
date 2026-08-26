"""Command-line entry points for the FoodStateEdit v0.1 prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from foodstateedit import Scene, evaluate_constraints, simulate


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def require_new_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {path}")
    path.mkdir(parents=True)


def simulate_command(args: argparse.Namespace) -> None:
    scene = Scene.from_dict(load_json(args.scene))
    result = simulate(scene)
    acceptance = evaluate_constraints(result.before, result.after)
    require_new_directory(args.output_dir)
    write_json(args.output_dir / "before.json", result.before.to_dict())
    write_json(args.output_dir / "after.json", result.after.to_dict())
    write_json(args.output_dir / "trace.json", result.trace)
    write_json(args.output_dir / "acceptance_report.json", acceptance)
    print(
        json.dumps(
            {
                "scene_id": scene.scene_id,
                "status": acceptance["status"],
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
        )
    )


def boolean_check(
    check_id: str,
    category: str,
    actual: Any,
    passed: bool,
    criterion: str,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "category": category,
        "passed": bool(passed),
        "actual": actual,
        "criterion": criterion,
    }


def adapt_ramen_command(args: argparse.Namespace) -> None:
    geometry = load_json(args.geometry_manifest)
    metrics = load_json(args.deformable_metrics)["candidate"]
    manual = load_json(args.manual_audit)
    source_core = metrics["source_only_core"]
    outside = metrics["outside_alpha"]
    feather = metrics["feather_band"]
    solver = geometry["solver"]

    hard_checks = [
        boolean_check(
            "strand_length_preserved",
            "geometry",
            geometry["length_ratio"],
            0.98 <= float(geometry["length_ratio"]) <= 1.02,
            "0.98 <= target/source length <= 1.02",
        ),
        boolean_check(
            "segment_lengths_preserved",
            "geometry",
            solver["max_segment_relative_error"],
            float(solver["max_segment_relative_error"]) <= 1e-4,
            "maximum segment relative error <= 1e-4",
        ),
        boolean_check(
            "grip_constraint_satisfied",
            "geometry",
            solver["grip_position_error"],
            float(solver["grip_position_error"]) <= 1e-4,
            "3D grip error <= 1e-4",
        ),
        boolean_check(
            "lift_moves_toward_camera",
            "geometry",
            geometry["out_of_plane_depth_displacement"],
            float(geometry["out_of_plane_depth_displacement"]) < 0.0,
            "target grip depth displacement < 0",
        ),
        boolean_check(
            "old_source_removed_numeric",
            "edit",
            source_core["original_to_proxy_projection"],
            float(source_core["original_to_proxy_projection"]) >= 0.8,
            "source-core original-to-proxy projection >= 0.8",
        ),
        boolean_check(
            "old_source_not_correlated",
            "edit",
            source_core["gray_correlation_with_original"],
            float(source_core["gray_correlation_with_original"]) <= 0.5,
            "source-core gray correlation with original <= 0.5",
        ),
        boolean_check(
            "outside_edit_region_locked",
            "compositing",
            outside["max_pixel_difference"],
            int(outside["max_pixel_difference"]) == 0,
            "maximum outside-alpha pixel difference == 0",
        ),
    ]
    for check_id in (
        "exactly_two_chopsticks",
        "no_forbidden_objects",
        "continuous_noodle_bowl_to_grip",
        "grip_between_chopsticks",
        "correct_contact_z_order",
        "old_source_duplicate_absent",
    ):
        hard_checks.append(
            boolean_check(
                check_id,
                "manual_action_audit",
                manual[check_id],
                manual[check_id] is True,
                "manual visual audit must pass",
            )
        )

    photo_checks = [
        boolean_check(
            "feather_band_mae",
            "appearance",
            feather["rgb_mae_from_original"],
            float(feather["rgb_mae_from_original"]) <= 6.0,
            "feather-band RGB MAE <= 6.0",
        ),
        boolean_check(
            "feather_band_p95",
            "appearance",
            feather["rgb_abs_difference_p95"],
            float(feather["rgb_abs_difference_p95"]) <= 25.0,
            "feather-band absolute-difference P95 <= 25",
        ),
    ]
    for check_id in (
        "source_hole_texture_natural",
        "no_visible_source_hole_patch",
        "contact_and_root_shadows_natural",
    ):
        photo_checks.append(
            boolean_check(
                check_id,
                "manual_photo_audit",
                manual[check_id],
                manual[check_id] is True,
                "manual 200% crop audit must pass",
            )
        )

    hard_success = all(item["passed"] for item in hard_checks)
    photo_success = all(item["passed"] for item in photo_checks)
    report = {
        "schema_version": "foodstateedit.acceptance.v0.1",
        "scene_id": manual.get("scene_id", "ramen_geoedit"),
        "dish_type": "ramen",
        "materials": ["rigid", "strand", "granular", "liquid"],
        "action": "lift_strand_with_chopsticks",
        "inputs": {
            "geometry_manifest": str(args.geometry_manifest.resolve()),
            "deformable_metrics": str(args.deformable_metrics.resolve()),
            "manual_audit": str(args.manual_audit.resolve()),
        },
        "hard_action_success": hard_success,
        "photo_real_completion_success": photo_success,
        "strict_end_to_end_success": hard_success and photo_success,
        "hard_checks": hard_checks,
        "photo_checks": photo_checks,
        "verdict": (
            "strict_pass"
            if hard_success and photo_success
            else "action_pass_photo_incomplete"
            if hard_success
            else "action_failed"
        ),
    }
    require_new_directory(args.output_dir)
    write_json(args.output_dir / "acceptance_report.json", report)
    failed = [
        item["id"]
        for item in hard_checks + photo_checks
        if not item["passed"]
    ]
    markdown = "\n".join(
        [
            "# FoodStateEdit unified acceptance report",
            "",
            f"- Scene: `{report['scene_id']}`",
            f"- Dish: `{report['dish_type']}`",
            f"- Verdict: `{report['verdict']}`",
            f"- Hard action success: `{str(hard_success).lower()}`",
            f"- Photo-real completion success: `{str(photo_success).lower()}`",
            f"- Strict end-to-end success: `{str(hard_success and photo_success).lower()}`",
            f"- Failed checks: `{', '.join(failed) if failed else 'none'}`",
            "",
            "The strict gate passes only when both the state/action gate and the",
            "photo-real completion gate pass.",
            "",
        ]
    )
    (args.output_dir / "REPORT.md").write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "scene_id": report["scene_id"],
                "verdict": report["verdict"],
                "failed_checks": failed,
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    simulate_parser = subparsers.add_parser("simulate", help="apply state actions")
    simulate_parser.add_argument("--scene", required=True, type=Path)
    simulate_parser.add_argument("--output-dir", required=True, type=Path)
    simulate_parser.set_defaults(func=simulate_command)

    ramen_parser = subparsers.add_parser(
        "adapt-ramen", help="evaluate an existing GeoEdit noodle experiment"
    )
    ramen_parser.add_argument("--geometry-manifest", required=True, type=Path)
    ramen_parser.add_argument("--deformable-metrics", required=True, type=Path)
    ramen_parser.add_argument("--manual-audit", required=True, type=Path)
    ramen_parser.add_argument("--output-dir", required=True, type=Path)
    ramen_parser.set_defaults(func=adapt_ramen_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
