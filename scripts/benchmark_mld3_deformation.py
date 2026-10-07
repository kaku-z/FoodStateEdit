"""Known synthetic soft-block proxy benchmark; no photographic ground truth."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodstateedit.material_lineage.deformable import solve_deformation, support_gaps, mesh_edges


def soft_block(nx=25, ny=19):
    x, y = np.meshgrid(np.linspace(-.45, .45, nx), np.linspace(-.30, .30, ny))
    low = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    top = low.copy()
    top[:, 2] = .13 + .016 * np.cos(top[:, 0] * 4) * np.cos(top[:, 1] * 4)
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            faces.extend([[a, a + 1, a + nx], [a + 1, a + nx + 1, a + nx]])
    f = np.asarray(faces, dtype=np.int64)
    directed = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    boundary = directed[counts[inverse] == 1]
    n = len(top)
    side = np.asarray([[a, a + n, b] for a, b in boundary] + [[b, a + n, b + n] for a, b in boundary])
    return np.r_[top, low], np.r_[f, f[:, ::-1] + n, side]


def spoon_for(vertices, translation, radius=(.37, .27), axes=None):
    axes = np.eye(3) if axes is None else np.asarray(axes)
    q = (vertices + translation) @ axes
    center = q.mean(axis=0)[:2]
    radial = np.sum(((q[:, :2] - center) / radius) ** 2, axis=1)
    inside = radial <= 1
    curvature = .05
    return dict(spoon_axes_camera=axes.tolist(), center_local=center.tolist(), radius_ab=list(radius),
                floor_local=float(np.min(q[inside, 2] - curvature * radial[inside])), bowl_curvature_height=curvature)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/material_lineage_deformable_20261004/deformation_benchmark"))
    parser.add_argument("--iterations", type=int, default=180)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    vertices, faces = soft_block()
    translation = np.array([.1, .08, .80])
    spoon = spoon_for(vertices, translation)
    variants = {
        "rigid": dict(compliance=0),
        "soft_full": dict(compliance=.35),
        "cohesive_meat": dict(compliance=.08),
        "flexible": dict(compliance=.80),
        "no_contact": dict(compliance=.35, contact=False),
        "no_gravity": dict(compliance=.35, gravity_strength=0),
        "no_volume": dict(compliance=.35, preserve_volume=False),
        "no_bending": dict(compliance=.35, bending_strength=0),
        "double_gravity": dict(compliance=.35, gravity_strength=.18),
        "smaller_support": dict(compliance=.35),
    }
    rows = {}
    arrays = dict(rest_xyz=vertices, rigid_xyz=vertices + translation, faces=faces,
                  edges=mesh_edges(faces), material_vertex_ids=np.arange(len(vertices)))
    for name, options in variants.items():
        local_spoon = spoon_for(vertices, translation, (.29, .20)) if name == "smaller_support" else spoon
        started = time.perf_counter()
        result, metrics = solve_deformation(vertices, faces, np.eye(3), translation, local_spoon,
                                            iterations=args.iterations, **options)
        metrics["runtime_seconds"] = time.perf_counter() - started
        metrics["spoon_info"] = local_spoon
        rows[name] = metrics
        arrays[name + "_xyz"] = result
        arrays[name + "_gaps"], arrays[name + "_inside"] = support_gaps(result, local_spoon)
        print(name, json.dumps({k: metrics[k] for k in ["runtime_seconds", "normalized_nonrigid_displacement_rms", "edge_strain_rms", "volume_relative_error", "penetration_depth_normalized", "supported_vertex_fraction", "face_orientation_reversal_count"]}), flush=True)
    np.savez_compressed(args.output / "deformation_states.npz", **arrays)
    # A second source orientation, bowl inclination and lift pose, with all four
    # primary ablations on exactly the same material IDs and action.
    angle = np.deg2rad(14.)
    food_axes = np.array([[1., 0., 0.], [0., np.cos(angle), -np.sin(angle)], [0., np.sin(angle), np.cos(angle)]])
    angle = np.deg2rad(-8.)
    bowl_axes = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
    rotated_source = vertices @ food_axes.T
    translation_b = np.array([.14, -.05, .95])
    spoon_b = spoon_for(rotated_source, translation_b, (.43, .34), bowl_axes)
    second_rows = {}
    second_arrays = dict(rest_xyz=rotated_source, rigid_xyz=rotated_source + translation_b, faces=faces,
                         edges=mesh_edges(faces), material_vertex_ids=np.arange(len(vertices)))
    for name in ["rigid", "soft_full", "no_contact", "no_gravity"]:
        started = time.perf_counter()
        p, m = solve_deformation(rotated_source, faces, food_axes, translation_b, spoon_b,
                                iterations=args.iterations, **variants[name])
        m["runtime_seconds"] = time.perf_counter() - started
        m["spoon_info"] = spoon_b
        second_rows[name] = m
        second_arrays[name + "_xyz"] = p
        second_arrays[name + "_gaps"], second_arrays[name + "_inside"] = support_gaps(p, spoon_b)
        print("tilted_pose", name, json.dumps({k: m[k] for k in ["runtime_seconds", "normalized_nonrigid_displacement_rms", "edge_strain_rms", "volume_relative_error", "penetration_depth_normalized", "supported_vertex_fraction", "face_orientation_reversal_count"]}), flush=True)
    np.savez_compressed(args.output / "tilted_deformation_states.npz", **second_arrays)
    dense_v, dense_f = soft_block(61, 45)
    dense_spoon = spoon_for(dense_v, translation)
    started = time.perf_counter()
    dense_p, dense_metrics = solve_deformation(dense_v, dense_f, np.eye(3), translation, dense_spoon,
                                               compliance=.35, iterations=120)
    dense_metrics["runtime_seconds"] = time.perf_counter() - started
    np.savez_compressed(args.output / "dense_deformation_states.npz", rest_xyz=dense_v, rigid_xyz=dense_v + translation,
                        deformed_xyz=dense_p, faces=dense_f, material_vertex_ids=np.arange(len(dense_v)))
    print("dense_5490", json.dumps({k: dense_metrics[k] for k in ["runtime_seconds", "normalized_nonrigid_displacement_rms", "edge_strain_rms", "volume_relative_error", "penetration_depth_normalized", "face_orientation_reversal_count"]}), flush=True)
    claims = dict(
        known_synthetic_proxy=True, no_measured_real_material_parameters=True,
        source_ids_and_face_topology_preserved=all(r["material_vertex_ids_preserved"] and r["face_topology_preserved"] for r in rows.values()),
        full_changes_shape_beyond_global_translation=rows["soft_full"]["normalized_nonrigid_displacement_rms"] > 1.e-3,
        gravity_ablation_changes_shape_less=rows["no_gravity"]["normalized_nonrigid_displacement_rms"] < rows["soft_full"]["normalized_nonrigid_displacement_rms"],
        cohesive_case_has_lower_edge_strain=rows["cohesive_meat"]["edge_strain_rms"] < rows["soft_full"]["edge_strain_rms"],
        full_does_not_penetrate_bowl=rows["soft_full"]["penetration_depth_normalized"] < 1.e-10,
        full_has_support=rows["soft_full"]["supported_vertex_fraction"] > .01,
        full_volume_drift_below_two_percent=rows["soft_full"]["volume_relative_error"] < .02,
        smaller_support_changes_output=float(np.sqrt(np.mean((arrays["smaller_support_xyz"] - arrays["soft_full_xyz"]) ** 2))),
        tilted_full_has_support=second_rows["soft_full"]["supported_vertex_fraction"] > .01,
        tilted_full_no_penetration=second_rows["soft_full"]["penetration_depth_normalized"] < 1.e-10,
        tilted_full_changes_shape=second_rows["soft_full"]["normalized_nonrigid_displacement_rms"] > 1.e-3,
        dense_5490_no_orientation_reversals=dense_metrics["face_orientation_reversal_count"] == 0,
    )
    result = dict(status="complete", scope="Two synthetic fixed-source/action poses plus 5490-vertex scale test; no empirical photorealism or true food stiffness claim", iterations=args.iterations, cases=rows, tilted_pose_cases=second_rows, dense_scale_test=dense_metrics, checks=claims)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    panels = ["rigid", "soft_full", "cohesive_meat", "flexible", "no_contact", "no_gravity", "no_volume", "no_bending", "double_gravity", "smaller_support"]
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), sharex=True, sharey=True)
    n = len(vertices) // 2
    for ax, name in zip(axes.ravel(), panels):
        p = arrays[name + "_xyz"]
        central = np.abs(vertices[:n, 1]) < .001
        for offset, color in [(0, "#cd884e"), (n, "#6a3d22")]:
            ax.plot(p[:n][central, 0] if offset == 0 else p[n:][central, 0],
                    p[:n][central, 2] if offset == 0 else p[n:][central, 2], color=color, marker=".", ms=3)
        sp = rows[name]["spoon_info"]
        bx = np.linspace(-sp["radius_ab"][0], sp["radius_ab"][0], 80)
        by = sp["floor_local"] + sp["bowl_curvature_height"] * (bx / sp["radius_ab"][0]) ** 2
        ax.plot(bx + sp["center_local"][0], by, color="#586e83", lw=3)
        ax.set_title(name.replace("_", " ") + "\nstrain={:.3f}, vol={:.2%}".format(rows[name]["edge_strain_rms"], rows[name]["volume_relative_error"]))
        ax.grid(alpha=.2)
        ax.set_xlim(-.48, .68)
        ax.set_ylim(.45, 1.03)
    axes[0, 0].set_ylabel("lifted camera z (proxy)")
    axes[1, 0].set_ylabel("lifted camera z (proxy)")
    fig.suptitle("Persistent material mesh: bowl contact, elastic relaxation, and fixed-source ablations\nNo-contact falls below the plotting window; all source vertex IDs and topology remain fixed", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.output / "deformation_comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(claims, indent=2), flush=True)


if __name__ == "__main__":
    main()
