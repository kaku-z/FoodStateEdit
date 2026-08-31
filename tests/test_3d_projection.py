import hashlib
import json
import unittest
from pathlib import Path

try:
    import numpy as np
    from PIL import Image

    from foodstateedit.projection3d import (
        PinholeCamera,
        helix_about_axis,
        interpolate_points,
        zbuffer_splat,
    )
except ModuleNotFoundError:
    np = None
    Image = None


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "fork_3d_projection_v0.json"
VACE_COMPARE_CONFIG = ROOT / "configs" / "vace_fork_3d_projection_compare_v0.json"
VACE_PREFLIGHT_REPORT = ROOT / "results" / "day8_vace_fork_3d_compare_preflight_gp39_v0.json"
VACE_RESULT_ROOT = ROOT / "results" / "day8_vace_fork_3d_compare_v0"
RESULT_ROOT = ROOT / "results" / "day8_fork_3d_projection_v0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Projection3DTests(unittest.TestCase):
    @unittest.skipIf(np is None, "requires the frozen geometry runtime with NumPy")
    def test_camera_round_trip_and_interpolation(self):
        camera = PinholeCamera.normalized_relative(736, 544, 1.2)
        uv = np.asarray([[0.0, 0.0], [368.0, 272.0], [735.0, 543.0]])
        depths = np.asarray([0.8, 1.0, 1.2])
        xyz = camera.backproject(uv, depths)
        recovered, recovered_depth = camera.project(xyz)
        np.testing.assert_allclose(recovered, uv, atol=1e-10)
        np.testing.assert_allclose(recovered_depth, depths, atol=1e-12)
        np.testing.assert_allclose(interpolate_points(xyz, xyz + 1.0, 0.0), xyz)
        np.testing.assert_allclose(interpolate_points(xyz, xyz + 1.0, 1.0), xyz + 1.0)

    @unittest.skipIf(np is None, "requires the frozen geometry runtime with NumPy")
    def test_helix_has_true_out_of_plane_motion_and_zbuffer_prefers_near_point(self):
        helix = helix_about_axis(
            np.asarray([0.0, 0.0, 1.0]),
            np.asarray([1.0, 0.0, 0.0]),
            radius=0.05,
            pitch=0.01,
            turns=2.25,
            samples=160,
        )
        self.assertLess(float(helix[:, 2].min()), 1.0)
        self.assertGreater(float(helix[:, 2].max()), 1.0)
        canvas = np.zeros((3, 3, 3), dtype=np.uint8)
        rendered, visible = zbuffer_splat(
            canvas,
            np.asarray([[1.0, 1.0], [1.0, 1.0]]),
            np.asarray([1.0, 0.5]),
            np.asarray([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
        )
        self.assertEqual(rendered[1, 1].tolist(), [0, 255, 0])
        self.assertEqual(visible.tolist(), [False, True])

    def test_frozen_fork_pilot_is_claim_limited_and_not_fake_reconstruction(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["anchor_id"], "pasta_fork_001")
        self.assertEqual(
            config["scientific_status"],
            "relative_3d_geometry_pilot_not_reconstructed_scene_depth",
        )
        self.assertIn("does not claim monocular scene reconstruction", config["claim_limit"])
        self.assertEqual(config["spaghetti"]["strand_count"], 4)
        self.assertGreaterEqual(
            config["spaghetti"]["helix_turns"],
            config["projection"]["minimum_projected_turns"],
        )

    @unittest.skipIf(np is None, "requires the frozen geometry runtime with NumPy")
    def test_completed_fork_projection_is_traceable_and_geometrically_valid(self):
        manifest = json.loads((RESULT_ROOT / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "complete_relative_3d_projection_pilot_requires_visual_and_learned_render_review")
        self.assertEqual(manifest["geometry_source"], "relative_3d_normalized_pinhole")
        self.assertLess(manifest["fork_final_reprojection_max_error_px"], 1e-9)
        self.assertTrue(manifest["selected_depth_crosses_fork"])
        depth_min, depth_max = manifest["selected_helix_depth_range"]
        self.assertLess(depth_min, manifest["selected_fork_depth"])
        self.assertGreater(depth_max, manifest["selected_fork_depth"])
        self.assertEqual(manifest["outside_motion_support_max_pixel_difference"], 0)
        self.assertEqual(manifest["frames"][0], {"index": 0, "phase": "source_anchor", "changed_pixels": 0})
        self.assertEqual(len(manifest["selected_strand_lengths_3d"]), 4)
        for name, record in manifest["files"].items():
            tracked = RESULT_ROOT / name
            if tracked.exists():
                self.assertEqual(tracked.stat().st_size, record["size_bytes"])
                self.assertEqual(sha256_file(tracked), record["sha256"])

        with np.load(RESULT_ROOT / "geometry_3d.npz") as geometry:
            self.assertEqual(geometry["intrinsic"].shape, (3, 3))
            self.assertEqual(geometry["helix_points"].shape, (4, 160, 3))
            self.assertEqual(geometry["tail_points"].shape, (4, 100, 3))
            self.assertGreater(float(np.ptp(geometry["helix_points"][..., 2])), 0.03)

        with Image.open(RESULT_ROOT / "projected_frame_18.png") as projected:
            self.assertEqual(projected.size, (736, 544))
            self.assertGreater(int(np.asarray(projected).max()), 0)

    def test_pilot_source_files_match_the_recorded_bytes(self):
        expected = {
            ROOT / "scripts" / "build_fork_3d_projection_control.py":
                "1a3cf2af199d4adf473d5ebaf507a061ce5dbe86f3c9d2f3e5e00ab1a061f459",
            ROOT / "foodstateedit" / "projection3d.py":
                "181727cb57c11a4253007268686ff8975695c7eafb683e160720cbb2b52176e0",
            CONFIG:
                "e737caa96490edee7e37babd5ee11af699c249a8e639235289e9fbe5f1be681a",
        }
        for path, expected_hash in expected.items():
            self.assertEqual(sha256_file(path), expected_hash)

    def test_builder_is_deterministic_offline_and_fail_closed(self):
        source = (ROOT / "scripts" / "build_fork_3d_projection_control.py").read_text(encoding="utf-8")
        self.assertIn("Refusing to reuse output directory", source)
        self.assertIn("PinholeCamera.normalized_relative", source)
        self.assertIn("helix_about_axis", source)
        self.assertIn("zbuffer_splat", source)
        self.assertNotIn("from_pretrained", source)
        self.assertNotIn("requests", source)
        self.assertNotIn("download", source.lower())

    def test_vace_comparison_is_same_seed_claim_limited_and_frozen(self):
        config = json.loads(VACE_COMPARE_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["source_parent_commit"], "63fa633")
        self.assertEqual(config["control"]["anchor_id"], "pasta_fork_001")
        self.assertEqual(config["control"]["geometry_source"], "relative_3d_normalized_pinhole")
        self.assertEqual(config["baseline"]["seed"], config["inference"]["seed"])
        self.assertEqual(config["inference"]["seed"], 1)
        self.assertEqual(config["inference"]["selected_frame_index"], 18)
        self.assertEqual(config["inference"]["num_inference_steps"], 20)
        self.assertFalse(config["inference"]["enable_ttm"])
        self.assertEqual(
            hashlib.sha256(config["inference"]["prompt"].encode("utf-8")).hexdigest(),
            config["inference"]["prompt_sha256_utf8"],
        )
        self.assertIn("not_generalization", config["scientific_status"])

    def test_vace_comparison_preflight_and_runner_are_offline_and_fail_closed(self):
        preflight = (ROOT / "scripts" / "preflight_vace_fork_3d_compare.py").read_text(encoding="utf-8")
        runner = (ROOT / "scripts" / "run_vace_fork_3d_compare.py").read_text(encoding="utf-8")
        self.assertIn("Refusing to overwrite preflight report", preflight)
        self.assertIn("require_no_compute_process", preflight)
        self.assertIn("required_gpu_name", preflight)
        self.assertIn("output_absent", preflight)
        self.assertIn("Preflight blocked inference; no output directory was created", runner)
        self.assertIn("Refusing to reuse output root", runner)
        self.assertIn("motion_union_alpha.png", runner)
        self.assertIn("outside_motion_support_max_pixel_difference", runner)
        for source in (preflight, runner):
            self.assertNotIn("from_pretrained", source)
            self.assertNotIn("snapshot_download", source)
            self.assertNotIn("requests", source)

    def test_remote_vace_preflight_failed_only_the_occupied_gpu_gate(self):
        report = json.loads(VACE_PREFLIGHT_REPORT.read_text(encoding="utf-8"))
        self.assertFalse(report["ready"])
        self.assertIsNone(report["selected_gpu"])
        self.assertEqual(report["config_sha256"], sha256_file(VACE_COMPARE_CONFIG))
        failed = [check for check in report["checks"] if not check["passed"]]
        self.assertEqual(len(report["checks"]), 36)
        self.assertEqual([check["id"] for check in failed], ["gpu_gate"])
        self.assertEqual(failed[0]["actual"]["safe_gpu_indices"], [])
        self.assertEqual(
            {process["owner"] for process in failed[0]["actual"]["processes"]},
            {"xiong-p"},
        )

    def test_completed_vace_comparison_is_hash_verified_and_exactly_protected(self):
        run = json.loads((VACE_RESULT_ROOT / "run_manifest.json").read_text(encoding="utf-8"))
        passing_preflight = json.loads((VACE_RESULT_ROOT / "preflight.json").read_text(encoding="utf-8"))
        evidence = json.loads((VACE_RESULT_ROOT / "evidence_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(run["status"], "complete_requires_separate_action_and_photo_review")
        self.assertTrue(passing_preflight["ready"])
        self.assertTrue(all(check["passed"] for check in passing_preflight["checks"]))
        self.assertEqual(len(passing_preflight["checks"]), 36)
        self.assertEqual(run["inference"]["pipeline_load_count"], 1)
        self.assertEqual(run["inference"]["decoded_frames"], 21)
        self.assertEqual(run["inference"]["outside_motion_support_max_pixel_difference"], 0)
        self.assertEqual(evidence["source_video"]["sha256"], "dbac7be2a7ffb4c1cb996f960e04f799542f1875e7fd16d231155af863b045c4")
        self.assertFalse(evidence["source_video"]["tracked_in_git"])
        self.assertEqual(
            evidence["builder_sha256"],
            sha256_file(ROOT / "scripts" / "make_day8_fork_3d_vace_review.py"),
        )
        for name, record in evidence["files"].items():
            path = VACE_RESULT_ROOT / name
            self.assertEqual(path.stat().st_size, record["size_bytes"])
            self.assertEqual(sha256_file(path), record["sha256"])
        self.assertFalse((VACE_RESULT_ROOT / "result.mp4").exists())

    def test_completed_vace_review_separates_rigid_signal_from_action_failure(self):
        review = json.loads((VACE_RESULT_ROOT / "internal_visual_review_v0.json").read_text(encoding="utf-8"))
        self.assertFalse(review["day6_planar"]["action_success"])
        self.assertFalse(review["day8_relative_3d"]["action_success"])
        self.assertFalse(review["day8_relative_3d"]["photo_success"])
        self.assertTrue(review["relative_conclusion"]["utensil_identity_improved"])
        self.assertTrue(review["relative_conclusion"]["utensil_trajectory_improved"])
        self.assertFalse(review["relative_conclusion"]["food_motion_improved"])
        self.assertIn("DO_NOT_EXPAND_SEEDS", review["decision"])

        baselines = json.loads((ROOT / "configs" / "baselines_v1.json").read_text(encoding="utf-8"))
        gate = baselines["day8_fork_3d_vace_gate"]
        self.assertEqual(gate["technical_complete"], 1)
        self.assertEqual(gate["provisional_action_success"], 0)
        self.assertEqual(gate["provisional_photo_success"], 0)
        self.assertEqual(gate["rigid_utensil_topology_signal"], 1)
        self.assertEqual(gate["deformable_payload_signal"], 0)


if __name__ == "__main__":
    unittest.main()
