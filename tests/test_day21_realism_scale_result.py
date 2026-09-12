import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/day21_realism_scale_sweep_v1/result.json"
CROSS_METHOD = ROOT / "results/day21_cross_method_evaluation_status_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Day21RealismScaleResultTests(unittest.TestCase):
    def setUp(self):
        self.result = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_frozen_implementation_and_verification_hashes(self):
        evidence = self.result["evidence"]
        self.assertEqual(
            evidence["config_sha256"],
            sha256(ROOT / "configs/day21_realism_scale_sweep_v1.json"),
        )
        self.assertEqual(
            evidence["runner_sha256"],
            sha256(ROOT / "scripts/run_realism_scale_sweep.py"),
        )
        self.assertEqual(
            evidence["dependency_sha256"],
            sha256(ROOT / "scripts/run_high_lift_vace_pilot.py"),
        )
        self.assertEqual(
            evidence["verifier_sha256"],
            sha256(ROOT / "scripts/verify_realism_scale_sweep.py"),
        )
        self.assertEqual(
            evidence["verification_sha256"],
            sha256(ROOT / "artifacts/day21_realism_scale_sweep_review_v1/verification.json"),
        )

    def test_all_run_packages_match_manifests_and_remote_listings(self):
        evidence = self.result["evidence"]
        for case in ("soup", "rice", "cake"):
            artifact_root = ROOT / "artifacts/day21_realism_scale_sweep_v1" / case
            manifest_path = artifact_root / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(sha256(manifest_path), evidence["run_manifest_sha256"][case])
            self.assertEqual(manifest["status"], "complete_requires_matched_baseline_review")
            self.assertEqual(manifest["pipeline_load_count"], 1)
            self.assertEqual(manifest["config_sha256"], evidence["config_sha256"])
            self.assertEqual(manifest["runner_sha256"], evidence["runner_sha256"])
            self.assertEqual(
                [row["vace_scale"] for row in manifest["completed_conditions"]],
                [0.6, 0.8],
            )
            for condition in manifest["completed_conditions"]:
                self.assertEqual(condition["frames"], 21)
                self.assertEqual(condition["steps"], 20)
                self.assertEqual(condition["seed"], 1)
                self.assertFalse(condition["lora_enabled"])
                self.assertFalse(condition["ttm_enabled"])
                self.assertEqual(condition["outside_support_max_pixel_difference"], 0)
                for record in condition["files"].values():
                    path = artifact_root / record["path"]
                    self.assertEqual(path.stat().st_size, record["size_bytes"])
                    self.assertEqual(sha256(path), record["sha256"])

            preflight = ROOT / "results/day21_realism_scale_sweep_v1" / f"{case}_preflight.json"
            preflight_data = json.loads(preflight.read_text(encoding="utf-8"))
            self.assertEqual(sha256(preflight), evidence["preflight_sha256"][case])
            self.assertTrue(preflight_data["passed"])
            self.assertEqual(preflight_data["gpu"]["name"], "NVIDIA RTX A6000")
            self.assertGreaterEqual(preflight_data["gpu"]["free_memory_mib"], 48000)
            self.assertLessEqual(preflight_data["gpu"]["utilization_percent"], 5)
            self.assertEqual(preflight_data["gpu"]["compute_processes"], [])
            self.assertGreaterEqual(preflight_data["host_available_memory_mib"], 80000)

            listing = ROOT / "results/day21_realism_scale_sweep_v1" / f"{case}_remote_sha256.txt"
            self.assertEqual(sha256(listing), evidence["remote_listing_sha256"][case])
            rows = [line.split(maxsplit=1) for line in listing.read_text().splitlines()]
            self.assertEqual(len(rows), 18)
            for expected, relative in rows:
                path = artifact_root / relative.removeprefix("./")
                self.assertTrue(path.is_file(), path)
                self.assertEqual(sha256(path), expected)

    def test_manual_gate_and_claim_boundary(self):
        aggregate = self.result["aggregate"]
        self.assertEqual(aggregate["cases_with_clear_final_image_photo_improvement"], 2)
        self.assertEqual(aggregate["cases_passing_strict_final_image_gate"], 1)
        self.assertEqual(aggregate["cases_passing_temporal_action_gate"], 0)
        self.assertFalse(aggregate["universally_safe_scale_found"])
        self.assertEqual(
            self.result["manual_review"]["cases"]["cake"]["selected_scale_for_final_image"],
            0.6,
        )
        decision = self.result["decision"]
        self.assertFalse(decision["run_blind_benchmark"])
        self.assertFalse(decision["generalization_claim_allowed"])
        self.assertFalse(decision["physical_correctness_claim_allowed"])
        self.assertFalse(decision["paper_level_photo_realism_claim_allowed"])


class Day21CrossMethodStatusTests(unittest.TestCase):
    def setUp(self):
        self.status = json.loads(CROSS_METHOD.read_text(encoding="utf-8"))

    def test_cross_method_figure_is_hash_traceable(self):
        figure = self.status["figure"]
        self.assertEqual(sha256(ROOT / figure["path"]), figure["sha256"])
        self.assertEqual(sha256(ROOT / figure["manifest_path"]), figure["manifest_sha256"])
        self.assertEqual(sha256(ROOT / figure["builder_path"]), figure["builder_sha256"])
        manifest = json.loads((ROOT / figure["manifest_path"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["rows"], 4)
        self.assertEqual(len(manifest["columns"]), 5)
        self.assertEqual(len(manifest["records"]), 20)
        for record in manifest["records"]:
            if record["column"] == "input":
                self.assertEqual(sha256(ROOT / record["source_path"]), record["source_sha256"])
            else:
                self.assertEqual(sha256(ROOT / record["path"]), record["sha256"])

    def test_cross_method_numbers_remain_descriptive(self):
        methods = self.status["methods"]
        self.assertEqual(methods["vanilla_geoedit"]["provisional_action_success"], 1)
        self.assertEqual(methods["vanilla_geoedit"]["provisional_photo_success"], 0)
        for name in (
            "geoedit_unified_action_mask",
            "foodstateedit_staged_exclusive",
            "vace_direct_static_proxy",
        ):
            self.assertEqual(methods[name]["provisional_action_success"], 0)
            self.assertEqual(methods[name]["provisional_photo_success"], 0)
        self.assertEqual(self.status["verification_status"], "ANALYZED")
        self.assertEqual(self.status["overall_confidence"], "CAUTION")
        self.assertTrue(self.status["statistical_interpretation"]["descriptive_only"])
        self.assertEqual(self.status["statistical_interpretation"]["fallacy_scan_coverage"], "11/11")
        self.assertFalse(self.status["formal_benchmark_gate"]["passed"])
        self.assertEqual(
            set(self.status["model_scope"]["not_executed"]),
            {"Qwen-Image-Edit", "FLUX-Kontext", "ChronoEdit"},
        )


if __name__ == "__main__":
    unittest.main()
