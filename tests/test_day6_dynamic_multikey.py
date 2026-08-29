import ast
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def constants(path: Path, wanted: set[str]):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                found[target.id] = ast.literal_eval(node.value)
    if found.keys() != wanted:
        raise AssertionError(f"Missing constants: {wanted - found.keys()}")
    return found


class DynamicMultikeyTests(unittest.TestCase):
    def test_schedule_is_contiguous_and_frozen(self):
        builder_constants = constants(
            SCRIPTS / "build_dynamic_multikey_proxy.py",
            {
                "FRAME_COUNT",
                "SOURCE_FRAME",
                "APPROACH_START",
                "APPROACH_END",
                "CONTACT_START",
                "CONTACT_END",
                "LIFT_START",
                "LIFT_END",
                "FINAL_START",
                "FINAL_END",
                "SELECTED_FRAME_INDEX",
            },
        )
        expected = (
            ["source_anchor"]
            + ["approach"] * 5
            + ["contact_hold"] * 3
            + ["lift"] * 8
            + ["final_hold"] * 4
        )
        actual = []
        for index in range(builder_constants["FRAME_COUNT"]):
            if index == builder_constants["SOURCE_FRAME"]:
                actual.append("source_anchor")
            elif builder_constants["APPROACH_START"] <= index <= builder_constants["APPROACH_END"]:
                actual.append("approach")
            elif builder_constants["CONTACT_START"] <= index <= builder_constants["CONTACT_END"]:
                actual.append("contact_hold")
            elif builder_constants["LIFT_START"] <= index <= builder_constants["LIFT_END"]:
                actual.append("lift")
            elif builder_constants["FINAL_START"] <= index <= builder_constants["FINAL_END"]:
                actual.append("final_hold")
        self.assertEqual(actual, expected)
        self.assertEqual(builder_constants["SELECTED_FRAME_INDEX"], 18)

    def test_config_matches_builder_and_worker(self):
        config = json.loads(
            (ROOT / "configs" / "vace_direct_dynamic_multikey_v1.json").read_text()
        )
        builder_constants = constants(
            SCRIPTS / "build_dynamic_multikey_proxy.py",
            {
                "SOURCE_FRAME",
                "APPROACH_START",
                "APPROACH_END",
                "CONTACT_START",
                "CONTACT_END",
                "LIFT_START",
                "LIFT_END",
                "FINAL_START",
                "FINAL_END",
            },
        )
        worker_path = SCRIPTS / "run_vace_dynamic_multikey_resident_batch.py"
        worker_constants = constants(
            worker_path,
            {
                "METHOD",
                "SEED",
                "NUM_FRAMES",
                "NUM_INFERENCE_STEPS",
                "VACE_SCALE",
                "SELECTED_FRAME_INDEX",
            },
        )
        self.assertEqual(config["method"], worker_constants["METHOD"])
        self.assertEqual(config["seed"], worker_constants["SEED"])
        self.assertEqual(config["num_frames"], worker_constants["NUM_FRAMES"])
        self.assertEqual(
            config["num_inference_steps"],
            worker_constants["NUM_INFERENCE_STEPS"],
        )
        self.assertEqual(config["vace_scale"], worker_constants["VACE_SCALE"])
        self.assertEqual(
            config["selected_frame_index"],
            worker_constants["SELECTED_FRAME_INDEX"],
        )
        self.assertEqual(
            config["phase_schedule"],
            {
                "source_anchor": [builder_constants["SOURCE_FRAME"], builder_constants["SOURCE_FRAME"]],
                "approach": [builder_constants["APPROACH_START"], builder_constants["APPROACH_END"]],
                "contact_hold": [builder_constants["CONTACT_START"], builder_constants["CONTACT_END"]],
                "lift": [builder_constants["LIFT_START"], builder_constants["LIFT_END"]],
                "final_hold": [builder_constants["FINAL_START"], builder_constants["FINAL_END"]],
            },
        )

    def test_worker_is_offline_resident_fixed_selection_and_fail_closed(self):
        source = (
            SCRIPTS / "run_vace_dynamic_multikey_resident_batch.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"DIFFSYNTH_SKIP_DOWNLOAD": "true"', source)
        self.assertIn('proxy_dir / "dynamic_control.mp4"', source)
        self.assertIn('proxy_dir / "dynamic_mask.mp4"', source)
        self.assertIn("pipeline_load_count", source)
        self.assertIn("Refusing to reuse output root", source)
        self.assertIn("verify_model_audit", source)
        self.assertIn("extract_selected_and_project", source)
        self.assertIn("deterministic_geometry_only_no_image_generation", source)
        self.assertNotIn("raw_last_frame.png", source)

    def test_builder_has_no_learned_model_or_imagegen_dependency(self):
        source = (SCRIPTS / "build_dynamic_multikey_proxy.py").read_text(
            encoding="utf-8"
        ).lower()
        self.assertNotIn("torch", source)
        self.assertNotIn("diffusers", source)
        self.assertNotIn("imagegen", source)
        self.assertIn("no_image_generation", source)
        self.assertIn("final-hold controls are exact copies", source)

    def test_completed_batch_and_review_are_traceable(self):
        root = ROOT / "results" / "day6_vace_dynamic_multikey_v1"
        summary = json.loads((root / "batch_summary_v1.json").read_text())
        self.assertEqual(summary["method"], "vace_direct_dynamic_multikey")
        self.assertEqual(summary["selected_frame_index"], 18)
        self.assertEqual(summary["case_count"], 4)
        self.assertEqual(summary["complete_count"], 4)
        self.assertEqual(summary["technical_failure_count"], 0)
        self.assertTrue(summary["resident_contract_passed"])
        self.assertTrue(summary["all_protected_pixels_exact"])
        self.assertTrue(summary["all_output_hashes_verified_after_transfer"])
        manifests = sorted(root.glob("gpu*/*/seed_1/run_manifest.json"))
        self.assertEqual(len(manifests), 4)
        for path in manifests:
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest["method"], "vace_direct_dynamic_multikey")
            self.assertEqual(manifest["status"], "complete")
            self.assertFalse(manifest["inference"]["enable_ttm"])
            self.assertEqual(manifest["inference"]["selected_frame_index"], 18)
            self.assertEqual(
                manifest["inference"]["outside_edit_alpha_max_pixel_difference"],
                0,
            )
            self.assertEqual(
                manifest["inference"]["pipeline_load_count_at_completion"], 1
            )
            self.assertEqual(
                manifest["environment"]["code_commit"],
                "7f11962f039136c0bf0e596265fc8b4f768fab26",
            )
            self.assertEqual(
                manifest["environment"]["worker_file"]["sha256"],
                "80b9d4b29a6fd7542a44d1fe5d9defba1cab9c6eb64218028a48d0345d0b1b30",
            )
        for worker in summary["workers"]:
            path = root / worker["tracked_path"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), worker["sha256"]
            )
        review = json.loads((root / "internal_pilot_review_v1.json").read_text())
        self.assertEqual(review["aggregate"]["provisional_action_success"], 0)
        self.assertEqual(review["aggregate"]["provisional_photo_success"], 0)
        self.assertEqual(review["aggregate"]["raw_topology_signal_count"], 1)


if __name__ == "__main__":
    unittest.main()
