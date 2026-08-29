import ast
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compose_motion_union_projection.py"
CONFIG = ROOT / "configs" / "motion_union_projection_diagnostic_v1.json"


class MotionUnionProjectionTests(unittest.TestCase):
    def test_config_is_explicitly_post_hoc_and_nonstochastic(self):
        config = json.loads(CONFIG.read_text())
        self.assertEqual(
            config["scientific_status"], "post_hoc_exploratory_not_preregistered"
        )
        self.assertFalse(config["stochastic_rerun"])
        self.assertEqual(config["source_seed"], 1)
        self.assertEqual(config["source_selected_frame_index"], 18)
        self.assertIn("confirmed prospectively", config["claim_limit"])

    def test_script_constants_match_config(self):
        config = json.loads(CONFIG.read_text())
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        wanted = {"METHOD", "SOURCE_METHOD", "SOURCE_SEED", "SOURCE_SELECTED_FRAME_INDEX"}
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in wanted:
                    constants[target.id] = ast.literal_eval(node.value)
        self.assertEqual(constants.keys(), wanted)
        self.assertEqual(config["method"], constants["METHOD"])
        self.assertEqual(config["source_method"], constants["SOURCE_METHOD"])
        self.assertEqual(config["source_seed"], constants["SOURCE_SEED"])
        self.assertEqual(
            config["source_selected_frame_index"],
            constants["SOURCE_SELECTED_FRAME_INDEX"],
        )

    def test_script_is_deterministic_fail_closed_and_has_no_model(self):
        source = SCRIPT.read_text(encoding="utf-8").lower()
        self.assertNotIn("torch", source)
        self.assertNotIn("diffusers", source)
        self.assertNotIn("imagegen", source)
        self.assertIn("refusing to reuse output root", source)
        self.assertIn("outside_motion_union_alpha_max_pixel_difference", source)
        self.assertIn("post_hoc_exploratory_not_preregistered", source)
        self.assertIn("verify_hash", source)

    def test_completed_diagnostic_is_traceable_and_claim_limited(self):
        root = ROOT / "results" / "day6_motion_union_projection_diagnostic_v1"
        summary_path = root / "motion_union_projection_summary.json"
        summary = json.loads(summary_path.read_text())
        self.assertEqual(summary["method"], "motion_union_projection_diagnostic")
        self.assertEqual(
            summary["scientific_status"],
            "post_hoc_exploratory_not_preregistered",
        )
        self.assertFalse(summary["stochastic_rerun"])
        self.assertEqual(summary["case_count"], 4)
        self.assertEqual(summary["complete_count"], 4)
        self.assertTrue(summary["all_outside_motion_union_exact"])
        self.assertEqual(
            hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "225dffe81f6cbb0165d337fd1eaaa0de40deff31d65758cd63f6db0ebe6cfedd",
        )
        manifests = sorted(root.glob("*/run_manifest.json"))
        self.assertEqual(len(manifests), 4)
        for path in manifests:
            manifest = json.loads(path.read_text())
            self.assertTrue(manifest["deterministic"])
            self.assertFalse(manifest["stochastic_rerun"])
            self.assertEqual(
                manifest["scientific_status"],
                "post_hoc_exploratory_not_preregistered",
            )
            self.assertEqual(
                manifest["invariants"][
                    "outside_motion_union_alpha_max_pixel_difference"
                ],
                0,
            )
        review = json.loads(
            (root / "internal_diagnostic_review_v1.json").read_text()
        )
        self.assertEqual(review["formal_success_claims"], 0)
        self.assertEqual(review["aggregate"]["diagnostic_topology_near_pass"], 1)
        self.assertEqual(review["aggregate"]["support_mismatch_confirmed"], 1)


if __name__ == "__main__":
    unittest.main()
