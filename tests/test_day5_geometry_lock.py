import ast
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def script_constants():
    source = (ROOT / "scripts" / "compose_geometry_lock.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    wanted = {"METHOD", "DONOR_METHOD", "MASK_FILES"}
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                constants[target.id] = ast.literal_eval(node.value)
    if constants.keys() != wanted:
        raise AssertionError(f"Missing constants: {wanted - constants.keys()}")
    return constants, source


class GeometryLockTests(unittest.TestCase):
    def test_frozen_config_and_script_contract_match(self):
        config = json.loads((ROOT / "configs" / "geometry_lock_v1.json").read_text())
        constants, source = script_constants()
        self.assertEqual(config["method"], constants["METHOD"])
        self.assertEqual(constants["DONOR_METHOD"], "foodstateedit_staged_exclusive")
        self.assertEqual(
            set(config["foreground_layers"]),
            {"rigid", "contact", "material"},
        )
        self.assertEqual(
            set(constants["MASK_FILES"]),
            {"rigid", "contact", "material", "hole", "edit_alpha"},
        )
        self.assertIn("before any geometry-lock v1 output", config["freeze_rule"])
        self.assertIn("result[semantic_core] = proxy[semantic_core]", source)
        self.assertIn("result[np.logical_not(support)] = source", source)
        self.assertIn("edge_donor = proxy_f", source)
        self.assertEqual(source.count("generated_f * generated_weight"), 1)

    def test_generated_donor_weight_is_bounded(self):
        config = json.loads((ROOT / "configs" / "geometry_lock_v1.json").read_text())
        self.assertGreater(config["generated_contact_donor_weight"], 0)
        self.assertLess(config["generated_contact_donor_weight"], 0.5)
        self.assertGreater(config["contact_halo_strength"], 0)
        self.assertLessEqual(config["contact_halo_strength"], 0.5)

    def test_completed_batch_and_review_are_traceable(self):
        root = ROOT / "results" / "day5_geometry_lock_v1"
        summary = json.loads((root / "geometry_lock_summary.json").read_text())
        self.assertEqual(summary["method"], "foodstateedit_geometry_lock")
        self.assertEqual(summary["case_count"], 4)
        self.assertEqual(summary["complete_count"], 4)
        self.assertTrue(summary["all_semantic_cores_exact"])
        self.assertTrue(summary["all_protected_pixels_exact"])
        self.assertTrue(summary["all_generated_influence_outside_core"])
        for case in summary["cases"]:
            manifest_path = root / case["anchor_id"] / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest, case)
            self.assertEqual(
                manifest["code_commit"],
                "29531326cf74aa1f9533a42cf0cb154baf4e011a",
            )
            self.assertEqual(
                manifest["config_sha256"],
                hashlib.sha256(
                    (ROOT / "configs" / "geometry_lock_v1.json").read_bytes()
                ).hexdigest(),
            )
        review = json.loads((root / "internal_pilot_review_v1.json").read_text())
        self.assertEqual(review["aggregate"]["provisional_action_topology_success"], 4)
        self.assertEqual(review["aggregate"]["provisional_photo_success"], 0)


if __name__ == "__main__":
    unittest.main()
