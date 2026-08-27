import ast
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_exclusive_projection_masks import owner_for_membership  # noqa: E402


def worker_constants():
    source = (ROOT / "scripts" / "run_geoedit_resident_batch_v2.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    wanted = {"METHOD", "OWNERSHIP_PRIORITY", "SCHEDULE", "MASKS"}
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                constants[target.id] = ast.literal_eval(node.value)
    if constants.keys() != wanted:
        raise AssertionError(f"Missing worker constants: {wanted - constants.keys()}")
    return constants


class ExclusiveProjectionMaskTests(unittest.TestCase):
    def test_priority_assigns_exactly_one_owner(self):
        self.assertEqual(owner_for_membership({"rigid"}), "rigid")
        self.assertEqual(owner_for_membership({"rigid", "material"}), "material")
        self.assertEqual(
            owner_for_membership({"rigid", "material", "contact"}),
            "contact",
        )
        self.assertEqual(
            owner_for_membership({"rigid", "material", "contact", "hole"}),
            "hole",
        )
        self.assertIsNone(owner_for_membership(set()))
        with self.assertRaises(ValueError):
            owner_for_membership({"unknown"})

    def test_builder_enforces_union_and_disjoint_invariants(self):
        source = (
            ROOT / "scripts" / "build_exclusive_projection_masks.py"
        ).read_text(encoding="utf-8")
        self.assertIn("Exclusive masks changed union coverage", source)
        self.assertIn("Exclusive masks overlap", source)
        self.assertIn("Exclusive mask became empty", source)
        self.assertIn("def save_mask", source)
        self.assertIn("import numpy as np\n    from PIL import Image", source)

    def test_frozen_v2_config_matches_worker(self):
        config = json.loads(
            (ROOT / "configs" / "staged_schedule_v2_exclusive.json").read_text()
        )
        worker = worker_constants()
        self.assertEqual(config["method"], worker["METHOD"])
        self.assertEqual(config["schedule"], worker["SCHEDULE"])
        self.assertEqual(config["mask_mapping"], worker["MASKS"])
        self.assertEqual(
            config["projection_mask_rule"]["ownership_priority_high_to_low"],
            list(worker["OWNERSHIP_PRIORITY"]),
        )
        self.assertIn("before any exclusive-mask stochastic output", config["freeze_rule"])


if __name__ == "__main__":
    unittest.main()
