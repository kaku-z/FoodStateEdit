import ast
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


if __name__ == "__main__":
    unittest.main()
