import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def script_contract():
    source = (ROOT / "scripts" / "compose_material_render.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {"METHOD", "MASK_FILES"}:
                constants[target.id] = ast.literal_eval(node.value)
    return constants, source


class MaterialRenderTests(unittest.TestCase):
    def test_frozen_config_and_script_match(self):
        config = json.loads(
            (ROOT / "configs" / "material_render_v1.json").read_text()
        )
        constants, source = script_contract()
        self.assertEqual(config["method"], constants["METHOD"])
        self.assertEqual(
            set(constants["MASK_FILES"]),
            {"rigid", "contact", "material", "hole", "edit_alpha"},
        )
        self.assertEqual(config["utensil_material_rule"]["wood"], ["chopsticks"])
        self.assertIn("before any material-render v1 output", config["freeze_rule"])
        self.assertIn("result[~support] = source[~support]", source)
        self.assertIn("split_chopsticks", source)

    def test_parameters_are_bounded(self):
        config = json.loads(
            (ROOT / "configs" / "material_render_v1.json").read_text()
        )
        self.assertLessEqual(config["cast_shadow_opacity"], 0.25)
        self.assertLessEqual(config["contact_shadow_opacity"], 0.2)
        self.assertLess(config["steel_environment_mix"], 0.25)
        self.assertLess(config["wood_environment_mix"], 0.15)
        self.assertGreater(config["foreground_outer_feather_sigma"], 0)
        self.assertLessEqual(config["foreground_outer_feather_sigma"], 1.0)


if __name__ == "__main__":
    unittest.main()
