from __future__ import annotations

import json
import unittest
from pathlib import Path

from foodstateedit import Scene, evaluate_constraints, simulate


ROOT = Path(__file__).resolve().parents[1]


class FoodStateEditFrameworkTests(unittest.TestCase):
    def load_scene(self, name: str) -> Scene:
        value = json.loads((ROOT / "examples" / name).read_text(encoding="utf-8"))
        return Scene.from_dict(value)

    def test_soup_transfer_conserves_volume(self) -> None:
        result = simulate(self.load_scene("soup_transfer.json"))
        report = evaluate_constraints(result.before, result.after)
        self.assertTrue(report["state_success"])
        self.assertEqual(result.after.get_object("broth_in_bowl").state["volume_ml"], 350.0)
        self.assertAlmostEqual(
            result.after.get_object("broth_in_bowl").state["fill_level"], 0.7
        )

    def test_fried_rice_scoop_conserves_every_component(self) -> None:
        result = simulate(self.load_scene("fried_rice_scoop_mix.json"))
        report = evaluate_constraints(result.before, result.after)
        self.assertTrue(report["state_success"])
        self.assertAlmostEqual(
            result.after.get_object("rice_on_spatula").state["mass_g"], 80.0
        )
        self.assertAlmostEqual(
            result.after.get_object("rice_in_wok").state["mass_g"], 420.0
        )
        self.assertAlmostEqual(
            result.after.get_object("rice_on_spatula").state["mix_uniformity"], 0.8
        )

    def test_end_to_end_is_not_claimed_without_render_evaluation(self) -> None:
        result = simulate(self.load_scene("soup_transfer.json"))
        report = evaluate_constraints(result.before, result.after)
        self.assertIsNone(report["render_success"])
        self.assertFalse(report["strict_end_to_end_success"])


if __name__ == "__main__":
    unittest.main()
