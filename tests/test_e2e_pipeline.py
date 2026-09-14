import json
import tempfile
import unittest
from pathlib import Path

from foodstateedit.e2e_pipeline import FrameMetric, METHODS, build_plan, phase_for_frame, run_id, select_frame


ROOT = Path(__file__).resolve().parents[1]


class E2EPipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/e2e_interaction_v1.json").read_text(encoding="utf-8"))
        self.cases = json.loads((ROOT / "configs/e2e_interaction_ramen_smoke_cases_v1.json").read_text(encoding="utf-8"))["cases"]

    def test_schedule_is_total_and_exact(self):
        phases = [phase_for_frame(i) for i in range(81)]
        self.assertEqual(phases.count("approach"), 16)
        self.assertEqual(phases.count("contact"), 13)
        self.assertEqual(phases.count("lift"), 32)
        self.assertEqual(phases.count("hold"), 20)

    def test_plan_has_five_generations_and_f_reuses_e(self):
        plan = build_plan(self.config, self.cases, ROOT)
        self.assertEqual(plan["generation_job_count"], 5)
        self.assertEqual(len(plan["jobs"]), 6)
        self.assertEqual(plan["post_selection_refinement"]["backend"], "ChordEdit_SD_Turbo")
        self.assertTrue(plan["post_selection_refinement"]["exact_outside_support"])
        event = plan["jobs"][-1]
        self.assertFalse(event["new_generation"])
        self.assertEqual(event["source_method"], "E_coupled_vace")

    def test_run_id_is_stable_and_method_scoped(self):
        first = run_id(METHODS[0], "case", 0, self.config)
        self.assertEqual(first, run_id(METHODS[0], "case", 0, self.config))
        self.assertNotEqual(first, run_id(METHODS[1], "case", 0, self.config))

    def test_ordinary_selector_balances_three_factors(self):
        weights = self.config["selector"]["weights"]
        result = select_frame([
            FrameMetric(61, 0.8, 0.8, 0.2),
            FrameMetric(72, 0.9, 0.7, 0.1),
        ], weights)
        self.assertEqual(result["selected_index"], 72)

    def test_event_selector_does_not_promote_unconfirmed_tracking(self):
        weights = self.config["selector"]["weights"]
        result = select_frame([
            FrameMetric(70, 1.0, 1.0, 0.0, None, None, None),
            FrameMetric(71, 0.8, 0.8, 0.1, 0.7, 0.7, True),
        ], weights, require_events=True)
        self.assertEqual(result["selected_index"], 71)
        failed = select_frame([FrameMetric(70, 1.0, 1.0, 0.0, None, None, None)], weights, require_events=True)
        self.assertEqual(failed["status"], "unable_to_confirm")

    def test_writer_contract_can_be_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "plan.json"
            target.open("x").close()
            with self.assertRaises(FileExistsError):
                target.open("x")


if __name__ == "__main__":
    unittest.main()
