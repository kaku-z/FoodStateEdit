import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "day29_available_evaluation", ROOT / "scripts/build_day29_available_evaluation.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Day29AvailableEvaluationTests(unittest.TestCase):
    def test_percentile_interpolates_endpoints(self):
        self.assertEqual(MODULE.percentile([0.0, 1.0], 0.0), 0.0)
        self.assertEqual(MODULE.percentile([0.0, 1.0], 1.0), 1.0)
        self.assertEqual(MODULE.percentile([0.0, 1.0], 0.5), 0.5)

    def test_case_rates_require_all_cases(self):
        records = [
            {"case_id": case, "action_success": index % 2 == 0}
            for index, case in enumerate(MODULE.CASES)
        ]
        rates = MODULE.case_rates(records, "action_success")
        self.assertEqual(set(rates), set(MODULE.CASES))

    def test_cluster_bootstrap_is_deterministic(self):
        rates = {case: index / 3 for index, case in enumerate(MODULE.CASES)}
        left = MODULE.cluster_bootstrap(rates, iterations=100, seed=7)
        right = MODULE.cluster_bootstrap(rates, iterations=100, seed=7)
        self.assertEqual(left, right)

    def test_paired_difference_direction(self):
        qwen = {case: 1.0 for case in MODULE.CASES}
        chord = {case: 0.0 for case in MODULE.CASES}
        result = MODULE.paired_cluster_bootstrap(qwen, chord, iterations=100, seed=9)
        self.assertEqual(result["estimate"], 1.0)
        self.assertEqual(result["ci95_percentile"], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
