from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "day33_qwen_rice_utensil_geometry_repair_v1.json"
RUNNER_PATH = ROOT / "scripts" / "run_qwen_utensil_geometry_repair.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runner():
    spec = importlib.util.spec_from_file_location("day33_runner", RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Day33QwenRiceUtensilGeometryRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_frozen_hashes_and_single_case(self) -> None:
        config = self.config
        self.assertEqual(config["schema_version"], "foodstateedit.qwen_utensil_geometry_repair.v1")
        self.assertEqual(config["implementation"]["runner_sha256"], _sha256(RUNNER_PATH))
        self.assertEqual(
            config["implementation"]["base_runner_sha256"],
            _sha256(ROOT / "scripts" / "run_qwen_utensil_refinement.py"),
        )
        self.assertEqual(config["case_order"], ["rice_square_head_repair"])
        self.assertEqual(config["expected_generation_count"], 1)

    def test_direct_output_policy_is_explicit(self) -> None:
        policy = self.config["output_policy"]
        self.assertFalse(policy["automatic_acceptance_gate"])
        self.assertFalse(policy["rollback_on_metric_failure"])
        self.assertTrue(policy["write_candidate_as_development_final"])
        self.assertTrue(policy["human_review_required_before_paper_claim"])
        self.assertTrue(self.config["composition"]["exact_outside_hard_support"])

    def test_expanded_repair_support_covers_head_and_handle_locally(self) -> None:
        runner = _load_runner()
        mask = runner.base_runner.draw_support(
            (688, 512), self.config["cases"]["rice_square_head_repair"]["utensil_support"]
        )
        bbox = mask.getbbox()
        self.assertIsNotNone(bbox)
        assert bbox is not None
        self.assertLessEqual(bbox[0], 365)
        self.assertLessEqual(bbox[1], 55)
        self.assertGreaterEqual(bbox[2], 688)
        self.assertGreaterEqual(bbox[3], 230)
        nonzero = int(np.count_nonzero(np.asarray(mask)))
        self.assertGreater(nonzero, 20_000)
        self.assertLess(nonzero, 70_000)

    def test_output_contract_contains_direct_final_and_diagnostics(self) -> None:
        expected = set(self.config["output_contract_per_case"])
        self.assertIn("07_candidate.png", expected)
        self.assertIn("08_diagnostics.json", expected)
        self.assertIn("10_final_direct_candidate.png", expected)


if __name__ == "__main__":
    unittest.main()
