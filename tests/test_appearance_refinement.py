import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from foodstateedit.appearance_refinement import (
    REQUIRED_SEMANTIC_CHECKS,
    apply_decision,
    audit_refinement_candidate,
    strict_local_composite,
    validate_refinement_config,
)


ROOT = Path(__file__).resolve().parents[1]


class AppearanceRefinementTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (ROOT / "configs/day23_geometry_locked_chordedit_refinement_v1.json").read_text(encoding="utf-8")
        )

    def test_frozen_config_requires_one_candidate_and_complete_manual_gate(self):
        validate_refinement_config(self.config)
        self.assertEqual(self.config["candidate_policy"]["candidate_count"], 1)
        self.assertEqual(
            tuple(self.config["acceptance_gate"]["required_manual_checks"]),
            REQUIRED_SEMANTIC_CHECKS,
        )

    def test_composite_is_byte_exact_outside_hard_support(self):
        base = Image.fromarray(np.full((12, 12, 3), 20, dtype=np.uint8))
        edited = Image.fromarray(np.full((12, 12, 3), 220, dtype=np.uint8))
        support = np.zeros((12, 12), dtype=np.uint8)
        support[3:9, 4:10] = 255
        soft = Image.fromarray(np.full((12, 12), 255, dtype=np.uint8))
        output = np.asarray(strict_local_composite(base, edited, soft, Image.fromarray(support), alpha=0.5))
        outside = support == 0
        self.assertTrue(np.array_equal(output[outside], np.asarray(base)[outside]))
        self.assertTrue(np.any(output[~outside] != np.asarray(base)[~outside]))

    def test_incomplete_manual_review_rolls_back(self):
        arr = np.zeros((12, 12, 3), dtype=np.uint8)
        arr[4:8, 4:8] = 160
        base = Image.fromarray(arr)
        candidate_arr = arr.copy()
        candidate_arr[5:7, 5:7] = 180
        candidate = Image.fromarray(candidate_arr)
        mask_arr = np.zeros((12, 12), dtype=np.uint8)
        mask_arr[4:8, 4:8] = 255
        mask = Image.fromarray(mask_arr)
        audit = audit_refinement_candidate(
            base,
            candidate,
            mask,
            mask,
            {"min_changed_fraction": 0.01, "max_changed_fraction": 0.95, "min_structure_edge_cosine": 0.0},
            {},
        )
        self.assertFalse(audit["accepted"])
        self.assertEqual(audit["disposition"], "rollback_to_pre_refinement_frame")
        self.assertTrue(np.array_equal(np.asarray(apply_decision(base, candidate, audit)), arr))


if __name__ == "__main__":
    unittest.main()
