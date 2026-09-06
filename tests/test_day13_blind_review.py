import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/flexible_completion_blind_review_20260906_v1.json"


class BlindReviewTests(unittest.TestCase):
    def test_mapping_is_deterministic_bijective_and_frozen_before_output(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertTrue(config["frozen_before_evaluation_output_was_seen"])
        self.assertTrue(config["independent_review_required"])
        self.assertTrue(config["mapping_must_not_be_given_to_reviewers"])
        source = config["execution_config_sha256"]
        conditions = list(config["aliases"].values())
        expected = sorted(conditions, key=lambda name: hashlib.sha256(f"{source}\n{name}".encode()).hexdigest())
        self.assertEqual(list(config["aliases"]), ["R1", "R2", "R3", "R4", "R5"])
        self.assertEqual(conditions, expected)
        self.assertEqual(len(set(conditions)), 5)

    def test_review_dimensions_separate_semantics_and_photo(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["reviewer_count"], 2)
        self.assertIn("pinch_contact", config["review_dimensions"])
        self.assertIn("strand_continuity", config["review_dimensions"])
        self.assertIn("connection_to_bowl", config["review_dimensions"])
        self.assertIn("final_hold", config["review_dimensions"])
        self.assertIn("photo_realism", config["review_dimensions"])


if __name__ == "__main__":
    unittest.main()
