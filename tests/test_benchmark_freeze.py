import csv
import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkFreezeTests(unittest.TestCase):
    def load_csv(self, name: str):
        with (ROOT / "benchmark" / name).open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_inventory_has_32_traceable_candidates_per_family(self):
        rows = self.load_csv("candidate_inventory_v1.csv")
        self.assertEqual(len(rows), 128)
        self.assertEqual(
            Counter(row["family"] for row in rows),
            Counter({"liquid": 32, "granular": 32, "strand": 32, "strand_contact": 32}),
        )
        self.assertEqual(len({row["candidate_id"] for row in rows}), 128)
        self.assertTrue(all(len(row["source_sha256"]) == 64 for row in rows))
        self.assertEqual(len({row["source_sha256"] for row in rows}), 128)

    def test_visual_review_covers_every_candidate(self):
        inventory = self.load_csv("candidate_inventory_v1.csv")
        review = json.loads((ROOT / "benchmark" / "visual_review_v3.json").read_text(encoding="utf-8"))
        reviewed_ids = set()
        for family_review in review["families"].values():
            reviewed_ids.update(family_review["eligible"])
            reviewed_ids.update(family_review["excluded"])
        self.assertEqual(reviewed_ids, {row["candidate_id"] for row in inventory})

    def test_provisional_manifest_is_balanced_and_leakage_safe(self):
        rows = self.load_csv("data_manifest_provisional_v3.csv")
        self.assertEqual(len(rows), 60)
        self.assertEqual(Counter(row["family"] for row in rows), Counter({
            "liquid": 15,
            "granular": 15,
            "strand": 15,
            "strand_contact": 15,
        }))
        for family in {row["family"] for row in rows}:
            family_rows = [row for row in rows if row["family"] == family]
            self.assertEqual(Counter(row["split"] for row in family_rows), Counter({"pilot": 5, "test": 10}))
        self.assertTrue(all(row["has_target_utensil"] == "false" for row in rows))
        self.assertTrue(all(row["freeze_status"] == "provisional_frozen" for row in rows))
        self.assertEqual(len({row["source_sha256"] for row in rows}), 60)
        splits = {row["case_id"]: row["split"] for row in rows}
        self.assertEqual(splits["noodle_001"], "pilot")
        self.assertEqual(splits["noodle_002"], "pilot")


if __name__ == "__main__":
    unittest.main()
