import importlib.util
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "day30_heldout_readiness", ROOT / "scripts/audit_day30_heldout_readiness.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Day30HeldoutReadinessTests(unittest.TestCase):
    def test_frozen_inventory_is_real_balanced_and_unique(self):
        inventory, freeze = MODULE.load_heldout_inventory(
            ROOT / "benchmark/data_manifest_v1.csv",
            ROOT / "benchmark/canonical_input_manifest_v1.csv",
            ROOT / "benchmark/freeze_record_v1.json",
        )
        self.assertEqual(len(inventory), 40)
        self.assertEqual(freeze["test_count"], 40)
        self.assertEqual(
            Counter(item["family"] for item in inventory),
            Counter(MODULE.EXPECTED_FAMILIES),
        )
        self.assertEqual(len({item["case_id"] for item in inventory}), 40)
        self.assertEqual(len({item["canonical_input_sha256"] for item in inventory}), 40)

    def test_pending_annotations_keep_generation_locked(self):
        inventory, _ = MODULE.load_heldout_inventory(
            ROOT / "benchmark/data_manifest_v1.csv",
            ROOT / "benchmark/canonical_input_manifest_v1.csv",
            ROOT / "benchmark/freeze_record_v1.json",
        )
        self.assertEqual(Counter(item["annotation_status"] for item in inventory), Counter({"pending": 40}))


if __name__ == "__main__":
    unittest.main()
