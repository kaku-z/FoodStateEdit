import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "day25_collection", ROOT / "scripts/verify_day25_ablation_collection.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Day25AblationCollectionTests(unittest.TestCase):
    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bin"
            path.write_bytes(b"foodstateedit")
            self.assertEqual(
                MODULE.sha256_file(path),
                "f56fc9143dde1d82310c8fc0a61232a2f7de282b11648a7e7848fda11c8f329c",
            )


if __name__ == "__main__":
    unittest.main()
