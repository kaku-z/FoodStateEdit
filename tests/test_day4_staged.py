import hashlib
import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_worker_constants():
    source = (ROOT / "scripts" / "run_geoedit_resident_batch.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    wanted = {"EXPECTED_GEOEDIT_FILES", "SCHEDULE", "MASKS"}
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                constants[target.id] = ast.literal_eval(node.value)
    if constants.keys() != wanted:
        raise AssertionError(f"Missing worker constants: {wanted - constants.keys()}")
    return constants


class Day4StagedTests(unittest.TestCase):
    def test_schedule_was_frozen_and_matches_worker(self):
        config = json.loads((ROOT / "configs" / "staged_schedule_v1.json").read_text())
        worker = load_worker_constants()
        self.assertEqual(config["seed"], 1)
        self.assertEqual(config["schedule"], worker["SCHEDULE"])
        self.assertEqual(config["mask_mapping"], worker["MASKS"])
        self.assertFalse(config["warm_start"])
        self.assertEqual(config["replace_mode"], "mask_new")
        self.assertIn("before any four-layer stochastic output", config["freeze_rule"])

    def test_day3_override_snapshots_are_byte_exact(self):
        root = ROOT / "results" / "day3_geoedit_overrides_v1" / "GeoEdit"
        expected = {
            "geoedit/inference.py": "f78488216536e70744e839c0294319f334d541adc36f9e76136c14e7ffd328fa",
            "diffsynth/pipelines/wan_video.py": "28a1b5e3939983139e2dcd4a7f19d085fcec0017434793bac163767b7ae29c00",
            "diffsynth/utils/data/__init__.py": "fe056b4a675a345cf02d6c76d327e8434103ccf2a4066ff208ce41368771e360",
            "tests/test_masks.py": "57176ea2595f236846e1d38a79f58e3d586a99e67ce7ef0df4747af768a982da",
        }
        for relative, digest in expected.items():
            self.assertEqual(sha256(root / relative), digest)

    def test_four_layer_runtime_hashes_match_worker_contract(self):
        worker = load_worker_constants()
        root = ROOT / "vendor_overrides" / "GeoEdit"
        for relative, digest in worker["EXPECTED_GEOEDIT_FILES"].items():
            self.assertEqual(sha256(root / relative), digest)

    def test_pipeline_uses_all_four_half_open_windows(self):
        pipeline = (
            ROOT / "vendor_overrides" / "GeoEdit" / "diffsynth" / "pipelines" / "wan_video.py"
        ).read_text(encoding="utf-8")
        inference = (
            ROOT / "vendor_overrides" / "GeoEdit" / "geoedit" / "inference.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def active_ttm_layer_names", pipeline)
        for name in ("rigid", "contact", "material", "hole"):
            self.assertIn(f'"{name}"', pipeline)
        self.assertIn("replace_start <= global_step < endpoint", pipeline)
        self.assertIn('"--contact-mask"', inference)
        self.assertIn('"--hole-mask"', inference)


if __name__ == "__main__":
    unittest.main()
