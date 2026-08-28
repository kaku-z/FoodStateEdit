import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def worker_constants():
    source = (ROOT / "scripts" / "run_vace_direct_resident_batch.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    wanted = {"METHOD", "SEED", "NUM_FRAMES", "NUM_INFERENCE_STEPS", "VACE_SCALE"}
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                constants[target.id] = ast.literal_eval(node.value)
    if constants.keys() != wanted:
        raise AssertionError(f"Missing constants: {wanted - constants.keys()}")
    return constants, source


class VaceDirectTests(unittest.TestCase):
    def test_frozen_config_matches_worker(self):
        config = json.loads(
            (ROOT / "configs" / "vace_direct_static_proxy_v1.json").read_text()
        )
        constants, source = worker_constants()
        self.assertEqual(config["method"], constants["METHOD"])
        self.assertEqual(config["seed"], constants["SEED"])
        self.assertEqual(config["num_frames"], constants["NUM_FRAMES"])
        self.assertEqual(
            config["num_inference_steps"], constants["NUM_INFERENCE_STEPS"]
        )
        self.assertEqual(config["vace_scale"], constants["VACE_SCALE"])
        self.assertFalse(config["enable_ttm"])
        self.assertIn("enable_ttm=False", source)
        self.assertIn('proxy_dir / "motion_signal.png"', source)
        self.assertIn('proxy_dir / "edit_alpha.png"', source)

    def test_worker_is_offline_resident_and_fail_closed(self):
        _, source = worker_constants()
        self.assertIn('"DIFFSYNTH_SKIP_DOWNLOAD": "true"', source)
        self.assertIn("pipeline_load_count", source)
        self.assertIn("Refusing to reuse output root", source)
        self.assertIn("verify_model_audit", source)
        self.assertIn("extract_and_project", source)


if __name__ == "__main__":
    unittest.main()
