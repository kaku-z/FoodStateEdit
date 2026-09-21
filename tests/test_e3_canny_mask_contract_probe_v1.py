import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_runner():
    sys.path.insert(0, str(SCRIPTS))
    helper_stub = types.ModuleType("run_high_lift_vace_pilot")
    helper_stub.resource_preflight = lambda *args, **kwargs: None
    helper_stub.run_text = lambda *args, **kwargs: ""
    helper_stub.sha256 = lambda *args, **kwargs: ""
    helper_stub.validate_file = lambda *args, **kwargs: None
    helper_stub.write_json = lambda *args, **kwargs: None
    sys.modules["run_high_lift_vace_pilot"] = helper_stub
    spec = importlib.util.spec_from_file_location(
        "run_e3_canny_mask_contract_probe_v1",
        SCRIPTS / "run_e3_canny_mask_contract_probe_v1.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class E3CannyMaskContractProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(
            (ROOT / "configs/e3_canny_mask_contract_probe_v1.json").read_text(encoding="utf-8")
        )
        cls.runner_text = (SCRIPTS / "run_e3_canny_mask_contract_probe_v1.py").read_text(encoding="utf-8")
        cls.runner = load_runner()

    def test_probe_changes_only_the_mask_contract(self):
        self.assertEqual(self.config["cases"], ["cake", "noodle"])
        self.assertEqual(self.config["source_controls"]["required_control"], "fullframe_canny")
        self.assertEqual(self.config["inference"]["seeds"], [1])
        self.assertEqual(self.config["inference"]["num_frames"], 21)
        self.assertEqual(self.config["inference"]["num_inference_steps"], 20)
        self.assertEqual(self.config["inference"]["vace_scale"], 1.0)
        self.assertFalse(self.config["inference"]["lora_enabled"])
        self.assertFalse(self.config["inference"]["enable_ttm"])

    def test_full_generation_mask_is_independent_of_edit_alpha(self):
        black = Image.new("L", (8, 8), 0)
        white = Image.new("L", (8, 8), 255)
        policy = self.config["mask_contract"]["vace_conditioning_mask_policy"]
        self.assertIsNone(self.runner.resolve_vace_conditioning_mask(policy, black, 21))
        self.assertIsNone(self.runner.resolve_vace_conditioning_mask(policy, white, 21))
        with self.assertRaises(ValueError):
            self.runner.resolve_vace_conditioning_mask("legacy_local_alpha", white, 21)

    def test_runner_does_not_reintroduce_legacy_local_vace_mask(self):
        self.assertIn("vace_video_mask=conditioning_mask", self.runner_text)
        self.assertNotIn("vace_video_mask=[alpha]", self.runner_text)
        self.assertNotIn("vace_video_mask=[edit_alpha]", self.runner_text)
        self.assertIn("project_lossless(", self.runner_text)

    def test_post_generation_projection_is_lossless_outside_support(self):
        reference_array = np.full((8, 8, 3), 100, dtype=np.uint8)
        raw_array = np.full((8, 8, 3), 200, dtype=np.uint8)
        control = np.zeros((1, 8, 8, 3), dtype=np.uint8)
        alpha_array = np.zeros((8, 8), dtype=np.uint8)
        alpha_array[2:6, 2:6] = 255
        reference = Image.fromarray(reference_array)
        raw = Image.fromarray(raw_array)
        alpha = Image.fromarray(alpha_array)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = self.runner.project_lossless([raw], reference, alpha, control, root, [0])
            projected = np.asarray(Image.open(root / "projected_frames/00.png").convert("RGB"))
        protected = alpha_array == 0
        active = alpha_array > 0
        self.assertTrue(np.array_equal(projected[protected], reference_array[protected]))
        self.assertTrue(np.array_equal(projected[active], raw_array[active]))
        self.assertEqual(metrics["outside_support_max_pixel_difference"], 0)


if __name__ == "__main__":
    unittest.main()
