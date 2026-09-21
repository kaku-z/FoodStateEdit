import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class E2ControlRepresentationPairedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/e2_control_representation_paired_v1.json").read_text(encoding="utf-8"))
        cls.builder = (ROOT / "scripts/build_e2_control_representation_paired_v1.py").read_text(encoding="utf-8")
        cls.runner = (ROOT / "scripts/run_e2_control_representation_paired_v1.py").read_text(encoding="utf-8")

    def test_frozen_matched_design(self):
        self.assertEqual(self.config["cases"], ["cake", "noodle"])
        self.assertEqual([arm["id"] for arm in self.config["arms"]], ["appearance_laden_rgb", "fullframe_canny"])
        self.assertEqual(self.config["inference"]["seeds"], [1, 2, 3])
        self.assertEqual(self.config["inference"]["reference_count_per_arm"], 1)
        self.assertFalse(self.config["inference"]["lora_enabled"])
        self.assertFalse(self.config["inference"]["enable_ttm"])

    def test_builder_uses_full_frame_canny_without_e1_union_outline(self):
        self.assertIn("cv2.Canny", self.builder)
        self.assertIn("np.repeat(edges", self.builder)
        self.assertNotIn("MORPH_GRADIENT", self.builder)
        self.assertNotIn("control[active] = 0", self.builder)

    def test_runner_preserves_pairing_and_lossless_evaluation(self):
        self.assertIn('vace_reference_image=[reference]', self.runner)
        self.assertIn('for seed in inference_cfg["seeds"]', self.runner)
        self.assertIn('project_lossless(', self.runner)
        self.assertIn('projected_frames', self.runner)
        self.assertIn('pipeline_load_count', self.runner)
        self.assertIn('assert_no_foreign_process', self.runner)


if __name__ == "__main__":
    unittest.main()
