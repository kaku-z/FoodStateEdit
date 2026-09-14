from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "day28_qwen_utensil_refinement_four_case_v3.json"
RUNNER_PATH = ROOT / "scripts" / "run_qwen_utensil_refinement.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("day28_runner", RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Day28QwenUtensilRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_frozen_runner_hash_and_non_overwrite_paths(self) -> None:
        config = self.config
        self.assertEqual(config["schema_version"], "foodstateedit.qwen_utensil_refinement.v1")
        self.assertEqual(config["implementation"]["runner_sha256"], _sha256(RUNNER_PATH))
        self.assertEqual(
            config["implementation"]["appearance_refinement_sha256"],
            _sha256(ROOT / "foodstateedit" / "appearance_refinement.py"),
        )
        self.assertEqual(config["case_order"], ["ramen", "soup", "rice", "cake"])
        self.assertEqual(config["expected_generation_count"], 4)
        self.assertNotEqual(config["output_root"], config["runtime_root"])
        self.assertNotIn(config["preflight_path"], {config["output_root"], config["runtime_root"]})

    def test_refinement_is_single_candidate_offline_and_fail_closed(self) -> None:
        config = self.config
        self.assertTrue(all(case["seed"] == 1 for case in config["cases"].values()))
        self.assertTrue(config["backend"]["offline_only"])
        self.assertTrue(config["backend"]["downloads_forbidden"])
        self.assertTrue(config["composition"]["exact_outside_hard_support"])
        self.assertTrue(config["acceptance_gate"]["uncertain_is_failure"])
        self.assertTrue(config["acceptance_gate"]["rollback_on_any_failure"])

    def test_frozen_utensil_support_is_local_and_nonempty(self) -> None:
        runner = _load_runner()
        mask = runner.draw_support((688, 512), self.config["cases"]["cake"]["utensil_support"])
        bbox = mask.getbbox()
        self.assertIsNotNone(bbox)
        assert bbox is not None
        self.assertGreaterEqual(bbox[0], 350)
        self.assertLessEqual(bbox[1], 50)
        self.assertLessEqual(bbox[2], 660)
        self.assertLessEqual(bbox[3], 190)
        nonzero = int(np.count_nonzero(np.asarray(mask)))
        self.assertGreater(nonzero, 5_000)
        self.assertLess(nonzero, 30_000)

    def test_output_contract_keeps_proposal_and_conservative_final(self) -> None:
        expected = set(self.config["output_contract_per_case"])
        self.assertTrue({
            "00_pre_refinement.png",
            "05_raw_qwen_crop.png",
            "06_aligned_qwen_crop.png",
            "07_candidate.png",
            "08_automatic_gate.json",
            "09_manual_review.json",
            "10_final.png",
            "condition_manifest.json",
        } <= expected)

    def test_numpy_translation_registration_recovers_shift(self) -> None:
        runner = _load_runner()
        base = Image.new("RGB", (96, 96), "white")
        draw = ImageDraw.Draw(base)
        draw.rectangle((8, 10, 35, 30), fill=(20, 80, 180))
        draw.ellipse((55, 52, 82, 81), fill=(180, 40, 70))
        edited = Image.new("RGB", base.size, "white")
        edited.paste(base.crop((0, 0, 89, 92)), (7, 4))
        support = Image.new("L", base.size, 0)
        aligned, metrics = runner.align_crop(
            base, edited, support, max_translation_px=12, search_downsample=1
        )
        self.assertEqual(metrics["translation_xy_pixels"], [-7, -4])
        difference = np.abs(
            np.asarray(aligned, dtype=np.int16) - np.asarray(base, dtype=np.int16)
        )
        self.assertLess(float(difference[:, :85].mean()), 1.0)


if __name__ == "__main__":
    unittest.main()
