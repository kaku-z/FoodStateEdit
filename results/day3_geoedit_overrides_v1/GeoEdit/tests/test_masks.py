import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from geoedit.inference import compute_hole_masks, load_video_or_image, validate_args


def _mask(array):
    return Image.fromarray(np.asarray(array, dtype=np.uint8) * 255, mode="L")


class MaskTests(unittest.TestCase):
    def test_still_image_is_repeated_as_video_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth.png"
            Image.new("RGB", (3, 2), (10, 20, 30)).save(path)

            frames = load_video_or_image(path, height=8, width=16, num_frames=5)

        self.assertEqual(len(frames), 5)
        self.assertTrue(all(frame.mode == "RGB" for frame in frames))
        self.assertTrue(all(frame.size == (16, 8) for frame in frames))
        self.assertTrue(all(frame is not frames[0] for frame in frames[1:]))

    def test_hole_is_old_mask_minus_new_mask(self):
        old = _mask([[1, 1], [0, 0]])
        new = _mask([[0, 1], [0, 0]])
        result = compute_hole_masks([old], [new], dilation=1)[0].convert("L")
        self.assertEqual(
            (np.asarray(result) > 127).tolist(),
            [[True, False], [False, False]],
        )

    def test_valid_timestep_range(self):
        args = argparse.Namespace(
            num_frames=81,
            num_inference_steps=50,
            tweak_index=3,
            tstrong_index=15,
        )
        validate_args(args)

    def test_invalid_frame_count(self):
        args = argparse.Namespace(
            num_frames=80,
            num_inference_steps=50,
            tweak_index=3,
            tstrong_index=15,
        )
        with self.assertRaisesRegex(ValueError, "4n\\+1"):
            validate_args(args)

    def test_valid_staged_material_schedule(self):
        args = argparse.Namespace(
            num_frames=21,
            num_inference_steps=20,
            tweak_index=3,
            tstrong_index=12,
            material_mask=Path("mask_material.png"),
            material_tstrong_index=8,
            replace_mode="mask_new",
        )
        validate_args(args)

    def test_material_schedule_requires_material_mask(self):
        args = argparse.Namespace(
            num_frames=21,
            num_inference_steps=20,
            tweak_index=3,
            tstrong_index=12,
            material_mask=None,
            material_tstrong_index=8,
            replace_mode="mask_new",
        )
        with self.assertRaisesRegex(ValueError, "requires --material-mask"):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
