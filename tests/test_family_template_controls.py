import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.build_family_template_controls import (
    FRAME_COUNT,
    feather_union,
    phase,
    render_solid,
    render_strands,
    smoothstep,
    working_size,
    write_video,
)


class FamilyTemplateControlsTest(unittest.TestCase):
    def setUp(self):
        y, x = np.mgrid[:192, :256]
        self.source = np.stack(
            [80 + x // 3, 70 + y // 3, np.full_like(x, 110)], axis=2
        ).clip(0, 255).astype(np.uint8)

    def test_working_size_is_divisible_and_bounded(self):
        width, height = working_size(1200, 800)
        self.assertEqual((width, height), (688, 464))
        self.assertEqual(width % 16, 0)
        self.assertEqual(height % 16, 0)

    def test_phase_and_smoothstep_contract(self):
        self.assertEqual([phase(i) for i in (0, 1, 6, 9, 16)], ["source", "approach", "contact", "lift", "hold"])
        self.assertEqual(smoothstep(-1.0), 0.0)
        self.assertEqual(smoothstep(0.0), 0.0)
        self.assertAlmostEqual(smoothstep(0.5), 0.5)
        self.assertEqual(smoothstep(2.0), 1.0)

    def test_all_renderers_obey_frame_and_support_contract(self):
        products = {}
        for family in ("liquid", "granular"):
            for mode in ("planar", "relative3d"):
                products[f"{family}-{mode}"], geometry = render_solid(self.source, family, mode)
                self.assertEqual(geometry["mode"], mode)
        for family in ("strand", "strand_contact"):
            for mode in ("planar", "relative3d"):
                products[f"{family}-{mode}"], geometry = render_strands(self.source, family, mode)
                self.assertEqual(geometry["mode"], mode)
        for frames in products.values():
            self.assertEqual(len(frames), FRAME_COUNT)
            self.assertTrue(np.array_equal(frames[0], self.source))
            self.assertTrue(any(not np.array_equal(frame, self.source) for frame in frames[1:]))
        alpha = feather_union(products, self.source)
        self.assertEqual(alpha.shape, self.source.shape[:2])
        self.assertGreater(np.count_nonzero(alpha), 0)
        for frames in products.values():
            for frame in frames:
                self.assertTrue(np.array_equal(frame[alpha == 0], self.source[alpha == 0]))

    def test_video_writer_emits_decodable_21_frame_video(self):
        frames, _ = render_solid(self.source, "liquid", "planar")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "control.mp4"
            write_video(path, frames)
            capture = cv2.VideoCapture(str(path))
            count = 0
            while capture.read()[0]:
                count += 1
            capture.release()
            self.assertEqual(count, FRAME_COUNT)


if __name__ == "__main__":
    unittest.main()
