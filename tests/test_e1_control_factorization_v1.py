import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_e1_control_factorization_v1 import payload_reference, structure_scribble_frames


class E1ControlFactorizationTests(unittest.TestCase):
    def test_structure_control_preserves_inactive_pixels(self):
        source = np.full((32, 48, 3), 80, dtype=np.uint8)
        appearance = np.stack([source.copy() for _ in range(3)])
        appearance[1:, 10:22, 16:32] = 220
        alpha = np.zeros((32, 48), dtype=np.uint8)
        alpha[6:26, 12:36] = 255
        cfg = {
            "structure_blur_kernel": 11,
            "structure_canny_low": 60,
            "structure_canny_high": 150,
            "structure_edge_dilation": 3,
        }
        result = structure_scribble_frames(source, appearance, alpha, cfg)
        self.assertTrue(np.array_equal(result[0], source))
        inactive = alpha == 0
        self.assertTrue(np.all(result[:, inactive] == source[inactive]))
        self.assertTrue(set(np.unique(result[1][alpha > 0])).issubset({0, 255}))

    def test_payload_reference_has_frozen_background_and_source_content(self):
        source = np.zeros((40, 60, 3), dtype=np.uint8)
        source[10:30, 20:40] = [20, 100, 220]
        result = payload_reference(source, [[0.3, 0.2], [0.7, 0.2], [0.7, 0.8], [0.3, 0.8]], [127, 127, 127])
        self.assertEqual(result.shape, source.shape)
        self.assertTrue(np.any(np.all(result == [20, 100, 220], axis=2)))
        self.assertTrue(np.any(np.all(result == [127, 127, 127], axis=2)))


if __name__ == "__main__":
    unittest.main()
