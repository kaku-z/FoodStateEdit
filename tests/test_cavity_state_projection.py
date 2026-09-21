import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from cavity_state_projection import select_material_patch, soft_cavity_projection
from original_reference_compositor import frame_edit_alpha


class CavityRegressionTests(unittest.TestCase):
    def test_dynamic_compositor_leaves_untouched_first_frame_and_reveals_background(self):
        ref = np.zeros((80,100,3),np.uint8)
        repair = np.zeros((80,100),np.uint8)
        payload = np.zeros_like(repair);payload[20:30,20:30]=255
        self.assertFalse(frame_edit_alpha(ref,ref,repair,payload).any())
        proxy=ref.copy();proxy[20:30,20:30]=100
        alpha=frame_edit_alpha(ref,proxy,repair,payload)
        self.assertTrue((alpha[20:30,20:30]==255).all())
        self.assertFalse(alpha[:,70:].any())

    def test_rejects_plate_pixels_inside_rectangular_crop(self):
        image = np.full((40, 50, 3), 240, np.uint8)
        mask = np.zeros((40, 50), bool)
        mask[4:29, 8:34] = True
        mask[23:29, 8:15] = False
        image[mask] = [180, 140, 70]
        patch, box = select_material_patch(image, mask)
        self.assertTrue(np.all(patch == [180, 140, 70]))
        self.assertTrue(mask[box[1]:box[3], box[0]:box[2]].all())

    def test_missing_material_fails_closed(self):
        with self.assertRaises(ValueError):
            select_material_patch(np.zeros((20, 20, 3), np.uint8), np.zeros((20, 20), bool))

    def test_projection_is_local_bounded_and_releases(self):
        z = torch.randn(1, 3, 4, 12, 12)
        ref = torch.randn_like(z)
        mask = torch.zeros(1, 1, 4, 12, 12)
        mask[:, :, 2:, 3:9, 3:9] = 1
        trace = []
        out = soft_cavity_projection(z, ref, mask, 0, 8, trace=trace)
        outside = (mask == 0).expand_as(z)
        self.assertTrue(torch.equal(out[outside], z[outside]))
        self.assertLessEqual(trace[0]['max_weight'], .350001)
        self.assertTrue(torch.equal(soft_cavity_projection(z, ref, mask, 8, 8), z))
        later = soft_cavity_projection(z, ref, mask, 7, 8)
        self.assertLess((later-z).abs().max(), (out-z).abs().max())

    def test_reference_prefix_is_protected_and_identity_is_fixed_point(self):
        z = torch.randn(1, 3, 7, 8, 8)
        mask = torch.ones(1, 1, 7, 8, 8); mask[:, :, :1] = 0
        out = soft_cavity_projection(z, z+2, mask, 0, 8)
        self.assertTrue(torch.equal(out[:, :, :1], z[:, :, :1]))
        self.assertTrue(torch.equal(soft_cavity_projection(z, z, mask, 0, 8), z))


if __name__ == '__main__':
    unittest.main()
