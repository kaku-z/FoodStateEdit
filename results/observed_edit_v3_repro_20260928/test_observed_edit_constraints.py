import unittest
from types import SimpleNamespace
import numpy as np
import torch
from foodstateedit.observed_edit.constrained import poisson_merge, neutral_plate, composite_layers
from foodstateedit.observed_edit.projection import RegionProjection


class BoundaryTests(unittest.TestCase):
    def test_removed_interior_does_not_leak_through_poisson(self):
        base = np.full((60, 70, 3), 170, np.uint8)
        mask = np.zeros((60, 70), bool); mask[15:45, 20:50] = True
        base[mask] = [255, 0, 0]
        proposal = np.full_like(base, 110)
        out = poisson_merge(base, proposal, mask)
        np.testing.assert_array_equal(out, np.full_like(base, 170))

    def test_linear_gradient_matches_boundary_without_edge_jump(self):
        yy, xx = np.mgrid[:60, :70]
        base = np.repeat((70+xx+yy)[..., None], 3, axis=2).astype(np.uint8)
        proposal = np.clip(base.astype(int)+40, 0, 255).astype(np.uint8)
        mask = np.zeros((60, 70), bool); mask[10:50, 10:60] = True
        np.testing.assert_array_equal(poisson_merge(base, proposal, mask), base)

    def test_known_food_is_a_hole_in_the_tool_edit_domain(self):
        base = np.full((60, 70, 3), 160, np.uint8)
        food = np.zeros((60, 70), bool); food[22:32, 27:37] = True
        base[food] = [180, 80, 15]
        free = np.zeros_like(food); free[10:50, 10:60] = True; free[food] = False
        proposal = np.full_like(base, 120)
        hard, smooth = composite_layers(base, proposal, free)
        for out in (hard, smooth):
            np.testing.assert_array_equal(out[~free], base[~free])

    def test_plate_fit_does_not_use_removed_food_rgb(self):
        a = np.full((160, 180, 3), 165, np.uint8)
        mask = np.zeros(a.shape[:2], bool); mask[60:100, 65:105] = True
        b = a.copy(); a[mask] = [200, 30, 10]; b[mask] = [20, 10, 240]
        out1, hole1, _ = neutral_plate(a, mask)
        out2, hole2, _ = neutral_plate(b, mask)
        np.testing.assert_array_equal(hole1, hole2)
        np.testing.assert_array_equal(out1, out2)


class ProjectionTests(unittest.TestCase):
    def test_next_sigma_and_free_region(self):
        clean = torch.tensor([[[2., 4., 6., 8.]]])
        noise = torch.tensor([[[10., 12., 14., 16.]]])
        free = torch.tensor([[[0., 1., .5, 0.]]])
        callback = RegionProjection(clean, noise, free)
        pipe = SimpleNamespace(scheduler=SimpleNamespace(sigmas=torch.tensor([1., .25, 0.])))
        incoming = torch.full_like(clean, 30)
        out = callback(pipe, 0, torch.tensor(1000), {'latents': incoming})['latents']
        torch.testing.assert_close(out, torch.tensor([[[4., 30., 19., 10.]]]))
        final = callback(pipe, 1, torch.tensor(250), {'latents': incoming})['latents']
        torch.testing.assert_close(final[..., [0,3]], clean[..., [0,3]])
        self.assertEqual(callback.audit[-1]['pinned_latent_max_error'], 0.)

    def test_reject_misaligned_latent_mask(self):
        with self.assertRaises(ValueError):
            RegionProjection(torch.zeros(1,4,16), torch.zeros(1,4,16), torch.zeros(1,4,1))


if __name__ == '__main__':
    unittest.main()
