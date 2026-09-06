import unittest

try:
    import torch
    from foodstateedit.contact_guidance import (sample_curve, path_reliability,
        structure_energy, occlusion_visibility, rgb_observer)
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is needed for differentiable guidance tests")
class ContactGuidanceTests(unittest.TestCase):
    def setUp(self):
        self.field = torch.ones(1, 1, 20, 20)
        self.uv = torch.stack([torch.linspace(3, 16, 14), torch.full((14,), 10.)], -1)
        self.visible = torch.ones(14, dtype=torch.bool)
        self.tips = torch.tensor([[16., 8.], [16., 12.]])

    def energy(self, field, **kwargs):
        return structure_energy(field, self.field, self.uv, kwargs.pop("visible", self.visible),
                                self.tips, phase=kwargs.pop("phase", "lift"), **kwargs)

    def test_break_matters_even_when_pixel_fraction_is_small(self):
        broken = self.field.clone()
        broken[:, :, :, 8:11] = 0
        self.assertGreater(float(self.energy(broken)["total"]), float(self.energy(self.field)["total"]) + .5)

    def test_occlusion_does_not_hide_an_unrelated_break(self):
        field = self.field.clone()
        field[:, :, :, 8:11] = 0
        visible = self.visible.clone()
        visible[5:8] = False
        self.assertLess(float(self.energy(field, visible=visible)["connection"]), .001)
        field[:, :, :, 13:15] = 0
        self.assertGreater(float(self.energy(field, visible=visible)["connection"]), .5)

    def test_approach_has_zero_gradient(self):
        field = torch.zeros_like(self.field).requires_grad_()
        result = self.energy(field, phase="approach")
        result["total"].backward()
        self.assertFalse(result["active"])
        self.assertEqual(float(field.grad.abs().sum()), 0.)

    def test_depth_difference_without_screen_overlap_stays_visible(self):
        tips = torch.tensor([[0., 0.], [1., 0.]])
        handles = torch.tensor([[0., 5.], [1., 5.]])
        visibility = occlusion_visibility(self.uv, torch.full((14,), 2.), tips, handles, torch.ones(2), 1.)
        self.assertTrue(bool(visibility.all()))

    def test_only_behind_overlapping_stick_is_hidden(self):
        tips = torch.tensor([[8., 2.], [13., 2.]])
        handles = torch.tensor([[8., 18.], [13., 18.]])
        depth = torch.full((14,), 1.05)
        visibility = occlusion_visibility(self.uv, depth, tips, handles, torch.tensor([1., 1.1]), .5)
        self.assertFalse(bool(visibility[5]))  # x=8, behind near stick
        self.assertTrue(bool(visibility[10]))  # x=13, in front of far stick

    def test_hidden_missing_section_receives_no_false_repair_gradient(self):
        field = torch.full_like(self.field, .2).requires_grad_()
        visible = self.visible.clone()
        visible[5:8] = False
        self.energy(field, visible=visible)["total"].backward()
        self.assertEqual(float(field.grad[0, 0, 10, 8:11].abs().sum()), 0.)
        self.assertGreater(float(field.grad.abs().sum()), 0.)

    def test_out_of_image_is_error(self):
        with self.assertRaises(ValueError):
            sample_curve(self.field, self.uv + 100)

    def test_all_occluded_is_not_an_automatic_pass(self):
        with self.assertRaises(ValueError):
            path_reliability(torch.ones(1, 5), torch.zeros(5, dtype=torch.bool))

    def test_color_observer_receives_rgb_gradient(self):
        rgb = torch.full((1, 3, 20, 20), .7, requires_grad=True)
        noodle, utensil = rgb_observer(rgb)
        structure_energy(noodle, utensil, self.uv, self.visible, self.tips, phase="lift")["total"].backward()
        self.assertTrue(bool(torch.isfinite(rgb.grad).all()))
        self.assertGreater(float(rgb.grad.abs().sum()), 0.)


if __name__ == "__main__":
    unittest.main()
