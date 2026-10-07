import unittest
import numpy as np
from foodstateedit.material_transfer.core import transport
from foodstateedit.material_transfer.sensitivity import make_fixture, edit_metrics, planar_comparator, translate_image


class SensitivityTests(unittest.TestCase):
    def test_wrong_selection_detected_against_independent_truth(self):
        source, delta, _ = make_fixture(0)
        wrong, _, _ = make_fixture(0, boundary_error=1)
        metrics = edit_metrics(source, transport(source, delta), wrong, transport(wrong, delta))
        self.assertEqual(metrics["self_volume_error"], 0)
        self.assertGreater(metrics["true_source_residual_fraction"], 0)
        self.assertGreater(metrics["true_destination_error_fraction"], 0)
        self.assertLess(metrics["selection_recall"], 1)

    def test_thickness_errors_have_independent_reference(self):
        source, delta, _ = make_fixture(2)
        thick, _, _ = make_fixture(2, thickness_error=1)
        metrics = edit_metrics(source, transport(source, delta), thick, transport(thick, delta))
        self.assertEqual(metrics["self_volume_error"], 0)
        self.assertGreater(metrics["reference_volume_error"], 0)
        self.assertLess(metrics["final_occupancy_iou"], 1)

    def test_planar_no_edit_is_identity_and_translation_does_not_wrap(self):
        rgb = np.random.default_rng(0).random((32, 32, 3))
        result = planar_comparator(rgb, np.zeros((32, 32), bool), np.array([0, 0, 0]))
        np.testing.assert_allclose(rgb, result)
        point = np.zeros((8, 8, 1)); point[3, 7] = 1
        self.assertEqual(float(translate_image(point, 2, 0).sum()), 0)


if __name__ == "__main__":
    unittest.main()
