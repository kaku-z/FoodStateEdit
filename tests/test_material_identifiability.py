import unittest
import numpy as np
from foodstateedit.material_transfer.core import render, transport
from foodstateedit.material_transfer.sensitivity import make_fixture
from foodstateedit.material_transfer.identifiability import CutSurfaceAlternative, SourceAppearance, swept_box_hits, tool_probe


class IdentifiabilityTests(unittest.TestCase):
    def test_same_source_different_hidden_target(self):
        source, delta, _ = make_fixture(0)
        alternate = CutSurfaceAlternative(source)
        a, b = render(source, 64), render(source, 64, alternate)
        np.testing.assert_array_equal(a["rgb"], b["rgb"])
        target = transport(source, delta)
        self.assertGreater(np.abs(render(target, 64)["rgb"] - render(target, 64, alternate)["rgb"]).max(), .1)

    def test_source_only_estimator_is_independent_of_hidden_truth(self):
        source, delta, _ = make_fixture(1)
        a = SourceAppearance(render(source, 64))
        b = SourceAppearance(render(source, 64, CutSurfaceAlternative(source)))
        q = np.array([[25.5, 23., 8.2], [8., 15.5, 9.3]])
        normals = np.array([[0, -1, 0], [-1, 0, 0]])
        np.testing.assert_array_equal(a(q, normals), b(q, normals))

    def test_swept_box_detects_obstacle_between_clear_endpoints(self):
        obstacles = np.array([[3., 0, 0]])
        self.assertTrue(swept_box_hits([0, 0, 0], [1, 1, 1], [6, 0, 0], obstacles, obstacles + 1)[0])
        self.assertFalse(swept_box_hits([0, 0, 0], [1, 1, 1], [6, 0, 0], obstacles + [0, 1, 0], obstacles + [1, 2, 1])[0])

    def test_tool_clearance_and_blocker(self):
        source, delta, _ = make_fixture(2)
        self.assertTrue(tool_probe(source, delta, gap=2, thickness=1)["feasible"])
        self.assertFalse(tool_probe(source, delta, gap=0, thickness=1)["feasible"])
        self.assertFalse(tool_probe(source, delta, gap=2, thickness=1, blocked=True)["feasible"])


if __name__ == "__main__":
    unittest.main()
