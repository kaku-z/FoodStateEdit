import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from source_cavity_masks import source_reveal_mask, inward_alpha, hybrid_condition, composite_repair


class SourceCavityContractTest(unittest.TestCase):
    def test_reveal_retains_source_and_excludes_payload(self):
        source = np.zeros((32, 32), bool)
        source[12:24, 10:22] = True
        payload = np.zeros_like(source)
        payload[3:15, 10:22] = True
        self.assertFalse(source_reveal_mask(source, payload, lifted=False, source_radius=2, exclusion_radius=2).any())
        revealed = source_reveal_mask(source, payload, lifted=True, source_radius=2, exclusion_radius=2)
        self.assertFalse(revealed[payload].any())
        self.assertTrue(revealed[20, 15])

    def test_condition_branches_and_final_projection(self):
        base = np.full((24, 24, 3), 180, np.uint8)
        structure = np.zeros_like(base)
        support = np.zeros((24, 24), bool)
        support[6:18, 6:18] = True
        condition = hybrid_condition(base, structure, support)
        self.assertTrue(np.array_equal(condition[~support], base[~support]))
        self.assertFalse(condition[support].any())
        alpha = inward_alpha(support, 3)
        self.assertFalse(alpha[~support].any())
        self.assertEqual(alpha[12, 12], 255)
        self.assertLess(alpha[6, 6], 255)
        candidate = np.full_like(base, 40)
        final = composite_repair(base, candidate, alpha)
        self.assertTrue(np.array_equal(final[~support], base[~support]))
        self.assertTrue(np.array_equal(final[12, 12], candidate[12, 12]))

    def test_zero_support_is_exact_noop(self):
        rng = np.random.default_rng(1)
        base = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        alpha = inward_alpha(np.zeros((20, 20), bool), 4)
        self.assertTrue(np.array_equal(composite_repair(base, 255 - base, alpha), base))


if __name__ == '__main__':
    unittest.main()
