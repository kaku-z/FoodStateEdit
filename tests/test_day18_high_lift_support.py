"""Frozen-input checks plus optional byte/geometry audits on local evidence."""
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'artifacts/day18_high_lift_swept_support_v1'
OLD_DATA = ROOT / 'artifacts/day17_high_lift_relative3d_udon_v1'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


class Day18SupportContractTests(unittest.TestCase):
    def setUp(self):
        self.old = read_json(ROOT / 'configs/day17_high_lift_vace_pilot_v1.json')
        self.base = read_json(ROOT / 'configs/day18_high_lift_swept_support_v1.json')
        self.run = read_json(ROOT / 'configs/day18_high_lift_swept_support_gp40_v1.json')

    def test_inference_and_model_are_identical(self):
        self.assertEqual(self.old['inference'], self.run['inference'])
        self.assertEqual(self.old['runtime'], self.run['runtime'])
        self.assertEqual(self.old['implementation'], self.run['implementation'])
        for key in ('geometry', 'reference', 'strand_mask'):
            self.assertEqual(self.old['dataset']['files'][key], self.run['dataset']['files'][key])

    def test_host_variant_changes_only_output_path(self):
        original, variant = dict(self.base), dict(self.run)
        self.assertNotEqual(original.pop('output_root'), variant.pop('output_root'))
        self.assertEqual(original, variant)

    def test_two_raster_inputs_change(self):
        for key in ('edit_alpha', 'relative3d_control'):
            self.assertNotEqual(self.old['dataset']['files'][key]['sha256'],
                                self.run['dataset']['files'][key]['sha256'])
        self.assertTrue(self.run['support_repair']['not_a_mask_stage_isolation'])

    def test_safety_and_review_gates_not_relaxed(self):
        self.assertEqual(self.old['resource_gate'], self.run['resource_gate'])
        self.assertEqual(self.old['review_gate'], self.run['review_gate'])
        self.assertEqual(self.old['claim_limit'], self.run['claim_limit'])
        runner = ROOT / self.run['implementation']['runner']
        self.assertEqual(hashlib.sha256(runner.read_bytes()).hexdigest(),
                         self.run['implementation']['runner_sha256'])

    def test_review_records_repair_without_promoting_strict_success(self):
        result = read_json(ROOT / 'results/day18_high_lift_support_repair_20260908.json')
        self.assertTrue(result['review']['raised_utensil_visible_at_frames_15_and_20'])
        self.assertFalse(result['review']['strict_action_gate_passed'])
        self.assertFalse(result['decision']['learned_improvement_established'])
        self.assertFalse(result['decision']['novel_algorithmic_contribution_established'])
        self.assertFalse(result['decision']['blind_fork_allowed'])
        self.assertFalse(result['decision']['automatic_execution_resumed'])

    def test_result_distinguishes_lossy_video_from_exact_compositing(self):
        verification = read_json(ROOT / 'results/day18_high_lift_support_repair_20260908.json')['verification']
        self.assertEqual(verification['recomposited_outside_support_max_difference'], 0)
        self.assertGreater(verification['encoded_outside_support_max_difference'], 0)
        self.assertGreater(verification['native_outside_support_mae_to_source'], 0)

    def test_completed_result_has_single_run_and_hash_verified_evidence(self):
        result = read_json(ROOT / 'results/day18_high_lift_support_repair_20260908.json')
        self.assertEqual(result['execution']['new_vace_inference_count'], 1)
        self.assertEqual(result['execution']['pipeline_load_count'], 1)
        self.assertEqual(result['execution']['frames'], 21)
        self.assertEqual(result['verification']['mismatches'], 0)
        self.assertEqual(result['verification']['remote_output_directory_files_verified'], 8)
        self.assertEqual(result['verification']['preserved_day17_artifacts_reverified'], 6)


@unittest.skipUnless(DATA.is_dir() and OLD_DATA.is_dir(), 'local-only evidence not installed')
class Day18LocalSupportEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from PIL import Image
        cls.np = np
        cls.config = read_json(ROOT / 'configs/day18_high_lift_swept_support_gp40_v1.json')
        alpha_rel = cls.config['dataset']['files']['edit_alpha']['path']
        cls.new = np.asarray(Image.open(DATA / alpha_rel).convert('L'))
        cls.old = np.asarray(Image.open(OLD_DATA / alpha_rel).convert('L'))

    def test_every_frozen_input_and_builder_matches(self):
        records = list(self.config['dataset']['files'].values()) + [self.config['dataset']['manifest']]
        for rec in records:
            path = DATA / rec['path']
            self.assertEqual(path.stat().st_size, rec['size_bytes'])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), rec['sha256'])
        manifest = read_json(DATA / 'dataset_manifest.json')
        self.assertEqual(hashlib.sha256((ROOT / 'scripts/prepare_high_lift_support_pilot.py').read_bytes()).hexdigest(),
                         manifest['builder_sha256'])
        self.assertIn('inference only', manifest['scope'])

    def test_support_never_shrinks_and_growth_is_local(self):
        np = self.np
        self.assertTrue(np.all(self.new >= self.old))
        self.assertLess(float((self.new > 0).mean() - (self.old > 0).mean()), .02)
        self.assertGreater(float((self.new > 0).mean() - (self.old > 0).mean()), 0)

    def test_all_full_trajectory_stick_bodies_have_full_opacity(self):
        np = self.np
        sys.path.insert(0, str(ROOT / 'scripts'))
        try:
            from build_flexible_completion_dataset import draw_segment
        finally:
            sys.path.pop(0)
        geometry = read_json(ROOT / 'configs/flexible_completion_udon_relative3d_geometry_high_lift_v1.json')
        manifest = read_json(OLD_DATA / 'dataset_manifest.json')
        h, w = self.new.shape
        sticks = geometry['chopsticks']
        for i in range(1, 21):
            pinch = np.asarray(manifest['frame_records'][i]['pinch_uv'])
            body = np.zeros((h, w, 3), np.uint8)
            for side in (1, -1):
                start = pinch + [0, side * sticks['tip_separation_normalized'] * h / 2]
                end = pinch + np.asarray(sticks['handle_offset_normalized']) * [w, h] + [0, side * sticks['handle_separation_normalized'] * h / 2]
                draw_segment(body, start, end, (255, 255, 255), sticks['line_width_px'], np.ones((h, w), bool))
            mask = np.any(body > 0, axis=2)
            self.assertTrue(np.all(self.new[mask] == 255), f'frame {i}')
            if i == 20:
                self.assertEqual(int((mask & (self.old == 0)).sum()), 1409)
                self.assertEqual(int(mask.sum()), 4421)


class Day18ProjectionMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.np = np
        spec = importlib.util.spec_from_file_location(
            'day18_verify', ROOT / 'scripts/verify_high_lift_support_result.py')
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_zero_alpha_restores_source_and_full_alpha_keeps_raw(self):
        np = self.np
        raw = np.full((2, 1, 2, 3), 220, np.uint8)
        ref = np.full((1, 2, 3), 20, np.uint8)
        alpha = np.array([[0, 255]], np.uint8)
        output = self.module.composite(raw, ref, alpha)
        self.assertTrue(np.all(output[:, :, 0] == 20))
        self.assertTrue(np.all(output[:, :, 1] == 220))

    def test_partial_alpha_attenuates_visible_change(self):
        np = self.np
        raw = np.full((1, 1, 1, 3), 220, np.uint8)
        ref = np.full((1, 1, 3), 20, np.uint8)
        alpha = np.array([[47]], np.uint8)
        output = self.module.composite(raw, ref, alpha)
        self.assertTrue(np.all(output == round(220 * 47 / 255 + 20 * 208 / 255)))


if __name__ == '__main__':
    unittest.main()
