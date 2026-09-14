import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('evidence_summary', ROOT / 'scripts/summarize_multimaterial_evidence.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def fixture():
    records = []
    for case in summary.CASES:
        for arm in summary.ARMS:
            records.append(dict(condition=case+'__'+arm, frames=21, size=[688, 512], seed=1,
                                steps=20, source_dataset_case=case, support_fraction=0.2,
                                native_outside_mae=10.0, preencode_outside_max=0,
                                encoded_outside_mae=3.0, encoded_outside_max=50,
                                preencode_inside_mae=18.0, final_sha256='a'*64, root='fixture'))
    return {'schema_version': 'foodstateedit.non_noodle_result.v1', 'technical_metrics': records}


class EvidenceSummaryTests(unittest.TestCase):
    def run_summary(self, doc):
        return summary.summarize([('fixture', doc)])

    def test_complete_metadata_is_not_semantic_success(self):
        result = self.run_summary(fixture())
        self.assertTrue(result['complete_reported_metadata'])
        self.assertTrue(all(not x['causal_effect_or_semantic_gain_established_by_tool'] for x in result['pairs']))
        self.assertIn('NOT native model', ' '.join(result['limitations']))
        self.assertEqual(result['conditions'][0]['source_status'], 'missing')

    def test_missing_condition(self):
        doc = fixture(); doc['technical_metrics'].pop()
        result = self.run_summary(doc)
        self.assertEqual(result['missing_conditions'], ['cake__relative3d'])
        self.assertEqual(result['pairs'][2]['reported_metadata_comparability'], 'not_comparable')

    def test_duplicate_never_silently_selected(self):
        doc = fixture(); doc['technical_metrics'].append(copy.deepcopy(doc['technical_metrics'][0]))
        result = self.run_summary(doc)
        self.assertEqual(result['duplicate_conditions'], ['soup__planar'])
        self.assertEqual(result['conditions'][0]['occurrence_count'], 2)
        self.assertFalse(result['complete_reported_metadata'])

    def test_seed_mismatch_blocks_comparison(self):
        doc = fixture(); doc['technical_metrics'][1]['seed'] = 2
        self.assertEqual(self.run_summary(doc)['pairs'][0]['mismatched_fields'], ['seed'])

    def test_missing_metric_blocks_metadata_completeness(self):
        doc = fixture(); del doc['technical_metrics'][0]['native_outside_mae']
        self.assertIn('native_outside_mae', self.run_summary(doc)['conditions'][0]['missing_fields'])

    def test_invalid_numbers_and_hashes(self):
        for key, value in [('frames', True), ('support_fraction', 2), ('native_outside_mae', float('nan')),
                           ('final_sha256', 'not-a-hash'), ('size', [0, 512])]:
            with self.subTest(field=key):
                doc = fixture(); doc['technical_metrics'][0][key] = value
                self.assertIn(key, self.run_summary(doc)['conditions'][0]['invalid_fields'])

    def test_verifier_schema_supported(self):
        doc = fixture()
        doc['schema_version'] = 'foodstateedit.presentation_verification.v1'
        doc['conditions'] = doc.pop('technical_metrics')
        self.assertTrue(self.run_summary(doc)['complete_reported_metadata'])

    def test_identity_mismatch_and_wrong_frame_count(self):
        doc = fixture(); doc['technical_metrics'][0].update(source_dataset_case='cake', frames=20)
        fields = self.run_summary(doc)['conditions'][0]['invalid_fields']
        self.assertIn('source_dataset_case_identity', fields)
        self.assertIn('expected_21_frames', fields)

    def test_provenance_duplicate_not_hidden(self):
        dataset = {'schema_version': 'foodstateedit.multimaterial_dataset.v1',
                   'cases': [{'case_id': 'soup', 'source_kind': 'real'}, {'case_id': 'soup', 'source_kind': 'synthetic'}]}
        result = summary.summarize([('fixture', fixture())], dataset)
        self.assertEqual(result['conditions'][0]['source_status'], 'ambiguous')

    def test_unknown_schema_rejected_and_input_not_mutated(self):
        doc = fixture(); original = copy.deepcopy(doc)
        self.run_summary(doc)
        self.assertEqual(doc, original)
        with self.assertRaises(ValueError):
            self.run_summary({'schema_version': 'unknown'})


if __name__ == '__main__':
    unittest.main()
