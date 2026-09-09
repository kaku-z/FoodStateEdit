import hashlib
import json
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]

class RecoveryContractTests(unittest.TestCase):
    def test_only_missing_cake_arm_allowed(self):
        c=json.loads((ROOT/'configs/day20_cake_recovery_v1.json').read_text())
        self.assertEqual(c['selected_conditions'],['cake__relative3d'])
        self.assertEqual(c['expected_conditions'],1)
        old=json.loads((ROOT/'configs/day19_multimaterial_pilot_v1.json').read_text())
        for field in ['inference','dataset_manifest','dataset_root','runtime','resource_gate','matched_factors']:
            self.assertEqual(c[field],old[field])
        self.assertNotEqual(c['output_root'],old['output_root'])
        self.assertEqual(c['parent_config_sha256'],hashlib.sha256((ROOT/'configs/day19_multimaterial_pilot_v1.json').read_bytes()).hexdigest())

    def test_runtime_freeze_and_original_unchanged(self):
        c=json.loads((ROOT/'configs/day20_cake_recovery_v1.json').read_text())
        for name,digest in c['implementation_files'].items():
            self.assertEqual(hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest(),digest)
        src=(ROOT/'scripts/run_multimaterial_recovery.py').read_text()
        self.assertIn("if condition not in config['selected_conditions']:continue",src)
        self.assertIn("!= ['cake__relative3d']",src)
        self.assertEqual(src.count('inference.load_pipeline('),1)
        self.assertIn('resource_preflight(config,output',src)
        self.assertIn('exist_ok=False',src)
        self.assertNotIn('kill(',src)

    def test_failure_is_preserved_in_retrieval(self):
        p=ROOT/'artifacts/day19_multimaterial_gp40_recovered_20260909_v1'
        if not p.exists():self.skipTest('Local-only results not present')
        m=json.loads((p/'run_manifest.json').read_text())
        self.assertEqual(m['status'],'technical_failure_preserved')
        self.assertEqual(len(m['completed_conditions']),5)
        self.assertIn('BrokenPipeError',(p/'failure.txt').read_text())
        self.assertTrue((p/'FAILED').exists())
        self.assertFalse((p/'cake__relative3d').exists())

if __name__=='__main__':unittest.main()
