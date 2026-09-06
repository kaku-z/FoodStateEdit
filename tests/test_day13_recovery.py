import copy
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Day13RecoveryTests(unittest.TestCase):
    def test_recovery_changes_only_storage_paths_and_recovery_metadata(self):
        original = json.loads((ROOT / 'configs/flexible_completion_execution_v1.json').read_text(encoding='utf-8'))
        recovery = json.loads((ROOT / 'configs/flexible_completion_execution_recovery_20260906_v1.json').read_text(encoding='utf-8'))
        restored = copy.deepcopy(recovery)
        root = restored.pop('recovery')['persistent_root']
        self.assertTrue(root.startswith('/host/space0/guo-z/tf-ufi/outputs/'))
        self.assertTrue(restored['dataset']['remote_root'].startswith(root + '/'))
        for arm, values in restored['arms'].items():
            self.assertTrue(values['output_root'].startswith(root + '/training/'))
            values['output_root'] = original['arms'][arm]['output_root']
        self.assertTrue(restored['evaluation']['output_root'].startswith(root + '/'))
        restored['dataset']['remote_root'] = original['dataset']['remote_root']
        restored['evaluation']['output_root'] = original['evaluation']['output_root']
        restored['freeze_date'] = original['freeze_date']
        self.assertEqual(restored, original)


if __name__ == '__main__':
    unittest.main()
