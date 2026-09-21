import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('teacher_audit',ROOT/'scripts/audit_teacher_mechanism_20260914.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class TeacherAuditTests(unittest.TestCase):
    def test_unsigned_subtraction_does_not_wrap(self):
        result=audit.compare(np.zeros((2,2,3),np.uint8),np.full((2,2,3),255,np.uint8))
        self.assertEqual(result['mae'],255)
        self.assertEqual(result['max_difference'],255)
        self.assertEqual(result['exact_pixel_fraction'],0)

    def test_outside_and_inside_are_separate(self):
        x=np.zeros((2,2,3),np.uint8);y=x.copy();y[0,0]=[3,6,9]
        mask=np.array([[False,True],[True,True]])
        self.assertEqual(audit.compare(x,y,mask)['mae'],0)
        self.assertEqual(audit.compare(x,y,~mask)['mae'],6)

    def test_saved_reanalysis_has_expected_scope(self):
        result=json.loads((ROOT/'results/teacher_mechanism_audit_20260914_v1.json').read_text(encoding='utf-8'))
        self.assertFalse(result['new_model_inference'])
        self.assertFalse(result['training_performed'])
        self.assertEqual(len(result['records']),36)
        self.assertEqual(len({(r['case'],r['seed'],r['condition']) for r in result['records']}),36)
        for r in result['records']:
            self.assertEqual(r['frames'],21)
            self.assertTrue(r['composite_reproduced_exactly'])
            self.assertEqual(r['final_outside']['mae'],0)
            self.assertEqual(audit.sha(ROOT/r['source_final']),r['final_sha256'])
        for r in result['summaries']:
            self.assertEqual(r['n_outputs'],12)
            self.assertEqual(r['n_independent_inputs'],4)
        for r in result['geometry_checks']:
            self.assertLess(r['backproject_project_max_error_px'],1e-10)
            self.assertEqual(r['planar_relative3d_anchor_max_difference_px'],0)

if __name__=='__main__':unittest.main()
