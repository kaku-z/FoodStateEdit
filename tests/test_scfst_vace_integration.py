import sys
import unittest
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from scripts.scfst_vace_conditions import build_condition_video, build_packed
from scripts.prepare_scfst_vace_runtime import STEP_OLD, REF_OLD, BLOCK_OLD, STEP_NEW, REF_NEW, BLOCK_NEW


class ConditionTests(unittest.TestCase):
    def setUp(self):
        self.mask=np.zeros((5,16,20),np.uint8)
        self.payload=np.zeros_like(self.mask)
        self.source=np.zeros((16,20),np.uint8); self.source[8:12,5:10]=255
        for t in range(5):
            self.payload[t,8-t:12-t,5+t:10+t]=255
            if t>=2: self.mask[t,8:12,5:10]=255

    def test_roles_and_invalid_exposure(self):
        c=build_condition_video(self.mask,self.payload,self.source)
        self.assertEqual(c.shape,(5,16,20,10))
        np.testing.assert_allclose(c[...,:4].sum(-1),1)
        moved=self.payload>0
        self.assertTrue((c[...,1][moved]==1).all())
        exposed=(self.mask>0)&~moved
        self.assertTrue((c[...,3][exposed]==1).all())
        self.assertTrue((c[...,4:6][exposed]==2).all())
        self.assertTrue((c[...,8][exposed]==0).all())

    def test_packed_contract_has_inactive_reference_prefix(self):
        packed,active=build_packed(self.mask,self.payload,self.source,[0,4],(2,3),6)
        self.assertEqual(packed.shape,(1,18,10))
        self.assertEqual(active.shape,(1,18))
        self.assertFalse(active[0,:6].any())
        self.assertTrue(active[0,6:].all())

    def test_bad_masks_fail_closed(self):
        with self.assertRaises(ValueError):
            build_condition_video(self.mask,self.payload[:4],self.source)
        bad=self.mask.copy(); bad[0,0,0]=127
        with self.assertRaises(ValueError):
            build_condition_video(bad,self.payload,self.source)


class PatchContractTests(unittest.TestCase):
    def test_local_pipeline_has_unique_patch_sites_and_result_compiles(self):
        path=ROOT/'vendor_overrides/GeoEdit/diffsynth/pipelines/wan_video.py'
        code=path.read_text(encoding='utf-8')
        self.assertEqual(code.count(STEP_OLD),1)
        self.assertEqual(code.count(REF_OLD),1)
        self.assertEqual(code.count(BLOCK_OLD),1)
        patched=code.replace(STEP_OLD,STEP_NEW).replace(REF_OLD,REF_NEW).replace(BLOCK_OLD,BLOCK_NEW)
        compile(patched,str(path),'exec')
        self.assertIn('SCFST token mismatch',patched)
        self.assertIn('max_abs_delta',patched)


if __name__=='__main__':
    unittest.main()
