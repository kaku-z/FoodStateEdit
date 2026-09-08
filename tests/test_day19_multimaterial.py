import hashlib
import importlib.util
import json
import unittest
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'artifacts/day19_multimaterial_dataset_v3'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


class MultimaterialContractTests(unittest.TestCase):
    def test_exact_four_cases_two_arms_and_no_lora(self):
        config=read(ROOT/'configs/day19_multimaterial_pilot_v1.json')
        self.assertEqual(config['case_order'],['soup','rice','cake','noodle'])
        self.assertEqual(config['arms'],['planar','relative3d'])
        self.assertEqual(config['inference']['seed'],1)
        self.assertFalse(config['inference']['lora_enabled'])
        self.assertFalse(config['inference']['enable_ttm'])
        self.assertEqual(config['expected_conditions'],8)

    def test_scripts_and_safety_are_bound(self):
        config=read(ROOT/'configs/day19_multimaterial_pilot_v1.json')
        for name,digest in config['implementation_files'].items():
            self.assertEqual(hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest(),digest)
        self.assertEqual(config['resource_gate']['min_free_memory_mib'],48000)
        self.assertTrue(config['resource_gate']['require_zero_compute_process'])
        source=(ROOT/'scripts/run_multimaterial_pilot.py').read_text()
        self.assertIn('foreign_process_check',source)
        self.assertNotIn('kill(',source)
        self.assertEqual(source.count('inference.load_pipeline('),1)

    def test_synthetic_input_is_not_a_method_result(self):
        config=read(ROOT/'configs/day19_multimaterial_pilot_v1.json')
        self.assertTrue(config['claim_policy']['mixed_sources_report_separately'])
        self.assertFalse(config['claim_policy']['learned_efficacy_established'])
        self.assertFalse(config['claim_policy']['novelty_established'])
        self.assertFalse(config['claim_policy']['automatic_heartbeat_resumed'])


@unittest.skipUnless(DATA.exists(),'local-only data not installed')
class MultimaterialLocalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import cv2
        from PIL import Image
        cls.np=np;cls.cv2=cv2;cls.Image=Image
        sys.path.insert(0,str(ROOT/'scripts'))
        try:
            spec=importlib.util.spec_from_file_location('multi_builder',ROOT/'scripts/build_multimaterial_pilot.py')
            cls.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.module)
        finally:sys.path.pop(0)

    def test_all_files_exact_and_videos_complete(self):
        manifest=read(DATA/'dataset_manifest.json')
        self.assertEqual(len(manifest['cases']),4)
        for case in manifest['cases']:
            self.assertEqual(case['uncovered_changed_pixels'],0)
            for rec in case['files'].values():
                path=DATA/rec['path']
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),rec['sha256'])
                self.assertEqual(path.stat().st_size,rec['size_bytes'])
                if path.suffix=='.mp4':
                    cap=self.cv2.VideoCapture(str(path));count=0
                    while True:
                        ok,frame=cap.read()
                        if not ok:break
                        count+=1
                        self.assertEqual(frame.shape[:2],(case['height'],case['width']))
                    cap.release();self.assertEqual(count,21)

    def test_3d_transform_preserves_contact_identity_and_matches_anchor(self):
        m=self.module;np=self.np;camera=m.PinholeCamera.normalized_relative(688,512)
        spec=m.SPECS['rice'];center=np.asarray(spec['contact'])*[688,512]
        target,amount=m.motion(6,spec,camera)
        points=np.stack([center+[-20,15],center+[20,-15]])
        for arm in m.ARMS:
            np.testing.assert_allclose(m.transform(points,[1.07,1.13],target,amount,spec,camera,arm),points,atol=1e-9)
            target,amount=m.motion(20,spec,camera)
            expected=np.asarray(spec['final'])*[688,512]
            np.testing.assert_allclose(m.transform(center[None],1.1,target,amount,spec,camera,arm)[0],expected,atol=1e-9)
            target,amount=m.motion(6,spec,camera)

    def test_source_and_protected_pixels_exact_before_encoding(self):
        m=self.module;np=self.np
        source=np.asarray(self.Image.open(DATA/'cake/reference.png').convert('RGB'))
        alpha=np.asarray(self.Image.open(DATA/'cake/edit_alpha.png').convert('L'))
        for arm in m.ARMS:
            frames,_=m.render_solid_case(source,'cake',arm)
            self.assertTrue(np.array_equal(frames[0],source))
            for frame in frames:
                self.assertTrue(np.array_equal(frame[alpha==0],source[alpha==0]))
                changed=np.any(frame!=source,axis=2)
                self.assertTrue(np.all(alpha[changed]==255))


if __name__=='__main__':unittest.main()
