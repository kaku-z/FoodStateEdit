import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('caption_compiler',ROOT/'scripts/compile_mld4_captions.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def write(p,value):p.write_text(json.dumps(value),encoding='utf-8')


class CaptionCompilerTest(unittest.TestCase):
    def test_scalar_list_null_normalization(self):
        self.assertEqual(c.strings(None),[])
        self.assertEqual(c.strings(' white '),['white'])
        self.assertEqual(c.strings([None,'white',['brown'],4,{'text':'yellow'}]),['white','brown'])
        self.assertEqual(c.agree(None,[])[0],'food')
        self.assertEqual(c.agree(['egg white'],'fried egg white')[0],'egg white')

    def test_family_agreement_and_specific_parts(self):
        self.assertEqual(c.agree('chicken','meat slice')[0],'cooked meat')
        self.assertEqual(c.agree('crispy chicken skin','chicken skin')[0],'chicken skin')
        self.assertEqual(c.agree('udon','noodle')[0],'noodle')
        for a,b in [('noodle','cabbage'),('egg white','egg yolk'),('rice','liquorice'),('egg white','egg'),('fish and herbs','fish')]:
            self.assertEqual(c.agree(a,b)[0],'food')

    def test_attribute_text_cannot_inject_annotations_or_adjacent_food(self):
        a={'visible_textures':['smooth','glossy egg yolk','magenta outline','smooth cauliflower'],
           'visible_colors':['white','yellow yolk','red meat and chicken']}
        b={'visible_textures':['smooth','glossy','smooth cauliflower'], 'visible_colors':['white','yellow yolk']}
        self.assertEqual(c.attributes(a,b,'visible_textures'),'smooth')
        self.assertEqual(c.attributes(a,b,'visible_colors'),'white')

    def fixture(self,root,labels=('egg white','Egg white')):
        case='real_test';source=root/'source';source.mkdir();probe=root/'repaired';probe.mkdir()
        names=['source.png','source_reference.png','source_reference_mask.png','guide_manifest.json']
        for i,name in enumerate(names):(source/name).write_bytes(bytes([i,17,29]))
        hashes={name:c.sha(source/name) for name in names}
        provenance=dict(case_id=case,selection_pixels_exact=True,no_generated_image_input=True,
            source_folder=str(source),source_input_sha256=hashes,source_reference_bbox=[1,2,3,4],owned_pixel_count=9)
        write(probe/'input_provenance.json',provenance)
        measurement=dict(case=case,source_only=True,vlm_used=False,target_image_used=False,
            source_sha256=hashes['source_reference.png'],mask_sha256=hashes['source_reference_mask.png'],
            owned_pixels=dict(pixel_count=9,lab_mean=[70,-2,22],rgb_mean=[183,171,131],lab_chroma_percentiles=[5,10,15,22,26,33,41]))
        measured=root/'measured.json';write(measured,measurement)
        paths=[]
        for template,label in zip(['A','B'],labels):
            path=probe/('reply_'+template+'.json');paths.append(path)
            write(path,dict(case_id=case,source_only=True,template_id=template,parsed=dict(component_label=label,
                visible_colors=['yellow yolk'],visible_textures=['smooth','glossy egg yolk'],confidence=.9,
                visible_surroundings=['black pan beside a third yellow egg yolk'],
                source_context_caption='An extra third yellow yolk inside a magenta outline.')))
        return case,probe,paths,measured

    def test_owned_measured_fallback_and_no_interior_or_context_injection(self):
        with tempfile.TemporaryDirectory() as tmp:
            case,probe,paths,measured=self.fixture(Path(tmp),('noodle','cabbage'))
            result=c.compile_case(case_id=case,paths=paths,provenance_path=probe/'input_provenance.json',measurement_path=measured,strict=True)
            self.assertFalse(result['component_consistency']);self.assertTrue(result['ownership_provenance_verified'])
            self.assertIn('muted beige',result['head'])
            for word in ['cabbage','noodle','yolk','magenta','glossy','interior','opening','thickness']:
                self.assertNotIn(word,result['head']+' '+result['source'])
            self.assertIn('existing pan',result['source'])
            self.assertIn('original pan surface',result['source_surface'])
            for word in ['food','egg','yolk','white','beige','opening','hole','cup','thickness','smooth']:
                self.assertNotIn(word,result['source_surface'])
            bad=c.read(measured);bad['mask_sha256']='wrong';write(measured,bad)
            with self.assertRaisesRegex(ValueError,'different source ownership'):
                c.compile_case(case_id=case,paths=paths,provenance_path=probe/'input_provenance.json',measurement_path=measured,strict=True)

    def test_manifest_uses_explicit_repaired_paths_and_retains_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,probe,paths,measured=self.fixture(root)
            manifest=root/'manifest.json';output=root/'captions.json'
            write(manifest,dict(expected_cases=1,cases=[dict(case_id=case,reply_paths=[str(p) for p in paths],
                source_input_provenance=str(probe/'input_provenance.json'),measured_material_path=str(measured))]))
            command=[sys.executable,str(ROOT/'scripts/compile_mld4_captions.py'),'--manifest',str(manifest),'--output',str(output)]
            run=subprocess.run(command,capture_output=True,text=True);self.assertEqual(run.returncode,0,run.stderr)
            result=c.read(output)[case];self.assertEqual(result['component_label'],'egg white')
            self.assertTrue(all('/repaired/' in p.replace('\\','/') for p in result['inputs']))
            first=output.read_bytes();repeat=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(repeat.returncode,0);self.assertEqual(output.read_bytes(),first)

    def test_surface_prompt_unknown_and_broth_are_separate_from_food(self):
        self.assertIn('original underlying surface',c.source_surface({'visible_surroundings':None},{'visible_surroundings':[]}))
        self.assertEqual(c.source_surface({'visible_surroundings':['noodles in dark broth']},{'visible_surroundings':'bowl with broth'}),
            'A close-up photograph of the continuous original broth surface with natural texture and lighting.')


if __name__=='__main__':unittest.main()
