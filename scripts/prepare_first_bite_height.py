"""Prepare the predeclared, separate 12-cell height sensitivity diagnostic.

It does not edit the main bundle or any running experiment. All three heights
share the exact prompt and source; only the geometry reference changes.
"""
import argparse,copy,hashlib,json,shutil,sys
from pathlib import Path
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from foodstateedit.first_bite.geometry import render_control
from prepare_first_bite import prompts


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--main-bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    design=json.loads((root/'configs/first_bite_experiment_v1_20260928.json').read_text())
    c=json.loads((a.main_bundle/'config.json').read_text())
    a.output.mkdir(parents=True,exist_ok=False)
    cases=[];checks=[];overview=Image.new('RGB',(1200,320*4),'white');draw=ImageDraw.Draw(overview)
    for row,family in enumerate(design['families']):
        base=next(x for x in c['cases'] if x['case_id']==family+'_01')
        base_audit=json.loads((a.main_bundle/base['files']['geometry.json']['path']).read_text())
        prompt=prompts(base,design['families'][family],base_audit,diagnostic=True)['D_geometry_guide']
        # Remove the fixed projected y positions inherited from the main test.
        lines=prompt.splitlines()
        lines[4]=(f"5. Layout: take the selected portion from around ({base['source_center'][0]*100:.0f}% width, "
                  f"{base['source_center'][1]*100:.0f}% height) in the source. Follow the guide for utensil, "
                  "portion and hand placement, including their vertical positions. Keep the source change visible.")
        prompt='\n'.join(lines).replace('Show the instant before the first mouthful enters the mouth.',
            'Show the selected first mouthful supported by the eating utensil at the position specified by the guide.')
        prompt=prompt.replace('The lifted portion is','The selected portion is').replace('in the lifted portion','in the selected portion')
        prompt=prompt.replace('gold marks the lifted food','gold marks the selected food')
        assert 'clearly raised' not in prompt and 'unmistakable air gap' not in prompt
        assert 'lifted' not in prompt
        group=[]
        for col,height in enumerate(design['height_diagnostic']['normalized_heights']):
            case=copy.deepcopy(base);case['parent_case_id']=base['case_id']
            case['case_id']=base['case_id']+f'_h{round(height*100):02d}'
            case['diagnostic_height']=height;case['prompts']={'D_geometry_guide':prompt}
            d=a.output/'inputs'/case['case_id'];d.mkdir(parents=True)
            shutil.copyfile(a.main_bundle/base['files']['source.png']['path'],d/'source.png')
            W,H=base['output_size'];im,audit=render_control(W,H,base,family,'geometry',height)
            im.save(d/'geometry.png');(d/'geometry.json').write_text(json.dumps(audit,indent=2))
            case['files']={f.name:{'path':f.relative_to(a.output).as_posix(),'sha256':sha(f)} for f in sorted(d.iterdir())}
            cases.append(case)
            group.append({'height':height,'target_uv':audit['target_uv'],'hand_uv':audit['hand_uv'],
                          'guide_sha256':sha(d/'geometry.png'),'source_sha256':sha(d/'source.png'),
                          'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
            im.thumbnail((400,280));overview.paste(im,(400*col,320*row))
            draw.text((400*col+8,320*row+288),f'{base["case_id"]} height={height}',fill='black')
        assert len({g['source_sha256'] for g in group})==1
        assert len({g['prompt_sha256'] for g in group})==1
        assert len({g['guide_sha256'] for g in group})==3
        assert group[0]['target_uv'][1]>group[1]['target_uv'][1]>group[2]['target_uv'][1]
        checks.append({'family':family,'same_source_and_prompt':True,'distinct_monotonic_controls':True,'levels':group})
    diag=copy.deepcopy(c);diag.update(schema='first_bite_height_execution.v1',status='FROZEN_HEIGHT_DIAGNOSTIC',
        cases=cases,methods=['D_geometry_guide'],seeds=[design['height_diagnostic']['seed']],expected_calls=12,
        parent_config_sha256=sha(a.main_bundle/'config.json'),preparer_sha256=sha(Path(__file__)),
        role='Separate height sensitivity diagnostic; excluded from main success denominator.',
        generation_condition='Main D outputs visibly lift, verified by non-blind assistant inspection.',
        height_units='Assumed container diameter, relative to local source support level; not calibrated metric height.')
    (a.output/'config.json').write_text(json.dumps(diag,indent=2))
    (a.output/'invariant_checks.json').write_text(json.dumps(checks,indent=2))
    overview.save(a.output/'input_review.png')
    print(json.dumps({'cases':len(cases),'calls':12,'config_sha256':sha(a.output/'config.json'),'invariants':'passed'}))


if __name__=='__main__':main()
