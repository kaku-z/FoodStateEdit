"""Prepare a bounded post-hoc annotation repair and the unrun height diagnostic.

Original main inputs and outputs remain immutable. Repair results never replace
original failures. Inference settings and seeds are unchanged.
"""
import copy,datetime,hashlib,json,shutil,subprocess,sys
from pathlib import Path
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from foodstateedit.first_bite.geometry import render_control
from prepare_first_bite import prompts


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parents[1];original=root/'outputs/first_bite_20260928_bundle_v1'
    corrected=root/'outputs/first_bite_20260928_roi_bundle_v2'
    height=root/'outputs/first_bite_20260928_height_bundle_v2'
    out=root/'outputs/first_bite_20260928_supplement_bundle_v2'
    design=json.loads((root/'configs/first_bite_experiment_v1_20260928.json').read_text())
    fixes={'strand_01':{'source_center':[.43,.79],'source_box':[.35,.70,.54,.90],
                       'reason':'Original ROI mixed scallions with exposed udon; move to visible front udon.'},
           'strand_02':{'source_center':[.80,.48],'source_box':[.75,.455,.85,.51],
                       'reason':'Original ROI covered nori and the bowl rim, not exposed udon; choose visible right-side strands.'},
           'liquid_01':{'source_center':[.50,.54],'source_box':[.45,.50,.55,.58],
                       'reason':'Original ROI center fell on the cup rim; move to an interior liquid patch.'}}
    shutil.copytree(original,corrected);c=json.loads((corrected/'config.json').read_text());audit=[]
    for case in c['cases']:
        if case['case_id'] not in fixes:continue
        cid=case['case_id'];fix=fixes[cid];before={k:case[k] for k in ['source_center','source_box','bite_description']}
        case['source_center']=fix['source_center'];case['source_box']=fix['source_box']
        if cid=='strand_02':case['bite_description']=case['bite_description'].replace('near the front','near the right')
        d=corrected/'inputs'/cid;W,H=case['output_size']
        for mode in ['planar','geometry']:
            image,geom=render_control(W,H,case,case['family'],mode,.16)
            image.save(d/(mode+'.png'));(d/(mode+'.json')).write_text(json.dumps(geom,indent=2))
        case['prompts']=prompts(case,design['families'][case['family']],geom)
        mask=Image.open(d/'diagnostic_edit_region.png');md=ImageDraw.Draw(mask)
        x0,y0,x1,y1=case['source_box'];md.rectangle([x0*W,y0*H,x1*W,y1*H],fill=255);mask.save(d/'diagnostic_edit_region.png')
        case['files']={f.name:{'path':f.relative_to(corrected).as_posix(),'sha256':sha(f)} for f in d.iterdir()}
        assert sha(d/'source.png')==sha(original/'inputs'/cid/'source.png')
        oldgeom=json.loads((original/'inputs'/cid/'geometry.json').read_text())
        assert geom['target_uv']==oldgeom['target_uv'] and geom['hand_uv']==oldgeom['hand_uv']
        audit.append({'case_id':cid,'before':before,'after':{k:case[k] for k in before},'reason':fix['reason'],
                      'same_source_bytes':True,'same_target_and_hand_coordinates':True})
        im=Image.open(d/'source.png').convert('RGB');draw=ImageDraw.Draw(im)
        draw.rectangle([x0*W,y0*H,x1*W,y1*H],outline='red',width=4)
        x,y=case['source_center'];draw.ellipse([x*W-6,y*H-6,x*W+6,y*H+6],fill='red')
        im.save(root/'outputs/first_bite_20260928_annotation_audit'/(cid+'_accepted_roi_v2.png'))
    overview=Image.new('RGB',(960,280*8),'white');od=ImageDraw.Draw(overview)
    for i,case in enumerate(c['cases']):
        for j,name in enumerate(['source.png','planar.png','geometry.png']):
            im=Image.open(corrected/case['files'][name]['path']).convert('RGB');im.thumbnail((320,250))
            overview.paste(im,(320*j,280*i));od.text((320*j+4,280*i+254),case['case_id']+' '+name,fill='black')
    overview.save(corrected/'input_review.png')
    c['parent_annotations_sha256']=c.pop('annotations_sha256')
    c.update(schema='first_bite_corrected_annotations.v2',status='POSTHOC_ROI_REPAIR_FROZEN_BEFORE_NEW_OUTPUTS',
             preparer_sha256=sha(Path(__file__)),
             parent_config_sha256=sha(original/'config.json'),annotation_changes=audit,
             correction_scope='Source ROI/center and matching strand_02 front-to-right wording only; model, seeds, target, hand and inference unchanged.')
    (corrected/'config.json').write_text(json.dumps(c,indent=2))
    subprocess.run([sys.executable,str(root/'scripts/prepare_first_bite_height.py'),'--main-bundle',str(corrected),'--output',str(height)],check=True)
    hc=json.loads((height/'config.json').read_text());out.mkdir();combined_cases=[];jobs=[]
    for base,group,selected in [(corrected,'roi_repair',[x for x in c['cases'] if x['case_id'] in fixes]),(height,'height',hc['cases'])]:
        for case in selected:
            case=copy.deepcopy(case);case['experiment_role']=group
            for f in case['files'].values():
                dest=out/group/f['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(base/f['path'],dest)
                f['path']=dest.relative_to(out).as_posix();assert sha(dest)==f['sha256']
            combined_cases.append(case)
            methods=c['methods'] if group=='roi_repair' else ['D_geometry_guide']
            seeds=[281,913] if group=='roi_repair' else [2027]
            for method in methods:
                for seed in seeds:jobs.append({'case_id':case['case_id'],'method':method,'seed':seed,'experiment_role':group})
    assert len(jobs)==36 and sum(j['experiment_role']=='roi_repair' for j in jobs)==24
    combined=copy.deepcopy(c);combined.update(schema='first_bite_supplement.v2',cases=combined_cases,jobs=jobs,expected_calls=36,
        seeds=[281,913,2027],matrix_policy='Only explicit jobs are executed, not the cartesian product.',
        denominators={'posthoc_roi_repair':24,'height_diagnostic':12},excluded_from_original_64_denominator=True,
        height_config_sha256=sha(height/'config.json'),old_height_bundle_v1_status='SUPERSEDED_WITH_ZERO_MODEL_CALLS',
        decision_rule='Run all 36 once; retain every output, no replacement seed, no further repair batch in this experiment.')
    (out/'config.json').write_text(json.dumps(combined,indent=2))
    record={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'POSTHOC_INPUT_DEFECT_CONFIRMED',
        'main_calls_preserved':64,'repair_calls':24,'height_calls':12,'original_height_calls':0,
        'changes':audit,'claim_limit':'This repair is chosen after main outputs. It is diagnostic, not preregistered independent validation.',
        'confounding_note':'Wrong source annotations limit interpreting the original noodle/soup failures as model-only failures.',
        'config_sha256':sha(out/'config.json')}
    (root/'results/FIRST_BITE_ANNOTATION_AUDIT_20260928.json').write_text(json.dumps(record,indent=2))
    print(json.dumps({'supplement_calls':36,'config_sha256':sha(out/'config.json'),'original_bundle_unchanged':True}))


if __name__=='__main__':main()
