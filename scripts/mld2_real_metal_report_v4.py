"""Keep every metal trial and its fixed-geometry projection result."""
import argparse
import csv
import json
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw
from mld2_real_report import tile


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--candidate-root',type=Path,required=True);p.add_argument('--first',type=Path,required=True)
    args=p.parse_args();rows=json.loads((args.candidate_root/'v4_projection_manifest.json').read_text())['cases']
    rows=sorted(rows,key=lambda r:r['index'])
    notes=json.loads((args.candidate_root/'visual_review.json').read_text()) if (args.candidate_root/'visual_review.json').exists() else {}
    for row in rows:row['actual_visual_review']=notes.get(row['case_id'],'pending actual review')
    for row in rows:
        source=args.root/row['case_id']/'source.png'
        (args.candidate_root/row['case_id']/'source.png').write_bytes(source.read_bytes())
    report=dict(status='complete',case_count=len(rows),cases=rows,blind_holdout_claim=False,
        protocol='One new seed41/24steps Qwen2.1 metal-only candidate on frozen corrected v3 geometry. SAM3-selected spoon pixels intersect fixed visible spoon at unchanged image coordinates. No shape warp. Uncovered spoon remains opaque PBR.',
        food_state_changed=False,spoon_geometry_changed=False,
        mean_Qwen_metal_coverage_fraction=float(np.mean([r['Qwen_metal_coverage_fraction'] for r in rows])),
        candidate_failure_indices=[r['index'] for r in rows if not r['candidate_found']],
        food_vs_v3_max_mae=max(r['carried_food_vs_core_mae'] for r in rows),
        cavity_vs_v3_max_mae=max(r['source_cavity_vs_core_mae'] for r in rows),
        context_vs_source_max_mae=max(r['outside_context_vs_source_mae'] for r in rows),
        truth='Construction consistency only; no paired edited photos or measured lighting/depth.')
    (args.candidate_root/'metal_v4_results.json').write_text(json.dumps(report,indent=2))
    keys=list(dict.fromkeys(k for r in rows for k,v in r.items() if not isinstance(v,(list,dict))))
    with (args.candidate_root/'metal_v4_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    for start in range(0,len(rows),4):
        group=rows[start:start+4];sheet=Image.new('RGB',(1200,32+252*len(group)),'white');draw=ImageDraw.Draw(sheet)
        draw.text((8,5),'Exploratory metal-only candidate; food, cavity and utensil shape fixed',fill='black')
        for y,row in enumerate(group):
            cid=row['case_id'];i=row['index']
            entries=[(args.root/cid/'source.png','source'),(args.root/cid/'final_v3.png','v3 PBR'),
                (args.candidate_root/cid/'candidate_composite.png','new Qwen metal candidate'),(args.candidate_root/cid/'final_v4.png','v4 strict metal projection')]
            for x,(path,title) in enumerate(entries):sheet.paste(tile(path,f'{i:02d} {title}'),(300*x,32+252*y))
        sheet.save(args.candidate_root/f'metal_comparison_{start:02d}_{start+len(group)-1:02d}.jpg',quality=94)
    final_report=json.loads((args.root/'projected_real_results.json').read_text())
    base_rows={r['case_id']:r for r in final_report['cases']}
    for row in rows:
        for key,value in row.items():base_rows[row['case_id']]['v4_'+key]=value
        base_rows[row['case_id']]['actual_visual_review']=row['actual_visual_review']
        base_rows[row['case_id']]['v3_Qwen_RGB_used']=False
        base_rows[row['case_id']]['final_Qwen_RGB_used']=row['Qwen_metal_used_pixel_count']>0
        base_rows[row['case_id']]['final_analytic_PBR_spoon_fraction']=1-row['Qwen_metal_coverage_fraction']
    final_report['cases']=[base_rows[r['case_id']] for r in rows]
    final_report.update(case_count=len(rows),current_round='Exploratory v4; corrected v3 geometry plus new strictly projected utensil-only Qwen pixels',
        v4_protocol=report['protocol'],v4_metal_result_summary={k:v for k,v in report.items() if k!='cases'})
    final_report['final_provenance'].update(Qwen_RGB_used=any(r['Qwen_metal_used_pixel_count'] for r in rows),
        Qwen_palette_used=False,Qwen_role='New Qwen2.1 seed41/24step metal-only candidates, SAM3 intersection with fixed v3 utensil; no generated food pixels',
        utensil='Same v3 closed shallow spoon; accepted overlapping candidate metal pixels only, original PBR fills uncovered pixels',
        mean_Qwen_metal_pixel_fraction=report['mean_Qwen_metal_coverage_fraction'],
        mean_PBR_metal_pixel_fraction=1-report['mean_Qwen_metal_coverage_fraction'])
    final_report['comparison']='Source/core-v2/final-v4/original Qwen-only source-reference baseline. Whole systems with different geometry inputs and metal-specific final retouch prompt; not an identical-input pure model ablation.'
    final_report['all_case_macro_means']['final_analytic_PBR_spoon_fraction']=1-report['mean_Qwen_metal_coverage_fraction']
    final_report['code_sha256'].update({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('mld2_real_metal_generate_v4.py'),Path(__file__).with_name('mld2_real_metal_project_v4.py')]})
    (args.candidate_root/'projected_real_results.json').write_text(json.dumps(final_report,indent=2))
    keys=list(dict.fromkeys(k for r in final_report['cases'] for k,v in r.items() if not isinstance(v,(list,dict))))
    with (args.candidate_root/'projected_real_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,keys,extrasaction='ignore');writer.writeheader();writer.writerows(final_report['cases'])
    for start in range(0,len(rows),4):
        group=rows[start:start+4];sheet=Image.new('RGB',(1200,32+252*len(group)),'white');draw=ImageDraw.Draw(sheet)
        draw.text((8,5),'All sources retained; exploratory v4; Qwen only supplies accepted metal pixels',fill='black')
        for y,row in enumerate(group):
            cid=row['case_id'];i=row['index']
            entries=[(args.root/cid/'source.png','source'),(args.root/cid/'final_v3.png','v3 PBR (no Qwen)'),
                (args.candidate_root/cid/'final_v4.png','final-v4 (locked food)'),(args.first/cid/'qwen_only/composited.png','Qwen-only baseline')]
            for x,(path,title) in enumerate(entries):sheet.paste(tile(path,f'{i:02d} {title}'),(300*x,32+252*y))
        sheet.save(args.candidate_root/f'final_comparison_{start:02d}_{start+len(group)-1:02d}.jpg',quality=94)
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}))


if __name__=='__main__':main()
