"""Summarize every frozen real case; no image selection or metric-driven retries."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion


DEV = [2, 4, 7, 9]
SOURCE_MANIFEST = Path('/host/space0/guo-z/tf-ufi/material_lineage_training_20261004/real_probes/manifest.json')


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def masked_mae(a, b, mask):
    return float(np.abs(a.astype(float)-b.astype(float))[mask].mean()) if mask.any() else 0.


def image_metrics(folder, image_path, mask_path):
    source = np.asarray(Image.open(folder/'source.png').convert('RGB'))
    result = np.asarray(Image.open(image_path).convert('RGB').resize((source.shape[1], source.shape[0])))
    edit = np.asarray(Image.open(mask_path).convert('L')) > 0
    frozen = np.asarray(Image.open(folder/'frozen_edit_mask.png').convert('L')) > 0
    top = np.asarray(Image.open(folder/'observed_food_mask.png').convert('L')) > 0
    anchor = np.asarray(Image.open(folder/'raw.png').convert('RGB'))
    inputs = anchor if image_path.parent.name=='refined' else source
    return dict(outside_frozen_context_source_mae=masked_mae(source, result, ~frozen),
                outside_appearance_edit_input_mae=masked_mae(inputs, result, ~edit),
                observed_top_core_mae=masked_mae(anchor, result, top),
                observed_protected_top_core_mae=masked_mae(anchor, result, binary_erosion(top,iterations=3)),
                actual_changed_pixels=int(np.any(source != result, axis=-1).sum()))


def tile(path, title, width=300, height=252):
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(25, 25, 25))
    if path.exists():
        im = Image.open(path).convert('RGB')
        im.thumbnail((width-12, height-30), Image.Resampling.LANCZOS)
        canvas.paste(im, ((width-im.width)//2, 26+(height-30-im.height)//2))
    else:
        draw.text((12, 70), 'MISSING / FAILED; retained', fill=(180, 20, 20))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    cases = load(SOURCE_MANIFEST)['cases']
    rows = []
    for index, case in enumerate(cases):
        folder = root/case['case_id']
        core = load(folder/'geometry_raw.json')
        base = load(folder/'geometry_baseline.json')
        if not core:
            rows.append(dict(case_id=case['case_id'], index=index, split='development' if index in DEV else 'held_out', status='failed'))
            continue
        diagnostics = core['hidden_prior_diagnostics']
        row = dict(case_id=case['case_id'], index=index, split='development' if index in DEV else 'held_out',
                   status='complete', selected_food=load(folder/'segmentation.json').get('selected_prompt'),
                   segmentation_confidence=load(folder/'segmentation.json').get('confidence'),
                   normalized_lift=core['normalized_lift'], projected_bite_major_pixels=core['moved_projected_major_pixels'],
                   moved_visible_pixels=core['moved_visible_pixels'], spoon_visible_pixels=core['spoon_visible_pixels'],
                   source_target_pixel_overlap=core['source_target_mask_overlap'],
                   inverse_transport_error=core['exact_inverse_transport_max_error'],
                   relative_volume_transport_error=core['relative_volume_transport_error'],
                   core_volume=core['source_closed_volume'], uniform_volume=base.get('source_closed_volume'),
                   core_contact_gap=core['spoon']['inferred_contact_min_gap'],
                   core_penetration=core['spoon']['inferred_penetration_depth'],
                   uniform_contact_gap=base.get('spoon',{}).get('inferred_contact_min_gap'),
                   uniform_penetration=base.get('spoon',{}).get('inferred_penetration_depth'),
                   core_changes_outside_frozen_mask=core['core_change_pixels_outside_frozen_mask'],
                   pre_projection_changes_outside_frozen_mask=core.get('pre_observation_projection_changes_outside_frozen_mask'),
                   geometry_pixels_outside_frozen_mask=core.get('geometry_pixels_outside_frozen_mask'),
                   source_cut_pixels=core['source_pixels'],
                   frozen_edit_mask_used=core['frozen_edit_mask_used'])
        row.update(diagnostics)
        if base:
            extent=float(np.load(folder/'closure_raw.npz')['extent'])
            row['core_contact_gap_over_extent']=row['core_contact_gap']/extent
            row['uniform_contact_gap_over_extent']=row['uniform_contact_gap']/extent
            row['same_source_cut'] = core['source_bite_center']==base['source_bite_center'] and core['source_bite_radius']==base['source_bite_radius'] and core['source_pixels']==base['source_pixels']
            row['same_translation'] = bool(np.allclose(core['target_translation'],base['target_translation'],atol=0,rtol=0))
            row['same_spoon_mesh'] = (folder/'spoon_raw.ply').read_bytes()==(folder/'spoon_baseline.ply').read_bytes()
            row['learned_vs_uniform_volume_relative_delta'] = (core['source_closed_volume']-base['source_closed_volume'])/base['source_closed_volume']
        for method, output, mask in [('core', folder/'raw.png',folder/'frozen_edit_mask.png'),
                                     ('uniform',folder/'baseline.png',folder/'frozen_edit_mask.png'),
                                     ('refined',folder/'refined/composited.png',folder/'refined/actual_edit_mask.png'),
                                     ('qwen_only',folder/'qwen_only/composited.png',folder/'qwen_only/actual_edit_mask.png')]:
            if output.exists():
                row.update({method+'_'+k:v for k,v in image_metrics(folder,output,mask).items()})
        rows.append(row)
    complete = [r for r in rows if r['status']=='complete']
    scalar_names = [k for k,v in complete[0].items() if isinstance(v,(float,int)) and not isinstance(v,bool)] if complete else []
    means = {k:float(np.mean([r[k] for r in complete if r.get(k) is not None])) for k in scalar_names}
    means['learned_vs_uniform_volume_absolute_relative_delta']=float(np.mean([abs(r['learned_vs_uniform_volume_relative_delta']) for r in complete]))
    contribution_cases=sum(r['learned_vs_uniform_floor_mean_absolute_delta_over_extent']>1e-6 for r in complete)
    report = dict(case_count=16, development_indices=DEV, held_out_indices=[i for i in range(16) if i not in DEV],
                  model_receipt=load(root/'render_manifest.json').get('evidence',{}),
                  all_case_macro_means=means, cases=rows,
                  cases_with_learned_geometry_difference=contribution_cases,
                  cases_with_closure_identical_to_uniform=len(complete)-contribution_cases,
                  measured_edited_ground_truth=False, measured_depth_or_3d_truth=False,
                  interpretation='Internal rigid mesh and texture transport checks; learned bounded hidden closure and inferred tangent plane, not physical accuracy. Qwen output appearance has no conservation guarantee.',
                  fair_comparison='Same source cut, action, spoon, editable layout, prompt, seed41,24steps. Refined receives geometry control and observed top anchoring; Qwen-only receives source reference without geometry control. Complete-system comparison, not identical-input pure model ablation.')
    (root/'real_results.json').write_text(json.dumps(report,indent=2))
    columns = list(dict.fromkeys(k for r in rows for k,v in r.items() if not isinstance(v,list)))
    with (root/'real_results.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    methods=[('source','source.png'),('core (no Qwen)','raw.png'),('uniform (no MLD)','baseline.png'),('core + Qwen','refined/composited.png'),('Qwen-only','qwen_only/composited.png')]
    for start in range(0,16,4):
        sheet=Image.new('RGB',(1500,1048),'white');draw=ImageDraw.Draw(sheet)
        draw.text((8,5),f'Frozen real cases {start:02d}-{start+3:02d}; all outcomes retained; 4 development / 12 held-out',fill='black')
        for y,index in enumerate(range(start,start+4)):
            case=cases[index];folder=root/case['case_id']
            for x,(label,filename) in enumerate(methods):
                title=f'{index:02d} '+('DEV ' if index in DEV else 'TEST ')+label
                sheet.paste(tile(folder/filename,title),(300*x,32+252*y))
        sheet.save(root/f'formal_comparison_{start:02d}_{start+3:02d}.jpg',quality=94)
    (root/'REPORT_NOTES.txt').write_text('The 16 source photos were frozen before this experiment. Development cases 2/4/7/9 cover cohesive foods; the other 12 are held out. No paired real after-images, true depth, material, camera or force labels are available. Closed-volume, inverse transport and UV checks test construction consistency. Lift/spoon contact use inferred MoGe scale and plane. Noodles and loose granular food do not receive deformation or strand constraints. NN-only color diagnostics exclude source-color mean smoothing. All final images and failure cases are retained. Review each photographic outcome separately from internal geometric checks.\n')
    print(json.dumps(dict(status='complete',cases=len(rows),macro_means=means)))


if __name__ == '__main__':
    main()
