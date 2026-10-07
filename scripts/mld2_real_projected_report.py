"""All-case exploratory report; final utensil is independent of Qwen pixels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mld2_real_report import tile


DEV = [2, 4, 7, 9]


def load(path):
    return json.loads(path.read_text())


def pixels(path):
    return np.asarray(Image.open(path).convert('RGB'))


def mask(path):
    return np.asarray(Image.open(path).convert('L')) > 0


def mae(a, b, region):
    return float(np.abs(a.astype(float)-b.astype(float))[region].mean()) if region.any() else 0.


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--core', type=Path, required=True)
    p.add_argument('--final', type=Path, required=True)
    p.add_argument('--first', type=Path, required=True)
    args=p.parse_args()
    rows=[]
    notes=load(args.final/'visual_review.json') if (args.final/'visual_review.json').exists() else {}
    for index, folder in enumerate(sorted(f for f in args.core.glob('real_*') if f.is_dir())):
        final_folder=args.final/folder.name
        geo=load(folder/'geometry_raw.json')
        uniform=load(folder/'geometry_baseline.json')
        metal=load(folder/'metal_projected.json')
        final=load(final_folder/'final_v3.json')
        source=pixels(folder/'source.png'); core=pixels(folder/'raw.png')
        neural_metal=pixels(folder/'metal_projected.png'); final_rgb=pixels(final_folder/'final_v3.png')
        spoon=mask(folder/'spoon_mask.png'); food=mask(folder/'bite_mask.png')
        source_cut=mask(folder/'source_bite_mask.png')
        cavity=source_cut & ~spoon & ~food
        final_spoon=mask(final_folder/'spoon_mask.png'); final_food=mask(final_folder/'food_mask.png')
        mutual_food=final_food & food
        cavity_common=source_cut & ~spoon & ~food & ~final_spoon & ~final_food
        edit=mask(final_folder/'frozen_edit_mask.png')
        extent=float(np.load(folder/'closure_raw.npz')['extent'])
        row=dict(case_id=folder.name,index=index,original_split='development' if index in DEV else 'held_out_first_round',
            current_round='exploratory_after_first_round_inspection',status='complete',
            selected_food=load(folder/'segmentation.json').get('selected_prompt'),
            normalized_lift=geo['normalized_lift'],source_target_pixel_overlap=geo['source_target_mask_overlap'],
            projected_bite_major_pixels=geo['moved_projected_major_pixels'],
            inverse_transport_error=geo['exact_inverse_transport_max_error'],
            volume_transport_relative_error=geo['relative_volume_transport_error'],
            source_uv_projection_max_change=geo['local_scale_diagnostics']['source_uv_projection_max_change'],
            source_pose_full_patch_texture_render_mae=geo['source_pixel_anchor_mae'],
            source_pose_eroded_patch_texture_render_mae=geo['visible_correspondence_source_mae'],
            learned_vs_uniform_volume_relative_delta=(geo['source_closed_volume']-uniform['source_closed_volume'])/uniform['source_closed_volume'],
            same_cut_as_uniform=geo['source_bite_center']==uniform['source_bite_center'] and geo['source_bite_radius']==uniform['source_bite_radius'],
            same_action_as_uniform=bool(np.allclose(geo['target_translation'],uniform['target_translation'],rtol=0,atol=0)),
            same_v2_spoon_as_uniform=(folder/'spoon_raw.ply').read_bytes()==(folder/'spoon_baseline.ply').read_bytes(),
            v2_candidate_metal_found=metal['candidate_found'],v2_candidate_metal_score=metal['candidate_score'],
            v2_neural_metal_food_vs_core_mae=mae(neural_metal,core,food),
            v2_neural_metal_visible_cavity_vs_core_mae=mae(neural_metal,core,cavity),
            v2_neural_metal_source_cut_roi_including_utensil_mae=mae(neural_metal,core,source_cut),
            final_mutual_visible_food_vs_core_mae=mae(final_rgb,core,mutual_food),
            final_mutual_visible_cavity_vs_core_mae=mae(final_rgb,core,cavity_common),
            final_outside_context_source_mae=mae(final_rgb,source,~edit),
            final_food_state_geometry_max_change=final['food_geometry_max_change'],
            final_food_material_render_max_change=final['food_material_render_max_change'],
            final_spoon_contact_gap_over_extent=final['spoon']['inferred_contact_min_gap']/extent,
            final_spoon_penetration_over_extent=final['spoon']['inferred_penetration']/extent,
            final_spoon_watertight=final['spoon']['watertight'],
            final_spoon_connected_components=final['spoon']['connected_components'],
            final_spoon_visible_pixels=int(final_spoon.sum()),
            final_food_visibility_changed_pixels=final['food_visibility_changed_pixels'],
            final_Qwen_RGB_used=final['Qwen_RGB_used'],
            final_analytic_PBR_spoon_fraction=final['analytic_PBR_spoon_fraction'],
            final_generated_food_pixels_used=final['generated_food_pixels_used'],
            actual_visual_review=notes.get(folder.name, 'pending actual image review'))
        row.update(geo['hidden_prior_diagnostics'])
        row.update(geo['local_scale_diagnostics'])
        rows.append(row)
    numeric=[k for k,v in rows[0].items() if isinstance(v,(int,float)) and not isinstance(v,bool)]
    macro={k:float(np.mean([r[k] for r in rows if r.get(k) is not None])) for k in numeric}
    report=dict(status='complete',case_count=len(rows),current_round='Exploratory rounds 2 and 3; all original 16 cases retained',
        original_development_indices=DEV,original_heldout_indices=[i for i in range(16) if i not in DEV],
        blind_holdout_claim=False,selected_checkpoint=str(args.first.parent/'runs_continuous/shared_s41/final_checkpoint.pt'),
        core_model_receipt=load(args.core/'render_manifest.json')['evidence'],
        final_provenance=dict(food='Frozen v2 observed-source texture and bounded base65k MLD2 hidden material/closure',
            utensil='Unified closed shallow elliptical spoon, smooth PBR metallic1 roughness.2 and analytic strip lighting',
            source_photograph_palette=True,Qwen_RGB_used=False,Qwen_palette_used=False,
            Qwen_role='Separate source-only baseline and retained v2 metal-projection control',
            SAM3_role='Observed food segmentation; also candidate metal segmentation for v2 control only',
            MoGe2_role='Inferred source rays and local tangent plane; no measured real depth',
            generated_food_pixels_used=0),
        v2_metal_candidate_failure_count=sum(not r['v2_candidate_metal_found'] for r in rows),
        v2_metal_candidate_failure_indices=[r['index'] for r in rows if not r['v2_candidate_metal_found']],
        all_case_macro_means=macro,cases=rows,
        true_edited_ground_truth=False,true_camera_depth_or_material_ground_truth=False,
        interpretation='Transport, UV, volume, source consistency and contact are construction checks. Local source-ray relief and hidden closure were bounded after inspecting first-round failures; they do not establish accurate scanning or food deformation. Inferred unseen cut faces and analytic lighting remain approximations.',
        comparison='Same source/cut/action. Core-v2 and uniform use the same spoon. Final-v3 changes only the utensil; food state is frozen. Qwen-only is the original seed41/24-step source-reference system without 3D controls; a whole-system comparison.',
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('mld2_real_spoon_v3.py'),Path(__file__).with_name('mld2_real_geometry_v2.py')]})
    (args.final/'projected_real_results.json').write_text(json.dumps(report,indent=2))
    columns=list(dict.fromkeys(k for r in rows for k,v in r.items() if not isinstance(v,(list,dict))))
    with (args.final/'projected_real_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    for start in range(0,len(rows),4):
        sheet=Image.new('RGB',(1200,1048),'white');draw=ImageDraw.Draw(sheet)
        draw.text((8,5),f'Cases {start:02d}-{start+3:02d}; exploratory correction; every frozen source retained',fill='black')
        for y,row in enumerate(rows[start:start+4]):
            cid=row['case_id'];idx=row['index']
            paths=[(args.core/cid/'source.png','source'),(args.core/cid/'raw.png','core v2 (no Qwen)'),
                (args.final/cid/'final_v3.png','final v3 (PBR, no Qwen)'),(args.first/cid/'qwen_only/composited.png','Qwen-only baseline')]
            for x,(path,title) in enumerate(paths):sheet.paste(tile(path,f'{idx:02d} {title}'),(x*300,32+252*y))
        sheet.save(args.final/f'final_comparison_{start:02d}_{start+3:02d}.jpg',quality=94)
    print(json.dumps(dict(status='complete',case_count=len(rows),macro_means=macro)))


if __name__=='__main__':main()
