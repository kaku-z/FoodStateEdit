"""Package actual code dependencies, source inputs and runtime receipts."""
import ast, hashlib, json, platform, subprocess, time, zipfile
from pathlib import Path
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def main():
    names=['run_separate_action_nodes.py','run_separate_action_node_worker.py','audit_separate_action_nodes.py','observe_separate_action_nodes.py','run_real_source_coarse_action.py','audit_real_source_coarse_action.py','observe_real_source_coarse_action.py','project_cut_ray_visibility.py','compose_positive_light_neural_band.py','audit_cavity_material_transport.py','observe_cavity_material_transport.py','compose_screened_surface_transport.py','audit_screened_surface_transport.py','run_real_source_conditioned_worker.py','run_real_source_conditioned_action.py','audit_real_source_conditioned_action.py','observe_surface_conditioning.py','collect_continuation_state.py','aggregate_continuation_observers.py','extract_action_review_boards.py','run_paired_patch_action_refinement.py','audit_paired_patch_action.py','observe_paired_patch_action.py','build_first_bite_gallery.py','build_first_bite_gallery_v2.py','refresh_experiment_reports.py','supported_spoon_source_fit_geometry.py','supported_spoon_source_fit_ellipsoid_geometry.py','prepare_spoon_source_fit_ellipsoid_channels.py','prepare_source_fit_ellipsoid_fields.py','compose_label_consistent_food.py','run_label_consistent_material_refinement.py','run_source_material_full_noise.py','run_bite_material_sdedit_full_noise.py','project_neural_material_photometry.py','refit_source_camera_focal_float64.py','audit_material_refinement.py','audit_label_consistent_food.py','observe_new_appearance_variants.py','audit_source_registered_action_outputs.py','project_full_boundary_material_photometry.py','project_continuous_material_photometry.py','compose_joint_food_cut_light.py','run_joint_action_refinement.py','run_joint_action_full_noise.py','audit_continuous_material.py','audit_joint_action_outputs.py','audit_joint_action_full_noise.py','observe_joint_action_all_noise.py','observe_supported_spoon.py','run_multireference_geometry_qwen21.py','run_qwen_image_edit_direct_baseline.py','audit_full_boundary_material.py']
    queue=[ROOT/n for n in names];seen=set();graph={}
    while queue:
        p=queue.pop()
        if p in seen:continue
        assert p.is_file(),p;seen.add(p);tree=ast.parse(p.read_text());deps=[]
        for node in ast.walk(tree):
            modules=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for module in modules:
                q=ROOT/(module.split('.')[0]+'.py')
                if q.exists():deps.append(q.name);queue.append(q)
        graph[p.name]=sorted(set(deps))
    runtimes={}
    for label,py in [('geometry','/host/space0/guo-z/tf-ufi/food3d_pilot_20260928/venv/bin/python'),('qwen2511','/home/yanai-lab/guo-z/.conda/envs/flux_pure_env/bin/python'),('qwen21','/mnt/tmp/guo-z_first_bite_structure_20260930_runtime/venv/bin/python')]:
        version=subprocess.check_output([py,'-c','import sys;print(sys.version)'],text=True).strip()
        p=subprocess.run([py,'-m','pip','list','--format=json'],capture_output=True,text=True)
        runtimes[label]={'python_path':py,'python_version':version,'pip_list_status':p.returncode,'packages':json.loads(p.stdout) if p.returncode==0 else [],'pip_error':p.stderr[-1000:] if p.returncode else ''}
    protocol={'created_unix':time.time(),'server_root':str(ROOT),'stages':['gate_v36','gate_v37','gate_v38','gate_v39','gate_v40','gate_v41','gate_v42','gate_v43_source_camera_float64','gate_v44','gate_v45','gate_v46','gate_v47','gate_v48','gate_v49','gate_v50','gate_v51','gate_v52','gate_v53','gate_v54','gate_v55','gate_v56'],'all_eight_current_split':'development','original_manifest_split_field_is_historical':True,'paired_physical_ground_truth':False,'blind_human_acceptance':False,'actual_calibrated_3d':False,'geometry_and_garnish_guarantees':'Disclosed image compositor and inferred shared geometry only; approximate source-chroma garnish mask.','model_weight_files_not_in_bundle':True,'model_weight_provenance':'Stage config backend.weight_files and backend21 range manifest','code_dependency_graph':graph,'runtimes':runtimes,'system':{'platform':platform.platform(),'nvidia':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True)}}
    protocol['portable_one_command_reproduction']=False
    protocol['required_external_assets_not_fully_bundled']=['model weights','predecessor geometry and raster fields','predecessor generation and appearance outputs','configured absolute server runtime paths']
    vendor=Path('/mnt/tmp/guo-z_first_bite_structure_20260930_runtime/mitsuba_vendor')
    protocol['mitsuba_vendor_packages']=[{'metadata_path':str(p),'sha256':sha(p)} for p in sorted(vendor.glob('*.dist-info/METADATA'))]
    with zipfile.ZipFile(ROOT/'continuation_reproduction_code_inputs_20261001.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(seen):z.write(p,'code/'+p.name)
        inputs=[]
        for p in (ROOT/'inputs').rglob('*'):
            if p.is_file():z.write(p,p.relative_to(ROOT));inputs.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size})
        for name in ['sam3_observer_provenance.json','backend21/range_manifest_v3.json','original_input_integrity_20261001.json','ALL_CASE_ACCEPTANCE_20261001.json','CONTINUATION_REVIEW_20261001.json']:
            p=ROOT/name
            if p.exists():z.write(p,p.relative_to(ROOT))
        for gate in ['gate_v36','gate_v37','gate_v38','gate_v39','gate_v40','gate_v41','gate_v42','gate_v43_source_camera_float64','gate_v44','gate_v45','gate_v46','gate_v47','gate_v48','gate_v49','gate_v50','gate_v51','gate_v52','gate_v53','gate_v54','gate_v55','gate_v56']:
            for p in (ROOT/gate).glob('*.py'):z.write(p,p.relative_to(ROOT))
            for n in ['config.json','parent_config.json','frozen_plan.json','resource_preflight.json']:
                p=ROOT/gate/n
                if p.exists():z.write(p,p.relative_to(ROOT))
        protocol['code_files']=[{'path':p.name,'sha256':sha(p)} for p in sorted(seen)];protocol['inputs']=inputs
        z.writestr('reproduction_protocol.json',json.dumps(protocol,indent=2))
    (ROOT/'reproduction_protocol_20261001.json').write_text(json.dumps(protocol,indent=2));print('REPRODUCTION_BUNDLE',len(seen),len(inputs),flush=True)
if __name__=='__main__':main()
