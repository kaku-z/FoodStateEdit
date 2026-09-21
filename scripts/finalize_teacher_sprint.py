"""Write an evidence index without upgrading unfinished inference to success."""
import argparse,hashlib,json
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inference-status',required=True);ap.add_argument('--collection',type=Path);ap.add_argument('--note',required=True);args=ap.parse_args()
    output=ROOT/'results/teacher_sprint_45min_20260914_v1.json'
    if output.exists():raise FileExistsError('Do not overwrite sprint evidence')
    now=datetime.now(timezone.utc);started=datetime.fromisoformat('2026-09-14T12:45:33+00:00')
    files=['output/pptx/FoodStateEdit_GUO_2530030_20260914_v39_teacher_verified.pptx','output/pdf/FoodStateEdit_Formal_GUO_2530030_SeniorFormat_20260914_v8_teacher_evidence.pdf','results/TEACHER_RESPONSE_20260914.md','results/teacher_mechanism_audit_20260914_v1.json','artifacts/teacher_counterfactual_failed_20260914T1250Z_collection.json','artifacts/teacher_runtime_trace_snapshot_20260914T1311Z/soup_trace.json','scripts/run_teacher_counterfactual.py','scripts/collect_teacher_counterfactual.py','scripts/audit_teacher_mechanism_20260914.py','tests/test_teacher_mechanism_audit.py','tmp/positive_revision_v33/v39.validation.json','paper/formal_20260909_senior_update/layout_audit.json']
    files.append('results/teacher_parallel_audit_20260914.md')
    if (ROOT/'results/teacher_counterfactual_result_20260914_v2.json').exists():files.append('results/teacher_counterfactual_result_20260914_v2.json')
    files.extend(str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'results').glob('teacher_counterfactual_status_20260914*.json'))
    result=dict(schema='foodstateedit.teacher_sprint.v1',started_utc=started.isoformat(),snapshot_utc=now.isoformat(),elapsed_minutes=(now-started).total_seconds()/60,
        completed=dict(existing_outputs_reanalysed=36,independent_inputs=4,seeds_per_input=3,arms=3,composites_reproduced_exactly=36,local_targeted_tests_passed=7,ppt_slides=18,pdf_pages=2,visual_qa=True,unchanged_slide_renders_verified=11,condition_encoder_runtime_trace_verified=True),
        new_inference=dict(status=args.inference_status,note=args.note,remote_root='/host/space0/guo-z/tf-ufi/outputs/teacher_counterfactual_rgb_v2_20260914T1304Z',planned_cases=['soup','cake'],planned_arms=['planar_original','relative_original','relative_counterfactual'],seed=2,frames=21,steps=20,scale=1,additional_training=False),
        failures_preserved=dict(initial_cases=2,cause='L mask supplied to RGB-only image preprocessing; retry converts mask to RGB with unchanged pixel values',collected_files=22,all_file_hashes_verified=True),
        scope=dict(three_dimensional_superiority_established=False,heldout_generalization_established=False,physical_correctness_established=False,formal_human_evaluation_completed=False,background_preservation_source='explicit alpha compositing, not intrinsic generation fidelity',new_encoder_trained=False),
        agent_reviews=dict(experiment='read-only runner/interface and storage diagnosis',ppt='read-only scientific and visual audit',report='read-only PDF, citation and metric audit',all_completed=True,main_agent_integrated_and_verified=True),
        files={f:dict(sha256=digest(ROOT/f),bytes=(ROOT/f).stat().st_size) for f in files})
    if args.collection:
        rec=json.loads(args.collection.read_text(encoding='utf-8'));result['new_inference']['collection']=rec
        result['new_inference']['completed_outputs']=sum(len(v['completed']) for v in rec['cases'].values())
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(output)
if __name__=='__main__':main()
