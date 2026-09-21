"""Index a partial batch without counting its incomplete arms as results."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'artifacts/teacher_counterfactual_partial_20260914T1304Z'
collection=A.parent/(A.name+'_collection.json')
resultpath=ROOT/'results/teacher_counterfactual_result_20260914_v2.json'
if resultpath.exists():raise FileExistsError('Do not overwrite')
proof=json.loads(collection.read_text(encoding='utf-8'))
assert proof['all_file_hashes_verified']
cases={}
for case in ['soup','cake']:
    s=json.loads((A/case/'run_manifest.json').read_text(encoding='utf-8'))
    assert s['pipeline_load_count']==1
    names=[x['condition'] for x in proof['cases'][case]['completed']]
    assert names==['planar_original']
    assert s['status']=='technical_failure_preserved'
    assert s['wall_seconds']>=1490
    assert not (A/case/'resource_stop.txt').exists()
    cases[case]=dict(batch_status=s['status'],completed_conditions=names,incomplete_conditions=['relative_original','relative_counterfactual'],pipeline_load_count=1,wall_seconds=s['wall_seconds'],error=s['error'],timeout_evidence='Approximately1500 seconds; own alarm raised; no resource_stop file',completed_control_endpoints=s['control_endpoints'],successful_file_generation_not_semantic_success=True)
result=dict(schema='foodstateedit.teacher_counterfactual_result.v2',created_utc=datetime.now(timezone.utc).isoformat(),status='partial_batch_timeout_preserved',completed_generations=2,planned_generations=6,all_six_outputs_verified=False,manual_visual_review_completed=False,additional_training=False,cases=cases,
    all_collected_files_sha256_verified=True,collection_manifest=str(collection.relative_to(ROOT)),collection_sha256=hashlib.sha256(collection.read_bytes()).hexdigest(),
    conditions='Same seed2,21frames,20steps,VACEscale1,TTMoff,LoRAoff;three-armsharedbinarymask;freshuncompressedRGBcontrols',
    claim_limit='Only two Planar controls finished. No paired relative3D or counterfactual result; cannot establish control response, 3D advantage, photorealism, physical validity or generalization.',
    failure_history='Initial two cases failed at L-mask/RGB interface, preserved separately. Retry fixed format and reached generation; both batches stopped at1500-second own timeout after Planar completion.',
    next_step='Resume remaining two arms per case only on safe resources, new output roots and explicit accounting for cold high/low-noise weight loading. Never overwrite this partial run.')
resultpath.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(resultpath)
