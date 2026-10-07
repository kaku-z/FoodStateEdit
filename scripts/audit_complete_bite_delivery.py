"""Verify the finished artifact bundle and summarize actual compute and observer diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    import numpy as np
    from PIL import Image
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    b = a.bundle
    def read(rel):
        return json.loads((b/rel).read_text(encoding='utf-8'))
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    def local(remote):
        return b/remote.split('/first_bite_complete_20260929/', 1)[1]
    f = read('formal/FROZEN.json')
    checked = []
    def check(rel, expected):
        assert sha(b/rel) == expected, rel
        checked.append(str(rel))
    for backend, h in f['configuration_sha256'].items():
        check('formal/'+backend+'_frozen.json', h)
    for rel, h in f['code_sha256'].items():
        check(rel, h)
    check('inputs/manifest.json', f['source_selection_manifest_sha256'])
    check('geometry/manifest.json', f['geometry_manifest_sha256'])
    check('pilot/development_decision.json', f['development_decision_sha256'])
    check('model_cache_receipt.json', f['local_model_cache_receipt_sha256'])
    input_cases = {r['case_id']: r for r in read('inputs/manifest.json')['cases']}
    source_pixel_checks = []
    for source in read('formal/source_audit.json'):
        assert sha(local(source['source_file']['path'])) == source['source_file']['sha256']
        original = input_cases[source['case_id']]
        for name, h in original['files'].items():
            assert sha(b/'inputs'/source['case_id']/name) == h
        # The geometry builder re-encodes PNG bytes. Verify each file against
        # its own frozen digest and compare decoded pixels across both copies.
        geometry_source = np.asarray(Image.open(local(source['source_file']['path'])).convert('RGB'))
        normalized_source = np.asarray(Image.open(b/'inputs'/source['case_id']/'source.png').convert('RGB'))
        assert np.array_equal(geometry_source, normalized_source)
        source_pixel_checks.append({'case_id': source['case_id'], 'exact_pixel_match': True,
            'geometry_source_sha256': source['source_file']['sha256'],
            'input_source_sha256': original['files']['source.png']})
    inv = read('formal/inventory.json')
    assert inv['status'] == 'complete' and len(inv['cells']) == 192
    assert inv['counts'] == {'generated': 174, 'preprocessing_failed': 18}, inv['counts']
    reviewed = json.loads(a.review.read_text(encoding='utf-8'))
    notes = {(r['backend'], r['id']): r for r in reviewed['rows']}
    assert len(notes) == len(reviewed['rows']) == 174
    summaries = []
    for cell in inv['cells']:
        if cell['status'] != 'generated':
            continue
        raw = local(cell['raw_path'])
        assert sha(raw) == cell['raw_sha256']
        n = notes[cell['backend'], cell['id']]
        assert n['raw_sha256'] == cell['raw_sha256']
        assert len(n['ratings']) == 6 and set(n['ratings']) <= set('PFU')
        result = json.loads(raw.with_name('result.json').read_text())
        for name, h in result['files'].items():
            assert sha(raw.with_name(name)) == h, (cell['id'], name)
    for backend in ['qwen', 'vace']:
        for method in ['A_direct', 'B_planar', 'C_rgb3d', 'D_staged3d']:
            values = [r['seconds'] for r in inv['cells'] if r['backend'] == backend and r['method'] == method and r['status'] == 'generated']
            summaries.append(dict(backend=backend, method=method, generated=len(values),
                                  mean_seconds=statistics.mean(values), median_seconds=statistics.median(values),
                                  summed_generation_seconds=sum(values),
                                  allocated_gpu_hours=sum(values)*(2 if backend == 'qwen' else 1)/3600))
    supervisor = read('formal/supervisor.json')
    assert supervisor['status'] == 'workers_finished'
    completion = read('compute_completion.json')
    assert completion['status'] == 'compute_complete_visual_review_required', completion['status']
    replay = read('reproducibility/replay_audit.json')
    assert replay['status'] == 'complete' and len(replay['checks']) == 2
    assert all(r['status'] == 'complete' for r in replay['checks'])
    observer = read('observer_formal/observations.json')
    assert observer['status'] == 'complete' and len(observer['images']) == 182
    assert len({r['id'] for r in observer['images']}) == 182
    obs = {r['id']: r for r in observer['images']}
    diagnostics = []
    for backend in ['qwen', 'vace']:
        for method in ['A_direct', 'B_planar', 'C_rgb3d', 'D_staged3d']:
            generated = [r for r in inv['cells'] if r['backend'] == backend and r['method'] == method and r['status'] == 'generated']
            rr = [obs[backend+'__'+r['id']] for r in generated]
            assert all(o['sha256'] == c['raw_sha256'] for o, c in zip(rr, generated))
            diagnostics.append(dict(backend=backend, method=method, n_generated=len(rr),
                fork_detected=sum(r['prompts']['fork']['instance_count'] > 0 for r in rr),
                hand_detected=sum(r['prompts']['hand']['instance_count'] > 0 for r in rr),
                matched_food_component=sum(r.get('payload_diagnostic') is not None for r in rr)))
    result = {
        'status': 'verified_compute_and_assistant_review_complete',
        'frozen_hashes_checked': checked, 'source_hashes_checked': 8, 'raw_outputs_checked': 174,
        'source_normalization_pixel_checks': source_pixel_checks,
        'review_rows_checked': 174, 'all_intended_cells': 192, 'preprocessing_failures': 18,
        'compute_by_backend_method': summaries,
        'compute_limits': 'Allocated GPU-hours sum per-call durations times worker GPU count. Includes per-call encoding/writing, excludes model loading, preprocessing, pilot, observer, replay and idle allocation; not measured electrical energy or billable total.',
        'supervisor_started_unix': supervisor['started_unix'],
        'supervisor_finished_unix': supervisor.get('finished_unix'),
        'minimum_sampled_mem_available_mib': min(r['mem_available_mib'] for r in supervisor['resource_samples']),
        'replay': replay, 'observer_negative_controls': observer['negative_controls'],
        'observer_by_backend_method': diagnostics,
        'observer_limits': observer['limits'],
        'review_limits': 'One unblinded assistant diagnostic review; no completed independent human ratings or physical 3-D ground truth.'
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'source_hashes_checked', 'raw_outputs_checked', 'review_rows_checked']}))


if __name__ == '__main__':
    main()
