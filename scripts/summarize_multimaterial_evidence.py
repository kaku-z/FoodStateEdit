"""Read-only metadata summary; does not replace pixel/hash verification or review."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
CASES = ('soup', 'rice', 'cake')
ARMS = ('planar', 'relative3d')
MATCH_FIELDS = ('frames', 'size', 'seed', 'steps', 'source_dataset_case', 'support_fraction')
METRICS = ('native_outside_mae', 'preencode_outside_max', 'encoded_outside_mae',
           'encoded_outside_max', 'preencode_inside_mae')
REQUIRED = MATCH_FIELDS + METRICS + ('final_sha256', 'root')
SCHEMAS = {'foodstateedit.non_noodle_result.v1': 'technical_metrics',
           'foodstateedit.presentation_verification.v1': 'conditions'}


def field_valid(field, value):
    if value is None:
        return False
    if field in ('frames', 'steps'):
        return type(value) is int and value > 0
    if field == 'seed':
        return type(value) is int and value >= 0
    if field == 'size':
        return isinstance(value, list) and len(value) == 2 and all(type(x) is int and x > 0 for x in value)
    if field in METRICS or field == 'support_fraction':
        return (type(value) in (int, float) and math.isfinite(value) and value >= 0
                and (field != 'support_fraction' or value <= 1))
    if field == 'final_sha256':
        return isinstance(value, str) and re.fullmatch('[0-9a-fA-F]{64}', value) is not None
    return isinstance(value, str) and bool(value.strip())


def summarize(documents, dataset=None):
    """documents is [(source_label, parsed_json)]; no filesystem effects."""
    expected = [case + '__' + arm for case in CASES for arm in ARMS]
    grouped = defaultdict(list)
    sources = []
    for label, document in documents:
        if not isinstance(document, dict):
            raise ValueError('Evidence JSON must be an object')
        schema = document.get('schema_version')
        if schema not in SCHEMAS:
            raise ValueError('Unsupported evidence schema: ' + str(schema))
        records = document.get(SCHEMAS[schema])
        if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
            raise ValueError('Evidence conditions must be a list of objects')
        runs = document.get('runs', [])
        if not isinstance(runs, list) or any(not isinstance(run, dict) for run in runs):
            raise ValueError('Reported runs must be a list of objects')
        sources.append({'label': label, 'schema_version': schema,
                        'reported_status': document.get('status'),
                        'reported_original_batch_status': document.get('original_batch_status'),
                        'reported_runs': [{k: run.get(k) for k in ('root', 'status', 'pipeline_load_count',
                                                                  'verified_file_count')}
                                          for run in runs]})
        for record in records:
            name = record.get('condition')
            if not isinstance(name, str) or not name:
                raise ValueError('Every condition requires a nonempty string name')
            grouped[name].append({'evidence_source': label, 'reported': record})
    provenance = defaultdict(list)
    if dataset is not None:
        if not isinstance(dataset, dict) or dataset.get('schema_version') != 'foodstateedit.multimaterial_dataset.v1':
            raise ValueError('Unsupported dataset schema')
        cases = dataset.get('cases', [])
        if not isinstance(cases, list) or any(not isinstance(case, dict) or not isinstance(case.get('case_id'), str) for case in cases):
            raise ValueError('Dataset cases require string case_id values')
        for case in cases:
            provenance[case['case_id']].append({k: case.get(k) for k in
                ('source_kind', 'source_original_path', 'source_original_sha256', 'license')})
    rows = []
    for name in expected:
        records = grouped.get(name, [])
        row = {'condition': name, 'occurrence_count': len(records), 'occurrences': records}
        row['status'] = 'missing' if not records else 'duplicate' if len(records) > 1 else 'present'
        row['missing_fields'] = []
        row['invalid_fields'] = []
        if len(records) == 1:
            record = records[0]['reported']
            row['missing_fields'] = [k for k in REQUIRED if record.get(k) is None]
            row['invalid_fields'] = [k for k in REQUIRED if record.get(k) is not None and not field_valid(k, record[k])]
            if record.get('source_dataset_case') != name.split('__')[0]:
                row['invalid_fields'].append('source_dataset_case_identity')
            if record.get('frames') != 21:
                row['invalid_fields'].append('expected_21_frames')
            if row['missing_fields'] or row['invalid_fields']:
                row['status'] = 'incomplete_metadata'
        row['reported_source_candidates'] = provenance.get(name.split('__')[0], [])
        row['source_status'] = ('missing' if not row['reported_source_candidates'] else
                                'ambiguous' if len(row['reported_source_candidates']) > 1 else
                                'reported_not_artifact_verified')
        if row['source_status'] == 'reported_not_artifact_verified':
            candidate = row['reported_source_candidates'][0]
            if any(not isinstance(v, str) or not v.strip() for v in candidate.values()):
                row['source_status'] = 'incomplete_provenance'
            elif not field_valid('final_sha256', candidate['source_original_sha256']):
                row['source_status'] = 'incomplete_provenance'
        rows.append(row)
    by_name = {r['condition']: r for r in rows}
    pairs = []
    for case in CASES:
        left, right = [by_name[case + '__' + arm] for arm in ARMS]
        reasons = [r['condition'] + ': ' + r['status'] for r in (left, right) if r['status'] != 'present']
        differences = []
        if not reasons:
            a, b = left['occurrences'][0]['reported'], right['occurrences'][0]['reported']
            differences = [k for k in MATCH_FIELDS if a[k] != b[k]]
            reasons.extend('mismatched ' + k for k in differences)
        pairs.append({'case_id': case,
                      'reported_metadata_comparability': 'not_comparable' if reasons else 'matched_reported_fields_only',
                      'reasons': reasons, 'mismatched_fields': differences,
                      'causal_effect_or_semantic_gain_established_by_tool': False})
    return {'schema_version': 'foodstateedit.multimaterial_evidence_summary.v1',
            'scope': 'Six non-noodle planar/relative3d pilot conditions',
            'evidence_sources': sources, 'conditions': rows, 'pairs': pairs,
            'missing_conditions': [r['condition'] for r in rows if r['status'] == 'missing'],
            'duplicate_conditions': sorted(k for k, v in grouped.items() if len(v) > 1),
            'unexpected_conditions': sorted(set(grouped) - set(expected)),
            'complete_reported_metadata': all(r['status'] == 'present' for r in rows),
            'limitations': [
                'Metadata summary only: output hashes, video frames, source linkage and reported metrics are NOT reverified.',
                'Use verify_multimaterial_presentation.py for artifact/pixel verification.',
                'preencode_outside_max=0 is imposed by alpha compositing, NOT native model background preservation.',
                'Native and encoded pixel changes are diagnostics, not semantic quality or effectiveness scores.',
                'Matching listed fields does not verify model, prompt, source identity, support pixels or all inference factors.',
                'Sources are joined by case ID only; dataset-to-run SHA linkage is not checked here.',
                'Do not pool real previously-used sources with synthetic cake as a generalization benchmark.',
                'No automatic semantic scoring, winner selection, human rating or scientific success decision.']}


def read_document(path):
    raw = path.read_bytes()
    return json.loads(raw.decode('utf-8-sig')), hashlib.sha256(raw).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, action='append', help='Repeatable; overlapping conditions are flagged, never selected.')
    parser.add_argument('--dataset-manifest', type=Path, default=ROOT / 'artifacts/day19_multimaterial_dataset_v3/dataset_manifest.json')
    parser.add_argument('--strict', action='store_true', help='Exit 1 on incomplete, duplicate, unexpected or non-comparable metadata.')
    args = parser.parse_args(argv)
    paths = args.evidence or [ROOT / 'results/day20_non_noodle_result_20260909.json']
    try:
        documents, input_hashes = [], {}
        for path in paths:
            doc, digest = read_document(path)
            documents.append((str(path), doc))
            input_hashes[str(path)] = digest
        dataset = None
        if args.dataset_manifest.exists():
            dataset, digest = read_document(args.dataset_manifest)
            input_hashes[str(args.dataset_manifest)] = digest
        report = summarize(documents, dataset)
        report['input_json_sha256'] = input_hashes
        report['dataset_manifest_path'] = str(args.dataset_manifest)
        report['dataset_manifest_available'] = dataset is not None
        print(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False))
        bad = (not report['complete_reported_metadata'] or report['duplicate_conditions'] or
               report['unexpected_conditions'] or any(p['reported_metadata_comparability'] == 'not_comparable' for p in report['pairs']))
        return 1 if args.strict and bad else 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print('Evidence summary error: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
