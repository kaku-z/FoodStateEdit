"""Validate recorded, non-blind supplemental observations; never assign scores."""
import hashlib
import json
from pathlib import Path

FIELDS = ['lift', 'bite_and_support', 'source_change', 'food_and_scene', 'hand_and_utensil']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    read = lambda p: json.loads((root / p).read_text(encoding='utf-8'))
    review = read('results/FIRST_BITE_SUPPLEMENT_ASSISTANT_REVIEW_20260928.json')
    original = {r['run_id']: r for r in read('results/FIRST_BITE_ASSISTANT_REVIEW_20260928.json')['ratings']}
    config = read('outputs/first_bite_20260928_supplement_bundle_v2/config.json')
    roles = {c['case_id']: c['experiment_role'] for c in config['cases']}
    expected = {f"{j['case_id']}__{j['method']}__{j['seed']}" for j in config['jobs'] if roles[j['case_id']] == 'roi_repair'}
    ratings = {r['run_id']: r for r in review['ratings']}
    assert len(ratings) == len(review['ratings']) == 24 and set(ratings) == expected
    for name, r in ratings.items():
        assert all(r[f] in ['pass', 'fail', 'uncertain'] for f in FIELDS)
        assert r['joint_pass'] == (all(r[f] == 'pass' for f in FIELDS) and r['photo_score'] >= 4)
        paths = list((root/'outputs/first_bite_20260928_supplement_gp40').glob('*/'+name+'/raw.png'))
        assert len(paths) == 1 and sha(paths[0]) == r['raw_sha256']
        assert r['original_raw_sha256'] == original[name]['raw_sha256']
    def stats(rows):
        return {'n': len(rows), 'joint_pass': sum(r['joint_pass'] for r in rows),
                'pass': {f: sum(r[f] == 'pass' for r in rows) for f in FIELDS},
                'photo_at_least_4': sum(r['photo_score'] >= 4 for r in rows)}
    def comparison(names):
        return {'original': stats([original[n] for n in names]), 'repaired': stats([ratings[n] for n in names]),
                'gained_joint': [n for n in sorted(names) if ratings[n]['joint_pass'] and not original[n]['joint_pass']],
                'lost_joint': [n for n in sorted(names) if original[n]['joint_pass'] and not ratings[n]['joint_pass']]}
    observations = review['height_observations']
    assert len(observations) == 4 and {x['family'] for x in observations} == {'cohesive','granular','strand','liquid'}
    for item in observations:
        assert len(item['levels']) == 3 and {x['height'] for x in item['levels']} == {0, .08, .16}
        for level in item['levels']:
            paths = list((root/'outputs/first_bite_20260928_supplement_gp40').glob('*/'+level['run_id']+'/raw.png'))
            assert len(paths) == 1 and sha(paths[0]) == level['raw_sha256']
    verification = read('outputs/first_bite_20260928_supplement_review/verification.json')
    assert verification['calls'] == 36
    summary = {'scope': 'Post-hoc annotation repair and height sensitivity; assistant non-blind development observations only',
               'independent_ballots_received': 0, 'formal_unseen_unlocked': False, 'roi_repair': comparison(expected),
               'by_method': {m: comparison([n for n in expected if n.split('__')[1] == m]) for m in config['methods']},
               'by_case': {c: comparison([n for n in expected if n.split('__')[0] == c]) for c in ['strand_01','strand_02','liquid_01']},
               'height': observations, 'calls': 36, 'total_calls_including_main': 100}
    (root/'results/FIRST_BITE_SUPPLEMENT_SUMMARY_20260928.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({'repairs':summary['roi_repair'], 'height_families_reviewed':len(observations)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
