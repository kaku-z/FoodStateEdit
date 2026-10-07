"""Reserve source-image candidates from official RGB-D test IDs, metadata only.

One candidate per cafe/calendar day; all days exposed in earlier Nutrition5k
manifests are excluded. This conservatively groups neighboring scans but is
not a substitute for later source-only annotation and physical-object audit.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request


SOLIDS = {'apple', 'orange', 'pizza', 'corn on the cob', 'tofu', 'sausage',
          'chicken', 'grilled chicken', 'bread', 'bagel', 'brownie', 'cake',
          'potatoes', 'steak', 'salmon', 'baked potato'}


def day(dish):
    return datetime.fromtimestamp(int(dish[5:]), timezone.utc).date().isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--prior', type=Path, action='append', required=True)
    p.add_argument('--scope', choices=['official-test', 'custom-unseen'], default='official-test')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = 'https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/dish_ids/splits/'
    official, split_sources = set(), []
    split_names = ['depth_test_ids.txt'] if args.scope == 'official-test' else ['depth_test_ids.txt', 'depth_train_ids.txt']
    original_split = {}
    for split_name in split_names:
        url = base+split_name
        raw = urllib.request.urlopen(url, timeout=30).read()
        (args.output/('official_'+split_name)).write_bytes(raw)
        ids = set(raw.decode().split())
        official.update(ids)
        original_split.update({dish: split_name for dish in ids})
        split_sources.append({'url': url, 'sha256': hashlib.sha256(raw).hexdigest()})
    exposed_days = set()
    previous = []
    for manifest_path in args.prior:
        manifest = json.loads(manifest_path.read_text())
        for row in manifest['selected']:
            for key in ('source_after_addition', 'target_before_addition'):
                exposed_days.add((row['cafe'], day(row[key])))
                previous.append(row[key])
    groups = {}
    metadata_hashes = {}
    eligible = 0
    for name in ('dish_metadata_cafe1.csv', 'dish_metadata_cafe2.csv'):
        file = args.metadata/name
        metadata_hashes[name] = hashlib.sha256(file.read_bytes()).hexdigest()
        for row in csv.reader(file.open()):
            if row[0] not in official:
                continue
            contents = [(row[i], row[i+1], float(row[i+2])) for i in range(6, len(row), 7)]
            if len(contents) > 3 or not any(x[1] in SOLIDS for x in contents):
                continue
            group = (name, day(row[0]))
            if group in exposed_days:
                continue
            candidate = {'dish_id': row[0], 'cafe': name, 'day_group': group[1],
                         'original_dataset_split': original_split[row[0]],
                         'ingredients': contents, 'source_only_annotation': 'pending',
                         'rank': hashlib.sha256(('observed-edit-v2:'+row[0]).encode()).hexdigest()}
            groups.setdefault(group, []).append(candidate)
            eligible += 1
    chosen = [min(values, key=lambda r:r['rank']) for values in groups.values()]
    chosen.sort(key=lambda r:r['rank'])
    registry = {'status': 'candidate_reservation_only_not_unlocked_test',
                'selection_uses': 'metadata only; no candidate images downloaded or viewed',
                'scope': args.scope, 'official_split_files': split_sources,
                'metadata_sha256': metadata_hashes, 'allowed_food_names': sorted(SOLIDS),
                'prior_dishes_excluded': sorted(set(previous)),
                'excluded_cafe_day_groups': sorted(exposed_days), 'eligible_dishes': eligible,
                'independent_day_groups': len(chosen), 'candidates': chosen,
                'limitations': ['Source masks/targets/utensil size and suitability require annotation before any output generation.',
                               'Calendar-day grouping is conservative; it is not an observed plate identity label.',
                               'Foundation-model pretraining overlap is unknown.',
                               'No real spoon-lift target photographs or physical measurements are supplied.']}
    out = args.output/'candidate_registry.json'
    out.write_text(json.dumps(registry, indent=2), encoding='utf-8')
    print(json.dumps({'eligible': eligible, 'day_groups': len(chosen), 'registry_sha256': hashlib.sha256(out.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
