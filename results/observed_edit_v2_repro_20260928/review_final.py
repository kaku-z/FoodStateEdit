"""Verify outputs, score only the real paired source-hole task, prepare blind review."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodstateedit.observed_edit.appearance import relight_observed
import numpy as np
from PIL import Image, ImageDraw


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mae(a, b, mask):
    return float(np.abs(a.astype(float)-b.astype(float))[mask].mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--real-pairs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--original-run', type=Path)
    p.add_argument('--original-bundle', type=Path)
    args = p.parse_args()
    c = json.loads((args.bundle/'config.json').read_text())
    run = json.loads((args.run/'run_manifest.json').read_text())
    if run['status'] != 'complete_unreviewed' or len(run['completed']) != c['expected_generation_count']:
        raise ValueError('Run is not complete')
    if run['config_sha256'] != sha(args.bundle/'config.json'):
        raise ValueError('Config hash mismatch')
    checked = 0
    for record in run['completed']:
        for name, expected in record['files'].items():
            if sha(args.run/record['name']/name) != expected:
                raise ValueError('Output checksum mismatch')
            checked += 1
    if args.original_run:
        if not args.original_bundle:
            raise ValueError('Original baseline bundle required for verification')
        other = json.loads((args.original_run/'run_manifest.json').read_text())
        if other['status'] != 'complete_unreviewed' or len(other['completed']) != 2:
            raise ValueError('Original-only baseline is not complete')
        if other['config_sha256'] != sha(args.original_bundle/'config.json'):
            raise ValueError('Original baseline config mismatch')
        for record in other['completed']:
            for name, expected in record['files'].items():
                if sha(args.original_run/record['name']/name) != expected:
                    raise ValueError('Original baseline output checksum mismatch')
                checked += 1
    args.output.mkdir(parents=True, exist_ok=False)
    blind = args.output/'blind'
    blind.mkdir()
    rows, review_items, key, sources = [], [], [], {}
    cols = ['source', 'removal_reference', 'scaffold', 'direct_raw', 'direct_localized',
            'grounded_raw', 'grounded_localized', 'observed_locked', 'relit_locked']
    if args.original_run:
        cols[3:3] = ['original_raw', 'original_localized']
    sheet = Image.new('RGB', (320*len(cols), 262*len(c['cases'])), 'white')
    draw = ImageDraw.Draw(sheet)
    for ci, case in enumerate(c['cases']):
        case_id = case['case_id']
        folder = args.bundle/'inputs'/case_id
        source = Image.open(folder/'source.png').convert('RGB')
        source_arr = np.array(source)
        state = np.load(folder/'state.npz', allow_pickle=False)
        target = Image.open(args.real_pairs/f"pair_{case['dataset_index']:03d}"/'target_rgb.png').convert('RGB')
        target_arr = np.array(target)
        sources[case_id] = source
        variants = {'source': source, 'removal_reference': target,
                    'scaffold': Image.open(folder/'scaffold.png').convert('RGB')}
        for condition in c['conditions']:
            out = args.run/f"{case_id}_{c['seeds'][0]}_{condition}"
            variants[condition+'_raw'] = Image.open(out/'raw.png').convert('RGB').resize(source.size, Image.Resampling.LANCZOS)
            variants[condition+'_localized'] = Image.open(out/'localized.png').convert('RGB')
            if condition == 'grounded':
                variants['observed_locked'] = Image.open(out/'observed_locked.png').convert('RGB')
                adapted, parameters = relight_observed(state['transported_rgb'], np.array(variants['grounded_localized']),
                                                       state['target_mask'], state['payload_core'])
                variants['relit_locked'] = Image.fromarray(adapted)
                variants['relit_locked'].save(args.output/(case_id+'_relit_locked.png'))
                (args.output/(case_id+'_appearance_parameters.json')).write_text(json.dumps(parameters))
        if args.original_run:
            original = args.original_run/f"{case_id}_{c['seeds'][0]}_original"
            variants['original_raw'] = Image.open(original/'raw.png').convert('RGB').resize(source.size, Image.Resampling.LANCZOS)
            variants['original_localized'] = Image.open(original/'localized.png').convert('RGB')
        for name, img in variants.items():
            arr = np.array(img)
            if name != 'removal_reference':
                rows.append({'case': case_id, 'variant': name,
                             'source_hole_real_target_mae': mae(arr, target_arr, state['source_mask']),
                             'protected_vs_source_mae': mae(arr, source_arr, ~state['editable']),
                             'payload_copy_error_internal_only': mae(arr, state['transported_rgb'], state['payload_core'])})
            if name not in ('source', 'removal_reference', 'scaffold'):
                review_items.append((case_id, name, img))
        for j, name in enumerate(cols):
            sheet.paste(variants[name].resize((320,240)), (320*j, 262*ci))
            draw.text((320*j+5, 262*ci+243), case_id+' '+name, fill='black')
        comparison_cols = ['source', 'original_raw' if args.original_run else 'direct_raw', 'grounded_raw', 'relit_locked']
        comparison = Image.new('RGB', (1280, 262), 'white')
        compare_draw = ImageDraw.Draw(comparison)
        for j, name in enumerate(comparison_cols):
            comparison.paste(variants[name].resize((320,240)), (320*j, 0))
            compare_draw.text((320*j+5, 243), name, fill='black')
        comparison.save(args.output/(case_id+'_comparison.png'))
    sheet.save(args.output/'labeled_overview.png')
    # Blind package is for future independent raters; this script fills no judgments.
    random.Random(20260928).shuffle(review_items)
    for i, (case_id, variant, img) in enumerate(review_items):
        item = f'item_{i:03d}'
        img.save(blind/(item+'.png'))
        sources[case_id].save(blind/(item+'_source.png'))
        key.append({'item': item, 'case': case_id, 'variant': variant})
    with (blind/'ballot.csv').open('w',newline='',encoding='utf-8') as stream:
        fields = ['item','source_removed','identity_preserved','one_spoon','supported',
                  'target_position','photo_natural','comments']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in key:
            writer.writerow({'item': item['item']})
    (blind/'INSTRUCTIONS.md').write_text(
        'Review every item against its source and task. Do not guess the method.\n'
        'Each categorical field accepts PASS, FAIL, or UNKNOWN. UNKNOWN is not success.\n'
        'Task: move the pizza portion to (230,220), handle toward (45,390); or the round corn slice '
        'to (440,175), handle toward (594,385), in 640x480 source coordinates. Exactly one metal serving '
        'spoon must support the same intact food, and the old position must be empty. All other food stays.\n'
        'photo_natural means no obvious pasted boundary, duplicated object, inconsistent lighting, '
        'or deformed spoon. Record the concrete defect when FAIL or UNKNOWN.\n'
        'These are development images. The creator has already seen the sources.\n', encoding='utf-8')
    (args.output/'blinding_key.json').write_text(json.dumps(key,indent=2),encoding='utf-8')
    template = (Path(__file__).resolve().parents[1]/'benchmark/observed_edit_v2_review_template.html').read_text(encoding='utf-8')
    instructions = {'pizza':'移动披萨至 (230,220)，勺柄指向 (45,390)。坐标基于 640×480 输入。移除原处披萨，保留玉米。',
                    'corn':'移动玉米片至 (440,175)，勺柄指向 (594,385)。坐标基于 640×480 输入。移除原处玉米，保留披萨。'}
    public_items = [{'id': row['item'], 'task': instructions[row['case']]} for row in key]
    (blind/'index.html').write_text(template.replace('__ITEMS_JSON__',json.dumps(public_items,ensure_ascii=False)),encoding='utf-8')
    with (args.output/'metrics.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    report={'status':'outputs_verified_independent_review_pending','verified_output_files':checked,
            'config_sha256':run['config_sha256'],'cases':len(c['cases']),'metrics':rows,
            'limits':['Real target metric applies only to the source hole, not the moved food or spoon.',
                      'Copy error and protected-region identity are construction checks, not independent performance.',
                      'No independent ballots exist yet. Two cases are not a confirmatory sample.']}
    (args.output/'review_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'verified':checked,'review_items':len(key),'status':report['status']}))


if __name__ == '__main__':
    main()
