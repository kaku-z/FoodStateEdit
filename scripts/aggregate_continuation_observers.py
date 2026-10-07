"""Summarize fallible SAM diagnostics without converting them to success rates."""
import argparse, collections, json, statistics, time, zipfile
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
    path=r/'SAM3_DIAGNOSTICS_SUMMARY_20261001.json';summary=json.loads(path.read_text(encoding='utf8'))
    for v in [14,15,16,17,18,19]:
        p=r/f'continuation_diagnostics_v{v}.zip'
        if not p.exists():continue
        with zipfile.ZipFile(p) as z:obs=json.loads(z.read(f'spoon_observer_v{v}/observations/observations.json'))
        assert obs['status']=='complete'
        groups=collections.defaultdict(list)
        for row in obs['images']:
            role=row['role']
            if 'gate_v48' in role or 'gate_v49' in role:role+='__coarse_anchor' if '__joint_action_coarse_anchor__' in row['id'] else '__free'
            groups[role].append(row)
        summary['groups']=[g for g in summary['groups'] if g['observer_version']!=v]
        for role,rows in groups.items():
            detections=[x['payload_diagnostic'] for x in rows if x.get('payload_diagnostic')]
            ious=[x['target_mask_iou'] for x in detections]
            summary['groups'].append(dict(observer_version=v,role=role,cells=len(rows),detected_payloads=len(detections),no_payload_diagnostic_ids=[x['id'] for x in rows if not x.get('payload_diagnostic')],detected_hands=sum(x['prompts']['hand']['instance_count'] for x in rows),target_mask_iou_median_on_detections=statistics.median(ious) if ious else None,target_mask_iou_min_on_detections=min(ious) if ious else None,scope='Fallible text-only SAM3 diagnostics against an inferred proxy mask. Neither detection nor IoU is photographic realism, true geometry, physical contact, measured conservation or proof of absence of hands.'))
        print('OBSERVER_AGGREGATED',v,len(obs['images']),len(groups),flush=True)
    summary.update(updated_unix=time.time(),scope='Repeated source/empty-spoon controls across observer versions are duplicates, not independent images. Previously reviewed development only; no automatic success assignment.')
    path.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')

if __name__=='__main__':main()
