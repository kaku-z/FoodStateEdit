"""Collect every development endpoint, including technical failures."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile


def read(p):
    for i in range(8):
        try:
            return json.loads(p.read_text())
        except OSError as e:
            if e.errno != 116 or i == 7:
                raise
            time.sleep(.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--gate', default='gate_v1')
    ap.add_argument('--archive', action='store_true')
    a = ap.parse_args(); r = a.root
    c = read(r/(a.gate+'.json'))
    seen = {}; workers = []
    for p in sorted(r.glob(a.gate+'_worker_*/manifest.json')):
        m = read(p)
        workers.append({k:m.get(k) for k in ['pid','status','current','step','error','updated_unix']})
        for row in m['completed']:
            raw = p.parent/row['id']/'raw.png'
            h = hashlib.sha256(raw.read_bytes()).hexdigest()
            assert h == row['files']['raw.png']
            seen[row['id']] = dict(row, raw_path=str(raw), raw_sha256=h)
    inventory = {'status':'complete' if len(seen)==len(c['jobs']) else 'incomplete',
        'not_formal':c['not_formal'],'expected':len(c['jobs']),'generated':len(seen),'workers':workers,'results':list(seen.values())}
    (r/(a.gate+'_inventory.json')).write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps({k:inventory[k] for k in ['status','expected','generated','workers']}))
    if a.archive:
        assert inventory['status']=='complete'
        arc = r/(a.gate+'_results.zip')
        paths = [r/(a.gate+'.json'), r/(a.gate+'_inventory.json'), r/'run_coupled_bite_qwen.py']
        paths += [p for d in [r/(a.gate+'_inputs')]+list(r.glob(a.gate+'_worker_*')) for p in d.rglob('*') if p.is_file()]
        paths += list((r/'logs').glob(a.gate+'*.log'))
        with zipfile.ZipFile(arc,'w',zipfile.ZIP_DEFLATED) as z:
            for p in paths:
                z.write(p,p.relative_to(r))
        receipt = {'archive':arc.name,'size_bytes':arc.stat().st_size,'sha256':hashlib.sha256(arc.read_bytes()).hexdigest()}
        (r/(a.gate+'_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt))


if __name__ == '__main__':
    main()
