"""Server-side receipt and portable archive; callable while jobs run or after completion."""
import hashlib
import json
from pathlib import Path
import time
import zipfile

ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_complete_20260929')


def read_json(p):
    for attempt in range(8):
        try:return json.loads(p.read_text())
        except OSError as exc:
            if exc.errno!=116 or attempt==7:raise
            time.sleep(.25*(attempt+1))


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(16*1024**2),b''):h.update(x)
    return h.hexdigest()


def main():
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--archive',action='store_true');a=a.parse_args()
    expected=read_json(ROOT/'formal/expected_cells.json')
    found={};workers=[]
    for p in sorted((ROOT/'formal').glob('*_worker_*/manifest.json')):
        m=read_json(p);backend=p.parent.name.split('_worker_')[0]
        workers.append({'path':str(p),'backend':backend,'status':m['status'],'pid':m['pid'],'completed':len(m['completed']),'current':m.get('current'),'step':m.get('step'),'updated_unix':m.get('updated_unix'),'error':m.get('error')})
        for row in m['completed']:
            key=(backend,row['id']);assert key not in found,key
            folder=p.parent/row['id']
            for name,h in row['files'].items():assert sha(folder/name)==h,(key,name)
            found[key]={'status':'generated','raw_path':str(folder/'raw.png'),'raw_sha256':sha(folder/'raw.png'),'seconds':row['seconds'],'result_path':str(folder/'result.json')}
    rows=[]
    for cell in expected:
        x=dict(cell);key=(x['backend'],x['id'])
        if key in found:x.update(found[key])
        elif x['status']=='scheduled':
            claim=ROOT/'formal'/('claims_'+x['backend'])/x['id']/'owner.json'
            if claim.exists():
                owner=read_json(claim);m=read_json(Path(owner['output']).parent/'manifest.json')
                x.update(status='technical_failure' if m['status']=='technical_failure' else 'running',owner=owner)
        rows.append(x)
    done=all(x['status'] in ['generated','preprocessing_failed','technical_failure'] for x in rows)
    audit={'status':'complete' if done else 'running','created_unix':time.time(),'cells':rows,'workers':workers,
           'counts':{s:sum(x['status']==s for x in rows) for s in sorted(set(x['status'] for x in rows))},
           'hashes_verified':len(found),'no_best_seed_selection':True}
    (ROOT/'formal/inventory.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:audit[k] for k in ['status','counts','hashes_verified']}))
    if a.archive:
        assert done,'Refuse to label incomplete archive final'
        archive=ROOT/'complete_results_v1.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
            for p in sorted(ROOT.rglob('*')):
                if not p.is_file():continue
                rel=p.relative_to(ROOT)
                if any(x in rel.parts for x in ['vace_cache','__pycache__']):continue
                if p.suffix in ['.zip','.npz']:continue
                z.write(p,str(rel))
        receipt={'path':str(archive),'size_bytes':archive.stat().st_size,'sha256':sha(archive),'excluded':'Redundant earlier ZIPs, model caches, pycache and per-pixel/observer NPZ arrays. NPZ originals retained on server.'}
        (ROOT/'complete_results_v1_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
