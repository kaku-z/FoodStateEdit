"""Read actual server execution and checksum receipts; never rate image quality."""
import argparse, hashlib, json, subprocess, time
from pathlib import Path

REMOTE = '/host/space0/guo-z/tf-ufi/first_bite_structure_20260930'
SSH=['ssh','-o','BatchMode=yes','-o','ProxyJump=none','-o','HostName=127.0.0.1','-o','Port=22240','-o','HostKeyAlias=gp40','-T','gp40','/usr/bin/python3','-u','-']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--hash-archives',action='store_true');a=ap.parse_args();r=a.root
    names=[]
    if a.hash_archives:
        old=json.loads((r/'ARCHIVE_TRANSFER_AUDIT_20261001.json').read_text(encoding='utf8'))
        names=[x.get('file',x.get('name')) for x in old['archives']+old.get('new_completed_archives',[])]
        names+=['gate_v48_complete.zip','gate_v49_complete.zip','gate_v50_complete.zip','continuation_diagnostics_v14.zip','continuation_diagnostics_v15.zip','gate_v51_complete.zip','gate_v52_complete.zip','continuation_diagnostics_v16.zip','gate_v53_complete.zip','gate_v54_complete.zip','gate_v55_complete.zip','continuation_diagnostics_v17.zip','continuation_diagnostics_v18.zip','gate_v56_complete.zip','continuation_diagnostics_v19.zip']
        names=sorted({n for n in names if (r/n).is_file()})
    code="""
import json,time,hashlib
from pathlib import Path
r=Path(REMOTE)
def read(p):
 for _ in range(10):
  try:return json.loads(p.read_text())
  except (OSError,json.JSONDecodeError):time.sleep(.3)
 raise RuntimeError(str(p))
stages={}
for n in [41,42,44,45,46,47,48,49,50,51,52,53,54,55,56]:
 g=r/('gate_v'+str(n));p=g/'execution.json'
 if not p.exists():continue
 x=read(p);ws=[]
 for p in sorted(g.glob('worker_*/manifest.json')):
  z=read(p);ws.append(dict(name=p.parent.name,status=z['status'],completed=len(z.get('completed',[])),expected=len(z.get('expected_jobs',[])),step=z.get('step'),current=z.get('current'),pid=z.get('pid')))
 stages[g.name]=dict(execution=x['status'],raw_cells=x.get('raw_cells'),compositions=x.get('compositions'),workers=ws)
observers={}
for n in [12,14,15,16,17,18,19]:
 p=r/('spoon_observer_v'+str(n))/'observations/observations.json'
 if p.exists():
  x=read(p);observers[str(n)]=dict(status=x['status'],images=len(x['images']))
checks=[]
for n in NAMES:
 p=r/n;h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 checks.append(dict(name=n,bytes=p.stat().st_size,sha256=h.hexdigest()))
print(json.dumps(dict(created_unix=time.time(),live_stages=stages,observers=observers,remote_archive_checksums=checks,scope='Actual execution receipts and hashes; not realism acceptance.')))
""".replace('Path(REMOTE)','Path('+repr(REMOTE)+')').replace('in NAMES:','in '+repr(names)+':')
    p=subprocess.run(SSH,input=code,capture_output=True,text=True,check=True);x=json.loads(p.stdout)
    (r/'EXECUTION_STATE_20261001.json').write_text(json.dumps(x,indent=2)+'\n',encoding='utf8')
    if a.hash_archives:
        verified=[]
        for row in x['remote_archive_checksums']:
            path=r/row['name'];h=hashlib.sha256()
            with path.open('rb') as f:
                for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
            assert h.hexdigest()==row['sha256'] and path.stat().st_size==row['bytes'],row['name']
            verified.append(dict(row,transfer_verified=True))
        originals={v['file'] for v in old['archives']}
        old.update(new_completed_archives=[v for v in verified if v['name'] not in originals],new_completed_archive_count=sum(v['name'] not in originals for v in verified),updated_unix=time.time(),local_remote_byte_and_sha256_equal=True)
        (r/'ARCHIVE_TRANSFER_AUDIT_20261001.json').write_text(json.dumps(old,indent=2)+'\n',encoding='utf8')
    print('SERVER_STATE',[(n,s['execution'],sum(w['completed'] for w in s['workers'])) for n,s in x['live_stages'].items() if n in ['gate_v49','gate_v50']], 'HASH_VERIFIED',len(names),flush=True)

if __name__=='__main__':main()
