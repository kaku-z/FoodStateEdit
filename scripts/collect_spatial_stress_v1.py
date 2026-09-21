"""Snapshot declared jobs; collect stopped results into a fresh local directory."""
import argparse,hashlib,json,shlex,subprocess
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
REMOTE='/host/space0/guo-z/tf-ufi/outputs/spatial_stress_v1_20260914T1358Z'
PYTHON='/host/space0/guo-z/envs/geoedit/bin/python'
def ssh(code):
    return json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=40','gp40',PYTHON+' -c '+shlex.quote(code)],text=True,encoding='utf-8',timeout=90))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--collect',type=Path);a=ap.parse_args()
    code='''import json,subprocess,hashlib
from pathlib import Path
from datetime import datetime,timezone
r=Path(REMOTE)
states={p.parent.name:json.loads(p.read_text()) for p in r.glob('*/run_manifest.json')}
logs={str(p.relative_to(r)):p.read_text(errors='replace')[-600:] for p in r.glob('*/*/inference.log')}
pids=','.join(str(s['pid']) for s in states.values())
ps=subprocess.run(['ps','-p',pids,'-o','pid,etime,stat,rss,wchan:20'],capture_output=True,text=True).stdout
finished=len(states)==2 and all(s['status'] in ['complete_requires_visual_review','technical_failure_preserved'] for s in states.values())
files={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in r.rglob('*') if p.is_file() and 'cache' not in p.relative_to(r).parts} if finished else {}
print(json.dumps(dict(captured_utc=datetime.now(timezone.utc).isoformat(),states=states,logs=logs,processes=ps,stopped=finished,files=files)))
'''.replace('REMOTE',repr(REMOTE))
    data=ssh(code)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    snapshot=ROOT/'results'/('spatial_stress_snapshot_'+stamp+'.json')
    with snapshot.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2)
    print(json.dumps(dict(snapshot=str(snapshot),states={k:dict(status=s['status'],active=s.get('active'),completed=len(s['completed'])) for k,s in data['states'].items()},processes=data['processes'],logs=data['logs']),indent=2))
    if a.collect:
        if not data['stopped']:raise RuntimeError('Jobs still running; collection deferred')
        if a.collect.exists():raise FileExistsError(a.collect)
        a.collect.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['scp','-r','-o','BatchMode=yes','-o','ConnectTimeout=40','gp40:'+REMOTE,str(a.collect)],check=True,timeout=180)
        after=ssh(code)
        assert after['files']==data['files'],'Remote evidence changed during collection'
        actual={str(p.relative_to(a.collect)).replace('\\','/'):sha(p) for p in a.collect.rglob('*') if p.is_file() and 'cache' not in p.relative_to(a.collect).parts}
        assert actual==data['files'],'File SHA-256 mismatch'
        verified=dict(remote=REMOTE,local=str(a.collect),file_count=len(actual),all_files_sha256_verified=True,states=data['states'],files=actual)
        with (a.collect.parent/(a.collect.name+'_verified.json')).open('x') as f:json.dump(verified,f,indent=2)
        print('COLLECTED_AND_VERIFIED',len(actual))
if __name__=='__main__':main()
