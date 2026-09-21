"""Read-only status snapshot; does not imply output completion."""
import argparse,json,shlex,subprocess
from pathlib import Path
from datetime import datetime,timezone

REMOTE='/host/space0/guo-z/tf-ufi/outputs/teacher_counterfactual_rgb_v2_20260914T1304Z'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Fresh snapshot path required')
    code="""import json,subprocess
from pathlib import Path
from datetime import datetime,timezone
r=Path(REMOTE_ROOT)
states={c:json.loads((r/c/'run_manifest.json').read_text()) for c in ['soup','cake']}
pids=','.join(str(int(s['pid'])) for s in states.values())
logs={str(p.relative_to(r)):p.read_text(errors='replace')[-1800:] for p in r.glob('*/*/inference.log')}
files={str(p.relative_to(r)):p.stat().st_size for p in r.rglob('*') if p.is_file()}
ps=subprocess.run(['ps','-p',pids,'-o','pid,etime,stat,rss,wchan:24'],capture_output=True,text=True).stdout
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True).stdout
print(json.dumps(dict(captured_utc=datetime.now(timezone.utc).isoformat(),states=states,logs=logs,files=files,processes=ps,gpus=gpu)))
""".replace('REMOTE_ROOT',repr(REMOTE))
    result=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=40','gp40','/host/space0/guo-z/envs/geoedit/bin/python -c '+shlex.quote(code)],text=True,encoding='utf-8')
    data=json.loads(result);data['local_received_utc']=datetime.now(timezone.utc).isoformat();data['snapshot_only']=True
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(args.output),status={c:s['status'] for c,s in data['states'].items()},completed={c:len(s['completed']) for c,s in data['states'].items()},processes=data['processes']),ensure_ascii=False))
if __name__=='__main__':main()
