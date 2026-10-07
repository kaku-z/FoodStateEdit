"""Run independent fresh-photo MLD4 jobs from one frozen regression manifest."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, data):
    Path(p).write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def main(args):
    config=json.loads(args.protocol.read_text(encoding='utf-8'))
    root=Path(config['output'])
    rows=[]
    indices=config['shards'][args.shard]
    for case in config['cases']:
        if case['index'] not in indices:
            continue
        source=Path(case['image'])
        assert sha(source)==case['source_sha256']
        destination=root/'cases'/case['case_id']
        command=[config['geometry_python'],'-u',config['entrypoint'],
            '--image',str(source),'--food-prompt',case['food_prompt'],
            '--output',str(destination),'--gpu',str(args.gpu)]
        started=time.monotonic()
        row=dict(case_id=case['case_id'],index=case['index'],status='running',
            source=case,gpu=args.gpu,command=command,
            started_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        rows.append(row)
        write(root/f'worker_{args.shard}.json',dict(status='running',cases=rows))
        with (root/'logs'/f'{case["case_id"]}.log').open('x',encoding='utf-8') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        row.update(status='complete' if result.returncode==0 else 'failed',
            returncode=result.returncode,seconds=time.monotonic()-started,
            finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        if (destination/'status.json').exists():
            row['execution_status']=json.loads((destination/'status.json').read_text(encoding='utf-8'))
        write(root/f'worker_{args.shard}.json',dict(status='running',cases=rows))
        print(json.dumps(dict(case_id=case['case_id'],status=row['status'],seconds=row['seconds'])),flush=True)
    write(root/f'worker_{args.shard}.json',dict(status='complete',cases=rows,
        complete=sum(r['status']=='complete' for r in rows),failed=sum(r['status']=='failed' for r in rows)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--shard',type=int,required=True)
    p.add_argument('--gpu',type=int,required=True)
    main(p.parse_args())
