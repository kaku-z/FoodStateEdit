"""Read this experiment's status and collect completed, immutable raw cells."""
import argparse,datetime,hashlib,json,subprocess
from pathlib import Path


SSH=['-o','BatchMode=yes','-o','ProxyJump=none','-o','HostName=127.0.0.1','-o','Port=22240','-o','HostKeyAlias=gp40']
REMOTE='/host/space0/guo-z/tf-ufi/first_bite_20260928'
READ='''from pathlib import Path
import json,subprocess,time,os
r=Path("/host/space0/guo-z/tf-ufi/first_bite_20260928")
out={"utc_unix":time.time(),"shards":[]}
for i in range(4):
 p=r/("shard_%d_v1/run_manifest.json"%i)
 if not p.exists():continue
 text=p.read_text();m=json.loads(text)
 launch=json.loads((r/("launch_%d_v1.json"%i)).read_text())
 out["shards"].append({"index":i,"manifest_text":text,"manifest":m,"launcher_alive":Path("/proc/%d"%launch["pid"]).exists()})
out["gpus"]=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader"],text=True).strip().splitlines()
print(json.dumps(out))
'''


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--collect',action='store_true')
    group=p.add_mutually_exclusive_group();group.add_argument('--height',action='store_true');group.add_argument('--supplement',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    read=READ.replace('shard_%d_v1','height_shard_%d_v1').replace('launch_%d_v1','height_launch_%d_v1') if a.height else READ
    if a.supplement:read=READ.replace('shard_%d_v1','supplement_shard_%d_v2').replace('launch_%d_v1','supplement_launch_%d_v2')
    data=json.loads(subprocess.run(['ssh',*SSH,'-T','gp40','/usr/bin/python3 -u -'],input=read,capture_output=True,text=True,check=True,timeout=45).stdout)
    suffix='_supplement' if a.supplement else ('_height' if a.height else '')
    monitor=root/('outputs/first_bite_20260928'+suffix+'_monitor');monitor.mkdir(exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    (monitor/(stamp+'.json')).write_text(json.dumps(data,indent=2))
    downloaded=0
    for item in data['shards']:
        shard_name=('height_' if a.height else '')+'shard_%d_v1'%item['index']
        if a.supplement:shard_name='supplement_shard_%d_v2'%item['index']
        directory=root/('outputs/first_bite_20260928'+suffix+'_gp40')/shard_name
        if a.collect:
            directory.mkdir(parents=True,exist_ok=True)
            for row in item['manifest']['completed']:
                cell=directory/row['name'];cell.mkdir(exist_ok=True)
                missing=[]
                for name,h in row['files'].items():
                    path=cell/name
                    if path.exists():
                        if sha(path)!=h:raise ValueError('Existing output hash mismatch')
                    else:missing.append(name)
                if missing:
                    sources=[f"gp40:{REMOTE}/{shard_name}/{row['name']}/{name}" for name in missing]
                    subprocess.run(['scp',*SSH,*sources,str(cell)],capture_output=True,text=True,check=True,timeout=60)
                    downloaded+=int('raw.png' in missing)
                for name,h in row['files'].items():
                    if sha(cell/name)!=h:raise ValueError('Transferred file hash mismatch')
            (directory/'run_manifest.json').write_bytes(item['manifest_text'].encode())
            if item['manifest']['status']=='complete_unreviewed':
                for name in ['preflight.json']:
                    if not (directory/name).exists():subprocess.run(['scp',*SSH,f"gp40:{REMOTE}/{shard_name}/{name}",str(directory)],capture_output=True,check=True,timeout=45)
        m=item['manifest']
        print(json.dumps({'shard':item['index'],'status':m['status'],'completed':len(m['completed']),'current':m.get('current'),'step':m.get('step'),'launcher_alive':item['launcher_alive'],'error':m.get('error')}))
    print(json.dumps({'total_complete':sum(len(x['manifest']['completed']) for x in data['shards']),'new_downloaded_raws':downloaded,'snapshot':str(monitor/(stamp+'.json'))}))


if __name__=='__main__':main()
