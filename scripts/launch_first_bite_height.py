"""Start each frozen diagnostic shard only after its main GPU pair is finished."""
import json,subprocess
from monitor_first_bite import SSH

REMOTE_SCRIPT=r'''
from pathlib import Path
import hashlib,json,os,subprocess,time
r=Path('/host/space0/guo-z/tf-ufi/first_bite_20260928')
runtime=r/'runtime_v1';height=r/'height_runtime_v1'
receipt=json.loads((height/'results/FIRST_BITE_HEIGHT_FREEZE_20260928.json').read_text())
for name,h in receipt['files'].items():assert hashlib.sha256((height/name).read_bytes()).hexdigest()==h
assert hashlib.sha256((height/'scripts/run_first_bite.py').read_bytes()).hexdigest()==hashlib.sha256((runtime/'scripts/run_first_bite.py').read_bytes()).hexdigest()
responses=[]
for i in range(4):
 launch=r/('height_launch_%d_v1.json'%i);out=r/('height_shard_%d_v1'%i)
 if launch.exists():
  responses.append({'shard':i,'status':'already_launched','pid':json.loads(launch.read_text())['pid']});continue
 m=json.loads((r/('shard_%d_v1/run_manifest.json'%i)).read_text())
 if m['status']!='complete_unreviewed' or m['failed']:
  responses.append({'shard':i,'status':'waiting_for_main_shard'});continue
 assert not out.exists(),'Refuse to overwrite diagnostic output'
 command=json.loads((r/('launch_%d_v1.json'%i)).read_text())['command']
 command[1]='3600'
 command[command.index('--bundle')+1]=str(height/'outputs/first_bite_20260928_height_bundle_v1')
 command[command.index('--output')+1]=str(out)
 env=os.environ.copy();env['PYTORCH_KERNEL_CACHE_PATH']=str(r/'kernel_cache')
 with (r/('height_shard_%d_v1.log'%i)).open('xb') as log:
  process=subprocess.Popen(command,cwd=str(runtime),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 record={'shard':i,'pid':process.pid,'command':command,'started_unix':time.time(),
         'main_completed_cells':len(m['completed']),'deployment_files_verified':len(receipt['files'])}
 launch.write_text(json.dumps(record,indent=2));responses.append(dict(record,status='launched'))
print(json.dumps(responses))
'''


if __name__=='__main__':
    raise SystemExit('Height v1 was superseded before any calls. Use launch_first_bite_supplement.py for corrected v2 inputs.')
    result=subprocess.run(['ssh',*SSH,'-T','gp40','/usr/bin/python3 -u -'],input=REMOTE_SCRIPT,
                          capture_output=True,text=True,check=True,timeout=45)
    print(json.dumps(json.loads(result.stdout),indent=2))
