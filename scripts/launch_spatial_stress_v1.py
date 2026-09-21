"""Launch exactly the two predeclared stress jobs on fresh gp40 paths."""
import json,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
runtime='/host/space0/guo-z/tf-ufi/runtime/spatial_stress_v1_20260914T1358Z'
output='/host/space0/guo-z/tf-ufi/outputs/spatial_stress_v1_20260914T1358Z'
python='/host/space0/guo-z/envs/geoedit/bin/python'
code='''import subprocess,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
runtime=Path(RUNTIME);root=Path(OUTPUT)
assert hashlib.sha256((runtime/'bundle.tar').read_bytes()).hexdigest()=='10b33b3a7fbebc12644358be14ed34f6bef3697fb319572cb1b56850a76a0e16'
root.mkdir(exist_ok=False)
jobs=[]
for job,gpu in [('soup_toward_camera',0),('cake_yaw45',1)]:
    cmd=[PYTHON,str(runtime/'scripts/run_spatial_stress_v1.py'),'--base-config',str(runtime/'configs/day25_same_condition_ablation_v2_retry.json'),'--controls',str(runtime/'artifacts/spatial_stress_controls_v1_20260914T1352Z'),'--manifest-sha256','090c2a4cdab600ae5b16ed58a5d9925e464b29adf46f42fc2719c4cd9dcef674','--job',job,'--gpu',str(gpu),'--root',str(root)]
    with (root/(job+'_launch.log')).open('x') as log:
        p=subprocess.Popen(cmd,cwd=runtime,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    jobs.append(dict(job=job,gpu=gpu,pid=p.pid,command=cmd))
receipt=dict(started_utc=datetime.now(timezone.utc).isoformat(),host='gp40',root=str(root),runtime=str(runtime),jobs=jobs,status='dispatched_not_complete')
(root/'launch_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''.replace('RUNTIME',repr(runtime)).replace('OUTPUT',repr(output)).replace('PYTHON',repr(python))
result=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=40','gp40',python+' -c '+shlex.quote(code)],text=True,encoding='utf-8',timeout=90)
data=json.loads(result);dest=ROOT/'results/spatial_stress_v1_launch_20260914T1358Z.json'
with dest.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2)
print(result)
