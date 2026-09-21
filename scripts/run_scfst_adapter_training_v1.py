"""Safety-gated launcher for one SCFST adapter training run."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def atomic_json(path,obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8'); os.replace(tmp,path)


def nvidia_rows(kind):
    if kind=='gpu':
        query='index,uuid,name,memory.total,memory.free,utilization.gpu'
    else:
        query='gpu_uuid,pid,used_memory,process_name'
    text=subprocess.check_output(['nvidia-smi',f'--query-{kind}={query}','--format=csv,noheader,nounits'],text=True)
    return [[field.strip() for field in row.split(',')] for row in text.splitlines() if row.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--trainer',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--physical-gpu',type=int,required=True)
    p.add_argument('--preflight-report',type=Path,required=True)
    p.add_argument('--log',type=Path,required=True)
    a=p.parse_args(); cfg=json.loads(a.config.read_text(encoding='utf-8'))
    if a.output_root.exists() or a.preflight_report.exists() or a.log.exists():
        raise FileExistsError('fresh output, preflight, and log paths required')
    rows=nvidia_rows('gpu'); match=[row for row in rows if int(row[0])==a.physical_gpu]
    if len(match)!=1: raise ValueError('physical GPU not found')
    idx,uuid,name,total,free,util=match[0]
    processes=[row for row in nvidia_rows('compute-apps') if row[0]==uuid]
    available_kib=next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                       if line.startswith('MemAvailable:'))
    checks={
        'gpu_name':name=='NVIDIA RTX A6000','gpu_free_memory':int(free)>=48000,
        'gpu_utilization':int(util)<=5,'zero_compute_processes':not processes,
        'host_available_memory':available_kib//1024>=80000,
    }
    report={'schema_version':'foodstateedit.scfst_training_preflight.v1',
            'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'gpu':{'physical_index':int(idx),'uuid':uuid,'name':name,'total_mib':int(total),
                   'free_mib':int(free),'utilization':int(util),'compute_processes':processes},
            'host_available_memory_mib':available_kib//1024,'checks':checks,'passed':all(checks.values())}
    atomic_json(a.preflight_report,report)
    if not report['passed']: return 4
    env=os.environ.copy(); env.update({
        'CUDA_VISIBLE_DEVICES':str(a.physical_gpu),'DIFFSYNTH_SKIP_DOWNLOAD':'True',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
        'HF_HOME':'/host/space0/guo-z/cache/huggingface',
        'TRANSFORMERS_CACHE':'/host/space0/guo-z/cache/huggingface',
    })
    command=[sys.executable,str(a.trainer),'--config',str(a.config),'--output-root',str(a.output_root)]
    state={'schema_version':'foodstateedit.scfst_training_launcher.v1','status':'starting',
           'started_utc':datetime.now(timezone.utc).isoformat(),'command':command,
           'physical_gpu':a.physical_gpu,'preflight_report':str(a.preflight_report)}
    launch_manifest=a.preflight_report.with_name(a.preflight_report.stem+'_launcher.json')
    with a.log.open('w',encoding='utf-8') as log:
        child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,text=True)
        state.update(status='running',child_pid=child.pid); atomic_json(launch_manifest,state)
        deadline=time.monotonic()+int(cfg['training']['hard_timeout_seconds'])
        stop_reason=None
        while child.poll() is None:
            if time.monotonic()>=deadline:
                stop_reason='hard_timeout'; child.terminate(); break
            foreign=[row for row in nvidia_rows('compute-apps') if row[0]==uuid and int(row[1])!=child.pid]
            if foreign:
                stop_reason='foreign_compute_process_appeared'; child.terminate(); break
            time.sleep(10)
        try: return_code=child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill(); return_code=child.wait(); stop_reason=(stop_reason or 'termination_timeout')
    state.update(status='complete' if return_code==0 else 'technical_failure_preserved',
                 return_code=return_code,stop_reason=stop_reason,
                 finished_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(launch_manifest,state)
    return return_code


if __name__=='__main__': raise SystemExit(main())
