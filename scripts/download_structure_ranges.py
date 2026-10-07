"""Resume pinned public weights using independent, verified HTTP byte ranges."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import urllib.request

ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
BLOCK=16*1024**2

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    os.environ['HTTPS_PROXY']='http://127.0.0.1:22341'
    oldpid=json.loads((ROOT/'backend21_download_launch.json').read_text())['pid']
    proc=Path('/proc')/str(oldpid)/'cmdline'
    if proc.exists():
        assert b'download_structure_backend.py' in proc.read_bytes()
        os.kill(oldpid,signal.SIGTERM)
        for _ in range(50):
            if not proc.exists():break
            time.sleep(.2)
        assert not proc.exists(),'Previous downloader must stop before taking file ownership'
    prior=json.loads((ROOT/'backend21/download_manifest.json').read_text())
    receipt=ROOT/'backend21/range_manifest.json'
    if receipt.exists():raise RuntimeError('Use saved range manifest to resume; never infer completion from preallocated size')
    state={'status':'downloading','started_unix':time.time(),'files':[],'errors':[],'previous_manifest':prior,'chunk_bytes':BLOCK}
    work=[];handles={}
    for model in prior['models']:
        for item in model['files']:
            p=Path(model['directory'])/item['rfilename'];p.parent.mkdir(parents=True,exist_ok=True)
            expected=item.get('lfs',{}).get('sha256')
            if p.exists():
                assert p.stat().st_size==item['size']
                state['files'].append({'path':str(p),'size':item['size'],'status':'previously_verified','sha256':sha(p)});continue
            tmp=p.with_suffix(p.suffix+'.part');prefix=(tmp.stat().st_size//BLOCK)*BLOCK if tmp.exists() else 0
            fd=os.open(tmp,os.O_RDWR|os.O_CREAT,0o600);os.ftruncate(fd,item['size']);handles[str(tmp)]=fd
            row={'path':str(p),'temporary':str(tmp),'size':item['size'],'prefix_bytes':prefix,'completed_ranges':[],
                'expected_sha256':expected,'status':'downloading'}
            state['files'].append(row);idx=len(state['files'])-1
            url='https://huggingface.co/'+model['repo']+'/resolve/'+model['revision']+'/'+item['rfilename']
            for start in range(prefix,item['size'],BLOCK):work.append((idx,url,start,min(start+BLOCK,item['size'])-1))
    # Spread requests over all files instead of finishing one shard at a time.
    work.sort(key=lambda x:(x[2],x[0]))
    def download(job):
        idx,url,start,end=job;row=state['files'][idx]
        for attempt in range(4):
            try:
                req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','User-Agent':'FoodStateEdit-research/1.0'})
                with urllib.request.urlopen(req,timeout=90) as res:
                    assert res.status==206 or start==0 and end+1==row['size'],('Range not honored',res.status)
                    if res.status==206:assert res.headers['Content-Range'].startswith(f'bytes {start}-{end}/'),res.headers['Content-Range']
                    offset=start
                    while offset<=end:
                        b=res.read(min(1024**2,end-offset+1))
                        if not b:raise IOError('Early end of requested range')
                        wrote=os.pwrite(handles[row['temporary']],b,offset);assert wrote==len(b);offset+=wrote
                    assert not res.read(1),'Server exceeded requested range'
                return idx,start,end
            except Exception:
                if attempt==3:raise
                time.sleep(1+attempt)
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
        pending={pool.submit(download,j):j for j in work}
        while pending:
            done,_=concurrent.futures.wait(pending,timeout=20,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                j=pending.pop(f)
                try:
                    idx,s,e=f.result();state['files'][idx]['completed_ranges'].append([s,e])
                except Exception as exc:state['errors'].append({'job':[j[0],j[2],j[3]],'error':repr(exc)})
            state['received_bytes']=sum(x.get('prefix_bytes',x['size'] if x['status']=='previously_verified' else 0)+sum(e-s+1 for s,e in x.get('completed_ranges',[])) for x in state['files'])
            state['updated_unix']=time.time();receipt.write_text(json.dumps(state,indent=2)+'\n')
            print('RANGES',len(pending),'remaining; received GB',round(state['received_bytes']/1e9,2),'errors',len(state['errors']),flush=True)
    for fd in handles.values():os.close(fd)
    for row in state['files']:
        if row['status']=='previously_verified':continue
        received=row['prefix_bytes']+sum(e-s+1 for s,e in row['completed_ranges'])
        if received!=row['size']:continue
        tmp=Path(row['temporary']);h=sha(tmp)
        if row['expected_sha256'] and h!=row['expected_sha256']:
            state['errors'].append({'path':row['path'],'error':'Full SHA256 mismatch'});continue
        tmp.replace(row['path']);row.update(status='verified',sha256=h)
    state.update(status='complete_verified' if not state['errors'] else 'failed',finished_unix=time.time());receipt.write_text(json.dumps(state,indent=2)+'\n')

if __name__=='__main__':main()
