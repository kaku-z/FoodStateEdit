"""Pinned public model download with size/hash verification, bounded retries."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
DEST=Path('/mnt/tmp/guo-z_first_bite_structure_20260930_models')
SPECS=[('Qwen/Qwen-Image-2.1','d26bb61231c349cf6b7896fa83353113880e1ba3','qwen-image-2.1'),
       ('alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union','8a4702014d4dabb5f896fcba917e2ee0a961465f','controlnet-union')]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    os.environ['HTTPS_PROXY']='http://127.0.0.1:22341'
    DEST.mkdir(exist_ok=False);out=ROOT/'backend21';out.mkdir(exist_ok=False)
    state={'started_unix':time.time(),'status':'downloading','destination':str(DEST),'models':[],'completed':[],'errors':[]}
    work=[]
    for repo,rev,name in SPECS:
        with urllib.request.urlopen('https://huggingface.co/api/models/'+repo+'/revision/'+rev+'?blobs=true',timeout=45) as f:c=json.load(f)
        assert c['sha']==rev
        (out/(name+'_api.json')).write_text(json.dumps(c,indent=2)+'\n')
        files=[x for x in c['siblings'] if not x['rfilename'].startswith(('asset/','assets/','results/','.'))]
        state['models'].append({'repo':repo,'revision':rev,'directory':str(DEST/name),'files':files})
        for item in files:work.append((repo,rev,name,item))
    def download(spec):
        repo,rev,name,item=spec;rel=item['rfilename'];path=DEST/name/rel;path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix(path.suffix+'.part');url='https://huggingface.co/'+repo+'/resolve/'+rev+'/'+rel
        for attempt in range(3):
            try:
                offset=tmp.stat().st_size if tmp.exists() else 0
                headers={'User-Agent':'FoodStateEdit-research/1.0'}
                if offset:headers['Range']='bytes='+str(offset)+'-'
                with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=90) as response:
                    append=offset>0 and response.status==206
                    if append:assert response.headers['Content-Range'].startswith('bytes '+str(offset)+'-')
                    with tmp.open('ab' if append else 'wb') as f:
                        while True:
                            b=response.read(1024**2)
                            if not b:break
                            f.write(b)
                assert tmp.stat().st_size==item['size'],(rel,tmp.stat().st_size,item['size'])
                digest=sha(tmp);expected=item.get('lfs',{}).get('sha256')
                if expected:assert digest==expected,(rel,'sha256 mismatch')
                tmp.replace(path)
                return {'model':name,'file':rel,'size_bytes':path.stat().st_size,'sha256':digest,'expected_lfs_verified':expected is not None}
            except Exception:
                if attempt==2:raise
                time.sleep(3*(attempt+1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(download,x):x for x in work}
        while pending:
            done,_=concurrent.futures.wait(pending,timeout=20,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                spec=pending.pop(f)
                try:state['completed'].append(f.result())
                except Exception as exc:state['errors'].append({'file':spec[3]['rfilename'],'error':repr(exc)})
            state['downloaded_bytes']=sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file())
            state['updated_unix']=time.time()
            (out/'download_manifest.json').write_text(json.dumps(state,indent=2)+'\n')
            print('DOWNLOAD',len(state['completed']),'/',len(work),'GB',round(state['downloaded_bytes']/1e9,2),'errors',len(state['errors']),flush=True)
    state.update(status='complete_verified' if not state['errors'] else 'failed',finished_unix=time.time())
    (out/'download_manifest.json').write_text(json.dumps(state,indent=2)+'\n')
    if state['errors']:raise RuntimeError('Download incomplete; inspect saved errors')

if __name__=='__main__':main()
