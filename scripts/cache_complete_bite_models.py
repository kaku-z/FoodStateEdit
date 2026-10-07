"""Copy only this experiment's offline models to private local disk, hashing every byte."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_complete_20260929')
DEST=Path('/mnt/tmp/guo-z_first_bite_20260929_models')


def main():
    os.umask(0o077)
    assert shutil.disk_usage('/mnt/tmp').free>180*1024**3
    DEST.mkdir(exist_ok=False,mode=0o700)
    q=json.loads((ROOT/'qwen_runtime.json').read_text())
    v=json.loads((ROOT/'vace_runtime.json').read_text())
    # Runtime wrappers are inspected rather than guessed by the launch caller.
    if 'backend' in q:q=q['backend']
    if 'runtime' in v:v=v['runtime']
    tasks=[(Path(q['model_root']),DEST/'qwen-image-edit-2511',q['weight_files']),
           (Path(v['model_root']),DEST/'PAI/Wan2.2-VACE-Fun-A14B',v['model_files']),
           (Path(v['model_root']).parent.parent/'Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl',DEST/'Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl',{})]
    state={'status':'copying','started_unix':time.time(),'destination':str(DEST),'files':[],'pid':os.getpid()}
    mp=ROOT/'model_cache_receipt.json'
    def save():
        tmp=mp.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(mp)
    save()
    for source,target,expected in tasks:
        assert source.is_dir(),source
        for p in sorted(source.rglob('*')):
            if not p.is_file() or any(x in p.relative_to(source).parts for x in ['.cache','.git']):continue
            rel=p.relative_to(source);out=target/rel;out.parent.mkdir(parents=True,exist_ok=True)
            h=hashlib.sha256();total=0;state.update(current=str(p),current_bytes=0);save()
            with p.open('rb') as src,out.open('xb') as dst:
                while True:
                    data=src.read(16*1024**2)
                    if not data:break
                    dst.write(data);h.update(data);total+=len(data)
                    if total%(256*1024**2)==0:state['current_bytes']=total;save()
                dst.flush();os.fsync(dst.fileno())
            digest=h.hexdigest()
            if str(rel) in expected:
                assert digest==expected[str(rel)]['sha256'] and total==expected[str(rel)]['size_bytes']
            state['files'].append({'source':str(p),'destination':str(out),'size_bytes':total,'sha256':digest});save()
        for name,info in expected.items():assert (target/name).is_file()
    state.update(status='complete',finished_unix=time.time(),qwen_root=str(DEST/'qwen-image-edit-2511'),vace_root=str(DEST/'PAI/Wan2.2-VACE-Fun-A14B'));save()
    print(json.dumps({'status':'complete','files':len(state['files'])}))


if __name__=='__main__':main()
