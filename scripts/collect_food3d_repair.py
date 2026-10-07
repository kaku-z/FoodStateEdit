"""Archive repair outputs, failures and pinned runtime sources; verify all hashes."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

SSH=['-o','BatchMode=yes','-o','ProxyJump=none','-o','HostName=127.0.0.1','-o','Port=22240','-o','HostKeyAlias=gp40']
REMOTE=r'''
import hashlib,json,pathlib,zipfile
r=pathlib.Path('/host/space0/guo-z/tf-ufi/food3d_repair_20260928')
assert (r/'results/repair_v5/repair_overview.jpg').exists()
assert all(x['status']=='passed' for x in json.loads((r/'results/sensitivity_v1/sensitivity.json').read_text())['cases'])
paths=[]
for folder in ['inputs','results','logs','runtime_wheels']:
 paths.extend(p for p in (r/folder).rglob('*') if p.is_file())
for pattern in ['*.json','*.py','*_runtime.zip','REPORT.md','run_repair.sh']:
 paths.extend(r.glob(pattern))
paths=sorted(set(paths));manifest={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
archive=r/'food3d_repair_results_v1.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for p in paths:z.write(p,p.relative_to(r))
 z.writestr('ARCHIVE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
 z.writestr('README.txt','Private real-photo geometry repair. Final: results/repair_v5. Sensitivity: results/sensitivity_v1. Prior failed renders are retained.\nNo final photorealistic generation or measured 3-D ground truth. Model weights remain on the server, pinned in upstream_manifest.json.\n')
print(json.dumps({'path':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'bytes':archive.stat().st_size,'files':len(paths)}))
'''


def main():
    root=Path(__file__).resolve().parents[1];out=root/'outputs/food3d_repair_20260928'
    archive=out/'server_results.zip';assert not archive.exists()
    result=subprocess.run(['ssh',*SSH,'-T','gp40','/usr/bin/python3 -'],input=REMOTE,text=True,capture_output=True,check=True,timeout=240)
    receipt=json.loads(result.stdout)
    subprocess.run(['scp',*SSH,'gp40:'+receipt['path'],str(archive)],check=True,timeout=240)
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
    extracted=out/'server_results';extracted.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('ARCHIVE_MANIFEST.json'))
        for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
        for name in z.namelist():assert (extracted/name).resolve().is_relative_to(extracted.resolve()),name
        z.extractall(extracted)
    receipt.update(local_archive=str(archive),local_results=str(extracted),all_files_hash_verified=True)
    (out/'collection_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
