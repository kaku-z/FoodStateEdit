"""Wait for owned development jobs, then build source-only geometry controls."""
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import zipfile

ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
OLD=Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929')

def read(p):
    for i in range(8):
        try:return json.loads(p.read_text())
        except OSError as e:
            if e.errno!=116 or i==7:raise
            time.sleep(.3)

def main():
    while True:
        ms=list((ROOT/'gate_v1').glob('worker_*/manifest.json'))
        if len(ms)==4 and all(read(p)['status'] in ['complete_unreviewed','technical_failure'] for p in ms):break
        time.sleep(20)
    # Allow finished processes to release their CUDA contexts before using GPU 0.
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(10)
    import numpy as np
    import torch
    import build_structured_geometry_v3 as base
    from moge.model.v2 import MoGeModel
    inp=ROOT/'inputs';inp.mkdir(exist_ok=True)
    with zipfile.ZipFile(ROOT/'prospective_inputs.zip') as z:
        for name in z.namelist():assert (inp/name).resolve().is_relative_to(inp.resolve())
        z.extractall(inp)
    future=read(inp/'manifest.json');development=read(OLD/'inputs/manifest.json')
    for case in development['cases']:shutil.copytree(OLD/'inputs'/case['case_id'],inp/case['case_id'],dirs_exist_ok=True)
    allcases=development['cases']+future['cases']
    (inp/'manifest.json').write_text(json.dumps({'cases':allcases,'development_ids':[c['case_id'] for c in development['cases']],
        'prospective_ids':[c['case_id'] for c in future['cases']],'source_selection':future},indent=2)+'\n')
    out=ROOT/'geometry_v3';out.mkdir(exist_ok=False);rows=[]
    state={'status':'building','cases':rows};mp=out/'manifest.json';model=None
    torch.set_num_threads(4)
    for case in allcases:
        d=out/case['case_id'];d.mkdir();start=time.time()
        try:
            if case in development['cases']:
                maps=dict(np.load(OLD/'geometry'/case['case_id']/'maps.npz'))
            else:
                if model is None:
                    ck=torch.load(base.REPAIR/'models/model.pt',map_location='cpu',weights_only=True)
                    model=MoGeModel(**ck['model_config']);model.load_state_dict(ck['model'],strict=True);del ck
                    model=model.eval().cuda()
                from PIL import Image
                arr=np.asarray(Image.open(inp/case['case_id']/'source.png').convert('RGB'))
                x=torch.as_tensor(arr.copy(),device='cuda',dtype=torch.float32).permute(2,0,1)/255
                with torch.inference_mode():res=model.infer(x,resolution_level=9,use_fp16=True)
                maps={k:v.cpu().numpy() for k,v in res.items()};np.savez_compressed(d/'maps.npz',**maps)
            row=base.geometry(case,d,maps);row['seconds']=time.time()-start
        except Exception as e:
            row={'case_id':case['case_id'],'status':'geometry_failed','error':repr(e)}
            (d/'FAILED.txt').write_text(traceback.format_exc())
        rows.append(row);mp.write_text(json.dumps(state,indent=2)+'\n');print(case['case_id'],row['status'],row.get('error'),flush=True)
    state.update(status='complete',finished_unix=time.time());mp.write_text(json.dumps(state,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/'geometry_v3_controls.zip','w',zipfile.ZIP_DEFLATED) as z:
        for folder in ['geometry_v3','inputs']:
            for p in (ROOT/folder).rglob('*'):
                if p.is_file() and p.suffix not in ['.npz','.glb','.ply']:z.write(p,p.relative_to(ROOT))
    print('GEOMETRY_COMPLETE',flush=True)

if __name__=='__main__':main()
