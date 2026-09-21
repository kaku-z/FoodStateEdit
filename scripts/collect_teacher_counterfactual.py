"""Collect only stopped runs into fresh directories and compare every file hash."""
import argparse,hashlib,json,subprocess,shlex
from pathlib import Path
import numpy as np
from PIL import Image

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ssh(command):return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=40','gp40',command],text=True,encoding='utf-8')
def inventory(root):
    output=ssh("find '"+root+"' -type f -exec sha256sum {} +")
    return {line.split(None,1)[1].strip()[len(root)+1:]:line.split(None,1)[0] for line in output.splitlines() if line.strip()}

def stopped_snapshot(root):
    code=("import json,hashlib;from pathlib import Path;"
          "r=Path("+repr(root)+");"
          "states={c:json.loads((r/c/'run_manifest.json').read_text()) for c in ['soup','cake']};"
          "assert all(v['status'] in ['technical_failure_preserved','complete_requires_visual_review'] for v in states.values()),states;"
          "files={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in r.rglob('*') if p.is_file()};"
          "print(json.dumps(dict(states=states,files=files)))")
    return json.loads(ssh('/host/space0/guo-z/envs/geoedit/bin/python -c '+shlex.quote(code)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--remote',required=True);ap.add_argument('--local',type=Path,required=True);args=ap.parse_args()
    assert args.remote.startswith('/host/space0/guo-z/tf-ufi/outputs/teacher_counterfactual')
    if args.local.exists():raise FileExistsError('Fresh destination required')
    snapshot=stopped_snapshot(args.remote);states=snapshot['states']
    for c,s in states.items():assert s['status'] in ['technical_failure_preserved','complete_requires_visual_review'],(c,s['status'])
    before=snapshot['files'];args.local.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['scp','-r','-o','BatchMode=yes','-o','ConnectTimeout=40','gp40:'+args.remote,str(args.local)],check=True)
    after=inventory(args.remote);assert before==after,'Source changed during collection'
    local={str(p.relative_to(args.local)).replace('\\','/'):sha(p) for p in args.local.rglob('*') if p.is_file()}
    assert local==before,'Per-file SHA-256 mismatch'
    summary=dict(remote=args.remote,local=str(args.local),all_file_hashes_verified=True,file_count=len(local),sha256=before,cases={})
    for c,state in states.items():
        entry=dict(status=state['status'],pipeline_load_count=state['pipeline_load_count'],completed=[],shapes=[])
        trace=args.local/c/'condition_shape_trace.json'
        if trace.exists():entry['shapes']=json.loads(trace.read_text())
        for condition in state['completed']:
            folder=args.local/c/condition['condition']
            for name,rec in condition['files'].items():assert sha(folder/name)==rec['sha256']
            meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames','-of','json',str(folder/'raw.mp4')]))['streams'][0]
            assert int(meta['nb_read_frames'])==21
            assert (meta['width'],meta['height'])==(688,512)
            dataset=Path(__file__).resolve().parents[1]/'artifacts/day19_multimaterial_dataset_v3'/c
            reference=np.array(Image.open(dataset/'reference.png').convert('RGB'))
            alpha=np.array(Image.open(args.local/c/'shared_alpha.png').convert('L'))
            final=np.array(Image.open(folder/'projected_final_hold.png').convert('RGB'))
            assert sha(dataset/'reference.png')==state['input_sha256']
            assert np.array_equal(final[alpha==0],reference[alpha==0]),'Composite outside changed'
            entry['completed'].append(dict(condition=condition['condition'],frames=21,all_manifest_hashes_verified=True,final_outside_exact=True))
        summary['cases'][c]=entry
    result=args.local.parent/(args.local.name+'_collection.json')
    result.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(result=str(result),files=len(local),cases=summary['cases']),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
