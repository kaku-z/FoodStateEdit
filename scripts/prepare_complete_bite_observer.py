"""Build the fixed text-observer batch after the raw formal outputs are complete."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_complete_20260929')


def main():
    inv=json.loads((ROOT/'formal/inventory.json').read_text());assert inv['status']=='complete'
    sam=json.loads((ROOT/'sam3_runtime.json').read_text());images=[]
    for p in sorted((ROOT/'inputs').glob('*/source.png')):
        images.append({'id':p.parent.name+'__source','case_id':p.parent.name,'role':'unedited_source','path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    for r in inv['cells']:
        if r['status']!='generated':continue
        x={'id':r['backend']+'__'+r['id'],'case_id':r['case_id'],'role':'formal_raw','path':r['raw_path'],'sha256':r['raw_sha256']}
        target=ROOT/'geometry'/r['case_id']/'bite_mask.png'
        if target.exists():x['target_mask']=str(target)
        images.append(x)
    out=ROOT/'formal/observer_frozen.json';assert not out.exists()
    out.write_text(json.dumps({'sam3':sam['sam3'],'images':images,'prompts':['fork','tofu','hand'],'confidence_threshold':.5,'minimum_mask_pixels':50,
        'purpose':'Advisory text-only segmentation and source negative controls; no automatic success assignment','script_sha256':hashlib.sha256((ROOT/'observe_complete_bite.py').read_bytes()).hexdigest()},indent=2)+'\n')
    print(json.dumps({'images':len(images),'config':str(out),'python':sam['python']}))


if __name__=='__main__':main()
