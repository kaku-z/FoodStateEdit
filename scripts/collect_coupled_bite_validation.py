"""Collect all frozen endpoints and execute declared, auditable layer composition."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    for i in range(8):
        try:return json.loads(p.read_text())
        except OSError as e:
            if e.errno!=116 or i==7:raise
            time.sleep(.3)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--archive',action='store_true');a=ap.parse_args()
    root=a.root;v=root/'validation';cfg=read(v/'frozen.json');expected=read(v/'expected_cells.json')
    found={};workers=[]
    for p in sorted(v.glob('worker_*/manifest.json')):
        m=read(p);workers.append({k:m.get(k) for k in ['pid','status','current','step','error','updated_unix','started_unix','finished_unix']})
        for row in m['completed']:
            folder=p.parent/row['id']
            for name,h in row['files'].items():assert sha(folder/name)==h,(row['id'],name)
            found[row['id']]=dict(row,raw_path=str(folder/'raw.png'),raw_sha256=sha(folder/'raw.png'))
    out=v/'results';out.mkdir(exist_ok=True);rows=[]
    for cell in expected:
        row=dict(cell)
        if row['status']=='preprocessing_failed':rows.append(row);continue
        if not all(k in found for k in row['dependencies']):
            row['status']='technical_failure' if any(w['status']=='technical_failure' for w in workers) else 'pending'
            rows.append(row);continue
        d=out/row['id'];d.mkdir(exist_ok=True)
        components=[found[k] for k in row['dependencies']]
        if row['composited']:
            prep=Path(row['prepared']);src=np.asarray(Image.open(prep/'source.png').convert('RGB'))
            tr=read(prep/'transforms.json');final=src.astype(float);union=np.zeros(src.shape[:2],bool);steps=[]
            # Fixed order gives the source-removal state final ownership.
            for kind,component,mask_name in [('target',components[1],'target_composite_mask'),('hole',components[0],'hole_edit')]:
                x0,y0,x1,y1=tr[kind]['box_xyxy']
                local=np.asarray(Image.open(component['raw_path']).convert('RGB').resize((x1-x0,y1-y0),Image.Resampling.LANCZOS))
                generated=src.copy();generated[y0:y1,x0:x1]=local
                mask=np.asarray(Image.open(prep/(mask_name+'.png')))>0
                # A matte may be expanded, but never asks a crop for pixels it did not synthesize.
                crop_support=np.zeros_like(mask);crop_support[y0:y1,x0:x1]=True;mask &= crop_support
                alpha=np.clip(distance_transform_edt(mask)/cfg['composition']['feather_pixels'],0,1)
                final=final*(1-alpha[...,None])+generated*alpha[...,None];union|=mask
                steps.append({'region':kind,'component':component['id'],'raw_sha256':component['raw_sha256'],
                    'mask_sha256':sha(prep/(mask_name+'.png')),'crop':tr[kind]})
            final=np.uint8(np.round(final));assert np.array_equal(final[~union],src[~union])
            path=d/'layered.png'
            if path.exists():assert np.array_equal(np.asarray(Image.open(path)),final)
            else:Image.fromarray(final).save(path)
            audit={'composited':True,'steps':steps,'outside_source_exact':True,'outside_pixels':int((~union).sum()),
                'raw_generator_output':False,'physical_correspondence_guaranteed':False}
            (d/'composition.json').write_text(json.dumps(audit,indent=2)+'\n')
        else:
            path=Path(components[0]['raw_path'])
        row.update(status='generated',output_path=str(path),output_sha256=sha(path),
            raw_components=[{'id':x['id'],'path':x['raw_path'],'sha256':x['raw_sha256']} for x in components],
            sum_component_seconds=sum(x['seconds'] for x in components))
        rows.append(row)
    complete=all(x['status'] in ['generated','preprocessing_failed','technical_failure'] for x in rows)
    inv={'status':'complete' if complete else 'running','created_unix':time.time(),'cells':rows,'raw_results':list(found.values()),
         'workers':workers,'counts':{s:sum(x['status']==s for x in rows) for s in sorted({x['status'] for x in rows})},
         'expected_raw_calls':len(cfg['jobs']),'completed_raw_calls':len(found)}
    (v/'inventory.json').write_text(json.dumps(inv,indent=2)+'\n')
    print(json.dumps({k:inv[k] for k in ['status','counts','completed_raw_calls','expected_raw_calls','workers']}))
    if a.archive:
        assert complete
        archive=root/'validation_results.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
            for folder in ['validation','inputs','geometry_v4']:
                for p in sorted((root/folder).rglob('*')):
                    if p.is_file() and p.suffix not in ['.npz','.zip'] and '__pycache__' not in p.parts:z.write(p,p.relative_to(root))
            for p in sorted(root.glob('*.py')):z.write(p,p.name)
            for p in sorted((root/'logs').glob('validation*.log')):z.write(p,p.relative_to(root))
        receipt={'size_bytes':archive.stat().st_size,'sha256':sha(archive),'archive':archive.name}
        (root/'validation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
