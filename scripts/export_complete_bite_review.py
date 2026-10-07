"""Convert only explicitly inspected, hash-bound assistant notes into the result table."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--notes',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    inv=json.loads((a.bundle/'formal/inventory.json').read_text());assert inv['status']=='complete'
    notes=json.loads(a.notes.read_text());key={(r['backend'],r['id']):r for r in notes['rows']};assert len(key)==len(notes['rows'])
    generated=[r for r in inv['cells'] if r['status']=='generated'];assert len(key)==len(generated),(len(key),len(generated))
    gallery_path=a.output.parent/'review_key.json'
    gallery={(r['backend'],r['id']):r['review_id'] for r in json.loads(gallery_path.read_text())} if gallery_path.exists() else {}
    rows=[]
    for cell in inv['cells']:
        row={'review_id':gallery.get((cell['backend'],cell['id']),'')}
        row.update({k:cell[k] for k in ['backend','id','case_id','seed','method','status']})
        if cell['status']=='generated':
            note=key[cell['backend'],cell['id']];assert note['raw_sha256']==cell['raw_sha256']
            assert len(note['ratings'])==6 and set(note['ratings'])<=set('PFU')
            for dim,val in zip(notes['dimension_order'],note['ratings']):row[dim]=notes['encoding'][val]
            row['notes']=note['notes'];row['raw_sha256']=cell['raw_sha256']
        else:
            row.update({d:'no_output' for d in notes['dimension_order']});row['notes']=cell.get('reason',cell['status']);row['raw_sha256']=''
        rows.append(row)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'reviewed_generated':len(key),'intended_cells':len(rows),'independent_human_raters':0,'output':str(a.output)}))


if __name__=='__main__':main()
