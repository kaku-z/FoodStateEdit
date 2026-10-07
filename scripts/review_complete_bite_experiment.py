"""Create complete galleries, source-locality diagnostics and an unfilled review sheet."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import random


def main():
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);b=a.bundle
    inv=json.loads((b/'formal/inventory.json').read_text());rows=inv['cells']
    def local(remote):return b/remote.split('/first_bite_complete_20260929/',1)[1]
    metrics=[];review=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    dims=['lift','utensil_support','matching_source_notch','scene_identity','no_person','photographic_realism']
    blinded=rows.copy();random.Random(592029).shuffle(blinded)
    code={(r['backend'],r['id']):f'R{i+1:03d}' for i,r in enumerate(blinded)}
    for r in rows:
        rid=code[r['backend'],r['id']];cid=r['case_id']
        review.append(dict(review_id=rid,backend=r['backend'],id=r['id'],case_id=cid,seed=r['seed'],method=r['method'],status=r['status'],**{d:'' for d in dims},notes=''))
        if r['status']!='generated':continue
        path=local(r['raw_path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==r['raw_sha256']
        out=np.asarray(Image.open(path).convert('RGB'),dtype=np.float64)
        src=np.asarray(Image.open(b/'inputs'/cid/'source.png').convert('RGB'),dtype=np.float64)
        mp=b/'geometry'/cid/'edit_mask.png'
        if not mp.exists():mp=b/'formal'/('fallback_'+cid)/'edit_mask.png'
        outside=np.asarray(Image.open(mp).convert('L'))==0
        delta=out-src;mse=float(np.mean(delta[outside]**2));mae=float(np.mean(np.abs(delta[outside])))
        metrics.append(dict(review_id=rid,backend=r['backend'],id=r['id'],case_id=cid,seed=r['seed'],method=r['method'],outside_pixels=int(outside.sum()),outside_mae_255=mae,outside_psnr_db=float(10*np.log10(255**2/max(mse,1e-12))),whole_image_mae_255=float(np.mean(np.abs(delta))),raw_sha256=r['raw_sha256']))
    (a.output/'locality_metrics.json').write_text(json.dumps({'scope':'Raw outputs; source locality only, not task success or realism','rows':metrics},indent=2)+'\n')
    template=a.output/'review_template.csv'
    if not template.exists():
        with template.open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(review[0]));w.writeheader();w.writerows(review)
    blinded_template=a.output/'human_review_blinded.csv'
    if not blinded_template.exists():
        with blinded_template.open('w',newline='',encoding='utf-8-sig') as f:
            fields=['review_id']+dims+['notes']
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            w.writerows({k:r[k] for k in fields} for r in sorted(review,key=lambda r:r['review_id']))
    (a.output/'review_key.json').write_text(json.dumps(review,indent=2)+'\n')
    for backend in ['qwen','vace']:
        for cid in sorted(set(r['case_id'] for r in rows)):
            sheet=Image.new('RGB',(1280,1110),'#eeeeee');draw=ImageDraw.Draw(sheet)
            draw.text((10,5),f'{backend} / {cid} | rows: source + geometry, then seeds 41, 163, 907 | columns: A B C D',font=font,fill='black')
            for i,n in enumerate(['source.png','planar_control.png','rgb_control.png']):
                path=b/('inputs' if i==0 else 'geometry')/cid/n
                if path.exists():sheet.paste(Image.open(path).convert('RGB').resize((320,240)),(i*320,35))
            for si,seed in enumerate([41,163,907]):
                for mi,method in enumerate(['A_direct','B_planar','C_rgb3d','D_staged3d']):
                    r=next(r for r in rows if r['backend']==backend and r['case_id']==cid and r['seed']==seed and r['method']==method)
                    x,y=mi*320,290+si*270
                    draw.text((x+3,y),f'{code[backend,r["id"]]} {method} s{seed}',font=font,fill='black')
                    if r['status']=='generated':sheet.paste(Image.open(local(r['raw_path'])).convert('RGB').resize((320,240)),(x,y+25))
                    else:draw.text((x+5,y+100),r['status'],font=font,fill='red')
            sheet.save(a.output/f'{backend}_{cid}.jpg',quality=94)
    import os
    def rel(path):return Path(os.path.relpath(path,a.output)).as_posix()
    cards=[]
    for r in blinded:
        rid=code[r['backend'],r['id']];src=b/'inputs'/r['case_id']/'source.png'
        result=f'<img src="{html.escape(rel(local(r["raw_path"])),quote=True)}">' if r['status']=='generated' else f'<p class="failure">{r["status"]}</p>'
        cards.append(f'<article data-code="{rid}"><h2>{rid}</h2><div class="pair"><img src="{html.escape(rel(src),quote=True)}">{result}</div><details><summary>Reveal method</summary>{html.escape(r["backend"]+" / "+r["id"])}</details></article>')
    page='''<!doctype html><meta charset="utf-8"><title>Complete first-bite experiment — all cells</title><style>body{font:16px system-ui;background:#13161b;color:#eee;margin:24px}article{background:#222731;padding:16px;margin:20px 0;border-radius:10px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}.pair img{width:100%}summary{cursor:pointer}h2{font-size:18px}.failure{color:#ffa8a8}header{max-width:1000px}input{font:inherit}</style><header><h1>Complete first-bite experiment</h1><p>Left: real source photograph. Right: raw generated result. All 192 intended cells are retained, including preprocessing failures. Random display order; method hidden by default. This viewer has no completed independent human ratings.</p><p>Judge visible lift, utensil support, matching source notch, scene identity, absence of people, and photographic realism separately. Uncertain is not a pass. Image appearance does not prove physical 3-D correctness.</p><label>Find review ID <input id="find" placeholder="R001"></label></header>'''+''.join(cards)+'''<script>document.querySelector('#find').oninput=e=>document.querySelectorAll('article').forEach(a=>a.hidden=!a.dataset.code.includes(e.target.value.toUpperCase()))</script>'''
    (a.output/'all_results.html').write_text(page,encoding='utf-8')
    print(json.dumps({'cells':len(rows),'generated':len(metrics),'sheets':16,'output':str(a.output)}))


if __name__=='__main__':main()
