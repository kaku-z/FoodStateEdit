"""Complete source-paired gallery and hash-bound, non-blinded diagnostic review pages."""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DIMS=['lift','utensil_support','matching_source_notch','scene_identity','no_person','photographic_realism']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--notes',type=Path);a=ap.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    b=a.bundle;inv=json.loads((b/'validation/inventory.json').read_text());cases=json.loads((b/'inputs/manifest.json').read_text())['cases']
    source={c['case_id']:c for c in cases}
    def local(remote):return b/remote.split('/first_bite_coupled_20260929/',1)[1]
    def crop(im,cid):
        prep=source[cid]['preprocessing'];x,y=prep['pad_left_top'];w,h=prep['resized'];return im.crop((x,y,x+w,y+h))
    seen=set()
    if a.notes and a.notes.exists():seen={r['id'] for r in json.loads(a.notes.read_text())['rows']}
    images=a.output/'views';images.mkdir(exist_ok=True)
    for c in cases:
        cid=c['case_id'];crop(Image.open(b/'inputs'/cid/'source.png').convert('RGB'),cid).save(images/('source_'+cid+'.png'))
    shuffled=inv['cells'].copy();random.Random(609293).shuffle(shuffled)
    ids={r['id']:f'R{i+1:03d}' for i,r in enumerate(shuffled)}
    rows=[];metrics=[]
    for r in inv['cells']:
        row={'review_id':ids[r['id']],**r,**{d:'' for d in DIMS},'notes':''};rows.append(row)
        if r['status']!='generated':continue
        p=local(r['output_path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['output_sha256']
        im=Image.open(p).convert('RGB');crop(im,r['case_id']).save(images/(r['id']+'.png'))
        cid=r['case_id'];src=np.asarray(Image.open(b/'inputs'/cid/'source.png').convert('RGB')).astype(float);out=np.asarray(im).astype(float)
        prep=b/'validation/prepared'/cid
        edit=(np.asarray(Image.open(prep/'hole_edit.png'))>0)|(np.asarray(Image.open(prep/'target_composite_mask.png'))>0)
        data=source[cid]['preprocessing'];x,y=data['pad_left_top'];w,h=data['resized'];valid=np.zeros(edit.shape,bool);valid[y:y+h,x:x+w]=True
        outside=valid&~edit;delta=out-src
        metrics.append({'id':r['id'],'outside_valid_source_pixels':int(outside.sum()),'outside_mae_255':float(np.abs(delta[outside]).mean()),
            'outside_max_difference':float(np.abs(delta[outside]).max()),'scope':'Source locality in original content rectangle, excluding deterministic letterbox.'})
    (a.output/'review_key.json').write_text(json.dumps(rows,indent=2)+'\n')
    (a.output/'locality_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    for filename,fields in [('review_template.csv',['review_id','id','case_id','method','seed','status']+DIMS+['notes']),
                            ('human_review_unfilled.csv',['review_id']+DIMS+['notes'])]:
        p=a.output/filename
        if not p.exists():
            with p.open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in rows)
    cards=[]
    for r in shuffled:
        result='<img loading="lazy" src="views/'+html.escape(r['id'])+'.png">' if r['status']=='generated' else '<p>'+r['status']+'</p>'
        rawlink=''
        if r['status']=='generated':
            full=os.path.relpath(local(r['output_path']),a.output).replace('\\','/')
            rawlink='<p><a style="color:#9cd2ff" href="'+html.escape(full)+'">查看完整 640×480 输出（含预处理补边区域）</a></p>'
        cards.append('<article data-id="'+ids[r['id']]+'"><h2>'+ids[r['id']]+'</h2><div class="pair"><img loading="lazy" src="views/source_'+r['case_id']+'.png">'+result+'</div>'+rawlink+'<details><summary>展开方法信息</summary>'+html.escape(r['id'])+(' — 分层合成输出' if r['composited'] else ' — 生成器原始输出')+'</details></article>')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>第一口修复实验：全部结果</title><style>body{font:16px system-ui;background:#151923;color:#eee;margin:24px}article{background:#242b38;padding:16px;margin:20px 0;border-radius:10px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.pair img{max-width:100%;max-height:650px;justify-self:center}summary{cursor:pointer}input{font:inherit}</style><h1>第一口修复实验：全部 72 个条件</h1><p>左侧为真实原图，右侧为该条件的最终输出。所有图片按原始内容范围去掉统一补边，原始 640×480 文件保留。部分方法包含明确记录的图层合成。方法名默认隐藏；尚无独立人类评分。</p><p>分别判断：明显抬升、餐具承托、对应缺口、场景保持、没有人物、照片真实感。不确定不算通过。外观无法证明真实三维体积守恒。</p><label>查找编号 <input id="search" placeholder="R001"></label>'''+''.join(cards)+'''<script>document.querySelector('#search').oninput=e=>document.querySelectorAll('article').forEach(a=>a.hidden=!a.dataset.id.includes(e.target.value.toUpperCase()))</script></html>'''
    (a.output/'all_results.html').write_text(page,encoding='utf-8')
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);pages=[]
    def panel(board,p,xy,label,cid):
        d=ImageDraw.Draw(board);x,y=xy;d.text((x+6,y+4),label,font=font,fill='black')
        im=crop(Image.open(p).convert('RGB'),cid);im.thumbnail((640,480));board.paste(im,(x+(640-im.width)//2,y+32))
    for c in cases:
        cid=c['case_id'];todo=[r for r in inv['cells'] if r['case_id']==cid and r['status']=='generated' and r['id'] not in seen]
        for i in range(0,len(todo),4):
            group=todo[i:i+4];board=Image.new('RGB',(1280,1560),'#eee')
            panel(board,b/'inputs'/cid/'source.png',(0,0),cid+' / REAL SOURCE',cid)
            panel(board,b/'geometry_v4'/cid/'rgb_control.png',(640,0),'GEOMETRY CONTROL (NOT OUTPUT)',cid)
            for j,r in enumerate(group):panel(board,local(r['output_path']),((j%2)*640,520+(j//2)*520),ids[r['id']]+' '+r['method']+' s'+str(r['seed']),cid)
            name=cid+'_'+str(i//4+1)+'.jpg';board.save(a.output/name,quality=96)
            pages.append({'page':name,'cells':[{'id':r['id'],'sha256':r['output_sha256']} for r in group]})
    (a.output/'pages.json').write_text(json.dumps(pages,indent=2)+'\n')
    print(json.dumps({'cells':len(rows),'generated':len(metrics),'review_pages':len(pages)}))


if __name__=='__main__':main()
