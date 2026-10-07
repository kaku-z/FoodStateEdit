"""Verify every raw output and build complete case grids and blind review forms.

This script assigns no aesthetic/action scores. Empty ballots are not reviews.
"""
import argparse,base64,csv,hashlib,io,json,random
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def thumb(path,max_size=(640,640)):
    with Image.open(path) as im:
        im=im.convert('RGB');im.thumbnail(max_size,Image.Resampling.LANCZOS)
        buf=io.BytesIO();im.save(buf,format='JPEG',quality=91)
    return 'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()


HTML='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>第一口图像评价</title>
<style>body{font:17px system-ui;margin:20px;background:#f1f3f5;color:#182331}header{position:sticky;top:0;background:#f1f3f5;padding:10px 0;z-index:2}button,input,select{font:inherit;padding:7px;margin:4px}main{max-width:1500px;margin:auto}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}.pair img{width:100%;height:60vh;object-fit:contain;background:#ddd;cursor:zoom-in}.row{display:flex;justify-content:space-between;padding:5px;border-bottom:1px solid #ccc;gap:20px}p{line-height:1.5}.note{font-size:14px;color:#52616b}textarea{width:97%;min-height:65px;font:inherit}#status{color:#164d68}label{display:inline-block}h2{font-size:20px}</style>
<main><header><b>第一口入口前图像评价</b> <span id="counter"></span><br>
<label>评阅者编号 <input id="rater" placeholder="请填写本人编号"></label>
<button onclick="nav(-1)">上一张</button><button onclick="nav(1)">下一张</button><button onclick="download()">导出本人评分 JSON</button><span id="status"></span></header>
<p>同时查看原图与编辑图。评价一口食物是否被真实托起、尚未入口；平移、餐具放在盘面上均不算抬升。点击图片可放大。方法和 seed 已隐藏。所有评分只保存在本机浏览器；请导出保存。</p>
<div class="pair"><section><h2>原始照片</h2><img id="source" onclick="zoom(this.src)"></section><section><h2>待评价图 <span id="item"></span></h2><img id="result" onclick="zoom(this.src)"></section></div>
<p id="material"></p><div id="questions"></div><p>失败原因或不确定之处：</p><textarea id="notes" onchange="save()"></textarea>
<p class="note">完整通过：同一评阅者对五项均选通过，照片真实感至少 4 分。无法判断计入不通过，但另行记录。不需要猜测精确质量或毫米高度。该表不能替代另一位真实评阅者的独立判断。</p></main>
<script>const DATA=__DATA__;let idx=0;const KEY='first_bite_'+DATA.package;
let state;try{state=JSON.parse(localStorage.getItem(KEY))||{rater:'',ratings:{}}}catch(e){state={rater:'',ratings:{}}}
const fields=[['lift','明显抬升：负载和餐具工作端离开原支撑面'],['bite_and_support','一口份量、工具接触和数量合理'],['source_change','来源变化符合材料规则且与取食一致'],['food_and_scene','食物身份、其他食物和场景保持'],['hand_and_utensil','手部、握持与餐具形状自然']];
document.getElementById('rater').value=state.rater;document.getElementById('rater').onchange=save;
function zoom(src){const w=window.open();if(w){w.document.write('<img style="max-width:100%" src="'+src+'">')}}
function save(){let r={};for(const [key] of fields)r[key]=document.getElementById(key)?.value||'';r.photo_score=document.getElementById('photo_score')?.value||'';r.notes=document.getElementById('notes').value;r.reviewed_at=new Date().toISOString();state.rater=document.getElementById('rater').value;state.ratings[DATA.items[idx].id]=r;localStorage.setItem(KEY,JSON.stringify(state));document.getElementById('status').textContent='已在本机保存';}
function show(){const item=DATA.items[idx],r=state.ratings[item.id]||{};document.getElementById('counter').textContent=(idx+1)+' / '+DATA.items.length;document.getElementById('item').textContent=item.id;document.getElementById('source').src=DATA.sources[item.case];document.getElementById('result').src=item.image;document.getElementById('material').textContent=DATA.rules[item.family];let q=document.getElementById('questions');q.innerHTML='';for(const [key,label] of fields){let row=document.createElement('div');row.className='row';let text=document.createElement('span');text.textContent=label;let s=document.createElement('select');s.id=key;for(const [v,t] of [['','未评分'],['pass','通过'],['fail','不通过'],['uncertain','无法判断']]){let o=new Option(t,v);s.add(o)}s.value=r[key]||'';s.onchange=save;row.append(text,s);q.append(row)}let row=document.createElement('div');row.className='row';let t=document.createElement('span');t.textContent='照片真实感（1=明显不真实，5=像真实摄影）';let s=document.createElement('select');s.id='photo_score';s.add(new Option('未评分',''));for(let n=1;n<=5;n++)s.add(new Option(String(n),String(n)));s.value=r.photo_score||'';s.onchange=save;row.append(t,s);q.append(row);document.getElementById('notes').value=r.notes||'';document.getElementById('status').textContent='';}
function nav(d){save();idx=Math.max(0,Math.min(DATA.items.length-1,idx+d));show()}
function download(){save();const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({package:DATA.package,...state},null,2)],{type:'application/json'}));a.download=DATA.package+'_ratings.json';a.click();URL.revokeObjectURL(a.href)}show();</script></html>'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    c=json.loads((a.bundle/'config.json').read_text());expected={f"{case['case_id']}__{m}__{s}" for case in c['cases'] for m in c['methods'] for s in c['seeds']}
    found={};manifests=[]
    for directory in sorted(a.runs.glob('shard_*_v1')):
        mp=directory/'run_manifest.json';m=json.loads(mp.read_text())
        if m['status']!='complete_unreviewed':raise ValueError(f'Incomplete shard {directory}')
        if m['config_sha256']!=sha(a.bundle/'config.json'):raise ValueError('Different input configs')
        for cell in m['completed']:
            if cell['name'] in found:raise ValueError('Duplicate cell')
            for name,h in cell['files'].items():
                if sha(directory/cell['name']/name)!=h:raise ValueError('Output hash mismatch')
            case=next(x for x in c['cases'] if x['case_id']==cell['case_id'])
            if cell['raw_size']!=case['output_size']:raise ValueError('Output dimensions changed')
            if hashlib.sha256(case['prompts'][cell['method']].encode()).hexdigest()!=cell['prompt_sha256']:raise ValueError('Prompt changed')
            found[cell['name']]={'record':cell,'path':directory/cell['name']/'raw.png'}
        manifests.append({'path':str(mp),'sha256':sha(mp),'run':m})
    if set(found)!=expected:raise ValueError(f'Missing={expected-set(found)}, unexpected={set(found)-expected}')
    for case in c['cases']:
        for f in case['files'].values():
            if sha(a.bundle/f['path'])!=f['sha256']:raise ValueError('Input hash changed')
    a.output.mkdir(parents=True,exist_ok=False)
    for case in c['cases']:
        cells=[('SOURCE',a.bundle/case['files']['source.png']['path'])]
        cells += [(f'{method} seed {seed}',found[f"{case['case_id']}__{method}__{seed}"]['path']) for method in c['methods'] for seed in c['seeds']]
        canvas=Image.new('RGB',(1440,1320),'#f2f4f6');d=ImageDraw.Draw(canvas)
        for i,(label,path) in enumerate(cells):
            im=Image.open(path).convert('RGB');im.thumbnail((480,410),Image.Resampling.LANCZOS)
            x=i%3*480;y=i//3*440
            canvas.paste(im,(x+(480-im.width)//2,y));d.text((x+10,y+417),label,fill='black')
        canvas.save(a.output/(case['case_id']+'_all.png'))
    rules={'cohesive':'固体：一口小块应留下对应缺口/断面，不是搬走整块主体。',
           'granular':'米饭：一小勺与浅凹/局部重排对应；原有勺子应被拿起而不是复制。',
           'strand':'面条：两根筷子夹起一口束，自然下垂，允许尾部与碗中相连。',
           'liquid':'汤：勺碗盛液并高于原液面；不要求永久缺口，不应生成不自然的坑。'}
    source_images={case['case_id']:thumb(a.bundle/case['files']['source.png']['path']) for case in c['cases']}
    thumbnails={name:thumb(item['path'],(1152,1152)) for name,item in found.items()}
    blind_keys=[]
    for slot in [1,2,3]:
        order=sorted(found);random.Random(20260928+slot).shuffle(order);items=[]
        for index,name in enumerate(order):
            record=found[name]['record'];cid=record['case_id'];family=next(x['family'] for x in c['cases'] if x['case_id']==cid)
            blind_id=f'P{slot}-{index+1:03d}'
            items.append({'id':blind_id,'case':cid,'family':family,'image':thumbnails[name]})
            blind_keys.append({'package':f'first_bite_reviewer_{slot}','blind_id':blind_id,'run_id':name,'raw_sha256':sha(found[name]['path'])})
        data={'package':f'first_bite_reviewer_{slot}','items':items,'sources':source_images,'rules':rules}
        (a.output/f'blind_review_{slot}.html').write_text(HTML.replace('__DATA__',json.dumps(data,ensure_ascii=False)),encoding='utf-8')
    (a.output/'PRIVATE_blind_keys.json').write_text(json.dumps(blind_keys,indent=2))
    rows=[]
    for name in sorted(found):
        cell=found[name]['record'];rows.append({'run_id':name,'case_id':cell['case_id'],'method':cell['method'],'seed':cell['seed'],
                                               'raw_sha256':sha(found[name]['path']),'assistant_lift':'','assistant_joint':'','notes':''})
    with (a.output/'assistant_review_blank.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (a.output/'verification.json').write_text(json.dumps({'status':'64_NATIVE_OUTPUTS_HASH_VERIFIED_EVALUATION_PENDING','calls':len(found),
                  'source_images':len(c['cases']),'manifests':manifests,'independent_ballots_received':0,
                  'no_automatic_action_success_assigned':True,'all_method_seed_cells_retained':True},indent=2))
    print(json.dumps({'verified_raw_images':len(found),'case_grids':8,'blank_blind_packages':3}))


if __name__=='__main__':main()
