"""Verify all 36 supplemental cells and make unselected before/after galleries."""
import hashlib,json,os
from pathlib import Path
from PIL import Image,ImageDraw


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def grid(cells,path,cols,width=600,height=520):
    canvas=Image.new('RGB',(cols*width,((len(cells)+cols-1)//cols)*height),'#edf1f5');d=ImageDraw.Draw(canvas)
    for i,(label,p) in enumerate(cells):
        im=Image.open(p).convert('RGB');im.thumbnail((width,height-30),Image.Resampling.LANCZOS)
        x=i%cols*width;y=i//cols*height;canvas.paste(im,(x+(width-im.width)//2,y+(height-30-im.height)//2));d.text((x+8,y+height-22),label,fill='black')
    canvas.save(path)


def main():
    root=Path(__file__).resolve().parents[1];b=root/'outputs/first_bite_20260928_supplement_bundle_v2'
    c=json.loads((b/'config.json').read_text());by_id={x['case_id']:x for x in c['cases']};found={};manifests=[]
    for directory in sorted((root/'outputs/first_bite_20260928_supplement_gp40').glob('supplement_shard_*_v2')):
        mp=directory/'run_manifest.json';m=json.loads(mp.read_text());assert m['status']=='complete_unreviewed' and not m['failed']
        assert m['config_sha256']==sha(b/'config.json');manifests.append({'path':str(mp),'sha256':sha(mp),'run':m})
        for row in m['completed']:
            assert row['name'] not in found
            for name,h in row['files'].items():assert sha(directory/row['name']/name)==h
            case=by_id[row['case_id']];assert case['output_size']==row['raw_size']
            assert hashlib.sha256(case['prompts'][row['method']].encode()).hexdigest()==row['prompt_sha256']
            found[row['name']]={'record':row,'raw':directory/row['name']/'raw.png'}
    expected={f"{j['case_id']}__{j['method']}__{j['seed']}" for j in c['jobs']};assert set(found)==expected and len(found)==36
    for case in c['cases']:
        for f in case['files'].values():assert sha(b/f['path'])==f['sha256']
    out=root/'outputs/first_bite_20260928_supplement_review';out.mkdir(exist_ok=False)
    rel=lambda p:Path(os.path.relpath(p,out)).as_posix();items=[];height_items=[]
    for case in c['cases']:
        if case['experiment_role']!='roi_repair':continue
        cells=[('SOURCE',b/case['files']['source.png']['path'])];paired=[]
        for method in c['methods']:
            for seed in [281,913]:
                name=f"{case['case_id']}__{method}__{seed}";raw=found[name]['raw']
                old=next((root/'outputs/first_bite_20260928_gp40').glob('shard_*/'+name+'/raw.png'))
                cells.append((f'{method} seed {seed}',raw));paired.extend([(f'ORIGINAL {method} {seed}',old),(f'REPAIRED {method} {seed}',raw)])
                items.append({'id':name,'source':rel(b/case['files']['source.png']['path']),'old':rel(old),'repaired':rel(raw),'raw_sha256':sha(raw)})
        grid(cells,out/(case['case_id']+'_repair_all.png'),3)
        grid(paired,out/(case['case_id']+'_before_after.png'),2,width=600,height=480)
    for family in ['cohesive','granular','strand','liquid']:
        cases=[x for x in c['cases'] if x['experiment_role']=='height' and x['family']==family]
        assert len(cases)==3 and len({x['prompts']['D_geometry_guide'] for x in cases})==1
        assert len({x['files']['source.png']['sha256'] for x in cases})==1
        cells=[('SOURCE',b/cases[0]['files']['source.png']['path'])]
        for case in cases:
            raw=found[case['case_id']+'__D_geometry_guide__2027']['raw'];cells.append((f'h={case["diagnostic_height"]} seed 2027',raw))
            height_items.append({'case_id':case['case_id'],'height':case['diagnostic_height'],'raw':rel(raw),'raw_sha256':sha(raw)})
        grid(cells,out/(family+'_height_all.png'),2,width=800,height=640)
    data={'repair':items,'height':height_items};(out/'gallery_data.json').write_text(json.dumps(data,indent=2))
    html='''<!doctype html><meta charset="utf-8"><title>标注修正与高度诊断</title><style>body{font:16px system-ui;margin:25px;background:#eff3f6;color:#21313e}p{line-height:1.6}select{font:inherit;padding:10px;width:100%;max-width:900px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}img{width:100%;height:48vh;object-fit:contain;background:white}figure{margin:8px 0}a{line-height:2}@media(max-width:900px){.grid{grid-template-columns:1fr 1fr}.grid figure:first-child{grid-column:1/-1}.grid figure:first-child img{height:24vh}.grid figure:not(:first-child) img{height:38vh}}</style><h1>24 格标注修正 + 12 格高度诊断</h1><p>这是发现主实验输入缺陷后的开发复核，不替换原 64 格失败记录，也不是独立人评。左右结果使用同一 seed、同一源图和推理设置；只修正取食位置表示。</p><select id="item"></select><div class="grid"><figure><a id="sourceLink" target="_blank"><img id="source" alt="真实源图"></a><figcaption>真实源图</figcaption></figure><figure><a id="oldLink" target="_blank"><img id="old" alt="原标注结果"></a><figcaption>原标注结果</figcaption></figure><figure><a id="repairedLink" target="_blank"><img id="repaired" alt="修正标注结果"></a><figcaption>修正标注结果</figcaption></figure></div><p id="hash"></p><h2>完整总览</h2>__LINKS__<script>const DATA=__DATA__;const s=document.getElementById('item');for(const row of DATA.repair)s.add(new Option(row.id,row.id));function show(){const r=DATA.repair.find(x=>x.id===s.value);for(const id of ['source','old','repaired']){document.getElementById(id).src=r[id];document.getElementById(id+'Link').href=r[id]}document.getElementById('hash').textContent='修正 raw SHA256: '+r.raw_sha256}s.onchange=show;show();</script>'''
    links=''.join(f'<a href="{p.name}" target="_blank">{p.stem}</a><br>' for p in sorted(out.glob('*.png')))
    (out/'index.html').write_text(html.replace('__LINKS__',links).replace('__DATA__',json.dumps(data)),encoding='utf-8')
    (out/'verification.json').write_text(json.dumps({'status':'36_SUPPLEMENT_OUTPUTS_VERIFIED_NOT_HUMAN_RATED','calls':36,
        'roi_repair_calls':24,'height_calls':12,'original_64_unchanged':True,'independent_ballots_received':0,'manifests':manifests},indent=2))
    print(json.dumps({'verified':36,'roi_repair':24,'height':12,'assistant_review':'not assigned by this script'}))


if __name__=='__main__':main()
