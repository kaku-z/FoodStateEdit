"""Summarize recorded assistant diagnostics, retaining the evaluation boundary."""
import argparse,hashlib,json,os,statistics
from pathlib import Path

FIELDS=['lift','bite_and_support','source_change','food_and_scene','hand_and_utensil']
HTML='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>第一口实验 · 完整结果</title>
<style>body{margin:0;background:#eff3f6;color:#20313c;font:16px system-ui}main{max-width:1550px;margin:25px auto;padding:20px}h1{font-size:30px}p{line-height:1.6}.badge{background:#fff3c8;padding:12px;border-left:4px solid #cd9c1c}.controls{display:flex;gap:12px;flex-wrap:wrap;padding:15px 0;position:sticky;top:0;background:#eff3f6;z-index:2}select,button{font:inherit;padding:10px;background:white;border:1px solid #aab9c1;border-radius:5px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:15px}figure{margin:0;background:white;border-radius:8px;padding:10px}img{width:100%;height:48vh;object-fit:contain}figcaption{padding:8px}table{border-collapse:collapse;background:white}th,td{padding:9px 16px;border:1px solid #c4cfd5;text-align:center}.links{line-height:2}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:15px;font-size:14px}@media(max-width:520px){.pair{grid-template-columns:1fr}img{height:42vh}}</style>
<main><h1>第一口入口前 · 64 格完整实验</h1>
<p>8 张真实源照片 × 4 种方法 × 2 个固定 seed。每格仅生成一次，无挑图、无最终图像合成。左侧为原始照片的固定预处理输入，右侧为模型 raw 输出。</p>
<p class="badge">表中评分来自助手的非盲开发诊断，不能替代三名独立真人评价。独立评分收到 0 份；未见测试尚未解锁。</p>
<details><summary>查看 64 格方法统计（助手诊断）</summary><div id="summary"></div><p>抬升只指画面中的空间现象；夹错食物仍不算完整成功。原批次有 3 个取食区域标注缺陷，修正复核另行报告。</p></details><div class="controls"><select id="case"></select><select id="method"></select><select id="seed"></select><button id="prev">上一格</button><button id="next">下一格</button></div>
<div class="pair"><figure><a id="sourceLink" target="_blank"><img id="source" alt="真实源图"></a><figcaption>真实源照片（预处理输入）</figcaption></figure><figure><a id="resultLink" target="_blank"><img id="result" alt="模型原生输出"></a><figcaption id="caption"></figcaption></figure></div>
<details><summary>查看本格的助手诊断与输出哈希</summary><pre id="diagnostic"></pre></details>
<h2>不筛选的逐图总览</h2><div class="links" id="grids"></div>
<p>方法 A：同一完整任务的连续文字；B：同样信息分项表达；C：B 加二维位置示意；D：B 加相对三维/接触示意。引导图是第二张参考图，不是硬性三维约束。</p>
<p>“第一口”是预设动作条件，单张图不能证明此前无人吃过；高度以手工容器尺度为相对单位，没有毫米真值。豆腐、米饭、面条、汤各两张，不能视为大规模泛化结果。</p></main>
<script>const DATA=__DATA__;const cases=DATA.cases,methods=DATA.methods,seeds=DATA.seeds;const el=id=>document.getElementById(id);
for(const c of cases)el('case').add(new Option(c.id,c.id));for(const m of methods)el('method').add(new Option(m,m));for(const s of seeds)el('seed').add(new Option(s,s));
function update(){const key=el('case').value+'__'+el('method').value+'__'+el('seed').value,r=DATA.rows.find(x=>x.run_id===key),c=cases.find(x=>x.id===el('case').value);el('source').src=c.source;el('sourceLink').href=c.source;el('result').src=r.raw;el('resultLink').href=r.raw;el('caption').textContent=key;el('diagnostic').textContent=JSON.stringify(r.rating,null,2)}
for(const id of ['case','method','seed'])el(id).onchange=update;
function step(d){const ids=DATA.rows.map(x=>x.run_id),key=el('case').value+'__'+el('method').value+'__'+el('seed').value;let i=(ids.indexOf(key)+d+ids.length)%ids.length;let [c,m,s]=ids[i].split('__');el('case').value=c;el('method').value=m;el('seed').value=s;update()}
el('prev').onclick=()=>step(-1);el('next').onclick=()=>step(1);
let table='<p>助手诊断：每种方法 16 张，所有条件共同满足才记完整通过。</p><table><tr><th>方法</th><th>明显抬升</th><th>来源变化</th><th>场景保持</th><th>完整通过</th></tr>';
for(const m of methods){const r=DATA.summary[m];table+='<tr><td>'+m+'</td><td>'+r.pass.lift+'/16</td><td>'+r.pass.source_change+'/16</td><td>'+r.pass.food_and_scene+'/16</td><td>'+r.joint_pass+'/16</td></tr>'}el('summary').innerHTML=table+'</table>';
for(const c of cases){let a=document.createElement('a');a.href=c.grid;a.textContent=c.id+'：原图与全部 8 个结果';a.target='_blank';el('grids').append(a,document.createElement('br'))}update();</script></html>'''


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--review-dir',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];bundle=root/'outputs/first_bite_20260928_bundle_v1'
    c=json.loads((bundle/'config.json').read_text());review=json.loads((root/'results/FIRST_BITE_ASSISTANT_REVIEW_20260928.json').read_text())
    expected=[f"{case['case_id']}__{m}__{s}" for case in c['cases'] for m in c['methods'] for s in c['seeds']]
    ratings={r['run_id']:r for r in review['ratings']}
    if len(ratings)!=len(review['ratings']) or set(ratings)!=set(expected):raise ValueError('Missing or duplicate assistant diagnostics')
    data={'cases':[],'methods':c['methods'],'seeds':c['seeds'],'rows':[],'summary':{}}
    summary={'scope':'Assistant non-blind development diagnostic, not independent human evaluation','calls':64,
             'independent_ballots_received':0,'formal_unseen_unlocked':False,'methods':{},'families':{},'technical':{}}
    rel=lambda p:Path(os.path.relpath(p,a.review_dir)).as_posix()
    for case in c['cases']:
        data['cases'].append({'id':case['case_id'],'source':rel(bundle/case['files']['source.png']['path']),
                              'grid':case['case_id']+'_all.png'})
    for name in expected:
        r=ratings[name];paths=list((root/'outputs/first_bite_20260928_gp40').glob('shard_*/'+name+'/raw.png'))
        if len(paths)!=1 or sha(paths[0])!=r['raw_sha256']:raise ValueError('Output mismatch')
        if any(r[f] not in ['pass','fail','uncertain'] for f in FIELDS):raise ValueError('Incomplete rating')
        joint=all(r[f]=='pass' for f in FIELDS) and r['photo_score']>=4
        if joint!=r['joint_pass']:raise ValueError('Incorrect conjunction')
        data['rows'].append({'run_id':name,'raw':rel(paths[0]),'rating':r})
    def stats(rows):
        return {'n':len(rows),'pass':{f:sum(r[f]=='pass' for r in rows) for f in FIELDS},
                'uncertain':{f:sum(r[f]=='uncertain' for r in rows) for f in FIELDS},
                'joint_pass':sum(r['joint_pass'] for r in rows),'photo_at_least_4':sum(r['photo_score']>=4 for r in rows)}
    for m in c['methods']:
        rows=[r for r in ratings.values() if r['run_id'].split('__')[1]==m];s=stats(rows)
        s['cases_with_a_joint_success']=[x['case_id'] for x in c['cases'] if any(r['joint_pass'] and r['run_id'].split('__')[0]==x['case_id'] for r in rows)]
        s['internal_engineering_gate_met']=s['joint_pass']>=12 and len(s['cases_with_a_joint_success'])==8
        summary['methods'][m]=s;data['summary'][m]=s
    for f in ['cohesive','granular','strand','liquid']:
        summary['families'][f]=stats([r for r in ratings.values() if r['run_id'].startswith(f+'_')])
    summary['overall']=stats(list(ratings.values()))
    runs=[json.loads(p.read_text()) for p in (root/'outputs/first_bite_20260928_gp40').glob('shard_*/run_manifest.json')]
    durations=[r['duration_seconds'] for m in runs for r in m['completed']]
    summary['technical']={'completed':len(durations),'failed':sum(len(m['failed']) for m in runs),
        'model_calls':sum(m['model_calls'] for m in runs),'median_generation_seconds':statistics.median(durations),
        'generation_seconds_sum':sum(durations),'config_sha256':sha(bundle/'config.json')}
    summary['technical']['per_method_generation_seconds']={method:{
        'median':statistics.median([r['duration_seconds'] for m in runs for r in m['completed'] if r['method']==method]),
        'sum':sum(r['duration_seconds'] for m in runs for r in m['completed'] if r['method']==method)} for method in c['methods']}
    (root/'results/FIRST_BITE_ASSISTANT_SUMMARY_20260928.json').write_text(json.dumps(summary,indent=2))
    (a.review_dir/'index.html').write_text(HTML.replace('__DATA__',json.dumps(data,ensure_ascii=False)),encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':main()
