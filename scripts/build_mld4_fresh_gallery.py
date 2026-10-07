"""Build a Chinese, all-case gallery from audited fresh regression outputs."""
import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path


LABELS={'source_material_identity':'源材料相容','visible_lift':'明显离盘',
        'food_spoon_support':'餐具承托','source_removal':'原位置减少',
        'texture_and_seams':'纹理与接缝','no_added_humans':'无新增人物',
        'handle_frame_connection':'勺柄伸出画面'}
VERDICTS={'pass':'通过','fail':'失败','uncertain':'不确定'}


def build(root):
    audit=json.loads((root/'audit.json').read_text(encoding='utf-8'))
    reviews=json.loads((root/'visual_reviews.json').read_text(encoding='utf-8'))
    visual={r['case_id']:r for r in reviews['cases']}
    cards=[]
    for row in audit['cases']:
        cid=row['case_id']; v=visual.get(cid); prefix='assets/'+cid+'/'
        badges=''; condition='pending'; note='尚未完成直接视觉审查。'
        if v:
            condition='fail' if 'fail' in v['criteria'].values() else 'uncertain' if 'uncertain' in v['criteria'].values() else 'pass'
            note=v['note']
            badges=''.join(f'<span class="badge {value}">{LABELS[key]} · {VERDICTS[value]}</span>' for key,value in v['criteria'].items())
        if row['status']!='complete':
            cards.append(f'<article data-condition="pending"><h2>{cid}</h2><p>{escape(row["status"])} · {escape(str(row.get("error") or row.get("stage")))}</p></article>')
            continue
        figures=''.join(f'<figure><button class="image" data-src="{prefix+name}" aria-label="放大：{label}"><img loading="lazy" src="{prefix+name}" alt="{cid} {label}"></button><figcaption>{label}</figcaption></figure>' for label,name in [('真实原图','source.png'),('最终 B：源纹理约束','final.png')])
        detail=''.join(f'<figure><button class="image" data-src="{prefix+name}" aria-label="放大：{label}"><img loading="lazy" src="{prefix+name}" alt="{cid} {label}"></button><figcaption>{label}</figcaption></figure>' for label,name in [('原位置 · 输入','source_removal.png'),('原位置 · 最终','final_removal.png'),('选中的源材料','selected_reference.png'),('勺头 · 最终','final_head.png')])
        checks='通过' if row['all_checks'] else '存在失败项'
        cards.append(f'''<article data-condition="{condition}"><div class="case-title"><h2>{cid}</h2><span>输入提示：{escape(row['food_prompt'])}</span></div>
<div class="pair">{figures}</div><div class="badges">{badges}</div><p class="note">{escape(note)}</p>
<details><summary>查看源缺口、勺头和 A/B 消融</summary><div class="details-grid">{detail}</div>
<a href="{prefix}comparison.jpg" target="_blank">打开完整 3 × 3 对比图</a> · <a href="{prefix}A.png" target="_blank">生成 A</a> · <a href="{prefix}status.json">运行记录</a>
<p class="small">实现检查：{checks}；编辑区外变更 {row['pixels']['changed_outside_edit']} 像素；显式绑定核心 {row['pixels']['bound_core']} 像素。以上是代码约束，不是摄影效果评分。</p></details></article>''')
    counts={k:Counter(v['criteria'][k] for v in visual.values()) for k in LABELS}
    table=''.join(f'<tr><td>{LABELS[k]}</td><td>{counts[k]["pass"]}</td><td>{counts[k]["fail"]}</td><td>{counts[k]["uncertain"]}</td></tr>' for k in LABELS)
    entire=Counter('fail' if 'fail' in v['criteria'].values() else 'uncertain' if 'uncertain' in v['criteria'].values() else 'pass' for v in visual.values())
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MLD4 · 原图完整回归</title>
<style>*{box-sizing:border-box}body{margin:0;background:#edf1f6;color:#152338;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1160px;margin:auto;padding:36px 24px 80px}h1{font-size:34px;margin:0 0 14px}h2{font-size:20px;margin:0}.intro{max-width:850px}.eyebrow{color:#45617e;font-weight:650;letter-spacing:.08em}.stats{display:flex;flex-wrap:wrap;gap:12px;margin:24px 0}.stat,article,.summary{background:white;border:1px solid #d9e1ec;border-radius:14px;padding:22px}.stat{min-width:150px}.stat strong{display:block;font-size:30px}.small{font-size:13px;color:#4c5f75}.case-title{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:16px}.case-title span{font-size:14px;color:#526680}article{margin:22px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}figure{margin:0;min-width:0}figcaption{color:#526680;font-size:14px;margin-top:5px}.image{width:100%;border:0;padding:0;background:#eef1f5;cursor:zoom-in;border-radius:8px;overflow:hidden}.pair img{width:100%;height:340px;object-fit:contain;display:block}.details-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:15px 0}.details-grid img{width:100%;height:190px;object-fit:contain}.badges{display:flex;flex-wrap:wrap;gap:7px;margin-top:17px}.badge{font-size:12px;border-radius:20px;padding:4px 10px}.pass{background:#e0f2e9;color:#176344}.fail{background:#fbe6e6;color:#9d2b33}.uncertain{background:#fff1d7;color:#785217}.note{margin:12px 0;color:#324a65}details{border-top:1px solid #e2e7ef;padding-top:12px}summary{cursor:pointer;color:#345477}a{color:#1d5aa1}table{border-collapse:collapse;width:100%;max-width:650px}td,th{text-align:left;border-bottom:1px solid #e2e7ef;padding:8px 15px 8px 0}nav{display:flex;gap:8px;flex-wrap:wrap;margin-top:25px}nav button{border:1px solid #bacbdc;border-radius:20px;padding:8px 18px;background:white;color:#203f61;cursor:pointer}nav button[aria-pressed=true]{background:#203f61;color:white}dialog{max-width:96vw;max-height:95vh;border:0;border-radius:12px;padding:14px;background:#f3f6fb}dialog::backdrop{background:#081321ce}dialog img{max-width:90vw;max-height:82vh;display:block}dialog button{float:right;border:0;padding:8px 15px;background:white;cursor:pointer}[hidden]{display:none!important}@media(max-width:650px){main{padding:24px 12px}h1{font-size:27px}.pair{gap:7px}.pair img{height:220px}.details-grid{grid-template-columns:1fr 1fr}article{padding:14px}.stat{min-width:120px}}</style>
<main><div class="eyebrow">MLD4 / 2026-10-05 / GP40</div><h1>从真实原图重新执行完整算法</h1>
<p class="intro">同一入口、同一规则，重新运行分割、深度估计、材料几何、动作规划、语义路由和图像生成。全部 24 张此前已用于观察或开发，本页是回归实验，不能视作独立盲测。</p>
<div class="stats">STATISTICS</div><section class="summary"><strong>如何读这些结果</strong><p>最终 B 在可信可见区域使用源纹理，隐藏面和原位置缺口依赖生成。勺柄延伸在三维阶段完成。逐像素保护与来源检查通过，不代表整张照片真实。</p><p>以下评级来自助手逐张看图，是 AI 视觉筛查；没有人工受试者、同一食物真实前后配对图或称重/扫描真值。</p><table><thead><tr><th>视觉项目</th><th>通过</th><th>失败</th><th>不确定</th></tr></thead><tbody>TABLE</tbody></table><p class="small">七项全部通过：ALLPASS；有明确失败：ANYFAIL；其余含不确定：UNCERTAIN。只计已审查的 REVIEWED 张，不进行选优替换。</p></section>
<nav aria-label="案例筛选"><button data-filter="all" aria-pressed="true">全部 24 张</button><button data-filter="fail" aria-pressed="false">含明确失败</button><button data-filter="uncertain" aria-pressed="false">含不确定</button><button data-filter="pass" aria-pressed="false">七项通过</button></nav>
<section id="cases">CARDS</section><p class="small">冻结协议：PROTOCOL<br><a href="audit.json">实现审计 JSON</a> · <a href="visual_reviews.json">逐例视觉记录</a> · <a href="protocol.json">完整协议</a></p></main>
<dialog><button id="close">关闭 ×</button><img alt="放大实验图片"></dialog><script>const modal=document.querySelector('dialog');document.querySelectorAll('.image').forEach(b=>b.addEventListener('click',()=>{modal.querySelector('img').src=b.dataset.src;modal.showModal()}));document.querySelector('#close').onclick=()=>modal.close();modal.addEventListener('click',e=>{if(e.target===modal)modal.close()});document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{document.querySelectorAll('[data-filter]').forEach(x=>x.setAttribute('aria-pressed',x===b));document.querySelectorAll('article').forEach(x=>x.hidden=b.dataset.filter!=='all'&&x.dataset.condition!==b.dataset.filter)});</script></html>'''
    stats=''.join(f'<div class="stat"><strong>{n}</strong>{label}</div>' for n,label in [(audit['complete'],'完整运行'),(audit['failed'],'执行失败'),(len(visual),'已逐张审查'),(audit['pending'],'尚在运行')])
    for token,value in dict(STATISTICS=stats,TABLE=table,ALLPASS=str(entire['pass']),ANYFAIL=str(entire['fail']),UNCERTAIN=str(entire['uncertain']),REVIEWED=str(len(visual)),CARDS=''.join(cards),PROTOCOL=audit['protocol_sha256']).items():
        page=page.replace(token,value)
    (root/'gallery.html').write_text(page,encoding='utf-8')
    (root/'visual_summary.json').write_text(json.dumps(dict(cases=len(visual),all_criteria=dict(entire),criteria={k:dict(v) for k,v in counts.items()},scope='AI visual screen of previously observed regression data'),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(root/'gallery.html')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evaluation',type=Path,required=True)
    build(parser.parse_args().evaluation)
