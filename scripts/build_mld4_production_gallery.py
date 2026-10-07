"""Build a portable full-result gallery, preserving development/holdout separation."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path


LABELS={'source_material_identity':'选中材质','visible_lift':'明显抬升','food_spoon_support':'食物与餐具接触',
        'source_removal':'原位缺口','texture_and_seams':'纹理与接缝','no_added_humans':'未新增人物',
        'handle_frame_connection':'勺柄与画外连接'}
STATUS={'pass':'通过','fail':'失败','uncertain':'不确定','pending':'待审查'}


def build(a):
    r=a.evaluation.resolve();audit=json.loads((r/'audit.json').read_text(encoding='utf-8'))
    reviewfile=r/'visual_reviews.json';review=json.loads(reviewfile.read_text(encoding='utf-8')) if reviewfile.exists() else {'cases':[]}
    reviews={x['case_id']:x for x in review['cases']}
    groups=[];counts={};criterion_counts={}
    for group,title in [('development','开发集 · 16 张'),('frozen_new_source','新来源保留集 · 8 张')]:
        cards=[];counter=Counter();by_criterion={k:Counter() for k in LABELS}
        for row in [x for x in audit['cases'] if x['group']==group]:
            cid=row['case_id'];v=reviews.get(cid,{});criteria=v.get('criteria',{})
            for key in LABELS:by_criterion[key][criteria.get(key,{}).get('status','pending')]+=1
            statuses=[c.get('status','pending') if isinstance(c,dict) else c for c in criteria.values()]
            screen='fail' if 'fail' in statuses else ('uncertain' if 'uncertain' in statuses else ('pass' if len(statuses)==len(LABELS) and set(statuses)=={'pass'} else 'pending'))
            counter[screen]+=1;asset='assets/'+cid
            panels=''.join(f'<figure><a href="{asset}/{f}" target="_blank"><img loading="lazy" src="{asset}/{f}" alt="{html.escape(cid+label)}"></a><figcaption>{label}</figcaption></figure>' for label,f in [('原始真实照片','source.png'),('完整基础生成','base.png'),('统一协议最终图','final.png')])
            td=[]
            for key,label in LABELS.items():
                c=criteria.get(key,{'status':'pending','note':'尚未直接查看实际图像'})
                if isinstance(c,str):c={'status':c,'note':''}
                status=c['status'];td.append(f'<tr><th>{label}</th><td><span class="tag {status}">{STATUS[status]}</span></td><td>{html.escape(c.get("note",""))}</td></tr>')
            synopsis=html.escape(v.get('summary','等待对实际图像的独立审查；像素边界检查不代表视觉成功。'))
            pixels=row['pixel_diagnostics'];evidence=html.escape(json.dumps({'pixel_diagnostics':pixels,'all_provenance_checks':row['all_provenance_checks'],'route':row['route'],'final_sha256':row['artifacts']['final.png']},ensure_ascii=False,indent=2))
            cards.append(f'''<article id="{cid}" data-screen="{screen}"><header><h3>{cid}</h3><span class="tag {screen}">{STATUS[screen]}</span><span class="muted">自动描述：{html.escape(row['automatic_semantic_label'])}（非真值）</span></header><p>{synopsis}</p><div class="triptych">{panels}</div><details><summary>查看局部放大、逐项判断与证据</summary><a href="{asset}/comparison.jpg" target="_blank"><img class="comparison" loading="lazy" src="{asset}/comparison.jpg" alt="选中材质、原位缺口、抬升局部的并列比较"></a><table><tbody>{''.join(td)}</tbody></table><p class="muted">源区来自完整基础生成，本轮只精修抬升食物及餐具头部。最终图使用统一 unprojected 输出，不逐例挑选。</p><details><summary>文件与像素审计</summary><pre>{evidence}</pre><a href="{asset}/head_request.json">生成请求</a> · <a href="{asset}/state.json">状态与 SHA256</a></details></details></article>''')
        counts[group]=dict(counter)
        criterion_counts[group]={k:dict(c) for k,c in by_criterion.items()}
        groups.append(f'<section id="{group}"><h2>{title}</h2><p class="muted">实际完成 {sum(counter.values())} 张；视觉筛查：通过 {counter["pass"]}，失败 {counter["fail"]}，不确定 {counter["uncertain"]}，待审查 {counter["pending"]}。</p>{"".join(cards)}</section>')
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MLD4 · 全部24例真实图片实验</title><style>
*{box-sizing:border-box}body{margin:0;background:#f1f4f7;color:#172332;font:16px/1.6 system-ui,-apple-system,"Segoe UI","Microsoft Yahei",sans-serif}main{max-width:1500px;margin:auto;padding:30px}h1{font-size:30px;margin:0}h2{padding-top:32px;margin-bottom:4px}h3{font-size:20px;margin:0}p{margin:10px 0}.intro{padding:24px;border-radius:12px;background:#152b42;color:#fff}.intro a{color:#b7deff}.muted{color:#617080;font-size:14px}.intro .muted{color:#c0d0dc}nav{display:flex;gap:20px;margin-top:18px;flex-wrap:wrap}article{background:#fff;padding:20px;border-radius:12px;border:1px solid #d9e1e8;margin:20px 0}article header{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.triptych{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}figure{margin:0;background:#f4f6f8;border-radius:8px;padding:8px}figure img{display:block;width:100%;height:360px;object-fit:contain}figcaption{text-align:center;padding:8px}.tag{border-radius:5px;padding:2px 8px;font-size:13px;font-weight:600}.pass{background:#d9f1e3;color:#1c6c3e}.fail{background:#ffe0df;color:#9b2721}.uncertain{background:#ffedbf;color:#80550b}.pending{background:#e5e9ee;color:#586775}summary{cursor:pointer;padding:10px 0}.comparison{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #dce3e9;text-align:left;vertical-align:top}th{min-width:130px}pre{white-space:pre-wrap;font-size:12px;background:#f2f5f8;padding:12px}a{color:#175fa7}button{padding:8px 14px;border:1px solid #afbfce;background:white;border-radius:6px;cursor:pointer}.toolbar{position:sticky;top:0;background:#f1f4f7ee;padding:12px 0;z-index:2;display:flex;gap:10px;align-items:center}.hidden{display:none}@media(max-width:900px){main{padding:12px}.triptych{grid-template-columns:1fr}figure img{height:auto;max-height:500px}.intro{padding:16px}}
</style><main><div class="intro"><h1>全部 24 例真实图片实验</h1><p>目标：第一口食物由餐具托起、明显离开原位，同时保持选中食物的身份与自然外观。</p><p>统一冻结协议：基础生成 CFG 3 → 深度与材质锚点约束的头部精修 CFG 1；同一随机种子与参数，无 LoRA，无逐例挑图。</p><p class="muted">16 张开发图片与 8 张新来源保留图片分开报告。没有真实的对应“吃第一口后”照片、标定3D或实测质量；审查结果是单次视觉筛查。区域外0漂移只说明编辑边界得到保留，不证明食物或物理真实性。</p><nav><a href="#development">开发集</a><a href="#frozen_new_source">新来源保留集</a><a href="audit.json">审计JSON</a><a href="visual_reviews.json">逐图判断JSON</a></nav></div><div class="toolbar"><button onclick="filter('all')">显示全部</button><button onclick="filter('fail')">仅失败</button><button onclick="filter('uncertain')">仅不确定</button><span class="muted">点击照片查看原图；展开每例查看局部。</span></div>'''+''.join(groups)+'''<footer><p class="muted">另行完成的200步材质LoRA实验：已观察区域重建有小幅指标改善，但三张真实编辑对照没有明确任务收益，因此未用于这轮正式24例。</p></footer></main><script>function filter(s){document.querySelectorAll('article').forEach(x=>x.classList.toggle('hidden',s!=='all'&&x.dataset.screen!==s))}</script></html>'''
    summary_rows=[]
    for key,label in LABELS.items():
        cells=[]
        for group in ['development','frozen_new_source']:
            c=criterion_counts[group][key]
            cells.append(f'<td>{c.get("pass",0)} / {c.get("fail",0)} / {c.get("uncertain",0)}</td>')
        summary_rows.append(f'<tr><th>{label}</th>{"".join(cells)}</tr>')
    overview='<section><h2>逐项观察总览</h2><p>各格为通过 / 失败 / 不确定的图片数。由 AI 助手直接查看全部实际图片；不是人工受试者研究，也不代表总体成功率。材质相容不等于验证了同一物理食物。</p><table><thead><tr><th>判据</th><th>开发16张</th><th>新来源8张</th></tr></thead><tbody>'+''.join(summary_rows)+'</tbody></table></section>'
    document=document.replace('<section id="development">',overview+'<section id="development">',1)
    (r/'gallery.html').write_text(document,encoding='utf-8')
    (r/'visual_screen_counts.json').write_text(json.dumps({'counts':counts,'criterion_counts':criterion_counts,'interpretation':'Descriptive AI visual screen after direct inspection, not ground-truth task accuracy or human-rated study.'},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'gallery':str(r/'gallery.html'),'counts':counts},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);build(p.parse_args())
