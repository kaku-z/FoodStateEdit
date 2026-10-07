"""Build the actual MLD3 source/rigid/deformation/final gallery and QA sheets."""
import argparse
import html
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def h(value):
    return html.escape(str(value))


def fmt(value, digits=5):
    return '未测量' if value is None else f'{value:.{digits}g}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(); out = root/'evaluation'; out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((out/'integrated_summary.json').read_text(encoding='utf-8'))
    case_root = Path(summary['input_case_root'])
    review_path = root/'visual_review.json'
    review = json.loads(review_path.read_text(encoding='utf-8')) if review_path.exists() else {'notes': {}, 'observer': '待实际逐张检查'}
    notes = review['notes']; rows = summary['cases']
    complete_review = all(row['case_id'] in notes for row in rows)
    independent_link = '<a href="independent_final_visual_review.json">独立最终视觉与像素检查记录</a>' if (out/'independent_final_visual_review.json').exists() else ''
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>MLD3 完整算法与真实图片实验</title><style>body{font:16px/1.65 system-ui,sans-serif;background:#f6f7f9;color:#202a35;margin:0}main{max-width:1520px;margin:auto;padding:24px}h1{font-size:29px}h2{font-size:23px;margin-top:36px}p{max-width:1100px}section{background:white;border:1px solid #d8dee5;border-radius:10px;padding:18px;margin:24px 0}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.grid img{width:100%;height:auto;display:block;background:#e8edf2}.grid figure{margin:0}.grid figcaption{font-weight:600;margin:5px 0}.note{padding:10px 14px;background:#fff6d8;border-left:4px solid #d2a329}.data{font:14px/1.6 ui-monospace,monospace;overflow:auto;background:#edf3f8;padding:9px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:7px;border-bottom:1px solid #dce3eb;text-align:left}.muted{color:#647283}a{color:#155ca2}.links{display:flex;gap:14px;flex-wrap:wrap}details{margin-top:14px}@media(max-width:800px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}main{padding:12px}}</style><main>',
        '<h1>MLD3：材料身份约束的食物变形与接触算法</h1>',
        f'<p>本图册保留全部 {len(rows)} 张真实来源图的刚性、完整变形、最终外观与接触/重力消融。算法操纵同一来源材料状态；来源图没有实测编辑后真值。本轮沿用此前已看过的图像，因此属于探索验证。</p>',
        '<p>流程：源图观测锚定 → 不确定背面闭合 → 源材料分配 → 固体网格或可见曲线变形 → 勺面接触和重力 → 来源纹理运输 → 仅餐具外观投影。MLD2 权重只提供学习到的来源先验，Qwen 只提供可选金属外观候选。</p>',
        '<p class="note">身份、UV、分配预算和背景像素不变属于构造一致性；应变、体积与接触距离属于建模代理。它们不能证明真实隐藏形状、物理质量或照片级真实感。视觉缺陷必须读逐例记录。</p>',
        f'<p>根代理实际逐张检查记录：{len([r for r in rows if r["case_id"] in notes])}/{len(rows)}；检查者：{h(review["observer"])}；完成全部检查：{h(complete_review)}。</p>',
        f'<p class="links"><a href="integrated_summary.json">完整指标与状态 JSON</a><a href="integrated_cases.csv">逐例与消融 CSV</a><a href="../visual_review.json">根代理视觉检查记录</a>{independent_link}<a href="../algorithm.md">算法文档</a></p>',
        '<h2>量化结果及其范围</h2><table><tr><th>量</th><th>全部来源图宏平均</th><th>解释</th></tr>']
    metric_rows = [('分配丢失材料 ID', summary['allocation_macro'].get('missing_source_ids', {}), '源状态中的身份分配，不是真实质量测量'),
        ('源 UV 身份最大差', summary['allocation_macro'].get('source_uv_identity_max_difference', {}), '纹理坐标持久性'),
        ('纹理双线性来源移除域所有权', summary['allocation_macro'].get('source_texture_bilinear_owned_removed_fraction', {}), '检查所有非零权重来源 texel；来源食材语义仍是假设'),
        ('实际最终编辑像素占比', summary['image_macro'].get('actual_final_changed_fraction', {}), '所有像素的实际 RGB 差异'),
        ('最终图域外 MAE /255', summary['image_macro'].get('final_outside_edit_mae_255', {}), '域外背景保护'),
        ('完整算法边应变绝对均值', summary['variant_macro']['deformed'].get('edge_strain_abs_mean', {}), '源重建网格相对边长变化'),
        ('完整算法代理体积漂移', summary['variant_macro']['deformed'].get('mesh_volume_relative_drift', {}), '仅闭合且正确朝向网格具有通常的体积解释'),
        ('实际导出餐具最大穿透', summary['variant_macro']['deformed'].get('actual_spoon_mesh_max_penetration_bite_extent', {}), '从实际 PLY 上表面三角形独立计算；不是实测物理接触'),
        ('完整算法形变 / 第一口尺度', summary['variant_macro']['deformed'].get('shape_deformation_mean_bite_extent', {}), '减去整体沉降后的相对顶点位移'),
        ('完整算法整体沉降 / 第一口尺度', summary['variant_macro']['deformed'].get('centroid_settling_displacement_bite_extent', {}), '与同动作刚性基线的质心位移'),
        ('去重力与完整算法位移差', summary['ablation_macro'].get('full_vs_no_gravity_vertex_mean_bite_extent', {}), '包含整体沉降'),
        ('去重力与完整算法形状差', summary['ablation_macro'].get('full_vs_no_gravity_shape_difference_mean_bite_extent', {}), '扣除整体平移后的重力作用'),
        ('去接触与完整算法位移差', summary['ablation_macro'].get('full_vs_no_contact_vertex_mean_bite_extent', {}), '大值主要来自自由坠落'),
        ('去接触与完整算法形状差', summary['ablation_macro'].get('full_vs_no_contact_shape_difference_mean_bite_extent', {}), '扣除整体平移后的接触作用')]
    for name, metric, scope in metric_rows:
        parts.append(f'<tr><td>{h(name)}</td><td>{fmt(metric.get("mean"))} (n={metric.get("n", 0)})</td><td>{h(scope)}</td></tr>')
    parts.append('</table>')
    for row in rows:
        case_id = row['case_id']; path = os.path.relpath(case_root/case_id, out).replace(os.sep, '/')+'/'
        parts.append(f'<section id="{h(case_id)}"><h2>{h(case_id)} · {h(row["branch"])}</h2><div class="grid">')
        for filename, label in [('source.png', '原图'), ('rigid.png', '同一材料 · 刚性基线'),
                                ('deformed.png', '完整变形与接触'), ('final.png', '最终约束外观')]:
            parts.append(f'<figure><a href="{path}{filename}"><img loading="lazy" src="{path}{filename}" alt="{h(case_id+" "+label)}"></a><figcaption>{h(label)}</figcaption></figure>')
        parts.append('</div>')
        note = notes.get(case_id, '尚未由根代理实际逐张检查；自动指标不能代替照片检查。')
        if isinstance(note, dict):
            note = json.dumps(note, ensure_ascii=False)
        parts.append(f'<p class="note">{h(note)}</p>')
        g = row['variants']['deformed']; a = row['allocation']; im = row['image']
        parts.append('<p class="data">'+h(f'分配缺失={a["missing_source_ids"]}；重复分配={a["overlap_ids"]}；'
            f'UV差={fmt(a["source_uv_identity_max_difference"])}；平均边应变={fmt(g["edge_strain_abs_mean"])}；'
            f'表面体积代理漂移={fmt(g["mesh_volume_relative_drift"])}；去整体沉降后的形变={fmt(g["shape_deformation_mean_bite_extent"])}；'
            f'背景MAE={fmt(im["final_outside_edit_mae_255"])}')+'</p>')
        if row['branch'] == 'strand':
            parts.append('<p class="data">'+h(f'实际管状表面边应变={fmt(g["edge_strain_abs_mean"])}；'
                f'恢复中心线应变={fmt(g["centerline_edge_strain_abs_mean"])}；'
                f'半径相对差={fmt(g["ring_radius_relative_change_abs_mean"])}；'
                f'中心线锥台体积代理漂移={fmt(g["centerline_frustum_volume_proxy_relative_drift"])}；'
                f'可见食物像素={im["visible_carried_food_pixels"]}。中心线约束不等于表面无应变或语义选取正确。')+'</p>')
        parts.append('<details><summary>接触与重力消融、代理量和源状态</summary><div class="grid">')
        for filename, label in [('no_contact.png', '去接触'), ('no_gravity.png', '去重力'), ('edit_mask.png', '实际声明编辑域')]:
            parts.append(f'<figure><a href="{path}{filename}"><img loading="lazy" src="{path}{filename}" alt="{h(label)}"></a><figcaption>{h(label)}</figcaption></figure>')
        parts.append(f'</div><p class="links"><a href="{path}state.json">本例状态与参数</a><a href="{path}transport.npz">同身份坐标运输数组</a></p><pre class="data">{h(json.dumps(row["variants"], ensure_ascii=False, indent=2))}</pre></details></section>')
    parts.append('<p class="muted">旧轮次失败和训练记录保留在 material_lineage_full_20261004；本轮没有删除先前负结果，也没有把构造性质当作物理真值。</p></main></html>')
    (out/'gallery.html').write_text('\n'.join(parts), encoding='utf-8')

    font = ImageFont.load_default(size=20)
    small = ImageFont.load_default(size=17)
    for start in range(0, len(rows), 4):
        block = rows[start:start+4]
        canvas = Image.new('RGB', (1440, len(block)*300+45), '#edf1f5'); draw = ImageDraw.Draw(canvas)
        for c, label in enumerate(('Source', 'Rigid transport', 'Deformation + contact', 'Constrained appearance')):
            draw.text((c*360+10, 8), label, fill='#182c42', font=font)
        for r, row in enumerate(block):
            for c, filename in enumerate(('source.png', 'rigid.png', 'deformed.png', 'final.png')):
                im = Image.open(case_root/row['case_id']/filename).convert('RGB')
                im.thumbnail((350, 260)); x=c*360+(360-im.width)//2; y=r*300+45+(260-im.height)//2
                canvas.paste(im, (x, y))
            draw.text((12, r*300+308), row['case_id']+' / '+row['branch'], fill='#1c3449', font=small)
        canvas.save(out/f'comparison_{start:02d}_{start+len(block)-1:02d}.jpg', quality=93)
    (out/'gallery_manifest.json').write_text(json.dumps({'case_count': len(rows),
        'visually_reviewed_count': len([r for r in rows if r['case_id'] in notes]),
        'all_cases_actually_reviewed': complete_review, 'gallery': str(out/'gallery.html'),
        'scope': 'Actual output gallery; no automated photographic-perfection rating.'}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out/'gallery.html')


if __name__ == '__main__':
    main()
