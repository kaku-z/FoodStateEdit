"""Build the final experiment report from completed, downloaded result records."""
from pathlib import Path
import argparse
import html
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


NAMES = ['shared_s41', 'shared_s163', 'shared_s907', 'no_image_s41']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def paired_interval(a, b, seed=20261004):
    difference = np.asarray(a, dtype=float)-np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.asarray([rng.choice(difference, len(difference), replace=True).mean()
                        for _ in range(2000)])
    return {'mean_difference': float(difference.mean()),
            'scene_bootstrap_95_percent_interval': np.quantile(draws, [.025, .975]).tolist(),
            'scene_count': len(difference), 'bootstrap_draws': len(draws)}


def table(headers, rows):
    head = ''.join(f'<th>{html.escape(str(x))}</th>' for x in headers)
    body = ''.join('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in row)+'</tr>'
                   for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def main(root):
    root = root.resolve()
    inputs = read(root/'analysis_inputs.json')
    output = root/'results'
    output.mkdir(exist_ok=True)
    if 'real_gallery_cases' in inputs:
        sections = []
        for entry in inputs['real_gallery_cases']:
            panels = []
            for label, image_path in entry['images']:
                relative = Path('..')/Path(image_path).relative_to(root)
                panels.append(f'<figure><figcaption>{html.escape(label)}</figcaption><a href="{relative.as_posix()}"><img src="{relative.as_posix()}" loading="lazy"></a></figure>')
            sections.append(f'<section><h2>{html.escape(entry["case_id"])}</h2><div class="panels">{"".join(panels)}</div><p>{html.escape(entry["notes"])}</p></section>')
        gallery = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>16张真实来源图 · 最终探索对照</title><style>
body{margin:0;background:#edf1f4;color:#172736;font:16px/1.6 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1380px;margin:auto;padding:24px}h1{font-size:28px}h2{font-size:18px;margin:0 0 12px}section{padding:18px;margin:20px 0;background:white;border-radius:12px}.panels{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}figure{margin:0}figcaption{font-size:14px;margin-bottom:8px}img{display:block;width:100%;height:260px;object-fit:contain;background:#fafafa}p{color:#526471}a{color:#116284}@media(max-width:850px){.panels{grid-template-columns:repeat(2,minmax(0,1fr))}img{height:230px}}
</style><main><h1>16张真实来源图：第一口抬升对照</h1><p>所有案例全部保留；最终结果属于探索实验，没有编辑后实拍真值。点击图片可打开原尺寸。</p><p><a href="report.html">完整实验报告、训练与数值结果</a></p>'''+''.join(sections)+'</main></html>'
        (output/'real_gallery.html').write_text(gallery, encoding='utf-8')
    results = {name: read(root/'runs_continuous'/name/'evaluation.json') for name in NAMES}
    receipts = {name: read(root/'runs_continuous'/name/'training_receipt.json') for name in NAMES}
    per_scene = {name: read(root/'runs_continuous'/name/'per_scene_metrics.json') for name in NAMES}
    # The serializer stores arrays, preserving original scene order for pairing.
    keep = np.asarray(per_scene[NAMES[0]]['scene_indices']) >= 7176
    statistics = {}
    for metric in ['source_iou', 'source_soft_mass_l1_normalized', 'id_soft_carried_l1_normalized']:
        a = np.asarray(per_scene['shared_s41'][metric])[keep]
        for control in ['no_image_s41', 'id_prior', 'id_shuffle']:
            if control.startswith('id_'):
                if metric != 'id_soft_carried_l1_normalized':
                    continue
                b = np.asarray(per_scene['shared_s41'][control+'_carried_l1_normalized'])[keep]
            else:
                b = np.asarray(per_scene[control][metric])[keep]
            statistics[metric+'_vs_'+control] = paired_interval(a, b)
    remainder = {name: {key: float(np.asarray(arrays[key])[keep].mean())
                        for key in results[name]['macro_per_scene']}
                 for name, arrays in per_scene.items()}
    reference = read(root/'evaluation/v1_reference_bbox_v2_summary.json')
    additional = {entry['name']: read(root/entry['evaluation'])
                  for entry in inputs.get('additional_runs', [])}
    training = read(root/'executed_training_summary.json')
    replay = read(root/'evaluation/replay_three_seed_summary.json')
    color_baseline = read(root/'evaluation/source_projection_color_baseline/summary.json')
    residual_truth = read(root/'evaluation/residual_truth_shared_s41_65k_full1024/summary.json')
    first_real = read(root/'real_base65k/real_results.json')
    final_real = read(root/'exploratory_metal_v4/projected_real_results.json')
    final_metal = read(root/'exploratory_metal_v4/metal_v4_results.json')
    summary = {'status': inputs['status'], 'inputs': inputs,
               'completed_runs': {name: {key: receipts[name][key] for key in
                                         ['status', 'actual_steps', 'actual_additional_updates',
                                          'resume_optimizer_steps', 'parameter_count']}
                                  for name in NAMES},
               'blind_synthetic_remainder_scene_count': int(keep.sum()),
               'blind_synthetic_remainder': remainder,
               'paired_scene_bootstrap': statistics,
               'additional_run_metrics': additional,
               'unique_completed_optimizer_updates': training['formal_main_unique_updates']+training['independent_control_unique_updates'],
               'replay': replay,
               'source_projection_color_baseline': color_baseline,
               'residual_truth_full1024': residual_truth,
               'first_formal_real': first_real,
               'final_exploratory_real': final_real,
               'final_metal_projection': final_metal,
               'interpretation': ['Synthetic ground truth tests reconstruction and allocation.',
                                  'Real unpaired photographs have no ground-truth after state or calibrated mass.',
                                  'Analytic partition budgets and rigid pose replay are construction properties.',
                                  'V1 reference changed training data; it is not an architecture-only ablation.']}
    (output/'analysis_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    rows = []
    for name in NAMES:
        v = results[name]; m = v['macro_per_scene']; u = v['uniform_test']
        rows.append([name, f"{m['source_iou']:.4f}", f"{u['occupancy_iou']:.4f}",
                     f"{m['source_soft_mass_l1_normalized']:.4f}",
                     f"{m['id_soft_carried_l1_normalized']:.4f}",
                     f"{v['visible_linear_rgb_mse']:.5f}"])
    for name, v in reference.items():
        m = v['aggregate']
        rows.append(['MLD1 '+name, f"{m['occupancy_iou']:.4f}", '—',
                     f"{m['source_mass_l1_per_source_volume']:.4f}",
                     f"{m['actions']['id']['mass_l1_per_source_volume']:.4f}",
                     f"{m['visible_linear_rgb_mse']:.5f}"])
    for name, v in additional.items():
        m = v['macro_per_scene']; u = v['uniform_test']
        rows.append([name, f"{m['source_iou']:.4f}", f"{u['occupancy_iou']:.4f}",
                     f"{m['source_soft_mass_l1_normalized']:.4f}",
                     f"{m['id_soft_carried_l1_normalized']:.4f}",
                     f"{v['visible_linear_rgb_mse']:.5f}"])
    families = sorted(results[NAMES[0]]['per_family'])
    family_rows = []
    for family in families:
        s = results['shared_s41']['per_family'][family]
        n = results['no_image_s41']['per_family'][family]
        u = results['shared_s41']['uniform_test']['per_family'][family]
        family_rows.append([family, s['scene_count'], f"{s['source_iou']:.4f}",
                            f"{u['occupancy_iou']:.4f}", f"{n['source_iou']:.4f}",
                            f"{s['id_soft_carried_l1_normalized']:.4f}"])

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    labels = ['Seed 41', 'Seed 163', 'Seed 907', 'No image']
    color = ['#246c8e']*3+['#b56b38']
    metrics = [('source_iou', 'Lattice occupancy IoU (macro)', False),
               ('id_soft_carried_l1_normalized', 'Carried mass L1 / true source volume', False),
               ('occupancy_iou', 'Continuous occupancy IoU (micro)', True)]
    for ax, (key, title, uniform) in zip(axes, metrics):
        values = [results[n]['uniform_test'][key] if uniform else results[n]['macro_per_scene'][key] for n in NAMES]
        bars = ax.bar(labels, values, color=color)
        ax.set_title(title); ax.tick_params(axis='x', labelrotation=25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'{value:.3f}', ha='center', va='bottom')
        ax.set_ylim(0, max(values)*1.2+.001)
    fig.suptitle('MLD2 completed runs · all 1,024 synthetic test identities', fontsize=13)
    fig.tight_layout(); fig.savefig(output/'synthetic_metrics.png', dpi=180); plt.close(fig)

    family_models = dict(results, **additional)
    family_names = list(family_models)
    matrices = [np.asarray([[family_models[name]['per_family'][family][key] for family in families]
                            for name in family_names])
                for key in ['source_iou', 'id_soft_carried_l1_normalized']]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax, matrix, label, cmap in zip(axes, matrices,
            ['Occupancy IoU / scene mean (higher is better)', 'Carried L1 / true source volume (lower is better)'],
            ['YlGnBu', 'YlOrRd']):
        picture = ax.imshow(matrix, aspect='auto', cmap=cmap)
        ax.set_xticks(range(len(families)), families, rotation=40, ha='right')
        ax.set_yticks(range(len(family_names)), family_names)
        ax.set_title(label)
        for row in range(len(family_names)):
            for column in range(len(families)):
                ax.text(column, row, f'{matrix[row,column]:.2f}', ha='center', va='center',
                        color='white' if picture.norm(matrix[row,column])>.65 else '#12202b', fontsize=9)
        fig.colorbar(picture, ax=ax, shrink=.8)
    fig.suptitle('All eight trained variants · 128 test identities per family; different training steps', fontsize=13)
    fig.tight_layout(); fig.savefig(output/'family_metrics.png', dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, key, multiplier, title in zip(axes,
            ['source_iou_macro', 'occupied_linear_rgb_mse_macro'], [1, 1000],
            ['DDIM minus posterior occupancy IoU (macro)', 'DDIM minus posterior RGB MSE (macro) × 1,000']):
        differences = [(residual_truth['ddim_draw_metric_mean_per_family'][family][key]-
                        residual_truth['methods'][0]['per_family'][family][key])*multiplier for family in families]
        colors = ['#367f70' if (x>0 if multiplier==1 else x<0) else '#bc7140' for x in differences]
        ax.barh(families, differences, color=colors); ax.axvline(0, color='#46535b', linewidth=.7)
        ax.set_title(title); ax.invert_yaxis()
    fig.suptitle('Three fixed 50-step draws · all 1,024 identities, 128 per family · no best-draw selection', fontsize=12)
    fig.tight_layout(); fig.savefig(output/'residual_truth_by_family.png', dpi=180); plt.close(fig)

    receipt_rows = [[entry['run'], entry['parent_optimizer_updates'], entry['new_optimizer_updates_in_continuation'],
                     entry['final_effective_optimizer_updates'], 'complete'] for entry in training['runs']]
    baseline = color_baseline['all_1024_test']
    color_rows = [['直接源图投影（无网络）', f"{baseline['occupied_linear_rgb_mse_micro']:.6f}",
                   f"{baseline['visible_linear_rgb_mse_micro']:.6f}"]]
    for name in NAMES:
        v = results[name]
        color_rows.append([name, f"{v['source']['occupied_linear_rgb_mse']:.6f}",
                           f"{v['visible_linear_rgb_mse']:.6f}"])
    replay_rows = []
    for entry in replay['runs']:
        construction = entry['construction']; query = entry['query_replay']
        replay_rows.append([entry['seed'], entry['independent']['total_scenes'],
                            f"{query['sample_512chunk']['max_abs']:.2e}",
                            f"{construction['per_id_soft_budget_max_abs']:.1e}",
                            f"{construction['inverse_pose_max_abs']:.2e}",
                            f"{entry['independent']['independent_cut_linear_rgb_mae_mean']:.5f}"])
    observation_rows = []
    for entry in training['runs']:
        if 'observations_validation' in entry:
            v = entry['observations_validation']; m = entry['validation_macro']
            observation_rows.append([entry['run'], f"{v['mask_iou']:.4f}", f"{v['depth_mae']:.4f}",
                                     f"{v['depth_bias']:+.4f}", f"{m['source_iou']:.4f}",
                                     f"{m['id_soft_carried_l1_normalized']:.4f}"])
    residual_rows = []
    for entry in residual_truth['methods']:
        v = entry['all_1024']; label = 'Posterior' if entry['method']=='posterior' else 'DDIM seed '+str(entry['sampling_seed'])
        residual_rows.append([label, f"{v['source_iou_macro']:.6f}", f"{v['occupied_linear_rgb_mse_macro']:.6f}",
                             f"{v['source_center_soft_volume_l1_normalized_macro']:.6f}",
                             f"{v['id_carried_center_soft_volume_l1_normalized_macro']:.6f}"])
    v = residual_truth['ddim_draw_metric_mean']
    residual_rows.append(['三次采样指标平均', f"{v['source_iou_macro']:.6f}", f"{v['occupied_linear_rgb_mse_macro']:.6f}",
                         f"{v['source_center_soft_volume_l1_normalized_macro']:.6f}",
                         f"{v['id_carried_center_soft_volume_l1_normalized_macro']:.6f}"])
    residual_intervals = [[key, f"{v['paired_scene_mean_delta']:+.6f}",
                           '['+', '.join(f'{x:+.6f}' for x in v['percentile_95_CI'])+']']
                          for key,v in residual_truth['paired_scene_bootstrap_2000']['metrics'].items()]
    real_cases = final_real['cases']
    real_rows = [['完整来源图', len(real_cases)],
                 ['归一化抬升（按源食物范围）', f"{final_real['all_case_macro_means']['normalized_lift']:.2f}"],
                 ['源/目标第一口像素重叠最大值', max(c['source_target_pixel_overlap'] for c in real_cases)],
                 ['局部深度修正后源UV最大变动 / 像素', f"{max(c['source_uv_projection_max_change'] for c in real_cases):.2e}"],
                 ['闭合且单连通匙', str(sum(c['final_spoon_watertight'] and c['final_spoon_connected_components']==1 for c in real_cases))+'/16'],
                 ['食物状态几何最大变动（v3对v2）', max(c['final_food_state_geometry_max_change'] for c in real_cases)],
                 ['最终食物对v3最大MAE / 255', final_metal['food_vs_v3_max_mae']],
                 ['最终缺口对v3最大MAE / 255', final_metal['cavity_vs_v3_max_mae']],
                 ['域外背景对源图最大MAE / 255', final_metal['context_vs_source_max_mae']],
                 ['可用金属候选', str(16-len(final_metal['candidate_failure_indices']))+'/16'],
                 ['Qwen候选占固定匙像素（按图平均）', f"{100*final_metal['mean_Qwen_metal_coverage_fraction']:.2f}%"],
                 ['隐藏厚度上界裁剪（按图平均）', f"{100*final_real['all_case_macro_means']['posterior_thickness_upper_clip_fraction']:.2f}%"]]
    comparison_rows = [[key, f"{value['mean_difference']:.5f}",
                        '['+', '.join(f'{x:.5f}' for x in value['scene_bootstrap_95_percent_interval'])+']',
                        value['scene_count']] for key, value in statistics.items()]
    artifact_blocks = []
    for entry in inputs['reviewed_artifacts']:
        path = Path(entry['path']); relative = Path('..')/path.relative_to(root)
        caption = html.escape(entry['description'])
        if path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            artifact_blocks.append(f'<figure><img src="{relative.as_posix()}" loading="lazy"><figcaption>{caption}</figcaption></figure>')
        else:
            artifact_blocks.append(f'<p><a href="{relative.as_posix()}">{caption}</a></p>')
    findings = ''.join(f'<li>{html.escape(x)}</li>' for x in inputs['findings'])
    composition = ''.join(f'<li>{html.escape(x)}</li>' for x in inputs['method_composition'])
    real_findings = ''.join(f'<li>{html.escape(x)}</li>' for x in inputs['real_findings'])
    variant_description = html.escape(inputs.get('variant_description', ''))
    hero = ''
    if 'hero' in inputs:
        entry = inputs['hero']; relative = Path('..')/Path(entry['path']).relative_to(root)
        hero = f'<figure><img src="{relative.as_posix()}"><figcaption>{html.escape(entry["description"])}</figcaption></figure>'
    gallery_link = ''
    if 'final_gallery' in inputs:
        relative = Path('..')/Path(inputs['final_gallery']).relative_to(root)
        gallery_link = f'<p><a href="{relative.as_posix()}">打开全部16张真实来源图的最终探索图册</a></p>'
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MLD2 完整实验 · 2026-10-04</title><style>
body{{margin:0;background:#edf1f4;color:#172736;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1320px;margin:28px auto;padding:32px;background:white;border-radius:14px}}
h1{{font-size:30px;line-height:1.3}}h2{{margin-top:32px;font-size:22px}}p,li{{max-width:1080px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:11px;border-bottom:1px solid #d9e1e6;text-align:left}}th{{background:#eaf2f6}}
.scroll{{overflow:auto}}img{{width:100%;height:auto;border:1px solid #dbe2e8}}figure{{margin:20px 0}}figcaption{{font-size:14px;color:#526471}}
code{{background:#e9eef2;padding:2px 4px}}a{{color:#116284}}.note{{padding:16px;background:#f6f2e9;border-left:4px solid #b88631}}
</style><main><h1>MLD2 完整实验结果</h1>
<p>{html.escape(inputs['conclusion'])}</p>{hero}{gallery_link}<ul>{findings}</ul>
<h2>本次实际运行的方法</h2><ol>{composition}</ol>
<h2>执行范围与修订</h2><p>8,192 个多拓扑合成场景分为 6,144 训练、1,024 验证、1,024 测试；8 类各有 128 个测试场景。
密集侧视开发诊断发现规则格点与 Fourier 特征混叠，加入每场景独立的连续随机点监督后续训。
前 8 个合成测试身份用于开发诊断，其余 1,016 个身份单独统计。真实图首轮固定 4 张开发、12 张测试，五路结果全部保存；查看全部首轮结果后开展的后续修正明确标为探索轮，不能再称盲测。没有真实编辑后配对图。</p>
{table(['运行', '恢复时累计步数', '本次续训更新数', '最终累计步数', '状态'], receipt_rows)}
<p>完成 8 组运行，共 310,000 次独立优化更新；继承父模型的更新只计一次，不计废弃烟雾测试。主模型含 5,598,994 个可训练参数，两个深度控制另加 75,650 个参数。主运行 55,000 步之后增加 10,000 步残差噪声预测训练，源场仍继续更新。8 份推理权重均已同步本地、核对哈希并实际加载。</p>
<h2>合成场景重建与分配</h2><p>格点 IoU 是逐场景平均；连续点 IoU 汇总独立随机点的交并集，二者不能混作同一指标。
质量误差为逐格绝对分配误差之和除以该场景真实源体积，再按场景平均；采用单位密度代理体积。
MLD1 是原始训练权重的迁移参考，训练数据也发生变化，不能解释成纯架构消融。</p>
<p>{variant_description}</p>
{table(['模型', '格点 IoU ↑', '连续点 IoU ↑', '源质量 L1 ↓', '搬运质量 L1 ↓', '可见线性 RGB MSE ↓'], rows)}
<img src="synthetic_metrics.png" alt="synthetic metrics">
{table(['类别', '测试数', '格点 IoU ↑', '连续点 IoU ↑', '无图格点 IoU ↑', '搬运质量 L1 ↓'], family_rows)}
<img src="family_metrics.png" alt="All variant family metrics">
<h2>直接颜色投影基线</h2><p>在同一组真实合成占据点／可见表面点评分。基线仅按已知相机坐标双线性采样输入 RGB 并转换为线性颜色，不输入真值材料或几何。
目标是无光照材料，输入包含光照；网络的优势包含去光照和内部材料预测。该结果不等价于真实食物背面颜色正确，也不能把源像素复制算作网络原创。</p>
{table(['方法', '占据点 RGB MSE（micro）↓', '可见表面 RGB MSE ↓'], color_rows)}
<h2>失败修正的验证结果</h2><p>低通特征和平衡全体积采样均没有消除细体侧视薄板。第一次深度／轮廓门控会通过移动预测表面降低几何损失；断开几何损失对观测头的梯度后，轮廓和深度偏差恢复，隐藏形状仍未恢复。
因此真实实验使用预定 shared_s41 的 65k 基础权重。各控制训练步数不同，表中如实列出，不作为等算力比较。</p>
{table(['验证模型', '轮廓 IoU ↑', '深度 MAE ↓', '深度有符号偏差', '几何格点 IoU ↑', '搬运质量 L1 ↓'], observation_rows)}
<h2>未用于开发的 1,016 个合成身份</h2><p>同一身份逐场景配对差值；区间为 2,000 次场景级 bootstrap 的 95% 分位数。
查询点不是独立样本，三个训练种子也不当作数千次重复。差值按“共享模型 − 对照”计算：IoU 为正较好，误差为负较好。</p>
{table(['配对比较', '平均差', '95% 场景 bootstrap 区间', '场景数'], comparison_rows)}
<h2>共享状态与回放</h2><p>三个训练种子各取 64 个场景、三个固定空间噪声种子、50 步 DDIM，改变分块和查询顺序验证同一状态能被复用。
共享状态的切面材料差为零；独立重新采样会改变同一切面。质量预算相等、材质复用和逆变换精度属于构造性质，不能当作真实物理准确度。</p>
{table(['训练种子', '场景数', 'DDIM 分块最大差', '逐身份预算最大差', '逆 SE(3) 最大差', '独立重采样切面 RGB MAE'], replay_rows)}
<h2>残差扩散是否提高真值重建</h2><p>额外完成全部 1,024 个测试场景、每场景 4,096 个相同中心点、三个固定采样种子、50 步 DDIM 的比较。
逐场景比较三次采样的指标平均，不选择最好样本，也不是先平均三个场再计算指标。总体 IoU、材料、中心体积和中心搬运误差都退步；面条类 IoU 从 0.0390 升到 0.0619，其余七类下降，八类颜色误差都增大。
最终真实管线采用确定性 posterior，不把当前扩散模块列为重建增益。</p>
<p>下面两列体积误差仅按中心点先求和再取绝对误差，分母为真实中心占据数；正式质量指标使用八子格与逐格绝对误差之和，两种指标不能混用。全部场景真实中心占据均非空。</p>
{table(['方法', 'IoU 宏平均 ↑', '占据 RGB MSE 宏平均 ↓', '中心源体积相对 L1 ↓', '中心搬运体积相对 L1 ↓'], residual_rows)}
<img src="residual_truth_by_family.png" alt="Residual truth by family">
<p>下表是“三次采样指标均值 − posterior”的场景级配对差；2,000 次 bootstrap 以场景为单位，三个 draw 不当作独立样本。</p>
{table(['配对指标', '平均差', '95% 场景 bootstrap 区间'], residual_intervals)}
<h2>真实图像</h2><ul>{real_findings}</ul>
{table(['最终探索轮的构造／来源检查', '实际值'], real_rows)}
<p>这些数值衡量像素来源、几何重放和构造约束；餐具接触由规划确定，没有测量真实食物体积、克数或照明误差。“Qwen直接编辑”对照采用同一Qwen-Image-2.1 + Fun ControlNet-Union参考图管线，去掉MLD2几何控制，仍保留相同区域合成约束。</p>
<p class="note">真实分支以 SAM3 分割和 MoGe2 可见表面为观测锚，MLD2 提供有界的隐藏厚度／材料先验；餐具承托和抬升由显式几何控制。
首轮 Qwen-Image-2.1 + Fun ControlNet-Union 局部外观修饰会添加新食物。探索轮把 Qwen 的作用收缩为金属外观候选，投影回固定器具后，食物、切口和背景直接复用共享状态渲染。
第三轮进一步采用闭合匙几何与独立金属PBR。第四轮在新几何上验证局部金属候选；各图实际使用的像素来源和最终选择按实验记录列出。
局部尺度和可见表面正则化保持源 UV，但属于建模假设。没有实测背面、标定尺度和编辑后真实图，不能由这些照片推导真实 3D／克数精度。
源体分配、材料复用和 SE(3) 回放的精确一致性属于构造性质，不能当作学得物理正确性的证明。</p>
<h2>实际查看的结果与文件</h2>{''.join(artifact_blocks)}
<p><a href="analysis_summary.json">完整场景统计 JSON</a> · <a href="../experiment_plan.json">实验范围与修订记录</a></p>
</main></html>'''
    (output/'report.html').write_text(page, encoding='utf-8')
    print(output/'report.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
