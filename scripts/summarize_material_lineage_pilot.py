"""Write measured pilot results and the owner's explicit visual acceptance."""
import argparse,hashlib,html,json,time
from pathlib import Path
import numpy as np

OBSERVATIONS={
 'new_01_7442':'抬升与空缺清楚；新切面过度光滑，像规整模具。光线追踪勺子比平面金属着色柔和，但食物仍有 CG 感。',
 'new_02_7496':'勺上载物明确；原位置的新表面像均色扇形填片，缺少自然挖取纹理与厚度变化。',
 'new_03_7443':'没有明显新配料，载物与缺口对应；规则轮廓和光滑内壁仍不够像自然豆腐。',
 'new_04_7459':'几何抬升保留；切面和载物偏黄、均匀，光照与材质不能可靠分离。',
 'prospective_01_7473':'原有木勺继续保留，新增金属勺明确；角部缺口小而平滑，载物仍偏规则。',
 'prospective_02_7441':'载物清楚；缺口侧壁更像光滑垫块或插片，源图低分辨率影响判断。',
 'prospective_03_11160':'载物抬起，原有葱花布局保留；载物与新切面带原图色偏，表面过于均匀。',
 'prospective_04_7498':'暖色、浅景深背景保留；新增载物像规则薄块，缺口表面缺少真实组织细节。',
}

def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main(root):
 cfg=json.loads((root/'config.json').read_text());audit=json.loads((root/'audit.json').read_text());assert audit['status']=='verified_internal_artifacts'
 metal=json.loads((root/'metal_v2/audit.json').read_text());assert metal['images']==144
 rows=[];shared=[];independent=[];bounds=[]
 for cid in cfg['cases']:
  d=root/cid;t=json.loads((d/'training/training_receipt.json').read_text());b=json.loads((d/'budget.json').read_text());s=t['sampling_validation']
  rows.append({'case_id':cid,'optimizer_steps':t['actual_optimizer_steps'],'parameters':t['parameter_count'],
   'source_pixel_heldout_ddim_rgb_mse':s['heldout_ddim_rgb_mse'],'source_pixel_constant_rgb_mse':s['heldout_base_color_rgb_mse'],
   'rgb_mse_ratio':s['heldout_ddim_rgb_mse']/s['heldout_base_color_rgb_mse'],'source_mass_proxy_residual':b['relative_total_mass_residual'],
   'v1_centroid_carried_fraction':b['quadrature_convergence'][-1]['carried_mass_fraction'],'exact_geometry_carried_fraction':b['exact_mesh_volume_bite_fraction'],
   'visual_photographic_acceptance':False,'visual_observation':OBSERVATIONS[cid]})
  for p in d.glob('pose*/probe_s*.npz'):
   with np.load(p) as z:
    shared.append(float(np.max(np.abs(z['shared_remaining_rgb']-z['shared_carried_rgb']))))
    independent.append(float(np.mean(np.abs(z['independent_remaining_rgb']-z['independent_carried_rgb']))))
  for p in (d/'fields').glob('*.npz'):
   with np.load(p) as z:bounds.append(float(np.mean(np.any(z['raw_linear_rgb']!=z['bounded_linear_rgb'],axis=1))))
 review={'status':'complete_reviewed_reject_photographic_realism','reviewer':'Owner assistant qualitative visual inspection, no human-study or blind-rating claim',
  'v1_composited_images_viewed':144,'v1_raw_render_images_viewed':144,'v2_composited_images_viewed':144,
  'boards_viewed':24,'all_fixed_seeds_and_poses_retained':True,'new_hands_or_people_seen':False,
  'global_photographic_acceptance':False,'cases':[{k:r[k] for k in ['case_id','visual_photographic_acceptance','visual_observation']} for r in rows]}
 write(root/'visual_review.json',review)
 exact_path=root/'mass_partition_v3/manifest.json'
 exact=json.loads(exact_path.read_text())
 assert exact['status']=='complete_verified_proxy_volume'
 result={'status':'complete_reviewed','conclusion':'内部一致性原型运行成功，真实第一口照片目标尚未通过。',
  'actual_models':'Eight photo-specific source-conditioned DDPM epsilon MLPs, 57987 parameters each, 6000 updates each; DDIM 24 steps.',
  'actual_geometry':'Previous source-fit monocular frusta and shared Boolean scoop, three joint rigid translations; no learned 3D state operator.',
  'actual_renderer':'Canonical material grid, perspective z-buffer, source UV on observed inherited faces, ray ambient visibility; v2 ray-traced Al spoon; explicit source-background composition.',
  'full_MLD_trained':False,'qwen_calls_this_pilot':0,'vace_calls_this_pilot':0,'real_photo_sources':8,'v1_images':144,'v2_images':144,
  'scope':'Previously developed photos of tofu only. Same-photo pixel split is not unseen-food generalization. No paired real after-image, scan, calibrated camera, lighting or real-mass measurement.',
  'metrics':{'internal_checks':audit['checks'],'internal_failed_checks':audit['failed_checks'],'shared_paired_material_max_linear_rgb_error':max(shared),
   'independent_paired_material_mean_linear_rgb_error':float(np.mean(independent)),'mean_material_points_changed_by_source_bounds':float(np.mean(bounds)),
   'source_heldout_color_beats_constant_cases':sum(r['rgb_mse_ratio']<1 for r in rows),'non_action_changed_pixels':0},
  'metric_scope':'1906 checks are V1 internal-script artifact checks; V2 has a separate metal/pixel audit. Heldout RGB MSE measures same-photo angular-corrected linear-RGB color proxies, not measured albedo or unseen-food prediction.',
  'interpretation':'Shared hidden material removes branch/pose resampling by construction, not by learning physical law. The photos show only small differences versus independent resampling. Source-derived decoder bounds and retained source UV are substantial constraints, not learned full-image editing.',
  'exact_mass_partition':exact,'visual_acceptance':review,'cases':rows,'created_unix':time.time()}
 write(root/'RESULTS.json',result)
 table=''.join('<tr><td>'+html.escape(r['case_id'])+'</td><td>6000</td><td>'+f"{r['source_pixel_heldout_ddim_rgb_mse']:.6f}"+'</td><td>'+f"{r['source_pixel_constant_rgb_mse']:.6f}"+'</td><td>'+f"{r['rgb_mse_ratio']:.3f}"+'</td><td>未通过</td></tr>' for r in rows)
 body=f'''<!doctype html><meta charset="utf-8"><title>Material lineage pilot results</title><style>body{{max-width:1100px;margin:30px auto;padding:0 20px;font:17px system-ui;line-height:1.65;color:#182330}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{border:1px solid #b8c4d2;padding:7px}}img{{max-width:100%}}.warn{{background:#fff1cf;padding:14px}}code{{font-size:14px}}</style>
 <h1>真实原图的物质谱系小型实验</h1><p class="warn"><b>{result['conclusion']}</b> 已完成 8 张原图 × 3 个随机种子 × 3 个抬升位置 × 2 种材质对照，共 144 张初版和 144 张餐具渲染修正版；全部保留并查看。</p>
 <p>实际训练了每张原图各一个 57,987 参数材质去噪网络，各 6,000 步。物质 ID、质量份额、两侧切面和反事实姿态共享材料场已经写入可运行代码。本轮没有 Qwen 或 VACE 推理调用。完整的可泛化物质谱系扩散模型尚未训练，三维几何和操作仍使用源图拟合与刚体先验。</p>
 <h2>实测结果及边界</h2><p>V1 的内部一致性脚本审计 {audit['checks']} 项通过，检查权重和输入来源、真实 source pixel、三维射线、物质预算、同一界面及所有最终像素；V2 另有餐具射线及像素审计。共享材质两侧的最大未着色 RGB 差异为 {max(shared):.3g}；独立采样平均差异为 {np.mean(independent):.6f}。这是内部表示的一致性结果，不能证明真实内部组织或照片观感。</p>
 <p>材质预测在 8 张照片各自保留像素的角度校正颜色代理上都优于常量颜色基准。表中误差使用 linear-RGB，颜色代理不是真实测量的反照率。此处训练与验证来自同一张照片，既没有独立新食物，也没有真实的编辑后图片；不能当作泛化或新切面质量的证明。</p>
 <table><tr><th>原图</th><th>实际训练步数</th><th>DDIM 颜色代理 MSE</th><th>常量颜色代理 MSE</th><th>误差比</th><th>真实照片目标</th></tr>{table}</table>
 <p>学习场的输出还受原图材质颜色范围约束，平均 {np.mean(bounds)*100:.1f}% 的三维格点有至少一通道被约束改变。继承表面的原图投影、这个颜色约束和固定几何都发挥了重要作用，不能把全部效果归因于材质小模型。</p>
 <p>8 组都可见勺上载物抬起，原图动作区域之外逐像素不变，没有观察到新人物或手。光线追踪消除了多数勺子分面，但切面均色、规则轮廓、偏色和生硬边缘仍存在。共享与独立材质在这些豆腐照片上的视觉差别很小，目前没有证据说明新方案已经在照片真实感上胜出。</p>
 <h2>取走量</h2><p>最初离散单元的预算严格守恒，但用单元中心判断切割区域会低估部分取走量。随后以每个粗三维区域与取走网格的实际布尔交体积分配质量份额，并保存独立结果。它只对应所选几何模型与均匀密度先验；原图没有真实质量和三维真值。两种记录都保留。</p>
 <p>后续需要让可学习的三维材质与几何联合解释局部微结构、湿润边缘和光照，并用新食物及实际 before/after 观测检验。继续增加同一组原图的 seed 数量，不能验证这些问题。</p>
 <p><a href="gallery.html">全部逐张结果</a> · <a href="audit.json">内部审计</a> · <a href="metal_v2/audit.json">餐具及像素检查</a> · <a href="RESULTS.json">完整数据</a> · <a href="config.json">原始冻结计划</a></p><img src="comparison_all_sources.jpg" alt="八张原图及三个抬升位置，固定seed41，无最佳结果挑选">'''
 (root/'RESULTS.html').write_text(body,encoding='utf-8')
 print('RESULTS_REVIEWED',len(rows),'cases',len(shared),'paired probes',flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();main(a.root)
