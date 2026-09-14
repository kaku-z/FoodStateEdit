# Day 28：Qwen 餐具局部外观修复实验

## Material Passport

- Artifact type：Experiment Result
- Verification Status：ANALYZED
- Scope：拉面、汤、炒饭、蛋糕四个选定开发样本；每类固定 seed 1
- Human evaluation：仅内部非盲视觉复核，不是独立人工评价

## 结果

Qwen-Image-Edit-2511 在 gp40 的 RTX A6000 GPU6 上一次加载并串行生成四个局部候选，技术执行成功。全部输出只在冻结餐具掩膜内融合，保护区外最大像素差均为 0；远端与本地共核对 55 个 SHA-256，0 个不一致。

四类候选均未通过全部冻结自动门，因此正式 `10_final.png` 全部回退到修复前图。本轮不得写成“四类最终修复成功”。

| 类别 | 支持区变化率 | 结构边缘余弦 | 对齐相似度 | 内部视觉观察 | 最终处理 |
| --- | ---: | ---: | ---: | --- | --- |
| 拉面筷子 | 0.9980 | 0.8286 | 0.9804 | 木纹与高光略改善 | 回退 |
| 汤勺 | 0.9867 | 0.8888 | 0.9901 | 金属边缘与反射改善较清楚 | 回退 |
| 炒饭铲 | 0.9986 | 0.8043 | 0.9440 | 仍像多边形代理，没有清晰净改善 | 回退 |
| 蛋糕叉 | 0.9962 | 0.8115 | 0.9826 | 金属高光改善，但结构阈值略未通过 | 回退 |

冻结门要求支持区变化率不超过 0.98、结构边缘余弦不低于 0.82。拉面和汤只因变化率失败；炒饭和蛋糕同时未通过结构边缘门。人工外观观察不能覆盖自动门，因此候选只能作为诊断证据保留。

## 解释

原始 Qwen 局部提案说明模型具备较强的木纹和金属材质渲染能力。问题主要出在两个位置：

1. 扩散式重绘几乎会改变掩膜内每个像素，现有“变化像素比例上限”不适合衡量材质修复；
2. 窄掩膜保证了食物、接触和背景不变，但同时把可见改善限制得很小；炒饭的矩形铲几何本身不自然，单纯改材质无法修复。

因此下一版算法应在运行前重新冻结门控，而不是在看到结果后放宽本轮阈值：用边界位移/实例数量/接触关系作为结构门，用局部感知差异和盲评作为外观门；炒饭需要先改善代理几何，再交给 Qwen 修材质。

## 证据

- 原图、候选和差异：[review grid](../artifacts/day28_qwen_utensil_refinement_four_case_v3/review_grid_before_candidate_diff.png)
- Qwen 原始局部提案：[raw crop grid](../artifacts/day28_qwen_utensil_refinement_four_case_v3/review_grid_raw_qwen_crops.png)
- 运行清单：[run manifest](../artifacts/day28_qwen_utensil_refinement_four_case_v3/run_manifest.json)
- 机器可读结果：[result JSON](day28_qwen_utensil_refinement_four_case_result_v1.json)

## 结论边界

这是四个选定开发样本、每类一个 seed 的后处理诊断。它支持“Qwen 有局部餐具材质修复能力”这一有限判断，但不能证明泛化、方法优越性、物理正确性、独立真实感提升或论文级照片真实感。
