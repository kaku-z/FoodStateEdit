# MLD4 v3：24 张真实照片完整回归

本目录发布 2026-10-06（JST；运行目录使用 UTC 2026-10-05）完成的实验。24 张图均从原始规范化 RGB 和固定粗提示重新运行感知、几何、语义和生成，不读取旧 guides，不逐图改种子或选优。4 个 worker 在 gp40 完成全部运行，耗时 33 分 47.6 秒，0 执行失败。

**算法仍未达到整体摄影目标。** 所有图片曾经用于观察或开发，没有真实取食前后配对图、标定扫描或重量真值。以下是 AI 逐图筛查，不是人工研究或独立泛化结果。

| 视觉项目 | 通过 | 失败 | 不确定 |
|---|---:|---:|---:|
| 源材料的可见颜色/纹理相容 | 16 | 0 | 8 |
| 可见离盘构图 | 24 | 0 | 0 |
| 餐具承托外观 | 16 | 0 | 8 |
| 原位置合理减少 | 4 | 16 | 4 |
| 纹理与接缝自然 | 0 | 11 | 13 |
| 无新增手或人物 | 24 | 0 | 0 |
| 勺柄伸出画面 | 24 | 0 | 0 |

七项全部通过为 **0/24**。源缺口被补回、新配料、薄片状食物和不自然接缝仍是主要失败。可信核心共 34,121 像素满足显式源纹理公式，编辑区域外变化为 0；这些只验证代码约束。

## 阅读结果

- [全部 24 例的 GitHub 图片索引](CASES.md)，包括失败和不确定例。
- [交互结果页](gallery.html)：下载仓库后在浏览器打开，支持筛选、放大和 A/B 比较；GitHub 文件页只显示 HTML 源码。
- [详细算法报告](../../docs/MLD4_ALGORITHM_AND_RESULTS_20261006.md)。
- [实现审计](audit.json)、[逐例视觉判断](visual_reviews.json)、[汇总](visual_summary.json)、[实际完成记录](completion.json)。
- [冻结协议](protocol.json)、[运行环境版本](runtime_inventory.json)、[冻结源码哈希](frozen_code_manifest.json)。
- [厚度截断等诊断](diagnostic_summary.json)、[同图重复运行差异](repeatability_dev02.json)。
- `default_food_test/` 保存默认 `food` 提示单例及其失败判断；`negative_source_probes/` 保存未采用的源缺口分支清单和审查。后者的完整中间图仍在服务器。

`assets/<case_id>/` 保存原图、生成 A、默认最终 B、源缺口/勺头放大图、完整对比图和每次运行记录。B 对可信可见材料内部使用源纹理乘有界平滑标量照明增益，源缺口沿用 A；因此 B 不解决源区补回问题。

JSON 中的绝对路径是执行当时的服务器路径，作为原始证据保留。图片浏览使用相对路径。`frozen_code_manifest.json` 的 `code\` 前缀来自原发布包，对应本仓库根目录；哈希指原始字节。本目录及这 65 个源码文件通过 Git 属性保留字节，避免自动换行转换破坏证据。

## 在现有服务器环境运行

```bash
GEOPY=/host/space0/guo-z/tf-ufi/food3d_pilot_20260928/venv/bin/python
CODE=/absolute/path/to/FoodStateEdit
"$GEOPY" -u "$CODE/scripts/run_mld4_from_image.py" \
  --image /absolute/path/to/food.jpg \
  --food-prompt chicken \
  --output /absolute/path/to/a_new_output_directory \
  --gpu 3
```

使用实际空闲 GPU，输出目录须不存在。默认 seed 41、180 次代理变形迭代、生成长边 1024、两阶段各 40 步。省略粗提示使用 `food`，但会影响感知和几何选择；默认提示单例的运行完成不代表质量通过。

默认最终 B 位于 `material_bound/real/real_00_input/final.png`，生成 A 位于 `production/real/real_00_input/final.png`。`--free-material` 关闭 B；`--legacy-handle` 关闭上游柄部延伸。多图需使用不同外层输出目录。

依赖 SAM3、MoGe2、冻结 MLD2、GLM-4.6V-Flash、Qwen-Image-2.1 和 Fun ControlNet Union。VACE 和试训 LoRA 不参与当前默认流程。服务器模型及 runtime 路径写在入口和协议中；本仓库不分发权重、第三方 vendor、训练集或完整几何缓存。版本清单不是跨机器依赖锁，几何重复一致也不保证最终 RGB 逐字节复现。

## 发布前代码验证（2026-10-07）

在 gp40 原几何环境中执行本次新增的 21 个测试模块：**142 项通过，0 失败，0 跳过**。[验证记录](code_validation_20261007.json)覆盖材料切分、变形、有限深度支持、条件图、语义编译及训练数据/损失等代码行为，不是图像质量测试。发布时另核验 65 个冻结源码与 Git 暂存字节一致、288 个审计资产哈希一致，详情见[发布完整性记录](publication_validation.json)。
