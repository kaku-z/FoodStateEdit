# v3 排查、修改与真实图片复测

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: 故障排查、代码修改、服务器执行与结果核验
- Origin Date: 2026-09-28
- Verification Status: EXECUTED_REPAIR_WITH_REMAINING_VISUAL_FAILURES
- Version Label: observed_edit_v3 + screened boundary repair

## 结论

本轮修改和复测已完成。**原处重复披萨、目标方向冲突和两套食物轮廓，在这两张开发图的新生成中不再出现。** 但严格去噪投影带来新的白边、块状边缘和旧位置轮廓状亮斑；目前不能把它作为已通过画质验证的正式方法。

最直接的改善来自只给模型一个一致的终态输入。逐步投影显著减少了原始生成对已知背景的改变，同时存在视觉代价。最终像素复制为零误差、潜变量投影为零误差，均不能证明图像自然或物理正确。

另外修复了纯泊松合成的低频亮度漂移：增加固定的内部保真项后，玉米无投影分支的源处 RGB MAE 从 14.00 降至 3.03，接近原始生成的 2.96；没有增加模型采样或使用真实参考来拟合参数。

## 排查结果与对应修改

| 问题 | 代码/实验依据 | 修改及状态 |
|---|---|---|
| 原处食物被再次生成 | v2 同时输入旧状态原图和搬运状态；生成器没有区域约束 | v3 只输入搬运后状态。两种新输入分支的披萨原处均未再出现重复食物 |
| 贴回纹理出现双重轮廓 | 生成器改变食物方向，后处理仍按原平移复制 | 每步去噪固定参考潜变量，约束目标食物和已知背景；本轮未再出现大幅旋转冲突，但解码边缘仍有伪影 |
| 硬合成导致色块 | 全图生成的颜色与原图不同，边界直接截断 | 保留硬合成作对照，增加带边界条件的融合；不能据此宣称所有接缝已消失 |
| 泊松融合使整个修复区变亮/变暗 | 玉米无投影 raw MAE=2.96，pure Poisson=14.00 | 增加 screening=0.1 的内部亮度保真项，MAE=3.03；旧算法默认值 0 保留，四张旧合成逐像素复现一致 |
| 固定区域产生白边与阴影残留 | 玉米 projected raw 已出现白圈和旧位置亮斑，最终合成仍可见 | 尚未解决；推测与 VAE 边界感受野、硬区域划分及阴影区域定义有关，未把这一推测冒充已完成因果消融 |

去噪约束使用实际 scheduler 的下一步 sigma：

`known = (1 - sigma_next) * encoded_reference + sigma_next * initial_noise`

`latent = free_weight * latent + (1 - free_weight) * known`

因为 callback 在 scheduler 更新之后执行，使用当前 timestep 的噪声水平会产生错位。掩码通过实际 VAE 网格与 2×2 packing 映射；服务器安装的 pipeline 源文件哈希也作了核验。两张约束图共 80 个步骤均有审计记录，固定潜变量最大误差为 0；它只验证代码约束执行。

带内部保真项的合成求解：

`min_x Σ_edges |∇x - ∇proposal|² + 0.1 Σ_inside |x - proposal|²`

区域外和受保护食物保持给定像素；第二项抑制边界修正向整个内部传播。它不恢复真实光照或材质，也不会自动删除 proposal 中已经存在的伪影。

## 实验与修改时序

源图、源块、目标位置、seed=281 沿用 v2 的 pizza/corn；两张均为已经看过的 Nutrition5k 开发图片，未解锁未见测试。v3 额外用源图矩形保护其他食物，因此 v2→v3 的改善属于组合修改，不能全部归功于潜变量投影。

先固定两个条件：相同终态输入的 `scaffold_only` 与 `projected`，各两次生成，均为 40 步、true CFG=4、1184×896。实际输入包为 `outputs/observed_edit_v3_20260928_bundle_r3`；前两个包只做过输入预检查，没有调用模型。流程细节见 `OBSERVED_EDIT_V3_REPAIR_PROTOCOL_20260928.md`。

四次模型生成结束后，发现纯泊松漂移，再增加固定 screening=0.1 的后处理修复。参数一次固定并应用到两张图的两种条件，未作真实目标上的参数搜索；这是**明确的开发阶段事后修订**，不是初始预注册。原始生成、硬合成、纯泊松与修订版本全部保留。

## 数值结果及其边界

源处 RGB MAE 使用与 v2 相同的源掩码、真实配对餐盘参考，0–255 单位，越低越接近参考。它只测量原食物占据的局部，不覆盖全部旧阴影和远处背景，不能代替整体画质。

| 版本 | 披萨源处 MAE | 玉米源处 MAE |
|---|---:|---:|
| v2 relit_locked | 78.07 | 2.47 |
| v3 单状态输入 raw | 3.13 | 2.96 |
| v3 逐步投影 raw | 3.38 | 3.81 |
| v3 单状态输入 + 纯泊松 | 3.18 | 14.00 |
| v3 逐步投影 + 纯泊松 | 3.27 | 4.45 |
| v3 单状态输入 + 内部保真融合 | 3.14 | 3.03 |
| v3 逐步投影 + 内部保真融合 | 3.38 | 3.82 |

披萨源处明显改善；玉米这个指标没有超过 v2，不能只展示披萨而声称全面改善。玉米原始单状态输入图较自然，但加约束/合成后仍有轮廓状亮斑，更说明局部 MAE 很低也可能伴随明显视觉问题。

在 v3 同一个已知背景区域内，raw 图对源图的 MAE：披萨无投影 10.57、投影 0.64；玉米无投影 2.39、投影 0.64。对应保留食物矩形的 MAE：披萨 8.00→1.16，玉米 3.03→0.84。这是投影限制改动范围的证据，但不是画质获胜证据。

最终层合成的搬运食物、保护食物像素误差为 0，是按公式固定的工程性质。边界变化量也只是固定区域边缘上的诊断值，不是经过验证的感知指标。两个样本不计算总体成功率、显著性或可信泛化区间；没有独立人工评阅和真实托举终态。

## 可视结果

汇总图依次显示 v2 最终图、v3 单状态 raw、v3 投影 raw、v3 投影加内部保真融合。两张案例及失败版本均保留。

![v2/v3 修复对照](../outputs/observed_edit_v3_20260928_screened/repair_summary.png)

完整材料：

- `outputs/observed_edit_v3_20260928_gp40/`：4 张原始生成、4 份逐步审计、运行清单和预检。
- `outputs/observed_edit_v3_20260928_review/`：全部 raw/hard/pure-Poisson 对照、指标。
- `outputs/observed_edit_v3_20260928_screened/`：4 张修订融合图、两套比较图、指标和修订 manifest。

## 测试、环境与代码

16 项单元/回归测试通过。新增测试检查源食物不会从修复域内部混回、边界梯度、保护食物不被覆盖、下一步 sigma、潜变量形状，以及保真融合保留内部亮度。修改后重新检查四张旧 pure-Poisson 合成，逐像素完全一致。

gp40 两张空闲 A6000 同时执行，约 785 秒墙钟时间，包含权重校验及加载。模型调用各约 197–199 秒；不将其与 v2 冷启动差异当速度收益。torch=2.5.1+cu121，diffusers=0.38.0.dev0；每个作业的 13 个模型权重/索引文件通过哈希检查。下载的 8 份服务器输出（4 图、4 审计）逐文件核验通过，无崩溃或重试，结束后计算进程已释放。服务器项目路径：`/host/space0/guo-z/tf-ufi/repair_v3_20260928`。

修改代码：

- `foodstateedit/observed_edit/constrained.py`：源图背景初值、保护区、边界合成及 screening。
- `foodstateedit/observed_edit/projection.py`：逐步去噪约束。
- `scripts/prepare_observed_edit_v3.py`、`run_observed_edit_v3.py`：固定输入、预检、服务器推理。
- `scripts/review_observed_edit_v3.py`、`repair_observed_edit_boundaries.py`：完整比较和可复现后处理。
- `tests/test_observed_edit_constraints.py`：新增回归测试。

初始协议及代码快照在 `results/observed_edit_v3_repro_20260928/`；后处理修订记录在 `results/OBSERVED_EDIT_V3_BOUNDARY_FIX_20260928.json`。正式测试仍保持封存：主要重复故障有修复证据，但当前强约束版本仍未通过全部视觉条件。

## 重现新增边界修复

在仓库根目录执行，输出需使用新路径：

```powershell
py -3.13 -m unittest tests.test_observed_edit tests.test_observed_edit_constraints
py -3.13 scripts/repair_observed_edit_boundaries.py --bundle outputs/observed_edit_v3_20260928_bundle_r3 --runs outputs/observed_edit_v3_20260928_gp40 --previous-review outputs/observed_edit_v3_20260928_review --real-pairs outputs/nutrition5k_solid_pairs_20260928_v1 --output outputs/observed_edit_v3_screened_reproduction
```

该命令只重算已有生成的后处理，不下载模型、不新增生成，也不覆盖本轮证据。
