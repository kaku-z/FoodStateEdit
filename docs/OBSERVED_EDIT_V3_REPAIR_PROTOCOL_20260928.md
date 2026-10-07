# v3 故障修复与固定开发对照

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: 用户授权的故障排查、修改、执行
- Origin Date: 2026-09-28
- Verification Status: FROZEN_REPAIR_DEVELOPMENT_PROTOCOL
- Version Label: observed_edit_v3

## 已定位的问题

v2 生成器同时接收旧状态原图和搬运状态，缺少区域硬约束；它可以恢复原处食物、旋转目标食物。事后贴回的来源纹理与生成形状冲突。全图生成的色调变化又被局部硬合成截断，形成餐盘边界。

v3 预处理检查还发现：若空盘修复的泊松边界仍在食物边缘/阴影内，食物颜色会被带回移除区。未使用的 `bundle` 和 `bundle_r2` 保留这两轮输入检查记录，均未运行模型。本轮实际生成只使用 `bundle_r3`。这些是查看输入后的开发修改，不是盲测试或事后筛选模型结果。

## 修改

1. 模型只看已经搬运的状态图；旧状态原图不再作为第二份模型输入。
2. 源掩码、目标位置、唯一来源对应沿用 v2，不改变失败案例。增加两处源图标注的矩形保护其他食物，记录在 config 中；它们是额外标注，因此 v2→v3 是整套修复比较，不能将全部改善归因于单一算子。
3. 餐盘初值由附近中性色像素的稳健二次 RGB 曲面给出；覆盖完整源块及外扩 28px 的阴影区，仅在外侧 14px 过渡，内部不混回原食物。其他食物受保护。这个初值只是占位，允许模型继续修复餐盘。
4. 新增生成内约束。每步 scheduler 更新后，以实际下一步 sigma 构造 `known=(1-sigma)*reference_latent+sigma*initial_noise`，投影 `z=free*z+(1-free)*known`。free 区域是勺具/邻近阴影与源处餐盘，其余固定。掩码按 VAE 网格和 2×2 packing 映射，不能把像素掩码直接扁平化。每步记录固定区域误差。
5. 最终分层合成保持真实搬运食物和已知背景，仅接入 free 区域；同时保存硬合成及带 Dirichlet 边界的泊松合成，隔离边界处理的作用。泊松解只使用域外边界，不从被移除的原食物内部混色。

仍然是二维可见观测编辑，无真实反照率、完整三维或物理质量主张。来源编号、固定区不变及零投影误差仅是工程不变量。

## 固定对照

沿用 pizza、corn 两张已见真实源图，seed=281，40 步、true CFG=4、bfloat16；输出 1184×896，与 v2 原始生成一致。离线模型及其 13 个权重/索引文件沿用 v2。实际安装的 pipeline 源代码哈希也固定，防止 callback 时序或 packing 改版后仍误用。

- scaffold_only：新输入、相同初始噪声、无逐步投影。
- projected：新输入、相同初始噪声、每步投影。

每支 2 次，合计 4 次新模型调用；每次均保存 raw，并对两支执行相同 hard / poisson 合成。禁止换 seed、丢弃失败或在本批次修改参数。餐盘参考图不进入输入目录或推理代码。生成回收后，仅用于源处恢复的局部评价。

评阅同时比较 v2 与 v3、raw 与合成。检查重复/原处残留、目标轮廓、其他食物、勺具支撑、边界与光照；不能用复制误差代替语义评价。相同两张开发图不足以证明泛化或构成独立盲评，正式测试仍封存。

## 实现与执行

- 核心：`foodstateedit/observed_edit/constrained.py`、`projection.py`。
- 准备：`scripts/prepare_observed_edit_v3.py`。
- 执行：`scripts/run_observed_edit_v3.py`。
- 测试：原有 8 项与新增 6 项，合计 14 项通过；包含错误 next-sigma、掩码形状、源食物颜色泄漏及边界梯度的回归用例。
- 实际输入：`outputs/observed_edit_v3_20260928_bundle_r3`，配置 SHA-256 `2e7afd505a3868642e74668eeeba5e17326c8d0b71f1341bf39c2dd5f412f014`。
- gp40 根目录：`/host/space0/guo-z/tf-ufi/repair_v3_20260928`；空闲 A6000 GPU0/1 分别执行两个条件；各自 7200 秒硬超时、进程与 manifest 检查；不覆盖旧 v2 输出。
