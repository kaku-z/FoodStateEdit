# FoodStateEdit 20 天论文冲刺方案

版本：v1.0  
制定日期：2026-08-25  
目标周期：20 个自然日  
项目根目录：`noodle_image_benchmark_v1`

## 1. 二十天后的交付目标

目标不是在 20 天内证明“所有料理都通用”，而是形成一个主张清楚、实验闭环、
可复现、可投稿的研究包：

1. 一套统一的、训练自由的料理操作图像编辑方法；
2. 四种材料/工具组合的端到端结果；
3. 一个至少 60 个输入的分层测试集；
4. 统一基线、消融、自动指标、盲评和失败案例；
5. 论文初稿、主图、表格、补充材料和可运行代码快照。

二十天计划的现实目标级别：

- 优先目标：高质量 workshop / short paper / arXiv 完整稿；
- 冲刺目标：形成可继续扩展至 CVPR/ICCV/ECCV 主会的实验骨架；
- 不承诺：仅靠当前单案例结果直接达到主会录用标准。

## 2. 冻结后的论文定位

### 2.1 暂定题目

**FoodStateEdit: Training-Free Physically-Constrained Food Manipulation Editing with Material-Aware Staged Projection**

中文工作题目：

**FoodStateEdit：基于材料感知分阶段投影的训练自由料理操作图像编辑**

### 2.2 任务定义

输入不是“只有一句文本”，而是：

- 一张无目标餐具操作的料理图像；
- 结构化动作说明，例如 `scoop_liquid`、`scoop_granular`、
  `lift_strand`、`twirl_strand`；
- 工具类型和少量用户控制，例如接触点、目标方向或局部动作区域；
- 可自动生成或人工校正的食材/容器掩码。

输出是一张保持原场景身份和背景、同时显示工具正在操作食物的编辑图像。

这应明确写成 **structured-control image editing**，不能在尚未实现时声称为
纯文本、全自动或任意料理编辑。

### 2.3 核心研究假设

刚体工具、可变形食材、源位置修复和接触区域不应该共享同一个扩散约束窗口。
将料理操作分解为物理状态和材料层，并按层采用不同的代理注入时序，能够同时提高：

- 工具几何完整性；
- 食材转移与守恒；
- 接触和遮挡正确性；
- 源位置修复质量；
- 非编辑区域保持。

### 2.4 计划主张的三项贡献

1. **Food manipulation state representation**：统一表示刚体、液体、颗粒、
   细长可变形材料、容器、载荷、接触、支撑和守恒关系。
2. **Material-aware staged projection**：将刚体、接触、材料、源洞和保护区使用
   不同的 denoising 注入窗口投影至 GeoEdit/Wan 视频扩散先验。
3. **FoodManipBench-P**：包含工具—食材操作硬约束、材料专用指标、背景锁定和
   物理真实性盲评的料理操作编辑测试集。

不能把“修改 GeoEdit 的两个时间参数”单独写成贡献。方法必须与状态表示、材料层和
评价协议结合成一个完整系统。

## 3. 二十天内严格控制的研究范围

### 3.1 四个主场景

| 材料族 | 场景 | 工具 | 动作 | 关键难点 |
| --- | --- | --- | --- | --- |
| 液体 | 清汤/浓汤 | 勺子 | 舀起 | 容纳、液面、反光、无溢出 |
| 颗粒 | 炒饭 | 勺子或锅铲 | 混合并舀起 | 成分守恒、载荷、源位置 |
| 细长材料 A | 拉面/乌冬 | 筷子 | 夹起 | 连续性、长度、夹持、下垂 |
| 细长材料 B | 意面 | 叉子 | 卷起并抬起 | 叉齿、缠绕、接触遮挡、重力 |

软块材料（蛋糕、沙拉、牛排）只作为补充定性案例，不在前十天加入主 benchmark，
避免范围失控。

### 3.2 明确不做的事项

- 不训练新的基础扩散模型；
- 不重新搭建或重复下载现有 GeoEdit/Wan 权重；
- 不在 20 天内追求纯文本全自动感知；
- 不做任意复杂烹饪过程或多步视频生成；
- 不用大量 seed 搜索后只报告最好的一张；
- 不把 ImageGen 代理相似度作为唯一或主要成功指标。

## 4. 方法实现方案

### 4.1 统一场景状态

每个案例保存一个可复现 JSON：

```text
Scene
├── container: bowl / plate / wok
├── utensil: spoon / chopsticks / fork / spatula
├── food objects
│   ├── material: liquid / granular / strand
│   ├── amount or geometry
│   └── source region
├── action
│   ├── type
│   ├── contact anchors
│   ├── target pose
│   └── payload amount
└── constraints
    ├── conservation
    ├── contact and support
    ├── no duplication
    └── protected background
```

所有料理通过同一入口执行；禁止为每张测试图修改求解代码。允许材料族拥有独立求解器，
但接口和输出字段必须一致。

### 4.2 五层空间分解

每个案例至少生成以下五个空间层：

1. `M_rigid`：勺子、叉子、筷子、锅铲等刚体几何；
2. `M_contact`：叉齿—意面、筷尖—面条、勺沿—液体等接触/遮挡带；
3. `M_material`：勺内液体、米饭载荷、卷起或抬起的面条；
4. `M_hole`：被取走食材的原位置和需要重建的背景；
5. `M_protect`：容器、桌面和其余必须精确保持的区域。

第一阶段可以人工校正掩码，但必须记录：自动结果、人工修改范围和每例标注时间。
论文任务定义应允许结构化控制，不能把人工控制隐藏成全自动结果。

### 4.3 分阶段注入

统一形式：

```text
M(t) = I[t < τ_r] M_rigid
     ∪ I[t < τ_c] M_contact
     ∪ I[t < τ_m] M_material
     ∪ I[t < τ_h] M_hole
```

初始候选调度：

```text
τ_rigid   = 18 / 20
τ_contact = 15 / 20
τ_material=  8 / 20
τ_hole    =  6-8 / 20
warm start = false
```

现有勺子结果只验证了 `rigid=18/material=8`。接触和源洞的独立窗口必须通过
消融后才能写入最终方法。

### 4.4 材料专用状态/代理

- 液体：目标液面、勺容量、无溢出、漂浮配料独立锁定；
- 颗粒：载荷质量、成分比例、源位置减少、颗粒纹理随机性；
- 面条：3D/2.5D 中心线、长度保持、接触夹持、自然下垂；
- 叉子意面：叉齿数量和连接、缠绕圈数、叉齿部分遮挡、下垂面条连续性。

正式测试中的代理应尽量由确定性几何/状态求解器生成。ImageGen 结果只能作为：

- 受控 pilot 的条件参考；
- 定性 oracle；
- 一个独立基线。

不得同时把同一 ImageGen 代理作为生成条件和唯一评价真值。

### 4.5 最终二维投影

- 编辑区域内部使用扩散生成结果；
- `M_protect` 外部直接复制原图，保持 bit-exact；
- 羽化带单独测量，而不是把它隐藏在全图 SSIM 中；
- 每个输出保存原始视频、所有帧、选帧规则、正式 2D 图和 hash。

## 5. 数据集与测试协议

### 5.1 规模

主目标：60 个真实输入。

```text
汤/勺子       15
炒饭/勺或锅铲 15
拉面/筷子     15
意面/叉子     15
```

其中每类：

- 5 个 development/pilot 案例；
- 10 个冻结测试案例；
- 调度只允许在 pilot 上选择；
- test 案例禁止逐图调参。

如果第 8 天无法让四类中的三类通过 anchor gate，则降级为 45 个案例、三个材料族，
并将论文定位改为“材料感知料理编辑的初步系统”，不得保留虚假的 universal 主张。

### 5.2 数据来源与合规

每个输入记录：

- 来源数据集或拍摄来源；
- 原始文件名和 SHA-256；
- 使用许可；
- 是否包含人工编辑或生成内容；
- 是否进入 pilot 或 test；
- 是否进行人工 mask 校正。

生成图与真实图必须分开统计。主表不能只混合报告一个总平均数。

### 5.3 随机性协议

- 正式方法：每个 test 输入运行 3 个预先冻结 seed；
- baseline：随机模型同样运行 3 个 seed；
- 禁止根据输出反向更换正式 seed；
- 同时报告 per-run 与 per-source 成功率；
- 主表报告均值和 95% bootstrap confidence interval。

## 6. 基线设计

### 6.1 必须完成

1. `Input / no edit`：保持性上界和动作成功下界；
2. `Strong instruction editor`：当前可用的强通用图像编辑器；
3. `Vanilla GeoEdit`：原始统一前景/背景策略；
4. `GeoEdit + unified action mask`：与当前方法同代理、同 seed、同计算量；
5. `FoodStateEdit staged`：完整方法。

### 6.2 时间允许时加入

- FreeFine：训练自由几何编辑；
- ObjectMorpher：用于面条和意面等可变形对象；
- PhysEdit/PhysicEdit：代码和权重可复现时加入；
- VACE direct/static proxy：证明状态与分阶段策略的必要性。

第 3 天必须冻结正式基线集合。若某方法无法运行，要保存安装日志并在论文中说明，
不能在第 18 天临时更换比较对象。

### 6.3 公平性

- 相同输入分辨率；
- 相同或明确报告的用户控制；
- 相同随机 seed 数；
- 不允许只给本方法精细 mask 而不给可接受相同控制的基线；
- API 模型记录调用日期、版本、完整 prompt 和原始输出。

## 7. 消融实验

从四类各抽 4 个案例，共 16 个案例，每个 3 seeds。依次比较：

| 编号 | 变体 | 目的 |
| --- | --- | --- |
| A0 | Prompt/direct editor | 无结构控制下界 |
| A1 | 统一 action mask、统一时间窗 | 原始基线 |
| A2 | 分层 mask、共享时间窗 | 分解本身是否有效 |
| A3 | 刚体/材料分阶段 | 当前勺子结论 |
| A4 | A3 + 独立 source-hole | 源位置修复贡献 |
| A5 | A4 + 独立 contact schedule | 接触/遮挡贡献 |
| A6 | A5 去掉 exact protection | 背景锁定贡献和代价 |
| A7 | full method，warm start on/off | 初始化影响 |

主方法的调度参数只能由 pilot 集选择。正式 test 不做穷举后挑最好结果。

## 8. 评价体系

### 8.1 三层成功定义

```text
ActionSuccess
  = correct utensil count
  ∧ correct food payload
  ∧ valid contact/support
  ∧ source object removed or reduced
  ∧ no forbidden duplicate

PhotoSuccess
  = local material realism
  ∧ plausible lighting/reflection/shadow
  ∧ invisible source-hole patch
  ∧ no obvious diffusion artifact

StrictE2ESuccess
  = ActionSuccess ∧ PhotoSuccess ∧ PreservationSuccess
```

主论文必须报告 `StrictE2ESuccess`，不能用状态求解成功替代图像成功。

### 8.2 共通自动指标

- 工具数量和类型正确率；
- 接触关系和支撑关系通过率；
- 源位置残留率；
- 非编辑区域 max pixel difference、MAE、LPIPS；
- 羽化带 MAE/P95；
- instruction alignment；
- 失败分类：重复、漂浮、错接触、断裂、溢出、源洞、背景漂移。

### 8.3 材料专用指标

- 液体：目标液面/面积比、溢出像素、异常强边缘比例；
- 颗粒：载荷面积或质量代理、成分计数/比例、源区减少；
- 面条：长度比、连通性、夹持位置误差、z-order；
- 叉子：叉齿连通/数量、缠绕或穿刺接触、下垂面条连通性。

代理 MAE/F1 仅作为控制遵循指标，不能替代物理真实性。

### 8.4 人工盲评

在 30 个分层抽样 test 案例上做随机化盲评：

- 每个样本至少 3 名评审；
- 隐藏方法名，随机列顺序；
- 二元问题：动作是否完成、接触是否物理正确、是否有重复/残留；
- 1–5 分：局部照片真实性、整体偏好；
- 报告多数票、平均分、95% CI 和评审一致性；
- 保存原始匿名评分，而不是只保存汇总表。

评价问题应指向具体区域和物理现象，采用类似 PICA 的 grounded questions，
避免只问“哪张更好看”。

### 8.5 统计检验

- 二元成功率：paired McNemar test 或 bootstrap difference CI；
- 连续指标：paired Wilcoxon 或 paired bootstrap；
- 报告 effect size，不只报告 p-value；
- 四类材料分别报告，不允许总体平均掩盖某一类完全失败。

## 9. 20 天逐日执行表

### Day 1：冻结论文主张和仓库

- 创建正式 Git 仓库/分支和论文实验 tag；
- 保存当前工作树、环境、权重路径和已有结果清单；
- 冻结题目、任务输入、四个主场景和成功定义；
- 建立统一的 `case_manifest.json`、`run_manifest.json`、`metrics.json` schema；
- 输出：`SCOPE_FREEZE.md`、代码 commit、数据清单 v0。

验收：任何人能在不阅读聊天记录的情况下说清输入、输出和论文主张。

### Day 2：冻结 benchmark 清单

- 每类整理 20 个候选，筛选并冻结 15 个；
- 划分 5 pilot + 10 test；
- 完成来源、许可、hash、分辨率和无目标餐具检查；
- 为 4 个 anchor 案例完成结构化动作标注。

验收：60 个输入均可追溯；test 集不再因结果好坏更换。

### Day 3：基线可运行性审计

- 在 4 个 anchor 上跑 strong editor、vanilla GeoEdit、unified mask；
- 对 FreeFine/ObjectMorpher/PhysEdit 做安装和最小 smoke test；
- 冻结最终 baseline 集合和失败说明模板；
- 不允许在此后下载重复权重或无记录地改版本。

Gate 1：至少三条基线能够批处理运行，否则论文比较设计必须当天缩减和重写。

### Day 4：统一五层条件生成器

- 将 spoon/liquid/noodle 的案例脚本统一到一个 CLI；
- 输出 rigid/contact/material/hole/protect 五层 mask；
- 自动生成 contact sheet、面积统计和输入合法性检查；
- 为每个案例记录是否人工修正。

验收：四个 anchor 使用同一命令和 schema 构建，不修改 Python 源码。

### Day 5：液体与颗粒代理

- 整合已通过的液体填充、漂浮配料锁定和 no-warm 设置；
- 实现炒饭载荷、成分比例、源位置减少和颗粒纹理代理；
- 完成 soup/fried-rice anchor 的硬约束评估器。

验收：状态、代理、最终图像三者的度量字段一致。

### Day 6：筷子与叉子细长材料代理

- 将已有面条 3D 中心线/长度/接触求解器接入统一接口；
- 增加 fork geometry、叉齿/接触带和意面卷起/下垂代理；
- 完成 fork anchor 的无餐具 source、目标动作和独立验收器。

验收：拉面和意面均能产生连续材料 mask、源洞和合法 contact/z-order。

### Day 7：多层分阶段 GeoEdit 集成

- 将现有 rigid/material 双层扩展到 contact/hole 独立 schedule；
- 保持无新参数时与原 GeoEdit 行为兼容；
- 增加 schedule validation 和单元测试；
- 所有运行使用新目录、状态文件、日志和断点跳过。

验收：远端测试通过，四个 anchor 各完成一个 21×20 preview。

### Day 8：anchor 调度消融与范围决策

- 在 pilot anchor 上比较 unified、12/8、15/8、18/8 和 contact/hole 变体；
- 冻结每个材料族的一组全局调度；
- 人工 200% 局部检查工具、接触、材料、源洞和背景；
- 不根据 test 图结果再改调度。

Gate 2：至少 3/4 anchor 达到 ActionSuccess，且至少 2/4 达到 PhotoSuccess。
若未达到，立即缩减到三个材料族或把论文改为 benchmark/diagnostic paper。

### Day 9：批处理与评估基础设施

- 完成统一 launcher、GPU 分配、失败重试和结果下载；
- 完成自动选帧规则并冻结；
- 完成各材料指标、总表、contact sheet 和盲评页面/表格生成器；
- 在 pilot 上端到端 dry run。

验收：一个命令可以从 case list 生成结果和 metrics，不手工挑帧。

### Day 10：60 个案例条件生成

- 批量运行自动条件生成；
- 完成 soup/rice 的人工校正与 QC；
- 输出 mask 面积、越界、重叠和保护区检查。

### Day 11：条件生成收尾

- 完成 ramen/pasta 的人工校正与 QC；
- 二次审核 contact 和 source-hole；
- 冻结 benchmark v1，生成 hash manifest；
- 此后不得删除失败案例。

验收：60/60 案例通过输入合法性，不代表编辑成功。

### Day 12：正式运行第一批

- GPU 0–3：FoodStateEdit 三 seeds；
- GPU 4–5：vanilla/unified GeoEdit；
- GPU 6：strong editor/其他 baseline 整理；
- GPU 7：失败重试和监控；
- 夜间持续批处理，禁止重复下载模型。

### Day 13：正式运行第二批与补跑

- 完成全部 test 和 baseline；
- 只补跑技术失败，不补跑“结果不好”；
- 每个输出保存日志、配置、seed、运行时间、GPU 和 hash；
- 生成第一版总体结果表。

验收：所有方法拥有相同 case list；缺失结果有明确失败原因。

### Day 14：自动评价和 Gate 3

- 计算三层成功率、材料指标和 95% CI；
- 生成按材料分类的失败矩阵；
- 检查代理偏差、source-hole 和背景保持；
- 锁定 30 个盲评样本，不能按本方法表现挑选。

Gate 3：完整方法必须在至少 3/4 材料族提高 ActionSuccess；总体
StrictE2ESuccess 相比 strongest reproducible baseline 至少有稳定正增益。
若不成立，论文必须改写为 benchmark/failure-analysis，不得挑选子集冒充总体成功。

### Day 15：盲评

- 生成匿名、随机顺序的评审材料；
- 至少 3 名独立评审；
- 当日完成初轮评分和漏项检查；
- 保存所有原始回答。

### Day 16：正式消融

- 在冻结的 16-case 子集上运行 A0–A7；
- 三 seeds；
- 同时统计质量、成功率和运行时间；
- 验证“分层但共享时间窗”与“真正分阶段”的区别。

### Day 17：统计、失败分析和局限

- 完成人工评分汇总和一致性；
- 运行配对统计检验；
- 建立失败 taxonomy 和代表性失败图；
- 单列真实图、生成图、自动 mask、人工 mask 的结果；
- 写清不适用场景和失败率。

### Day 18：主图、表格和复现包

- Figure 1：四类料理 teaser；
- Figure 2：状态图—材料求解—五层 mask—分阶段投影；
- Figure 3：staged schedule 可视化；
- Figure 4：与基线定性比较；
- Figure 5：失败案例；
- Table 1：总体与分材料结果；
- Table 2：硬约束；
- Table 3：消融；
- Table 4：盲评和运行时间；
- 冻结代码 commit、环境和 artifact manifest。

### Day 19：完成论文初稿

- Abstract、Introduction、Related Work；
- Method、Benchmark、Experiments；
- Limitations、Ethics/Data、Conclusion；
- 补充材料：实现、完整案例、prompt、指标定义、更多失败；
- 所有数字从机器可读 JSON 自动生成，禁止手工复制不同版本数字。

### Day 20：内部审稿和最终冻结

- 以审稿人角度检查 novelty、实验公平性和 claim/evidence 对齐；
- 删除无法由实验支持的“universal”“photorealistic”“fully automatic”；
- 复跑一个随机案例验证 README；
- 检查匿名性、引用、图表清晰度和数据许可；
- 发布 arXiv/workshop 版本或形成可提交包。

## 10. GPU 与运行预算

当前 8 张 RTX A6000（每张约 48 GiB）足够支持这套规模，但必须避免无组织 seed sweep。

建议固定分工：

```text
GPU 0-2  ours, seeds 1/2/3
GPU 3    ours extra material family / failed technical retry
GPU 4-5  vanilla and unified GeoEdit baselines
GPU 6    ablations
GPU 7    smoke test, evaluator, emergency retry
```

所有 launcher 必须：

- 设置 `DIFFSYNTH_SKIP_DOWNLOAD=True`；
- 检查权重存在后再启动；
- 拒绝覆盖已有正式结果；
- 写 `RUNNING/COMPLETE/FAILED` 状态；
- 保存 stdout/stderr、配置、seed 和模型 hash；
- 技术失败可以重跑，视觉失败保留原结果。

## 11. 推荐目录结构

```text
paper_release/
├── README.md
├── environment/
├── configs/
│   ├── methods/
│   ├── schedules/
│   └── baselines/
├── benchmark/
│   ├── manifests/
│   ├── pilot_split.txt
│   ├── test_split.txt
│   └── licenses.csv
├── foodstateedit/
│   ├── state/
│   ├── solvers/
│   ├── conditioning/
│   ├── projection/
│   └── evaluation/
├── scripts/
├── tests/
├── outputs/
│   └── <method>/<case>/<seed>/
├── tables/
├── figures/
└── paper/
```

现有大量历史输出保留为 archive，但正式论文只引用 `paper_release` 中有 manifest 的结果。

## 12. 论文写作结构

### Abstract

五句话完成：任务缺口、关键困难、状态/材料分解、分阶段投影、测试结果。

### Introduction

- 通用图像编辑能满足语义但常违反料理操作的接触和材料约束；
- 刚体几何编辑不能直接处理汤、颗粒和面条；
- 料理场景适合显式状态与守恒；
- 列出三项贡献。

### Related Work

- instruction/interactive image editing；
- 3D-aware geometric editing：GeoEdit、FreeFine、ObjectMorpher；
- physics-aware editing：PhysicEdit、PhysEdit、PICA；
- food manipulation/physics representation。

### Method

- scene/action state；
- material solvers；
- layered proxy；
- staged injection；
- exact preservation；
- acceptance gates。

### Experiments

- benchmark；
- baselines；
- metrics and human study；
- main table；
- ablation；
- failure/generalization。

### Limitations

- structured control rather than text-only；
- masks may require correction；
- single-view depth ambiguity；
- transparent/reflective utensils；
- complex multi-step cooking not covered。

## 13. 风险与止损策略

| 风险 | 早期信号 | 处理方式 |
| --- | --- | --- |
| 叉子/意面失败 | Day 6 无连续接触代理 | 将叉子作为补充案例，主表保留三材料族 |
| 炒饭外观不自然 | Day 8 anchor 无 PhotoSuccess | 缩小动作到“舀起”，不同时声称真实混合过程 |
| 多层 schedule 不稳定 | A3–A5 无增益 | 保留已验证双层方法，论文收窄为 rigid/material |
| 基线无法复现 | Day 3 smoke test 失败 | 用可复现 baseline 替换并公开失败日志 |
| 自动 mask 泛化差 | Day 10 校正量过大 | 将用户 mask 明确定义为输入，不声称 fully automatic |
| 人工盲评来不及 | Day 14 未找到评审 | 立即减少方法数/样本但保持分层随机，不取消盲评 |
| 总体无稳定提升 | Day 14 Gate 3 失败 | 转为 benchmark/diagnostic paper，报告负结果 |

## 14. 论文可接受的最低完成线

二十天结束时，只有满足以下条件才建议写成“方法论文”：

- 至少三个材料族完成真实图端到端编辑；
- 不少于 45 个冻结输入；
- 完整方法和至少三个有效基线；
- 多 seed，不挑结果；
- 至少一个完整的 staged ablation；
- 自动硬约束和独立盲评同时存在；
- 完整方法在大多数材料族有稳定正增益；
- 代码、配置、数据和结果有 commit/hash；
- 失败案例和局限公开。

若只完成受控液体和勺子案例，应作为 pilot/workshop 报告，而不能写成 universal editor。

## 15. 第一优先级清单

未来 48 小时最重要的事情按顺序是：

1. 冻结论文任务定义和标题；
2. 建立正式 Git/manifest，而不是继续增加零散输出目录；
3. 冻结 60 个输入及 pilot/test 划分；
4. 把当前案例构建器统一成五层条件 CLI；
5. 构建叉子卷意面 anchor；
6. 在四个 anchor 上跑统一基线，决定第 8 天是否有资格扩大实验。

这六项完成前，不应继续进行无计划的 prompt、seed 或单图美化实验。
