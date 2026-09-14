# 子项目 A：FoodStateEdit 成果证据审计

日期：2026-09-09。范围：本地只读证据审计；唯一新增文件为本文。未连接 GPU、未下载、未运行推理、未修改报告或提交。本轮没有重新进行图像人工评审，以下视觉结论明确引用既有助手评审。

## 结论

当前可以主张：实现了材料动作的结构化相对 3D 控制、冻结 VACE 外观生成与显式支持区合成的可复现原型；已有汤、炒饭、预切蛋糕的同种子平面/相对 3D 六条件对照；已定位并修复一次支持区裁掉高举餐具的具体工程缺陷。

当前不能主张：稳定的相对 3D 优势、新扩散算法有效、拓扑加权学习有效、自然照片级结果、物理正确性、未见真实图片泛化或已测量的社会收益。可发表的是诚实的阶段性研究与故障分析，不是已经获证的通用方法优越性。

## 1. 最新六条件完整性

直接依据两份原始记录，而非仅凭 README：

- `artifacts/day19_multimaterial_gp40_recovered_20260909_v1/run_manifest.json`：expected_conditions=8，completed_conditions=5；状态 `technical_failure_preserved`；异常 `BrokenPipeError(32, 'Broken pipe')`。第五个条件为 cake__planar，原始八条件批次仍未完成。
- `artifacts/day20_cake_recovery_gp40_20260909_v1/run_manifest.json`：expected_conditions=1，completed_conditions=1；仅恢复 cake__relative3d。总时长 234.354 秒，条件推理时长 213.505 秒，二者不是矛盾。
- 两批次分别 pipeline_load_count=1；不能合并写成“六条件总共加载一次”。本次没有完成原 Day19 的两个 noodles 条件，也不能拿不同日程的 Day18 noodle 结果冒充同批次第七、第八条件。

| 条件 | 所属批次 | 帧数 / seed / steps | 最终 PNG SHA-256 |
| --- | --- | --- | --- |
| soup__planar | Day19 | 21 / 1 / 20 | `95f0c5e909bb168c9fb2c300f5c6fba50fe811c599dddb3a2d3a4cecef361cc7` |
| soup__relative3d | Day19 | 21 / 1 / 20 | `ac8ec1bf6a09aefe643e1aae6624384c203992df3cff7d4b2637540ce42df854` |
| rice__planar | Day19 | 21 / 1 / 20 | `b115b3f5dd15a7e9bfb159cf877a2189a47aed8a957c957a1648278c894a38b4` |
| rice__relative3d | Day19 | 21 / 1 / 20 | `f219c9768d7acde853d51a9c7bd201568880732849fddd39c40d547f3f7157d3` |
| cake__planar | Day19 | 21 / 1 / 20 | `f5f44ae9a1b4124b8c8adafff9d0dfdba8bed6500faa4e53c688069399b6188b` |
| cake__relative3d | Day20 recovery | 21 / 1 / 20 | `3d4dcead955ef5aebb7171687c8ee6bbabf57e58dac28465da3d348673711b73` |

每个最终 PNG 位于对应批次根目录的 `<condition>/projected_final_hold.png`。共同设置为 688×512、VACE scale=1、LoRA off、TTM off。帧数与技术指标的解码检查来自 `artifacts/day20_non_noodle_presentation_v1/verification.json` 和 `results/day20_non_noodle_result_20260909.json`；本轮未重复视频解码。

本轮重新计算 `verification.json` 中两批次 all_file_hashes 的全部 51 个本地文件：**51/51 匹配，0 不一致**。这独立确认本地文件仍与既有核验记录一致，不是本轮重新访问远端确认。既有远端清单为 `results/day20_day19_remote_sha256.txt` 和 `results/day20_cake_remote_sha256.txt`。

另重新计算：

- `results/day20_cake_recovery_preflight.json`：`8a2d3d3e7b4387a92ff52621c63dbb529679af6650418672909a2ba23c06800a`，与恢复 manifest 相同。
- `artifacts/day19_multimaterial_dataset_v3/dataset_manifest.json`：`7a71eb2c63bf52cbe22705d1b1aceca028c045fd9fdf6328a8ed12b3a5e56078`，与两次运行 manifest 相同。

## 2. 技术成功和视觉成功必须分开

来源：`results/DAY20_NON_NOODLE_RESULTS_20260909.md`、`results/day20_non_noodle_result_20260909.json`。

| 材料 | 既有助手评审：动作 / 接触 / 自然度 | 清晰 3D 优势 |
| --- | --- | --- |
| 汤 | 携液汤匙动作部分成立；接触部分；匙缘硬、液面平；照片自然度 fail | false |
| 炒饭 | 配料块随铲移动部分成立；接触部分；铲形框状、来源减少未建立；照片自然度 fail | false |
| 预切蛋糕 | 提块与对应缺口部分成立；叉齿承载未建立；照片自然度 partial | false |

该评审看 0、3、6、10、15、20 帧，是助手检查，不是独立人工盲评，且不能声称连续视频所有帧已验收。固定第 20 帧的三列总图为 `artifacts/day20_non_noodle_presentation_v1/non_noodle_comparison.png`。程序控制图和最终合成图均不能冒充原生模型输出。

原生支持区外 MAE（0–255 像素尺度）：汤 10.2507/10.2544，炒饭 7.8269/7.8688，蛋糕 10.7539/10.7038（顺序均 planar/relative3d）。编码前合成的区外最大差均为 0，来自显式投影约束；有损 MP4 编码后区外 MAE 约为 2.914/3.724/3.130，并非像素精确保护。该指标不是动作或审美分数。

结果 JSON 记录完整测试 180/180 通过、0 failures/errors/skips；这是先前执行的测试证据，本审计未重跑测试，不等价于算法有效性。

## 3. 历史正向信号及其边界

| 分支 | 可核实数据 | 正确解释 / 不得扩大 |
| --- | --- | --- |
| Day13 拓扑加权 | `results/DAY13_FLEXIBLE_COMPLETION_RESULT_20260906.json`：相对3D加权比匹配 uniform 的 topology MAE 改善 2.22296%，pinch MAE 改善 2.83497%；预先门槛分别 5%；positive_contribution_gate_pass=false | 匹配初始状态、噪声与 32 步曝光下的负主结果。不能转而挑 LoRA-off 比较来宣称已过关。六 checkpoint 有 160 tensors、80 层加载验证只证明实现可运行。 |
| Day16 SAM3 观察器 | `results/DAY16_GEOMETRY_PROMPTED_SAM3_RESULT_20260907.md`：7/9 gates；planar strand IoU=0/0；relative3d LoRA-off/uniform/weighted=0.7284/0.6954/0.7246 | 仅一个 seen synthetic、几何支架导出的提示和目标掩码上的表示信号；并非独立标签。weighted 比 off 低 0.0038，不支持学习贡献。 |
| Day16 双筷实例 | 同文件：控制图两筷 mask IoU=0.5295，大于门槛0.50；中心距7.30 px，小于8 px；weighted mask IoU=0.6831 | 总观察器 gate 失败，不能接入 VACE guidance；“SAM3 已识别出两根筷子”不成立。 |
| Day18 支持区修复 | `results/DAY18_HIGH_LIFT_SUPPORT_REPAIR_20260908.md`：旧/新 support=16.1877%/17.9857%；最终餐具像素区外 1409/4421→0/4421；全路径 35309/35309 完全不透明覆盖（alpha=255） | 同一 raw 重投影隔离出真实合成缺陷，这是目前最清楚的局部工程正向证据；非新学习算法。新推理同时改控制栅格、VACE mask、合成alpha，不能分别归因。 |

Day18 指定提举位移 64.181→150.314 px 是支架控制数值，不是生成图测量值。两筷夹持、面条接回碗内、照片真实感仍未过关。

`results/CURRENT_RESULTS_INVENTORY.md` 保留历史快照；其早期 “no rendered fried-rice case”“fork missing” 和后面的停电未完成叙述已被新日期结果超越。不要把历史段当作当前待办，也不要使用历史“三图前端 generalizes”措辞推出本项目总体泛化。

## 4. 图片来源与发布范围

直接来源：`artifacts/day19_multimaterial_dataset_v3/dataset_manifest.json`；条款记录：`benchmark/DATASET_PROVENANCE.md`。

- 汤：`artifacts/day2_anchor_sources_x4_v1/soup_002.jpg`，SHA `6f633e7416d069c149fecfb37f5e67d7f4b568487788ee73cba75b4702d188e3`。
- 炒饭：`artifacts/day2_anchor_sources_x4_v1/rice_007.jpg`，SHA `60c6e0870bf35caf686db9bcdf636f2cdd302762d5902a4b5937214f5c5b88cc`。
- 二者标记 `real_previously_used_upscaled_pilot`、`UECFOOD256_noncommercial_research_only_no_public_redistribution`；既非未见测试图，也非高分辨率真值。仓库记录的 OSEDiff 超分可能合成细节。
- 蛋糕：`artifacts/day19_multimaterial_sources_v1/cake_imagegen_input.png`，SHA `ab287a7b20fe428e907214509ca36e0fc9f9999f1a85588044f4d9edb31f5e21`；标记 `synthetic_imagegen_input`。生成的是输入，不得包装成真实拍摄或算法输出。该字段不是完整法律授权证明。

沿用当前本地条款记录，不在本审计作新的法律判断。源图及含源图的报告/对照图不应直接公开上传 GitHub；公开发表材料的使用许可需先确认。可以准备去图的元数据、路径、哈希及方法代码清单，不因此宣称任何新授权已取得。

## 5. 缺少的正向贡献及最小下一判据（建议，未冻结/未执行）

缺口不是“再得到几张图片”，而是一个可归因且不依赖目标支架自身打分的改善：对照同样预算、同样输入与支持区，仅改变所提方法；独立检查动作接触收益，且外观与背景不恶化。

建议先选单一主要假设，例如“显式接触/可见性控制改善双筷夹持”，不要同时改提示、mask、材质、采样seed和损失。先完成不需要 GPU 的评估前置项：

1. 为现有一组输出和反事实控制建立独立 strand centerline、两根筷子实例、接触点标签；审阅者不得直接复制生成控制的支架掩码。图像编号盲化，0/3/6/10/15/20 帧固定，不挑最好帧。
2. 观察器必须在未用于调提示的合成控制和擦除/实心块反例上通过全部冻结检查，尤其双筷分离；当前 7/9 不够。失败时只做评估器修复，不启动 guidance。
3. 最小机制试验为 baseline/proposed 两臂、同图、预先固定 3 seeds（建议 1/2/3），所有超参数与输出路径先冻结。该六次运行只能诊断同图机制，不能叫泛化。
4. 提议新主判据：两位独立评审均判定 proposed 在至少 2/3 配对 seed 的“两个独立餐具实例 + 可见食物接触 + 连续负载”组合检查改善；其余配对不能恶化，外观评分不得整体下降。应先明确定义评分和无法判断的处理，再冻结。此阈值是下一设计建议，不能追溯套用旧结果。
5. 若仍评估 Day13 拓扑加权，就沿用原主判据 topology/pinch 各改善至少 5%并有清晰可见语义增益；新的独立语义分数不能抹去旧数值失败。不要把像素 MAE 当接触真值。
6. 仅机制通过后才另行设计未见图、多材料与足够样本的评估；创作者耗时、编辑接受率、观看愉悦度需独立用户研究，不由图像示例推导。

所有 GPU 执行仍需主任务统一调度、用户授权范围与新的实时资源预检；本文件不启用自动心跳、不批准任何推理，也不解锁 blind fork。

## 本轮审计边界

已读 README、CURRENT_RESULTS_INVENTORY、Day20 MD/JSON、两运行 manifest、核验 JSON、冻结数据 manifest、Day13 JSON、Day16/18结果 MD、数据来源说明；51个输出重新校验。未重新检查远端、未独立标注、未重跑180项测试或重新人工评阅 contact sheets。结论中“已证实”仅限对应证据层级。
