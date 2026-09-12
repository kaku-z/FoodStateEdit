# FoodStateEdit 跨方法对比与评估数据状态

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-12
- Verification Status: ANALYZED
- Version Label: cross_method_validation_v1

## 结论

目前已经有一组可追溯的**同四张输入、单 seed 开发对比**，但还没有 Qwen-Image-Edit、FLUX Kontext、ChronoEdit 的原生结果，因此不能把现状写成“优于多个强模型”的正式 benchmark。

新整理的横向图为 [cross-method pilot grid](../artifacts/day21_cross_method_pilot_v1/cross_method_pilot_grid.png)。它包含 Input、Vanilla GeoEdit、GeoEdit + union mask、VACE static 和 Ours (dynamic 2-D)。16 张生成图均按各自原始 run manifest 的 SHA-256 重新核验；四张输入来自冻结锚点原图并按原方法尺寸重建。图中没有后期修图，也没有把 Day 21 蛋糕的单例最佳结果混入旧四样本比较。

## 已有同输入 pilot 数据

共同范围：四个挑选的开发锚点，seed 1，21 帧，20 steps；动作和照片判定来自一次内部、非盲人工诊断。

| 方法 | 技术完成 | 支持区外精确保留 | 暂定动作通过 | 暂定照片通过 | 正确解释 |
| --- | ---: | ---: | ---: | ---: | --- |
| Input / no edit | 4/4 | 4/4 原图一致 | 不适用 | 不适用 | 负控制 |
| Vanilla GeoEdit | 4/4 | 4/4 | 1/4 | 0/4 | 唯一已有的外部方法 baseline；汤动作暂定通过 |
| GeoEdit + unified action mask | 4/4 | 4/4 | 0/4 | 0/4 | 同一 GeoEdit/Wan 后端的掩码对照 |
| FoodStateEdit staged exclusive | 4/4 | 4/4 | 0/4 | 0/4 | 早期分层投影版本 |
| Native VACE static proxy | 4/4 | 4/4 | 0/4 | 0/4 | 去掉 GeoEdit TTM 后仍失败 |
| FoodStateEdit dynamic 2-D | 4/4 | 4/4 | 0/4 | 0/4 | raw 拉面帧有 1/4 拓扑信号，但投影结果不通过 |

这里的多数生成方法共享 `Wan2.2-VACE-Fun-A14B` 外观后端，比较的是**控制表示与投影策略**，不是多个独立基础模型的排行榜。`4/4` 区外精确保留由显式投影保证，不能解释为生成模型自主保持背景。

## 最新 3-D 与自然度数据（独立开发实验）

| 实验 | 样本 | 可报告数据 | 结论边界 |
| --- | ---: | --- | --- |
| Day 20 planar vs relative-3D | 汤、炒饭、蛋糕各 1 张 | clear 3-D semantic gain `0/3`；严格照片通过 `0/3` | 六条件技术完成，不支持稳定 3-D 优势 |
| Day 21 VACE scale 1.0/0.8/0.6 | 同三张开发图 | 最终图外观清晰改善 `2/3`；严格最终图通过 `1/3`；完整时序通过 `0/3` | 蛋糕 scale 0.6 是局部正向结果，不是通用胜率 |
| Day 13 topology weighting | 一个 seen-synthetic 面条样本 | topology MAE 改善 2.22%，pinch MAE 改善 2.83% | 均低于预设 5% 门槛，主结果为负 |
| Day 16 SAM3 observer | 同一 seen-synthetic 样本 | planar strand IoU 0；relative-3D 0.695--0.728；总门控 7/9 | 观察器依赖几何提示且未分离两根筷子，不能当独立成功率 |

Day 20 的原生支持区外 MAE 为汤 `10.251/10.254`、炒饭 `7.827/7.869`、蛋糕 `10.754/10.704`（planar/relative-3D）。投影编码前区外最大差均为 0；这只是合成完整性指标，不是动作、接触或真实感分数。

## 尚缺的强模型对比

| 模型 | 当前状态 | 为什么不能放结果列 |
| --- | --- | --- |
| Qwen-Image-Edit | 未执行 | 审计到的服务器路径没有可用完整权重；不能把占位目录当模型 |
| FLUX Kontext | 未执行 | 没有冻结的本地安装、版本和输出 |
| ChronoEdit | 未执行 | 没有冻结的本地安装、版本和输出 |
| GeoEdit | 已有 4 张单 seed 输出 | 可放 pilot 列，但不是三 seed/held-out 正式比较 |

因此，中期发表可以展示“Input → GeoEdit → VACE → Ours”的开发对比和失败分析；不能展示虚构的 Qwen、FLUX、ChronoEdit 图片，也不能写 `Ours > all baselines`。

## 评估可信度

当前结果只能做描述统计，不报告 p 值、置信区间或效应量。原因是样本为四个挑选的开发锚点、每种方法一个 seed，且评价者不是独立盲评。11 类统计谬误已全部检查；主要风险是开发样本选择偏差、多轮试验导致的 look-elsewhere effect、garden of forking paths，以及把蛋糕单例改善推广到全部材料。

要形成正式表格，最低还需：相同输入上的 Qwen/FLUX/ChronoEdit 输出，预先固定的三 seeds，每类未见图片，以及至少两名独立盲评者按动作、接触、食物守恒、来源变化、照片自然度分别评分。在这些条件满足前，正式强模型 benchmark gate 为 **fail**。

机器可读摘要：[day21_cross_method_evaluation_status_v1.json](day21_cross_method_evaluation_status_v1.json)。
