# Day20 非面条结果与发表材料

2026年9月9日。汤、炒饭、预切蛋糕三类的平面与相对3D对照共六个条件已经完成并取回。它们可用于展示多材料原型和分析局限；本次未观察到足以支持“3D明显优于平面”的清晰语义增益。

## 本地图片

[三类统一对照图](../artifacts/day20_non_noodle_presentation_v1/non_noodle_comparison.png)采用固定第20帧。左列为输入，中列为平面控制加VACE，右列为相对3D控制加VACE，不修图、不逐方法挑帧。

- [汤的四列对照](../artifacts/day20_non_noodle_presentation_v1/soup_comparison.png)
- [炒饭的四列对照](../artifacts/day20_non_noodle_presentation_v1/rice_comparison.png)
- [蛋糕的四列对照](../artifacts/day20_non_noodle_presentation_v1/cake_comparison.png)

四列图额外展示程序3D控制图，控制图不是模型生成结果。原始输出和投影后输出均保留，不把最终合成图当作原生VACE输出。

## 执行与恢复

Day19原始八条件任务在写完第五个条件 cake__planar 后，输出进度时发生 BrokenPipeError，状态为 technical_failure_preserved。原始任务仍是未完成批次，不能改称八条件成功。

本次新冻结的恢复配置只允许 cake__relative3d；输入、控制、提示、种子、模型、推理预算及投影方式均不变。没有重复已完成条件，没有执行两个面条条件，没有换种子，也没有下载模型。

恢复采用独立的新输出和预检路径，输出日志写入服务器文件并与SSH会话断开依赖。gp40 GPU5 预检通过：RTX A6000，空闲48539 MiB，利用率0%，零计算进程，主机可用249457 MiB。开始于06:39:20 UTC，完成于06:43:14 UTC，总耗时234.354秒。

资源审计同时覆盖gp38至gp42：gp38有他人负载且没有符合门槛的卡；gp39空闲但本次未使用；gp40的GPU5至7符合当次只读快照门槛；gp41是A40，gp42是Blackwell，均排除。未干预他人进程，自动心跳未恢复。

## 质量评审

评审者为本助手，检查每个条件的0、3、6、10、15、20帧和最终图；这不是独立人工盲评，也不是对视频所有帧逐帧人工验收。无法判断不计为通过。

| 材料 | 两组中可见的动作线索 | 尚未解决的问题 | 照片自然度 | 清晰3D语义优势 |
| --- | --- | --- | --- | --- |
| 汤 | 单个汤匙接近汤面并携带汤液移动 | 匙缘与手柄轮廓偏硬，液面偏平，真实接触仍不充分 | 未通过 | 未建立 |
| 炒饭 | 米粒配料块随铲面移动 | 铲子呈框状几何轮廓，承载阴影弱，来源减少不清楚 | 未通过 | 未建立 |
| 蛋糕 | 单个预切块被提起，可见对应缺口 | 叉齿及承载接触不明确，局部边缘与阴影仍不充分 | 部分 | 未建立 |

蛋糕是比较清楚的动作示例，不能称为完成自然切割或已具物理正确性。汤和炒饭有编辑原型结果，但不宜包装为商业级照片。两组主要表现出小幅外形与透视差异，本次不能将这种差异直接写成算法改进。

## 技术核验

旧批次40个文件和恢复批次11个文件逐文件与远端SHA-256一致；额外验证恢复预检文件，共52个文件。失败标记、异常堆栈、日志和已完成结果全部保留。

六个条件各有21帧，688×512，seed=1，20 steps，VACE scale=1，LoRA off，TTM off。两个独立批次分别只有一次pipeline加载；不是跨两个任务总共只加载一次。

最终PNG可由保存的raw.mp4、参考图和alpha逐像素重建。以下数值是0至255像素尺度的诊断，不是动作或真实感评分。

| 条件 | 原生输出支持区外MAE | 投影编码前区外最大差 | 视频编码后区外MAE |
| --- | ---: | ---: | ---: |
| soup planar | 10.251 | 0 | 2.914 |
| soup relative3d | 10.254 | 0 | 2.915 |
| rice planar | 7.827 | 0 | 3.724 |
| rice relative3d | 7.869 | 0 | 3.724 |
| cake planar | 10.754 | 0 | 3.130 |
| cake relative3d | 10.704 | 0 | 3.130 |

支持区外差为零来自最终投影，不证明模型自主保护背景；有损视频编码会重新引入差异。区内变化和全部输出哈希见核验JSON。此前不存在的文件不补造，原始失败时间线不改写。

完整测试180项通过，无失败、错误或跳过。

## 可报告的贡献和社会意义

本次增加了三类食物动作的可复现对照材料，把“是否有动作”“接触是否可信”“素材自然不自然”分开诊断。这支持多材料编辑原型和后续研究设计，但没有证明新扩散算法、学习增益、泛化或物理正确性。

社会用途聚焦于帮助食物内容创作者用现有图片表达盛取瞬间，并让观看者更清楚地理解动作、体验细节的趣味。节省制作时间、降低成本和增加观看愉悦感均是待验证目标，不是本次实验结论。

[正式发表准备计划书](FORMAL_PRESENTATION_PLAN_20260909.md)列出了两天日程、具体受益者、探索性反馈设计、日文动机和不能夸大的主张。PDF交付文件为 output/pdf/FoodStateEdit_presentation_plan_20260909_v2.pdf，已渲染检查五页。文档技能要求的配套Word渲染器不可用，因此改用PDF技能生成并核验，而没有交付未核验DOCX。

汤和炒饭沿用真实、已使用的研究图片；蛋糕输入为图像生成模型制作并已披露。来源限制仍有效，图片留在本地，未上传GitHub；公开发表文件的图片授权需另行确认。

## 证据位置

- 配置 configs/day20_cake_recovery_v1.json
- 结果摘要 results/day20_non_noodle_result_20260909.json
- 远端哈希 results/day20_day19_remote_sha256.txt 与 results/day20_cake_remote_sha256.txt
- 恢复预检 results/day20_cake_recovery_preflight.json
- 原始批次 artifacts/day19_multimaterial_gp40_recovered_20260909_v1
- 恢复批次 artifacts/day20_cake_recovery_gp40_20260909_v1
- 逐项核验 artifacts/day20_non_noodle_presentation_v1/verification.json
- 旧远端输出 /host/space0/guo-z/tf-ufi/outputs/day19_multimaterial_gp40_v1
- 新远端输出 /host/space0/guo-z/tf-ufi/outputs/day20_cake_recovery_gp40_v1

本次到此停止新增推理，先供用户审阅成果与计划。后续修正必须有明确缺陷和新冻结版本，不以无目标调参追逐正向结论。
