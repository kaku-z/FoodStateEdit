# 第一口实验的复现与评价

本轮为 8 张已见开发图的受控实验。实验对象是入口前、由一只手持餐具托起的一口食物。实际调用矩阵包括 64 张主结果、发现标注问题后追加的 24 张修正复核，以及 12 张高度诊断，共 100 次。三部分分别报告，不能合并为主方法成功率。

## 固定输入与生成

1. `benchmark/first_bite_20260928/cases.bound.v1.json` 记录真实源图位置、SHA256、材料、取食区域、容器与相机假设。图像授权说明见 `benchmark/DATASET_PROVENANCE.md`；不要公开再分发原始数据。
2. `scripts/prepare_first_bite.py` 生成 `outputs/first_bite_20260928_bundle_v1`。该目录已存在，不要覆盖重建。所有方法共用相同源图、尺寸、源位置标注、40 步与 CFG 4；种子为 281、913。
3. `results/FIRST_BITE_EXECUTION_FREEZE_20260928.json` 固定生成前的输入与程序。`config.json` 中固定模型文件和本机 diffusers pipeline 的 SHA256。服务器使用已有离线 Qwen-Image-Edit-2511，不下载或更新模型。
4. 在已确认空闲的两张 A6000 上运行 `scripts/run_first_bite.py --bundle BUNDLE --output NEW_DIR --physical-gpus 0,1 --shard 0 --shards 4`。其他三个分片分别使用不同 GPU 对和分片号。输出目录必须全新；禁止覆盖或替换失败 seed。
5. 本轮实际采用每个作业 7200 秒上限。每格保存 `raw.png` 与 `request.json`，`run_manifest.json` 保存参数、时长、哈希与失败记录。结果为生成器原生输出，没有最终像素贴回或中途潜变量投影。

## 高度诊断

`scripts/prepare_first_bite_height.py` 从冻结主输入生成单独的高度包：每种材料第一张源图 × h=0/0.08/0.16 × seed 2027。三档共用逐字相同的中性文字和逐字节相同的源图，仅改变第二张空间引导图。`invariant_checks.json` 记录这些不变量与投影位置。

高度单位为手工标注容器直径，相对于局部食物/液面支撑参考面，不是测量得到的厘米。h=0 是低高度对照；汤勺底低于边沿，不能把勺碗顶面的 h 当作其最低点离地距离。阴影、工具、负载和手柄随相对几何一起改变。若输出未跟随，只能否定本实现的控制效果，不能直接否定所有三维约束方法。

生成主结果前已约定：只有 D 在主实验中出现可见抬升才运行这 12 格。输入仍是预先指定的四个第一槽位，不因输出好坏选择图像。旧 `results/FIRST_BITE_HEIGHT_FREEZE_20260928.json` 冻结的 v1 高度包未产生任何模型调用。发现源区域错误后，该包保留并由 `results/FIRST_BITE_HEIGHT_SUPERSESSION_20260928.json` 声明替代；实际运行修正源区域的 v2 包，与 24 次标注复核共同冻结在 `results/FIRST_BITE_SUPPLEMENT_FREEZE_20260928.json`。四张照片、三个高度和 seed 没有改选。

## 标注修正补充矩阵

详见 `docs/FIRST_BITE_ANNOTATION_REPAIR_PROTOCOL_20260928.md`。`scripts/prepare_first_bite_supplement.py` 构造 36 格显式任务矩阵；现有冻结目录不能覆盖。实际 bundle 为 `outputs/first_bite_20260928_supplement_bundle_v2`，通过 `scripts/run_first_bite_supplement.py` 在四组互不重叠的 GPU 对上各运行 9 格，每个分片有 5400 秒上限。旧高度启动器有失效保护，不可再运行 v1。

24 格只修复三张图的源取食区域，使用原四方法、原 seed 281/913 和相同源图。它是输出检查之后设计的诊断，不是独立验证。12 格高度对照共用中性文字与源图字节，只改变软参考图，无法识别模型是否真正进行了三维推理。

```text
py -3.13 scripts/monitor_first_bite.py --supplement --collect
py -3.13 scripts/review_first_bite_supplement.py
py -3.13 scripts/report_first_bite_supplement.py
```

第二条要求全部 36 格完成且哈希一致，输出目录必须全新。第三条验证并汇总已经人工记录的助手观察，不会自动生成评分。修正前后配对图保留全部 24 对，另有四组各三档高度总览。

## 收集、完整性与评阅

本轮采集工具 `scripts/monitor_first_bite.py --collect` 使用已经授权的本机 SSH 隧道，读取四个分片状态，仅下载已经完成的 raw 文件并核验哈希。它不向生成器写入参数，也不自动评分。

所有主格完成后执行：

```text
py -3.13 scripts/review_first_bite.py --bundle outputs/first_bite_20260928_bundle_v1 --runs outputs/first_bite_20260928_gp40 --output outputs/first_bite_20260928_review
py -3.13 scripts/report_first_bite.py --review-dir outputs/first_bite_20260928_review
```

第一条要求所有预定格、分片状态、输出哈希、原图哈希和指令哈希完全匹配，然后生成 8 张不筛选总览和 3 份随机化空白评阅表。第二条只汇总已经记录的助手非盲诊断，不生成新评分。

独立人评需要三位真实评阅者分别填写 `blind_review_1.html`、`blind_review_2.html`、`blind_review_3.html` 并导出 JSON。不要将 `PRIVATE_blind_keys.json` 交给评阅者，不要让三位评阅者先看有方法名称和助手意见的总览网页。三份空白表并不代表已完成三人评价。

完整通过要求同一位评阅者对五个二元维度全部通过且照片真实感至少 4 分，再要求至少两位评阅者各自完整通过。禁止从不同评阅者那里逐项拼出一次完整成功。`scripts/score_first_bite_ballots.py` 检查空白、重复身份和逐人合取逻辑；它无法从文件内容自动证明实际评阅者独立性。

助手观察保存在 `results/FIRST_BITE_ASSISTANT_REVIEW_20260928.json`，每条绑定 raw 哈希。它是开发诊断，不是人类盲评，更不是论文中的独立测试。联合成功和各维度分别报告；无法判断单列并计入未通过。

## 结论边界

- 同一源图的两个 seed 不是两个独立菜品；8 张源图不能提供大规模泛化结论。
- 两张豆腐不能代表全部固体食物；低分辨率原图也会限制细节保持的可判断性。
- 图片里的“第一口”是动作条件，单帧无法证明真实进食历史。
- 场景重绘、缺口与负载不一致、餐具复制、假手形、未响应高度均应单独记录。
- 未收到真实独立评分前，不宣称人评通过；未满足开发晋级门槛，不解锁 480 张未见正式测试。
