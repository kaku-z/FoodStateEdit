# Day 23：几何锁定的餐具局部扩散修复预检

## 结论

已把“最终关键帧的局部写实化”加入实验设计，但尚未执行 GPU 推理。服务器的 `guo-z` 存储中没有找到完整可运行的 Qwen-Image-Edit 基座；现成可用且最接近需求的图像编辑器是 gp40 上的 ChordEdit 与完整 SD-Turbo fp16 组件。

新阶段不是再次生成整张图：VACE/relative-3D 结果锁定动作结构，ChordEdit 只提出一个局部外观候选，硬支持区外逐像素复制原图。自动结构门或人工语义门任一失败，`08_final.png` 必须回退为修复前关键帧。

## 权重审计

- Qwen-Image-Edit：未发现完整基座；PhysicEdit 仅有约 2.24 GiB LoRA，缺 Qwen-Image-Edit-2509 与 DINOv2，不能运行。
- ChordEdit：代码存在；`sd-turbo-fp16-prepared` 包含 UNet、VAE、CLIP text encoder、tokenizer、scheduler 与 `model_index.json`，选为 v1 后端。三份主权重分别为 1,731,904,736、680,820,392 与 167,335,342 bytes；其 SHA-256 已逐文件写入冻结配置。
- InteractDiffusion：权重完整，但属于带交互框的文生图，不是输入图局部编辑器。
- SDXL base 1.0：基座存在，可做普通 img2img，但当前没有已验证的几何锁定食物局部编辑入口。
- Z-Image：关键权重仍为 `.incomplete`，不可用。
- Wan VACE：权重完整，保留为视频生成阶段，不拿同一模型重复充当独立图像编辑对照。

## 固定试验

首例只检查 Day 19 soup-relative3D 的勺子写实化。固定单候选、seed 1、ChordEdit one-step、`step_scale=1.0`、融合强度 `0.65`。修复区由“最终帧与原图差异”经小幅膨胀后再与既有 allowed-edit region 相交得到，不允许全图重绘，也不允许从多张结果中挑最好的一张。

通过条件包括：支持区外最大差异为 0、局部改变量不过大、结构边缘相似度达标，并人工确认餐具类别/数量、接触、载荷、来源对应关系均未退化且照片真实感确有提升。任何不确定项按失败处理。

## 科学边界

这是一个单样本开发诊断。即使通过，也只能证明该接口在一个已见 soup 样本上有潜力；不能据此声称泛化、物理正确性、独立人评优势或论文级照片真实感。
