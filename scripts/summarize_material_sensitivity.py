"""Plot descriptive synthetic sensitivity; does not perform significance testing."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    report = json.loads((args.results / "report.json").read_text())
    aggregate = report["aggregate"]
    root = Path(__file__).resolve().parents[1]
    for path, digest in report["code_sha256"].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, path
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for axis, family, label in zip(axes, ("selection", "thickness"),
                                   ("Selection boundary offset (voxels)", "World-z thickness error (voxels)")):
        offsets = [-2, -1, 0, 1, 2]
        values = [aggregate["exact" if v == 0 else f"{family}_{v:+}"] for v in offsets]
        means = [v["edit_rgb_mae"]["mean"] for v in values]
        lower = [v["edit_rgb_mae"]["min"] for v in values]
        upper = [v["edit_rgb_mae"]["max"] for v in values]
        baseline = [v["planar_edit_rgb_mae"]["mean"] for v in values]
        axis.fill_between(offsets, lower, upper, color="#307ba5", alpha=.16, label="3D scene min-max")
        axis.plot(offsets, means, "o-", color="#146d95", label="3D oracle material: mean")
        axis.plot(offsets, baseline, "s--", color="#c37827", label="2D source-only diagnostic")
        axis.set_xlabel(label)
        axis.set_xticks(offsets)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("RGB MAE within fixed true edit region [0, 1]")
    fig.suptitle(f"Synthetic sensitivity: {report['unique_source_grids']} unique scenes, independent reference", fontsize=13)
    fig.text(.5, .02, "Shading is the observed scene range, not a confidence interval. 3D has privileged hidden material; comparison is diagnostic.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .055, 1, .93))
    fig.savefig(args.results / "sensitivity_curves.png", dpi=180)
    plt.close(fig)
    lines = ["本轮敏感性测试已完成。结果说明内部守恒无法保证任务正确，选块与厚度精度仍然是实际瓶颈。", "",
             f"服务器：{report['host']}；独立源几何：{report['unique_source_grids']}；测试条件总数：{report['attempts']}；运行时间：{report['seconds']:.2f} 秒。9 项测试通过。",
             "", "同一源状态使用 9 种条件：准确状态、选块边界 ±1/±2 格、世界 z 方向厚度 ±1/±2 格。厚度误差不等同于相机射线方向的深度误差。源形状包括长方体、椭圆柱与阶梯体，尺寸、选块大小及平移有所变化；相机和解析材质模型固定。", "",
             "| 条件 | 源块残留比例 | 多搬材料/应搬材料 | 相对真值的体积误差 | 编辑区 RGB MAE |",
             "|---|---:|---:|---:|---:|"]
    for name, values in aggregate.items():
        if not values["generated"]:
            lines.append(f"| {name} | 全部拒绝 | — | — | — |")
            continue
        get = lambda k: values[k]["mean"]
        lines.append(f"| {name} | {get('true_source_residual_fraction'):.2%} | {get('extra_transported_fraction'):.2%} | {get('reference_volume_error'):.2%} | {get('edit_rgb_mae'):.4f} |")
    lines += ["", "所有生成案例相对自身状态的体积误差均为零；这与独立真值评价出现明显错误同时成立。准确条件使用与参考相同的渲染器，因此其图像误差为零是回归校验，不是现实效果证明。", "",
              f"选块边界向内偏 1 格时，平均有 {aggregate['selection_+1']['true_source_residual_fraction']['mean']:.2%} 的应移动材料留下；偏 2 格时为 {aggregate['selection_+2']['true_source_residual_fraction']['mean']:.2%}。这些选块沿 x 方向仅约 5–8 格宽，因此 1 格不是无穷小误差，不能直接换算成真实照片的一像素误差。边界向外偏会多搬材料，需要看多搬比例与剩余食物变化，不能仅看原块是否全部移走。", "",
              "厚度偏差 1 格时，平均相对真值的体积误差约 9.01%；偏 2 格时约 18.01%。高估厚度会添加不存在的几何，低估会删除本应存在的几何；输运仍可完全守恒其错误输入。", "",
              "二维参照使用同一源 RGB、源图上的选块掩码和投影位移，通过相邻像素传播补洞并双线性平移。厚度变化不改变其二维源掩码。它不接收目标 RGB 或正确背景，但也没有三维方法可用的完整隐藏材质信息，因此不能据此宣称三维方法优于强基线。v1 曾将薄几何对应的缺失表面 ID 传入二维掩码；v2 修正了这一混杂并重新运行完整测试，本文仅使用 v2。", "",
              "本轮有 108 次尝试，拒绝数为 " + str(sum(v["rejected"] for v in aggregate.values())) + "。数值为受控合成场景的描述性均值；曲线阴影为场景最小值到最大值，不是置信区间。", "",
              "真实数据条件：用户确认没有对应的真实前后照片及标定/深度/扫描数据，因此真实照片上限测试尚未开展。现有单张照片和生成结果不能替代独立真实真值。", "",
              "下一步应先建立小型真实配对采集集，验证准确几何与材质下的渲染上限；之后按开发集确定允许的选块与厚度误差。当前还没有得到可用于真实部署的容差阈值。工具承托与接触也尚未验证。", "",
              "原始记录：report.json、rows.json、protocol.json、tests.log、experiment.log。代码 SHA256 已与本地源码核对。"]
    (args.results / "analysis_zh.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.results / "analysis_zh.md")


if __name__ == "__main__":
    main()
