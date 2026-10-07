"""Descriptive report for a deliberately constructed hidden-material ambiguity."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    report = json.loads((args.results / "report.json").read_text())
    rows = json.loads((args.results / "appearance_rows.json").read_text())
    tools = json.loads((args.results / "tool_rows.json").read_text())
    root = Path(__file__).resolve().parents[1]
    for path, digest in report["code_sha256"].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, path
    resolutions = sorted(int(r) for r in report["aggregate_by_resolution"])
    mean = lambda res, key: report["aggregate_by_resolution"][str(res)][key]["mean"]
    middle = resolutions[1]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for key, label, color in (("source_only_cut_mae_a", "Cut: layered world A", "#237c97"),
                              ("source_only_cut_mae_b", "Cut: alternate world B", "#c37128"),
                              ("source_only_other_food_mae_a", "Other food, world A", "#698556")):
        axes[0].plot(resolutions, [mean(r, key) for r in resolutions], "o-", label=label, color=color)
    axes[0].set_ylabel("Source-observed appearance RGB MAE")
    axes[0].set_xlabel("Output image resolution (pixels per side)")
    axes[0].legend(fontsize=8)
    for key, label in (("source_only_equal_pair_cut_mae", "Actual equal-pair cut MAE"),
                       ("equal_pair_cut_mae_lower_bound", "Ambiguity-pair L1 lower bound")):
        axes[1].plot(resolutions, [mean(r, key) for r in resolutions], "o-", label=label)
    axes[1].set_ylabel("Cut-surface RGB MAE (equal synthetic pair)")
    axes[1].set_xlabel("Output image resolution")
    axes[1].legend(fontsize=8)
    axes[2].plot(resolutions, [mean(r, "selection_plus1_edit_rgb_mae") for r in resolutions], "o-", color="#237c97")
    axes[2].set_ylabel("Edit RGB MAE with +1-cell selection error")
    axes[2].set_xlabel("Output image resolution")
    for axis in axes:
        axis.set_xticks(resolutions)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=.2)
    fig.suptitle("Same observed source, different hidden material: synthetic diagnostic", fontsize=14)
    fig.text(.5, .02, "Oracle geometry and background; source-only appearance. The voxel grid stays fixed. The lower bound applies only to the constructed equal pair.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .93))
    fig.savefig(args.results / "diagnostic_curves.png", dpi=170)
    plt.close(fig)
    max_source_error = max(r["source_pair_max_abs_difference"] for r in rows)
    min_gap = min(r["target_pair_cut_rgb_mae"] for r in rows)
    blocked = [r for r in tools if r["blocked"]]
    too_thick = [r for r in tools if not r["blocked"] and r["gap"] < r["thickness"]]
    clear = [r for r in tools if not r["blocked"] and r["gap"] >= r["thickness"]]
    lines = ["本轮实验支持一个明确判断：正确的材料输运不能补回源图没有提供的隐藏表面信息。算法需要区分可由观测验证的材料身份与未观测部分的合理生成。", "",
             "Material Passport：academic-research-suite / experiment-agent；2026-09-28；ANALYZED。依据保存的服务器结果、代码哈希和测试日志分析。", "",
             f"服务器 {report['host']}；{report['unique_geometries']} 个不同源几何；{report['appearance_trials']} 组外观/分辨率试验；{report['tool_trials']} 组工具几何试验；服务器运行 {report['seconds']:.2f} 秒；13 项代码测试通过。", "",
             "| 输出分辨率 | 两个源图最大差异 | 两种真值切面差异 MAE | 源图外观预测在 A 切面的 MAE | 在 B 切面的 MAE | 一格选块偏差的编辑区 MAE |",
             "|---|---:|---:|---:|---:|---:|"]
    for res in resolutions:
        lines.append(f"| {res} | {mean(res,'source_pair_max_abs_difference'):.6f} | {mean(res,'target_pair_cut_rgb_mae'):.4f} | {mean(res,'source_only_cut_mae_a'):.4f} | {mean(res,'source_only_cut_mae_b'):.4f} | {mean(res,'selection_plus1_edit_rgb_mae'):.4f} |")
    lines += ["", f"全部 {len(rows)} 组源图像素严格相同，最大差异 {max_source_error:g}；移开后，两种切面真值均有差异，最小组 MAE 为 {min_gap:.4f}。两种材质只在操作前完全位于内部的分区界面上不同，因此源图一致不是靠降低图像分辨率制造的。", "",
              "外观预测器只读取源图颜色及其已知规范坐标，按相同法线的邻近源样本给目标表面着色；没有读取目标 RGB 或完整隐藏材质。几何、材料坐标和背景依旧由 oracle 提供，因此这是外观信息消融，尚不是单图端到端编辑。", "",
              f"在 {middle} 像素分辨率下，对构造出的 A/B 两种世界等权评价，切面平均 L1 误差的下界为 {mean(middle,'equal_pair_cut_mae_lower_bound'):.4f}，本预测器为 {mean(middle,'source_only_equal_pair_cut_mae'):.4f}。由三角不等式，任何接收相同源图并输出同一结果的估计器，都不能同时零误差重建这两个不同真值。这个下界只针对人为构造的等权歧义对，不代表真实食物分布的误差下界。", "",
              "该结果不要求系统放弃生成合理切面。它要求把未观测部分标成先验假设或多个候选，而非声称恢复了唯一真实内部纹理。若任务需要恢复真实切面，必须增加可区分两种状态的观测。", "",
              "分辨率实验仅把输出图像采样从低分辨率提高到高分辨率，始终保留 56³ 体素网格和同一个一格选块错误。因此它检查栅格图像采样影响，不是体素精细化或状态估计精度实验。", "",
              f"工具试验中，可行 {report['tool_feasible']} 组，拒绝 {report['tool_rejected']} 组；与该构造场景的解析预期不一致 {report['tool_expected_mismatches']} 组。间隙不足的 {len(too_thick)} 组全部拒绝，入口阻挡的 {len(blocked)} 组全部拒绝，间隙足够且路径畅通的 {len(clear)} 组全部通过。", "",
              "工具被建模为有限厚度平托片，采用固定方向的直线插入和固定姿态平移。碰撞检测对移动 AABB 使用连续时间的区间相交；边界相切允许。支撑只检查几何质心投影，未验证摩擦、力矩、变形、勺柄、曲面勺碗或其他绕行路径。因此拒绝是指定路径不可行，不能解释为所有路径都不可达。", "",
              "当前实现改动：渲染器支持外观回调并输出表面法线；加入只读源 RGB 的外观估计、内部切面歧义构造以及连续平移托片碰撞检测；默认解析渲染行为保留。", "",
              "数据边界：用户已确认没有真实配对照片及几何数据，本次继续使用合成数据。没有训练模型，也没有证明真实照片质量或相对强基线的优势。", "",
              "实验档案：protocol.json、appearance_rows.json、tool_rows.json、report.json、tests.log、experiment.log；ambiguity_*.png 展示同一源图、两个不同目标真值及仅用源颜色的预测。"]
    (args.results / "analysis_zh.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(args.results / "analysis_zh.md"), "max_source_difference": max_source_error,
                      "middle_resolution": middle, "cut_difference_mae": mean(middle,"target_pair_cut_rgb_mae"),
                      "tool_feasible": report["tool_feasible"], "tool_rejected": report["tool_rejected"]}))


if __name__ == "__main__":
    main()
