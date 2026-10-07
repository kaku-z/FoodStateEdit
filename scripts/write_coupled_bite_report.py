"""Write the final report from reviewed outcomes and verified execution records."""
import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
    s=json.loads((r/'analysis/summary.json').read_text());v=json.loads((r/'analysis/integrity_audit.json').read_text());assert v['status']=='verified'
    labels={'A_direct':'A：直接编辑','C_prior_two_reference':'C：旧双参考协议','E_full_single_reference':'E：新状态提示词整图编辑',
            'L0_no_cut_rgb':'L0：局部编辑，去掉缺口 RGB','L1_free_hole':'L1：几何缺口＋自由局部生成','L2_context_locked':'L2：几何缺口＋固定上下文'}
    dims=['lift','utensil_support','matching_source_notch','scene_identity','no_person','photographic_realism'];tab={x['method']:x for x in s['summary']}
    lines=['# 第一口食物图片：联合切取与局部生成修复实验','',
        '本轮已完成设计、修改、服务器生成、全部逐图检查、预先指定复跑和证据归档。问题得到部分改善，尚未彻底解决。',
        '',f"**严格六项同时通过：L2 {tab['L2_context_locked']['successes']}/12，L1 {tab['L1_free_hole']['successes']}/12；直接编辑 A {tab['A_direct']['successes']}/12，旧协议 C {tab['C_prior_two_reference']['successes']}/12。** 这些是同一批四张真实照片、三个固定种子的助手未盲诊断，不是独立人类评审或总体成功率。",'',
        '[查看全部 72 个条件](review/all_results.html) · [逐项评分 CSV](analysis/assistant_review.csv) · [方法与开发记录](METHOD_AND_DEVELOPMENT.md) · [复现说明](REPRODUCE.md)','',
        '## 这次具体改了什么','',
        '先在同一个估计三维实体里切出一口，并为抬起的块规划可见位置和餐具接触。接着分别生成源处缺口与目标第一口的外观，最终把它们合成回原图，源缺口拥有最后的覆盖权。固定上下文只作用于编辑区外，避免把可编辑材质强行拉回人工渲染颜色。',
        '', '修正了坐标轴手性处理、竖图内的目标取景和切块尺寸规划。取景失败时共同调整切走与抬起的块，不会只缩小展示的一口。四张新图均完成几何准备；旧失败图虽然不再报坐标错误，拟合轮廓 IoU 仍只有 0.6193，不能认为它的三维质量已解决。',
        '', '本项目实现的是切取/抬升规划、分支组织、上下文约束与可审计合成；外观生成仍由现有 Qwen 完成，没有训练新的基础生成模型。尚未验证方法新颖性。',
        '', '## 同一批照片上的完整结果','',
        '每行分母均为 12。表内单项为“通过”数量；不确定按未通过处理。全部生成成功，没有技术失败被从分母中删除。',
        '', '| 方法 | 六项同时通过 | 抬升 | 承托 | 对应缺口 | 场景保持 | 无人物 | 真实感 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in s['summary']:
        lines.append('| '+labels[x['method']]+' | '+str(x['successes'])+'/12 | '+' | '.join(str(x['dimensions'][d]['P']) for d in dims)+' |')
    lines+=['','![逐项通过数](figures/criteria_passes.png)','','![逐照片严格通过数](figures/per_photo_success.png)','',
        '三个 L 方法共享同一目标分支。L0 仅去掉源缺口 RGB，仍使用几何掩码、裁剪和目标，因此不能解释为完全无三维消融。L 方法使用两个生成调用，未与 A/C/E 做等算力比较。每张照片的三个种子是重复条件，不能当作三个独立场景。','',
        'L2 在三张照片上优于旧 C，但在 7442 上从 C 的 3/3 退化为 1/3。因此它并非逐场景全面优于旧方法；目标分支的材质与尺寸变化抵消了这张图上的结构收益。','',
        '## 改善与残留问题','',
        '同源切取和分层合成减轻了“只添一块食物，原块没有缺口”的问题。固定上下文能改善部分深洞和洞底不稳定，但它只固定区外，仍不能保证模型一定保住缺口。',
        '', '目标分支仍会缩小第一口、产生融化状边缘，或改变餐具方向。图 7443、种子 41 的原始目标输出已丢失叉柄，只留下孤立叉齿；这是生成阶段的结构失败，不能靠扩大合成掩码补救。图 7496、种子 41 的源缺口仍被补回。图 7459 的部分种子洞底过深或被酱汁遮住，无法确认对应关系。',
        '', '后续应优先验证对餐具连续轮廓、接触区域及缺口深度的直接约束，并把食物材质保留为可编辑部分。当前结果不支持继续只改提示词，也不支持把三维代理几何中的守恒直接当成最终图像中的守恒。',
        '', '## 图片展示','',
        '下面固定展示种子 41、全部四张照片与全部方法，包含失败，不挑最好种子。完整图集包含三个种子；可另外打开原始 640×480 文件，检查预处理补边内外的内容。','',
        '![四图固定种子完整比较](figures/all_sources_seed41.jpg)','',
        '## 执行与复现','',
        f"正式 84 次原始 Qwen 调用生成 72 个评价端点，其中 36 个为分层合成端点。四个 worker 均正常退出，正式计算历时 {v['formal_worker_wall_seconds']/60:.1f} 分钟；这不含前期开发、几何、复跑和评审。八张 A6000 分四组，每组两张卡。",'',
        f"新进程复跑预先指定的 `{v['replay']['endpoint']}`：两个原始部件文件和最终合成文件均与初次结果 SHA256 完全一致，像素最大差为 0。这个核验仅覆盖一张图、一个种子、两个部件，不能扩大为全部图像或全部硬件的确定性保证。",'',
        '对全部 72 个输出与 4 个原图完成 SAM3 观察。检测到叉子不能证明承托正确，未检测到手也不能单独证明没有人物，故检测器不决定任务成功。',
        '', '全部输入、冻结配置、声明的代码文件、84 个生成请求及其输出哈希已核对。36 个合成端点的编辑区外像素与源图完全相同；这是合成规则的机械性质，不是图像质量评分。',
        '',f"冻结配置 SHA256：`{v['frozen_configuration_sha256']}`。",'',
        '[执行与文件完整性核验](analysis/integrity_audit.json) · [完整统计与限制](analysis/summary.json) · [技术事件记录](TECHNICAL_INCIDENTS.json)','',
        '## 结论边界','',
        '本轮四张真实照片参与了生成前的几何取景修正，因此是前瞻生成验证，而非完全未触碰的端到端留出测试。仅限有明显块状结构的豆腐，使用近似人工轮廓和面锚点，没有真实切取前后配对照片、标定、深度真值或扫描。单目隐藏表面、绝对尺度和新暴露材质仍不可验证。没有独立人类盲评，模型历史训练是否见过这些图未知；VACE 在本轮没有修复验证。',
        '', 'Material Passport：academic-research-suite / experiment-agent，2026-09-29，coupled_bite_repair_v1；VERIFIED 仅指本报告中的执行记录、文件完整性和指定复跑，视觉判断与真实物理正确性仍属有限证据。','']
    (r/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print(r/'REPORT.md')


if __name__=='__main__':main()
