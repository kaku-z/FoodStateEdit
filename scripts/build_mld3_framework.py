"""Export the implemented MLD3 framework as a standalone PNG and SVG."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--font', type=Path, default=Path('C:/Windows/Fonts/msyh.ttc')
                   if Path('C:/Windows/Fonts/msyh.ttc').exists()
                   else Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'))
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    font = FontProperties(fname=str(args.font))
    fig, ax = plt.subplots(figsize=(17, 6), dpi=160)
    fig.patch.set_facecolor('#f4f7fb'); ax.set_facecolor('#f4f7fb')
    ax.set(xlim=(0, 17), ylim=(0, 6)); ax.axis('off')
    def box(x, y, width, title, body, color='#e1edf8'):
        ax.add_patch(FancyBboxPatch((x,y),width,1.65,boxstyle='round,pad=.07,rounding_size=.12',
            facecolor=color,edgecolor='#59718a',linewidth=1.1))
        ax.text(x+width/2,y+1.30,title,ha='center',va='center',fontproperties=font,fontsize=12,weight='bold',color='#19344e')
        ax.text(x+width/2,y+.65,body,ha='center',va='center',fontproperties=font,fontsize=10,linespacing=1.7,color='#243749')
    def arrow(a,b,dashed=False):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=16,color='#506a83',
            linewidth=1.4,linestyle='--' if dashed else '-'))
    ax.text(.2,5.65,'MLD3 · 来源材料约束的第一口食物编辑',fontproperties=font,fontsize=19,weight='bold',color='#19344e')
    ax.text(.2,5.22,'已实现流程：输入真实照片，输出同一材料的抬升、形变、缺口与对照',fontproperties=font,fontsize=11,color='#506276')
    box(.2,2.6,2.5,'来源观测','SAM3：食物与食材掩码\nMoGe2：相机与深度\n全部来自输入照片')
    box(3.2,2.6,2.7,'固定材料状态','MLD2：受约束隐藏先验\n来源身份、UV、颜色、权重\n选中材料 / 剩余材料')
    box(6.4,2.6,3.0,'拓扑与形变','连通闭合网格 / 可见曲线束\n材料连接、弯曲、重力\n同一勺面接触约束','#d9f0e9')
    box(9.9,2.6,2.8,'一致几何渲染','来源纹理运输与遮挡\n对应缺口、表面投影阴影\n背景保护与材料记录','#d9f0e9')
    box(13.2,2.6,3.3,'最终图与验证','可选 Qwen：仅金属外观\n刚性 / 完整 / 无接触 / 无重力\n实际数组检查与逐图审查','#f8eccf')
    for a,b in [(2.75,3.1),(5.95,6.3),(9.45,9.8),(12.75,13.1)]:arrow((a,3.4),(b,3.4))
    ax.text(8.0,1.95,'本次算法实现：来源材料分配 + 拓扑分支 + 变形接触 + 对应缺口 + 受限外观投影',
        fontproperties=font,fontsize=12,ha='center',color='#236750')
    ax.text(8.0,1.25,'固定来源状态与动作，四路对照复用身份、UV、颜色、餐具与相机',fontproperties=font,fontsize=11,ha='center',color='#364b61')
    ax.text(8.0,.58,'单图背面、材料参数与质量均为假设；可执行闭环不等于已经达到通用照片级真实感。',
        fontproperties=font,fontsize=10,ha='center',color='#7a5942')
    fig.savefig(args.output/'framework.png',bbox_inches='tight',facecolor=fig.get_facecolor())
    fig.savefig(args.output/'framework.svg',bbox_inches='tight',facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == '__main__':
    main()
