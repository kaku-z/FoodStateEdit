"""Draw the proposed MLD architecture and the actually executed pilot separately."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import base64
import html
import io
import math

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'material_lineage_framework_20261004'
PILOT = ROOT / 'outputs' / 'material_lineage_pilot_20261003'
W, H = 1800, 1260
INK = '#193044'
MUTED = '#526579'
BLUE = '#296b9b'
BLUE_BG = '#eef6fc'
ORANGE = '#b76b27'
ORANGE_BG = '#fff7ed'
LINE = '#9daebc'
WHITE = '#ffffff'
FONT = 'C:/Windows/Fonts/msyh.ttc'
BOLD = 'C:/Windows/Fonts/msyhbd.ttc'
image = Image.new('RGB', (W, H), WHITE)
draw = ImageDraw.Draw(image)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
       '<title>Material-Lineage Diffusion: proposed architecture and executed pilot</title>',
       '<desc>Dashed orange modules are proposed. Solid blue modules show implemented mechanisms. The pilot has not established photographic realism or cross-food generalization.</desc>',
       f'<rect width="{W}" height="{H}" fill="{WHITE}"/>']

def text(x, y, s, size=28, color=INK, bold=False, anchor='left'):
    f = ImageFont.truetype(BOLD if bold else FONT, size)
    width = draw.textlength(s, font=f)
    tx = x if anchor == 'left' else x-width/2 if anchor == 'middle' else x-width
    # Text positioning is explicitly top-aligned in both outputs.
    assert tx >= -1 and tx+width <= W+1, (s, tx, width)
    draw.text((tx, y), s, font=f, fill=color, anchor='lt')
    a = {'left':'start', 'middle':'middle', 'right':'end'}[anchor]
    svg.append(f'<text x="{x}" y="{y}" dominant-baseline="text-before-edge" text-anchor="{a}" fill="{color}" font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}">{html.escape(s)}</text>')

def path(points, color=LINE, width=3, dashed=False):
    if not dashed:
        draw.line(points, fill=color, width=width, joint='curve')
    else:
        for a,b in zip(points, points[1:]):
            dx,dy=b[0]-a[0],b[1]-a[1]; length=math.hypot(dx,dy)
            for start in range(0, int(length), 15):
                end=min(start+8,length)
                draw.line([(a[0]+dx*start/length,a[1]+dy*start/length),(a[0]+dx*end/length,a[1]+dy*end/length)],fill=color,width=width)
    d=' '.join(('M' if i==0 else 'L')+f'{x},{y}' for i,(x,y) in enumerate(points))
    svg.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"'+(' stroke-dasharray="8 7"' if dashed else '')+'/>')

def rect(x,y,w,h,fill=WHITE,stroke=LINE,dashed=False,r=14,width=2):
    draw.rounded_rectangle((x,y,x+w,y+h),radius=r,fill=fill,outline=None if dashed else stroke,width=width)
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"'+(' stroke-dasharray="9 7"' if dashed else '')+'/>')
    if dashed:
        # Dashed borders avoid relying on color alone for implementation status.
        path([(x+r,y),(x+w-r,y)],stroke,width,True)
        path([(x+w,y+r),(x+w,y+h-r)],stroke,width,True)
        path([(x+w-r,y+h),(x+r,y+h)],stroke,width,True)
        path([(x,y+h-r),(x,y+r)],stroke,width,True)

def polygon(points,fill,stroke=LINE,width=2):
    draw.polygon(points,fill=fill)
    draw.line(points+[points[0]],fill=stroke,width=width)
    svg.append('<polygon points="'+' '.join(f'{x},{y}' for x,y in points)+f'" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

def ellipse(x,y,w,h,fill,stroke=LINE,width=2):
    draw.ellipse((x,y,x+w,y+h),fill=fill,outline=stroke,width=width)
    svg.append(f'<ellipse cx="{x+w/2}" cy="{y+h/2}" rx="{w/2}" ry="{h/2}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

def arrow(points,color=LINE,width=3,dashed=False):
    path(points,color,width,dashed)
    x,y=points[-1];px,py=points[-2];a=math.atan2(y-py,x-px)
    p=[(x,y),(x-12*math.cos(a-.5),y-12*math.sin(a-.5)),(x-12*math.cos(a+.5),y-12*math.sin(a+.5))]
    polygon(p,color,color,1)

def photo(p,x,y,w,h):
    im=Image.open(p).convert('RGB')
    im.thumbnail((w,h),Image.Resampling.LANCZOS)
    ox=x+(w-im.width)/2;oy=y+(h-im.height)/2
    image.paste(im,(round(ox),round(oy)))
    b=io.BytesIO();im.save(b,'PNG');data=base64.b64encode(b.getvalue()).decode('ascii')
    svg.append(f'<image x="{ox}" y="{oy}" width="{im.width}" height="{im.height}" href="data:image/png;base64,{data}"/>')

def cube(x,y,w=120,h=90,color=BLUE_BG,edge=BLUE):
    off=28
    polygon([(x,y),(x+off,y-off),(x+w+off,y-off),(x+w,y)],color,edge)
    polygon([(x+w,y),(x+w+off,y-off),(x+w+off,y+h-off),(x+w,y+h)],'#d9e8f3',edge)
    polygon([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],color,edge)

# Title and implementation-status legend.
text(44,28,'物质谱系扩散框架',48,bold=True)
text(46,91,'Material-Lineage Diffusion (MLD) · 完整设计与已执行原型',30,color=MUTED)
rect(1250,40,30,24,BLUE_BG,BLUE,r=4)
text(1293,33,'已实现机制',28,color=BLUE)
rect(1490,40,30,24,ORANGE_BG,ORANGE,True,r=4)
text(1535,33,'完整模型待实现',28,color=ORANGE)

# Full proposed inference architecture. Its learned modules are intentionally dashed.
text(44,143,'完整算法设计',32,bold=True)
Y,BH=195,330
rect(44,Y,264,BH,BLUE_BG,BLUE)
text(63,Y+20,'源图 I + 动作 a',30,bold=True)
photo(PILOT/'new_01_7442/source.png',64,Y+67,224,173)
text(64,Y+252,'取第一口并抬起',28)
text(64,Y+291,'固定视角与场景',28,color=MUTED)

rect(350,Y,305,BH,ORANGE_BG,ORANGE,True)
text(369,Y+20,'三维物质后验',30,bold=True)
cube(413,Y+109,126,82)
for xx,yy in [(432,Y+128),(467,Y+128),(502,Y+128),(432,Y+163),(467,Y+163),(502,Y+163)]:
    ellipse(xx,yy,11,11,BLUE,BLUE,1)
text(502,Y+202,'ID / 参考坐标 / 份额',25,anchor='middle')
text(369,Y+244,'初始状态 S0',28)
text(369,Y+283,'隐藏材质样本 H',28)

rect(700,Y,340,BH,ORANGE_BG,ORANGE,True)
text(719,Y+20,'物质谱系状态转移',30,bold=True)
text(719,Y+71,'动作、拓扑、变形与接触',26)
rect(784,Y+120,171,48,WHITE,ORANGE,r=8)
text(870,Y+127,'父物质 ID',27,anchor='middle')
arrow([(820,Y+170),(780,Y+213)],ORANGE)
arrow([(925,Y+170),(962,Y+213)],ORANGE)
rect(722,Y+214,135,47,WHITE,BLUE,r=8)
rect(884,Y+214,135,47,WHITE,BLUE,r=8)
text(790,Y+221,'盘中剩余',26,anchor='middle')
text(951,Y+221,'取走一口',26,anchor='middle')
text(719,Y+283,'保守转移 P · 来源可追溯',26)

rect(1080,Y,330,BH,ORANGE_BG,ORANGE,True)
text(1099,Y+20,'谱系约束联合扩散',30,bold=True)
text(1099,Y+71,'共享 H 与 ε(ID, seed)',26)
for xx in range(1130,1375,30):
    for yy in range(Y+125,Y+197,30):
        ellipse(xx,yy,9,9,BLUE if (xx//30+yy//30)%2 else '#8aadc7',BLUE,1)
rect(1103,Y+224,126,47,WHITE,ORANGE,r=8)
rect(1260,Y+224,126,47,WHITE,ORANGE,r=8)
text(1166,Y+231,'几何 / 状态',23,anchor='middle')
text(1323,Y+231,'三维材质',25,anchor='middle')
arrow([(1234,Y+246),(1255,Y+246)],ORANGE,width=2)
text(1099,Y+288,'跨分支共同生成与约束',26)

rect(1460,Y,296,BH,BLUE_BG,BLUE)
text(1479,Y+20,'渲染与原图保持',30,bold=True)
# Schematic target: clearly a diagram rather than a claimed successful photograph.
ellipse(1497,Y+198,216,46,'#e2e9ed',LINE)
cube(1560,Y+163,80,46,'#edf2e3',BLUE)
ellipse(1493,Y+115,97,28,'#d5dfe7',MUTED)
path([(1494,Y+129),(1478,Y+129)],MUTED,4)
cube(1512,Y+88,44,28,'#edf2e3',BLUE)
text(1479,Y+259,'目标：同一口真实抬升',25)
text(1479,Y+292,'上图为结构示意',25,color=MUTED)

for x1,x2 in [(312,344),(659,694),(1044,1074),(1414,1454)]:
    arrow([(x1,Y+166),(x2,Y+166)],MUTED)

# The shared hidden cause is shown explicitly, with three observation/action branches.
text(44,566,'核心：分支共享同一个隐藏材质状态',34,bold=True)
rect(643,625,514,66,BLUE_BG,BLUE,r=10)
text(900,639,'H → 规范物质坐标中的材质场 M',29,anchor='middle')
path([(900,693),(900,719)],BLUE,3)
path([(390,719),(1410,719)],BLUE,3)
for xx in (390,900,1410):arrow([(xx,719),(xx,741)],BLUE)

branches=[(180,'盘中剩余食物','继承原始物质 ID 与份额','新切面查询同一材质场'),
          (690,'勺上的第一口','抬升只改变空间位置','新切面仍有共同来源'),
          (1200,'其他动作分支','同一 H，改变动作与姿态','其他姿态已测；其他切法待测')]
for x,title,a,b in branches:
    rect(x,746,420,118,BLUE_BG,BLUE,r=10)
    text(x+210,757,title,30,bold=True,anchor='middle')
    text(x+210,799,a,27,anchor='middle')
    text(x+210,832,b,26,anchor='middle')
text(900,879,'对应的是物质身份和材质；两侧最终像素不要求相同，光照、损伤和法线可以不同。',27,color=MUTED,anchor='middle')

# What the pilot actually executed, with a real, predetermined result thumbnail.
text(44,930,'本轮实际执行：仅验证核心表示与材质原型',32,bold=True)
PY,PH=982,198
rect(44,PY,306,PH,BLUE_BG,BLUE)
text(62,PY+20,'源图拟合三维代理',28,bold=True)
text(62,PY+73,'单图几何先验',27)
text(62,PY+113,'固定相机与背景',27)
text(62,PY+153,'尚无真实三维真值',27,color=MUTED)

rect(395,PY,354,PH,BLUE_BG,BLUE)
text(413,PY+20,'布尔切割 + 刚体抬升',28,bold=True)
text(413,PY+73,'持久 ID 与保守份额',27)
text(413,PY+113,'代理网格交体积分区',27)
text(413,PY+153,'不是实测质量',27,color=MUTED)

rect(794,PY,390,PH,BLUE_BG,BLUE)
text(812,PY+20,'单图 DDPM / DDIM 材质',28,bold=True)
text(812,PY+73,'每张照片独立训练 6,000 步',25)
text(812,PY+113,'共享噪声与规范材质场',27)
text(812,PY+153,'尚无跨食物泛化训练',27,color=MUTED)

rect(1229,PY,527,PH,BLUE_BG,BLUE)
text(1247,PY+20,'几何约束渲染与原图合成',27,bold=True)
text(1247,PY+72,'透视遮挡',27)
text(1247,PY+111,'光线追踪勺',27)
text(1247,PY+151,'固定 seed 41',25,color=MUTED)
photo(PILOT/'metal_v2/new_01_7442/pose0/shared_s41/view.png',1480,PY+64,257,125)
for a,b in [(354,389),(753,788),(1188,1223)]:arrow([(a,PY+99),(b,PY+99)],BLUE)

text(44,1210,'当前边界：守恒与共享由构造保证；物理状态转移尚未学到，真实照片目标尚未通过。',29,color=MUTED)

OUT.mkdir(parents=True,exist_ok=True)
svg.append('</svg>')
(OUT/'material_lineage_framework.svg').write_text('\n'.join(svg),encoding='utf-8')
image.save(OUT/'material_lineage_framework.png')
print(OUT/'material_lineage_framework.png')
print(OUT/'material_lineage_framework.svg')
