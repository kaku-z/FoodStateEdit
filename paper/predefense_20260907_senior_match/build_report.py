"""Reference-matched two-page report, with editable vector scientific figures.

All measured images come from frozen local evidence; schematic geometry is
explicitly labelled as a proxy. No model inference or image retouching occurs.
"""
from pathlib import Path
import sys, re, json, math, hashlib
from xml.sax.saxutils import escape

sys.path.insert(0, r"C:\Users\kaku\.cache\foodstateedit_pdf_dependencies")
from fontTools.ttLib import TTFont as FTFont
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.pens.cu2quPen import Cu2QuPen
from PIL import Image
import numpy as np
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph, Table, TableStyle
import reportlab.platypus.paragraph as paragraph_engine
import reportlab.lib.textsplit as textsplit_engine
from pypdf import PdfReader

# ReportLab's stock Japanese prohibition list omits full-width comma/period.
# Extend this process's line-breaking policy, without modifying the library.
paragraph_engine.ALL_CANNOT_START += '，．！？'
textsplit_engine.ALL_CANNOT_START += '，．！？'

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FIG=HERE/'figures'
OUT=ROOT/'output/pdf/FoodStateEdit_PreDefense_GUO_2530030_SeniorFormat_20260907.pdf'
DATA=ROOT/'artifacts/day13_3d_guided_flexible_completion_udon_v1'
R13=ROOT/'results/day13_flexible_completion_20260906'
R14=ROOT/'results/day14_contact_guidance_observer_20260907'
OBS=ROOT/'artifacts/day14_contact_guidance_observer_20260907_v3/rgb_observer_counterfactuals.png'
SOURCE=DATA/'vace_reference_image/udon_chopsticks_imagegen_pseudo_v1.png'
TARGET=DATA/'target_keyframe/udon_chopsticks_imagegen_pseudo_v1.png'
CONTROL=DATA/'review/relative3d_control_contact_sheet.png'
MASKS=DATA/'review/topology_mask_contact_sheet.png'
SYNTH=R14/'synthetic_mask_optimization.png'
W,H=595.28,841.89
LX,RX,CW=48.905,304.724,246.25
BLACK=colors.black
RED=colors.HexColor('#D6352B'); BLUE=colors.HexColor('#467BB5')
GREEN=colors.HexColor('#3A9561'); ORANGE=colors.HexColor('#CB7B1C')
BGRED=colors.HexColor('#F7DEDC'); BGBLUE=colors.HexColor('#E1EAF9')
BGGREEN=colors.HexColor('#E7F1E6'); GRAY=colors.HexColor('#767676')
AUDIT=[]; BODYTEXT=[]


def convert_font(src, dest, family):
    """Outline-format conversion of only needed glyphs (OFL retained, renamed)."""
    text=Path(__file__).read_text(encoding='utf-8')
    chars=set(map(ord,text))|set(range(32,127))
    if dest.exists() and dest.stat().st_mtime>Path(__file__).stat().st_mtime:
        return
    f=FTFont(src); cmap={u:g for u,g in f.getBestCmap().items() if u in chars}
    order=['.notdef']+sorted(set(cmap.values())-{'.notdef'})
    gs=f.getGlyphSet(); glyphs={}
    for name in order:
        pen=TTGlyphPen(None)
        gs[name].draw(Cu2QuPen(pen, max_err=.5, reverse_direction=True))
        glyphs[name]=pen.glyph()
    b=FontBuilder(f['head'].unitsPerEm,isTTF=True)
    b.setupGlyphOrder(order); b.setupCharacterMap(cmap); b.setupGlyf(glyphs)
    b.setupHorizontalMetrics({g:f['hmtx'].metrics[g] for g in order})
    b.setupHorizontalHeader(ascent=f['hhea'].ascent,descent=f['hhea'].descent)
    b.setupNameTable({'familyName':family,'styleName':'Regular','uniqueFontIdentifier':family,
        'fullName':family,'psName':family,'copyright':'Derived from Harano Aji Fonts / Source Han; SIL OFL 1.1. See fonts/LICENSE.'})
    b.setupOS2(sTypoAscender=f['OS/2'].sTypoAscender,sTypoDescender=f['OS/2'].sTypoDescender,
        usWinAscent=f['OS/2'].usWinAscent,usWinDescent=f['OS/2'].usWinDescent)
    b.setupPost(); b.setupMaxp(); b.save(dest)


def fonts():
    for original,name in [('HaranoAjiMincho-Regular','FSERefMincho'),('HaranoAjiGothic-Medium','FSERefGothic')]:
        dest=HERE/'fonts'/f'{name}.ttf'
        convert_font(HERE/'fonts'/f'{original}.otf',dest,name)
        pdfmetrics.registerFont(TTFont(name,str(dest)))
    texfonts=Path(r'C:\Users\kaku\AppData\Local\Programs\MiKTeX\fonts')
    for name,stem in [('Roman','utmr8a'),('RomanB','utmb8a'),('Italic','utmri8a')]:
        face=pdfmetrics.EmbeddedType1Face(str(texfonts/'afm/urw/times'/f'{stem}.afm'),
            str(texfonts/'type1/urw/times'/f'{stem}.pfb'))
        pdfmetrics.registerTypeFace(face)
        pdfmetrics.registerFont(pdfmetrics.Font(name,face.name,'WinAnsiEncoding'))
    pdfmetrics.registerFontFamily('Roman',normal='Roman',bold='RomanB',italic='Italic',boldItalic='Italic')
    for name,file in [('Arial','arial.ttf'),('ArialB','arialbd.ttf')]:
        pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file))
    pdfmetrics.registerFontFamily('FSERefMincho',normal='FSERefMincho',bold='FSERefGothic')


def txt(c,s,x,y,size=19,font='Arial',col=BLACK,align='left'):
    c.setFillColor(col); c.setFont(font,size)
    if align=='center': c.drawCentredString(x,y,s)
    elif align=='right': c.drawRightString(x,y,s)
    else: c.drawString(x,y,s)


def box(c,x,y,w,h,fill=colors.white,stroke=BLACK,r=10,lw=1.5,dash=False):
    c.saveState(); c.setFillColor(fill); c.setStrokeColor(stroke); c.setLineWidth(lw)
    if dash:c.setDash(7,4)
    c.roundRect(x,y,w,h,r,fill=1,stroke=1);c.restoreState()


def group(c,x,y,w,h,title,col,fill=colors.white,ts=22):
    box(c,x,y,w,h,fill,col,18,2.0)
    tw=pdfmetrics.stringWidth(title,'ArialB',ts)
    c.setFillColor(colors.white); c.rect(x+19,y+h-8,tw+12,20,fill=1,stroke=0)
    txt(c,title,x+25,y+h-5,ts,'ArialB',col)


def arrow(c,pts,col=BLACK,dash=False,lw=1.8):
    c.saveState();c.setStrokeColor(col);c.setFillColor(col);c.setLineWidth(lw)
    if dash:c.setDash(6,4)
    for a,b in zip(pts,pts[1:]):c.line(*a,*b)
    c.setDash();(x1,y1),(x2,y2)=pts[-2:]
    a=math.atan2(y2-y1,x2-x1);p=c.beginPath();p.moveTo(x2,y2)
    p.lineTo(x2+9*math.cos(a+2.7),y2+9*math.sin(a+2.7))
    p.lineTo(x2+9*math.cos(a-2.7),y2+9*math.sin(a-2.7));p.close()
    c.drawPath(p,fill=1,stroke=0);c.restoreState()


def image_crop(c,path,rect,x,y,w,h):
    with Image.open(path) as im:
        panel=im.crop(rect) if rect else im.copy()
        c.drawImage(ImageReader(panel),x,y,w,h)


def cold(c,x,y):
    # Vector frozen-weight icon, not an emoji dependent on font substitution.
    for a in [0,math.pi/3,2*math.pi/3]:
        c.setStrokeColor(BLUE);c.setLineWidth(1.7)
        c.line(x-6*math.cos(a),y-6*math.sin(a),x+6*math.cos(a),y+6*math.sin(a))


def train(c,x,y):
    c.setFillColor(ORANGE);c.setStrokeColor(RED)
    p=c.beginPath();p.moveTo(x,y+8);p.curveTo(x-9,y,x-7,y-8,x,y-8)
    p.curveTo(x+9,y-6,x+5,y+2,x,y+8);c.drawPath(p,fill=1,stroke=1)


def geom_proxy(c,x,y,w,h):
    """Scientific schematic of rigid utensils and a flexible 3D curve."""
    o=(x+25,y+20)
    for dx,dy,t in [(48,0,'x'),(-15,28,'z'),(0,65,'y')]:
        arrow(c,[o,(o[0]+dx,o[1]+dy)],GRAY,lw=1)
        txt(c,t,o[0]+dx+2,o[1]+dy,13,col=GRAY)
    c.setStrokeColor(colors.HexColor('#B9BFC5'));c.setLineWidth(1)
    c.ellipse(x+30,y+9,x+w-15,y+53,stroke=1,fill=0)
    # Two rigid sticks with distinct tips; green visible strand and hidden dash.
    for dy in [0,13]:
        c.setStrokeColor(colors.HexColor('#88502B'));c.setLineWidth(6)
        c.line(x+107,y+91+dy,x+w-6,y+132+dy)
    c.setStrokeColor(GREEN);c.setLineWidth(4)
    p=c.beginPath();p.moveTo(x+90,y+34);p.curveTo(x+65,y+54,x+132,y+60,x+107,y+91)
    c.drawPath(p)
    c.setFillColor(BLUE);c.circle(x+90,y+34,5,fill=1,stroke=0)
    c.setFillColor(RED);c.circle(x+107,y+91,5,fill=1,stroke=0)
    txt(c,'source',x+102,y+27,15,col=BLUE)
    txt(c,'pinch',x+35,y+100,15,col=RED)
    txt(c,'relative 3D proxy',x+w/2,y+h-11,17,'ArialB',align='center')


def system_figure(c):
    # Coordinates are 900 x 440; scaled as a vector figure to one report column.
    box(c,8,20,180,393,colors.HexColor('#F8F8F8'),BLACK,18)
    txt(c,'Input',98,390,23,'ArialB',align='center')
    box(c,22,324,152,46,colors.HexColor('#FFF4CF'),ORANGE,2,1)
    txt(c,'Lift a noodle',98,345,20,align='center')
    txt(c,'Action prompt',98,306,17,align='center')
    image_crop(c,SOURCE,None,22,148,152,127)
    txt(c,'Source image',98,131,17,align='center')
    image_crop(c,DATA/'edit_alpha/udon_chopsticks_imagegen_pseudo_v1.png',None,22,40,87,73)
    txt(c,'Edit',146,88,18,align='center');txt(c,'support S',146,67,18,align='center')

    group(c,218,229,670,190,'Geometry / control module',RED)
    geom_proxy(c,231,246,219,159)
    box(c,472,285,142,77,BGRED,RED,8)
    txt(c,'Depth-aware',543,336,20,align='center');txt(c,'projection',543,311,20,align='center')
    arrow(c,[(449,321),(471,321)])
    for i,(cc,rr) in enumerate([(2,0),(0,1),(2,1)]):
        image_crop(c,CONTROL,(cc*688+360,rr*576+105,cc*688+650,rr*576+395),
                   638+i*77,280,73,73)
    txt(c,'contact / lift / hold',755,259,17,align='center')
    txt(c,'RGB control sequence',755,379,19,'ArialB',align='center')
    arrow(c,[(615,323),(635,323)])

    group(c,218,20,670,184,'Diffusion completion module',BLUE)
    box(c,234,91,130,72,BGBLUE,BLUE,8)
    txt(c,'Condition',299,137,19,align='center');txt(c,'encoder',299,111,19,align='center')
    box(c,390,70,174,94,BGBLUE,BLUE,8)
    txt(c,'VACE backbone',467,141,18,'ArialB',align='center');cold(c,551,151)
    box(c,407,82,140,34,BGRED,RED,4,1)
    txt(c,'rank-8 LoRA',474,92,17,align='center');train(c,539,99)
    txt(c,'high-noise branch only',478,48,15,align='center')
    box(c,590,92,110,70,BGGREEN,GREEN,6)
    txt(c,'Support',645,139,19,align='center');txt(c,'composite',645,113,19,align='center')
    image_crop(c,R13/'relative3d_topology_weighted_step32_final_hold.png',(330,110,650,430),730,57,123,123)
    txt(c,'Output (existing run)',790,35,16,align='center')
    for a,b in [(364,390),(564,590),(700,728)]:arrow(c,[(a,126),(b,126)])
    arrow(c,[(188,342),(205,342),(205,334),(230,334)])
    arrow(c,[(174,195),(210,195),(210,127),(232,127)],BLUE,lw=1.4)
    arrow(c,[(759,230),(759,217),(297,217),(297,165)],BLUE)
    arrow(c,[(109,49),(206,49),(206,7),(646,7),(646,90)],GREEN,lw=1.4)


def constraint_figure(c):
    group(c,7,16,884,417,'Phase- and visibility-aware structural guidance',RED,colors.HexColor('#FBFCFD'))
    group(c,21,229,428,183,'Expected structure (3D prior)',GREEN)
    # Depth diagram, with explicit observed overlap and visible samples.
    txt(c,'C: projected curve',132,382,18,'ArialB',align='center')
    pts=[(43+q*19,270+q*7+9*math.sin(q*.7)) for q in range(9)]
    c.setStrokeColor(GREEN);c.setLineWidth(3)
    for a,b in zip(pts,pts[1:]):c.line(*a,*b)
    c.setFillColor(colors.HexColor('#A98970'));c.rect(134,280,32,69,fill=1,stroke=0)
    for i,(px,py) in enumerate(pts):
        c.setFillColor(GRAY if i in [5,6] else GREEN);c.circle(px,py,4,fill=1,stroke=0)
    txt(c,'visible',65,245,16,col=GREEN);txt(c,'occluded',157,245,16,col=GRAY)
    box(c,240,285,191,83,BGGREEN,GREEN,5)
    txt(c,'screen overlap',335,346,19,align='center')
    txt(c,'+ nearer utensil',335,323,19,align='center')
    txt(c,'visibility V',335,301,19,'ArialB',align='center')
    arrow(c,[(211,307),(238,307)],GREEN)
    txt(c,'Not reconstructed metric 3D',335,253,14,col=GRAY,align='center')

    group(c,469,229,405,183,'Image evidence (independent)',BLUE)
    image_crop(c,OBS,(448,95,723,435),483,258,93,116)
    txt(c,'RGB',529,240,16,align='center')
    box(c,597,319,136,55,BGBLUE,BLUE,6)
    txt(c,'Color observer',655,350,16,align='center');cold(c,719,359)
    txt(c,'current diagnostic',665,332,14,align='center')
    box(c,597,248,136,51,colors.white,BLUE,6,1.2,True)
    txt(c,'SAM3',665,279,19,'ArialB',align='center');cold(c,719,282)
    txt(c,'planned validation',665,262,14,align='center')
    arrow(c,[(576,317),(585,317),(585,346),(595,346)])
    arrow(c,[(576,317),(585,317),(585,274),(595,274)],BLUE,True)
    for yy,t,col in [(332,'P noodle',GREEN),(266,'P utensil',ORANGE)]:
        box(c,759,yy-16,95,48,colors.white,col,5)
        txt(c,t,806,yy+2,16,align='center')
    arrow(c,[(734,346),(757,346)])
    arrow(c,[(734,346),(744,346),(744,280),(757,280)])

    box(c,25,75,153,116,colors.HexColor('#F3F3F3'),GRAY,7)
    txt(c,'Phase gate g',101,169,19,'ArialB',align='center')
    txt(c,'approach: OFF',101,143,17,align='center')
    txt(c,'contact / lift',101,119,17,align='center')
    txt(c,'hold: ON',101,96,17,align='center')
    group(c,201,71,431,119,'Differentiable surrogate',ORANGE)
    box(c,216,98,179,62,colors.white,ORANGE,4)
    txt(c,'Weakest-link path',305,137,18,align='center');txt(c,'on visible samples',305,114,16,align='center')
    box(c,421,98,195,62,colors.white,ORANGE,4)
    txt(c,'Endpoint + two tips',518,137,18,align='center');txt(c,'expected contact sites',518,114,16,align='center')
    txt(c,'+',408,122,26,align='center')
    arrow(c,[(176,131),(201,131)],GRAY)
    arrow(c,[(335,229),(190,229),(190,169),(201,169)],GREEN)
    arrow(c,[(672,229),(672,214),(645,214),(645,166),(632,166)],BLUE)
    box(c,660,84,207,94,colors.HexColor('#FFF3E5'),RED,8)
    txt(c,'Validation gate',763,153,21,'ArialB',align='center')
    txt(c,'synthetic masks: pass',763,127,17,col=GREEN,align='center')
    txt(c,'RGB semantics: fail',763,105,17,col=RED,align='center')
    arrow(c,[(632,130),(658,130)])
    txt(c,'Latent guidance in VACE: not executed',540,40,19,'ArialB',RED,align='center')
    arrow(c,[(764,83),(764,43),(735,43)],RED,True)


def composite_figure(c):
    # Reference-style qualitative figure: image row + labelled findings.
    box(c,112,679,770,50,colors.HexColor('#F1F1F1'),GRAY,7)
    txt(c,'Action',72,696,22,'ArialB',align='center')
    txt(c,'Lift one noodle with chopsticks (final hold, f = 20)',493,696,22,align='center')
    panels=[(SOURCE,None,'Input'),(TARGET,None,'Synthetic target'),
            (R13/'relative3d_uniform_step32_final_hold.png',None,'Uniform'),
            (R13/'relative3d_topology_weighted_step32_final_hold.png',None,'Weighted')]
    for i,(p,_,name) in enumerate(panels):
        xx=13+i*221
        image_crop(c,p,(335,108,651,424),xx,436,208,208)
        txt(c,name,xx+104,413,21,'ArialB' if i==3 else 'Arial',align='center')
    for yy,t,body,fill,stroke in [(366,'Uniform','No clear, large semantic lift.',BGBLUE,BLUE),
                                (325,'Weighted','Small change; primary 5% gate not met.',BGRED,RED)]:
        txt(c,t,8,yy+7,19,'ArialB');box(c,136,yy,750,32,fill,stroke,6,1)
        txt(c,body,149,yy+8,19)
    txt(c,'RGB observer counterexample (not generated output)',450,291,22,'ArialB',align='center')
    for i,(rect,title,score) in enumerate([((448,95,723,435),'Target','E = 1.1930'),
             ((448,495,723,835),'Solid-color block','E = 0.5180'),
             ((888,495,1163,835),'Geometry overlay','alignment audit')]):
        xx=68+i*287
        image_crop(c,OBS,rect,xx,53,171,211)
        txt(c,title,xx+86,32,18,align='center');txt(c,score,xx+86,10,17,col=RED if i==1 else BLACK,align='center')


def mask_comparison(c):
    for i,title in enumerate(['No correction','Fixed constraints','Phase / visibility']):
        x=i*300+5
        txt(c,title,x+145,213,23,'ArialB',align='center')
        image_crop(c,SYNTH,(480*i,60,480*(i+1),348),x,26,290,174)
    txt(c,'Synthetic probability fields only; green: noodle, red: utensil',450,5,18,align='center')


def placed_figure(c,draw,x,top,native_size):
    scale=CW/native_size[0]
    c.saveState();c.translate(x,H-top-native_size[1]*scale);c.scale(scale,scale);draw(c);c.restoreState()
    return top+native_size[1]*scale


def mixed(s,size=9.212,roman_size=9.963):
    parts=re.split(r'([\x20-\x7e]+)',s)
    return ''.join(f'<font name="Roman" size="{roman_size}">{escape(p)}</font>' if p and all(ord(q)<128 for q in p) else escape(p) for p in parts)


def para(c,s,x,top,kind='body',indent=True,gap=2):
    fs,lead=(9.212,12.752) if kind=='body' else ((8.291,10.3) if kind=='caption' else (6.974,8.1))
    markup=mixed(s,fs,fs if kind=='ref' else fs*10/9.246)
    p=Paragraph(markup,ParagraphStyle('p',fontName='FSERefMincho',fontSize=fs,leading=lead,
        wordWrap='CJK',alignment=TA_JUSTIFY if kind=='body' else TA_LEFT,
        firstLineIndent=fs if indent and kind=='body' else 0))
    _,ph=p.wrap(CW,1000)
    assert top+ph<=775,('Text overflow',s[:30],top+ph)
    p.drawOn(c,x,H-top-ph)
    AUDIT.append({'page':c.getPageNumber(),'type':kind,'x':x,'top':top,'bottom':top+ph,'text':s})
    BODYTEXT.append(s)
    return top+ph+gap


def head(c,s,x,top,sub=False):
    size=9.212 if sub else 11.055
    txt(c,s,x,H-top-size*.88,size,'FSERefGothic')
    BODYTEXT.append(s)
    return top+(16 if sub else 20.5)


def equation(c,s,x,top,num):
    p=Paragraph(s,ParagraphStyle('eq',fontName='Roman',fontSize=9.963,leading=13,alignment=TA_CENTER))
    _,ph=p.wrap(CW-27,100);p.drawOn(c,x,H-top-ph)
    txt(c,f'({num})',x+CW-1,H-top-10,9.963,'Roman',align='right')
    return top+ph+8


def table(c,rows,x,top,widths,shade=(),size=8.0):
    data=[]
    for r,row in enumerate(rows):
        data.append([Paragraph(mixed(str(v),size,size*1.081),ParagraphStyle('t',fontName='FSERefMincho',
            fontSize=size,leading=size+1.6,alignment=TA_LEFT if i==0 else TA_CENTER,wordWrap='CJK')) for i,v in enumerate(row)])
    t=Table(data,colWidths=widths)
    styles=[('LINEABOVE',(0,0),(-1,0),.8,BLACK),('LINEBELOW',(0,0),(-1,0),.4,BLACK),
        ('LINEBELOW',(0,-1),(-1,-1),.8,BLACK),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),
        ('TOPPADDING',(0,0),(-1,-1),2.5),('BOTTOMPADDING',(0,0),(-1,-1),2.5)]
    for r in shade:styles.append(('BACKGROUND',(0,r),(-1,r),colors.HexColor('#E6E6E6')))
    t.setStyle(TableStyle(styles));_,th=t.wrap(CW,1000)
    assert top+th<=775
    t.drawOn(c,x,H-top-th);AUDIT.append({'page':c.getPageNumber(),'type':'table','top':top,'bottom':top+th})
    return top+th+4


def first_page(c):
    txt(c,'2026年度 修士論文中間発表会 資料',LX,H-73.85,9.212,'FSERefMincho')
    txt(c,'2530030',550.97,H-74.90,9.963,'Roman',align='right')
    for top,title in [(104.17,'相対3次元動作状態と拡散モデルを用いた柔軟食品編集における'),
                      (120.90,'動作段階と可視性に基づく構造補完')]:
        txt(c,title,W/2,H-top-13.266*.88,13.266,'FSERefGothic',align='center')
    txt(c,'発表者：所属［要確認］  学籍番号2530030  GUO ZHENGPENG',W/2,H-160.5,11.055,'FSERefMincho',align='center')
    txt(c,'主指導教員：［要確認］    副指導教員：［要確認］',54.88,H-174.36,11.055,'FSERefMincho')
    y=head(c,'1  はじめに',LX,196.62)
    y=para(c,'麺を箸で持ち上げる画像編集では，道具の移動だけでなく，柔軟な麺の変形，器との連結，把持点の接触を同時に保つ必要がある．麺は細長く，箸によって隠されるため，画像上の欠落が断線か遮蔽かを区別することも重要となる．',LX,y)
    y=para(c,'本研究FoodStateEditは，剛体の箸と柔軟な麺を相対3次元状態で記述し，幾何制御だけでは表せない外観を拡散モデルで補完することを目指す．研究の焦点は3D化そのものではなく，動作段階と可視性に基づいて，どの構造をいつ補完するかを定める点にある．本稿では局所重み付き学習の比較実験と，接触・遮蔽制約の予備検証を報告する．',LX,y)
    y=para(c,'評価では，見た目の自然さと動作の成立を分ける．背景が自然に見えても，麺が持ち上がらない，空把持となる，麺が器から切れる場合には，動作編集の成功とはみなさない．',LX,y)
    y=head(c,'2  関連研究',LX,y+5)
    y=para(c,'VACE [1]は，画像・動画条件を統合して生成・編集を行う拡散モデルである．LoRA [2]は少数の追加パラメータによる適応を可能にする．本研究では，これらを柔軟食品編集の生成基盤として利用する．',LX,y)
    y=para(c,'一方，SAM3 [3]は概念プロンプトから物体領域を検出・分割・追跡する．ただし領域の検出は，麺の連結や箸との接触の正しさを保証しない．clDice [4]のような中心線に基づく評価も参考に，3D状態から得る期待構造と，出力画像から観測する証拠を分けて評価する必要がある．',LX,y)
    y=para(c,'既存モデルは生成器・観測器として利用し，本研究ではその間に置く柔軟物体の構造制約を検討する．幾何表現の追加，学習重みの変更，観測器の変更を分離し，どの要素が寄与したかを比較可能にする．',LX,y)
    y=head(c,'3  提案手法',LX,y+5)
    y=head(c,'3.1  相対3次元状態を用いた拡散補完',LX,y,True)
    y=para(c,'全体構成を図1に示す．入力画像と動作から，麺を可変形曲線，箸を剛体として表現し，先端，器内の接続点，相対深度を管理する．この状態は計画用の相対3D代理表現であり，実シーンの計測復元ではない．',LX,y)
    y=para(c,'同じ状態からRGB制御系列と麺・把持・接続領域のマスクを投影する．高ノイズ側のVACE部にrank-8 LoRAを加え，局所重み付き誤差で学習する．推論後は編集領域外を原画像と厳密に合成する．領域外の画素差0は保全処理の性質であり，生成能力の改善とは区別する．',LX,y)

    yy=placed_figure(c,system_figure,RX,187.9,(900,440))
    yy=para(c,'図1  FoodStateEditの全体構成',RX,yy+3,'caption',False,7)
    yy=placed_figure(c,constraint_figure,RX,yy,(900,445))
    yy=para(c,'図2  段階・遮蔽対応制約と観測器の検証',RX,yy+3,'caption',False,7)
    yy=head(c,'3.2  局所構造重み付き学習',RX,yy,True)
    yy=para(c,'麺，箸との把持点，器内の接続点を重視するため，各領域マスクから画素平均が1になる重みを構成する．',RX,yy)
    yy=equation(c,'W = (1 + 3M<sub>strand</sub> + 7M<sub>pinch</sub> + 5M<sub>source</sub>)/Z',RX,yy,1)
    yy=para(c,'Zは分子の画素平均である．この重みを速度予測誤差に適用し，均一重みと同じ初期値・乱数系列で比較する．',RX,yy)
    yy=head(c,'3.3  動作段階・遮蔽対応制約',RX,yy+4,True)
    yy=para(c,'図2では，予測した麺の確率を投影曲線上でサンプリングし，弱い箇所を強調する滑らかな経路スコアを用いる．接触項は，期待位置にある麺端と2本の箸先の証拠を測る．接近中は無効とし，接触・持上げ・保持で有効とする．',RX,yy)
    yy=para(c,'遮蔽は画面上の重なりと箸が手前にあることの両方で判定し，既知の遮蔽点を経路制約から除外する．画像中の麺の消失だけから遮蔽を推定しない．',RX,yy)
    yy=para(c,'経路スコアは温度0.08の正規化soft-minを用い，可視点の低い麺確率を強調する．ただし画像全体から連結経路を発見する処理ではなく，予定経路の証拠を測る代理指標である．',RX,yy)
    yy=para(c,'青の結晶は凍結部，橙の印は学習部である．破線のSAM3と拡散誘導は次段階とし，現行の色尤度観測器では単色や未編集画像への誤判定を先に検証する．',RX,yy)


def second_page(c,r13,r14):
    y=para(c,'表1  VACE比較実験の領域別RGB MAE（低いほど良い）．',LX,90.82,'caption',False,5)
    rows=[['Method','Support','Topo.','Pinch']]
    for key,title in [('planar_lora_off','Planar / LoRA off'),('planar_uniform_step32','Planar / uniform'),
                     ('relative3d_lora_off','3D / LoRA off'),('relative3d_uniform_step32','3D / uniform'),
                     ('relative3d_topology_weighted_step32','3D / weighted')]:
        a=r13['conditions'][key];rows.append([title,*[f'{a[k]:.3f}' for k in ['support_mae','topology_mae','pinch_mae']]])
    y=table(c,rows,LX,y,[105,48,47,CW-200],shade=(5,),size=8.291)
    y=para(c,'構造エネルギーは，段階ゲートg，可視点集合V，投影曲線C，画像から得る麺・箸の確率場Pを用いて次式で表す．',LX,y+5)
    y=equation(c,'E<sub>f</sub> = g<sub>f</sub> [E<sub>path</sub>(P<sub>n</sub>, C<sub>f</sub>, V<sub>f</sub>) + 0.5E<sub>tip</sub>(P<sub>n</sub>, P<sub>u</sub>)]',LX,y,2)
    y=para(c,'本制約は固定経路上の代理指標であり，任意の連結性や物体識別を保証しない．現在は合成確率場とRGB観測器を個別に検証した段階であり，拡散過程への勾配誘導は未実施である．',LX,y)
    y=head(c,'4  実験',LX,y+5)
    y=head(c,'4.1  実験設定と合成マスク検証',LX,y,True)
    y=para(c,'VACE実験では，既知の合成うどん1例を2行に複製し，平面制御・均一重み，相対3D・均一重み，相対3D・局所重みの3群を32更新で学習した．初期値とサンプル・ノイズ系列を揃え，LoRAなしを含む5条件を評価した．',LX,y)
    y=para(c,'推論条件はRTX A6000，seed 1，21フレーム，20 steps，VACE scale 1，TTM offで固定した．一つのpipeline読込で5条件を実行し，全条件で領域外の画素差は0であった．',LX,y)
    y=para(c,'合成マスク実験では，同じ断線・箸欠落確率場から，補正なし，固定制約，段階・遮蔽対応を比較した．補正条件はCPU上で60回更新した．図4と表2に示すように，固定制約は遮蔽部も埋めるが，遮蔽対応制約はその確率を初期値0.0200に維持した．',LX,y)
    y=placed_figure(c,mask_comparison,LX,y+3,(900,238))
    y=para(c,'図4  合成確率場の機構検証．VACE生成画像ではない．',LX,y+3,'caption',False,5)
    y=para(c,'表2  制約ごとのエネルギーEと既知の遮蔽部の麺確率．',LX,y,'caption',False,4)
    rows=[['条件','E','遮蔽部の麺確率']]
    opt=r14['synthetic_mask_experiment']['optimization']
    for key,title in [('none','補正なし'),('fixed','固定制約'),('phase_visibility','段階・遮蔽対応')]:
        a=opt[key];rows.append([title,f'{a["total"]:.6f}',f'{a["hidden_region_predicted_noodle_mean"]:.4f}'])
    y=table(c,rows,LX,y,[103,65,CW-168],shade=(3,),size=8.0)
    y=para(c,'Eは条件ごとに異なる目的関数値であり，直接の優劣比較には用いない．4つのマスク検証を通過したが，映像の自然さを示す結果ではない．',LX,y)
    y=para(c,'相対3D・均一重みは平面・均一重みよりSupport誤差が16.23%大きかった．制御の3D化自体の有効性もこの例では未確認であり，教師画像との対応を再点検する．',LX,y)

    yy=placed_figure(c,composite_figure,RX,90.82,(900,735))
    yy=para(c,'図3  上：保持フレームの生成比較（同位置・同倍率）．下：RGB観測器の診断．単色ブロックは反実仮想であり，生成結果ではない．右下は期待曲線の位置ずれを示す．',RX,yy+3,'caption',False,6)
    yy=head(c,'4.2  生成結果と観測器の評価',RX,yy,True)
    yy=para(c,'表1のSupportは編集領域，Topo.は局所重み付き領域，Pinchは把持領域のRGB誤差であり，意味的な成功率ではない．相対3Dの均一重みに対して，局所重みはTopo.を2.22%，Pinchを2.83%低減したが，事前設定した双方5%以上の条件を満たさなかった．図3でも明瞭な意味的改善は確認できず，2名の独立盲検評価は未完了である．',RX,yy)
    yy=para(c,'RGB色尤度観測器では，合成教師画像のE=1.1930より単色ブロックのE=0.5180を好み，4つの意味的検証が全て失敗した．有限で非零の勾配が得られても，誤った評価の最適化は危険である．教師画像と期待曲線の位置ずれも観察しており，失敗をVACEだけに帰因しない．',RX,yy)
    yy=head(c,'5  おわりに',RX,yy+5)
    yy=para(c,'本研究は，3D状態に基づく段階・遮蔽対応制約を提案し，合成マスク上で遮蔽と欠損を区別する挙動を確認した．一方，拡散生成への有効性は未確立である．結果は既知合成例の能力診断に限定し，汎化，実データ性能，物理的正しさ，写真品質は主張しない．',RX,yy)
    yy=para(c,'今後は既存SAM3を凍結した候補分割器として用い，人手で麺中心線，箸2本の領域・先端，接触を検証する．マスクIoU，中心線一致，箸先誤差，接触判定を個別に評価する．設計用と検証用を分離し，未編集・単色・断線・空把持・箸融合・遮蔽の対照と幾何対応の監査を先に行う．SAM3導入自体を新規性や成功の証明とはせず，硬いマスクをそのまま微分可能な誘導器としない．',RX,yy)
    yy=head(c,'参考文献',RX,yy+5)
    for ref in ['[1] Z. Jiang et al. VACE: All-in-One Video Creation and Editing. ICCV, 2025.',
                '[2] E. J. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR, 2022.',
                '[3] N. Carion et al. SAM 3: Segment Anything with Concepts. arXiv:2511.16719, 2025.',
                '[4] S. Shit et al. clDice - A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation. CVPR, 2021.']:
        yy=para(c,ref,RX,yy,'ref',False,1)


def build():
    fonts();FIG.mkdir(parents=True,exist_ok=True);OUT.parent.mkdir(parents=True,exist_ok=True)
    r13=json.loads((ROOT/'results/DAY13_FLEXIBLE_COMPLETION_RESULT_20260906.json').read_text(encoding='utf-8'))
    r14=json.loads((R14/'result.json').read_text(encoding='utf-8'))
    assert not r13['primary_comparison']['numeric_gate_pass']
    assert r14['new_vace_inference_count']==0
    for name,draw,size in [('framework',system_figure,(900,440)),('constraint_detail',constraint_figure,(900,445)),
                           ('qualitative',composite_figure,(900,735)),('mask_ablation',mask_comparison,(900,238))]:
        f=canvas.Canvas(str(FIG/(name+'.pdf')),pagesize=size);draw(f);f.showPage();f.save()
    c=canvas.Canvas(str(OUT),pagesize=(W,H),pageCompression=1)
    c.setTitle('FoodStateEdit - Senior-format two-page interim report')
    c.setAuthor('GUO ZHENGPENG 2530030')
    first_page(c);c.showPage();second_page(c,r13,r14);c.showPage();c.save()
    assert len(PdfReader(OUT).pages)==2
    (HERE/'report_ja.md').write_text('\n\n'.join(BODYTEXT),encoding='utf-8')
    sources=[SOURCE,TARGET,CONTROL,MASKS,OBS,SYNTH,
        ROOT/'results/DAY13_FLEXIBLE_COMPLETION_RESULT_20260906.json',R14/'result.json']
    (HERE/'layout_audit.json').write_text(json.dumps({'reference':'t2430084_doc.pdf',
        'reference_japanese_body_size_pt':9.212,'body_leading_pt':12.752,'title_size_pt':13.266,
        'left_x_pt':LX,'right_x_pt':RX,'page1_header_top_pt':64.64,'page2_content_top_pt':90.82,
        'font_note':'Harano Aji original outlines converted to renamed TrueType subsets; embedded Nimbus Roman Type 1, matching the reference font families.',
        'blocks':AUDIT,'sources_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'output_sha256':hashlib.sha256(OUT.read_bytes()).hexdigest()},ensure_ascii=False,indent=2),encoding='utf-8')
    print(OUT)


if __name__=='__main__':build()
