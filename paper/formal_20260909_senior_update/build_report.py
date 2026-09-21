"""Reference-matched two-page report, with editable vector scientific figures.

All measured images come from frozen local evidence; schematic geometry is
explicitly labelled as a procedural control. No model inference or image retouching occurs.
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
TITLE='3次元動作制御に基づく食物画像編集'
OUT=ROOT/'output/pdf/FoodStateEdit_Formal_GUO_2530030_SeniorFormat_20260914_v8_teacher_evidence.pdf'
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
        convert_font(ROOT/'paper/predefense_20260907_senior_match/fonts'/f'{original}.otf',dest,name)
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
    txt(c,'relative 3D control',x+w/2,y+h-11,17,'ArialB',align='center')


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


def table(c,rows,x,top,widths,shade=(),highlight=(),size=8.0):
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
    for r in highlight:
        styles += [('BACKGROUND',(0,r),(-1,r),colors.HexColor('#F7DEDC')),
                   ('TEXTCOLOR',(0,r),(-1,r),RED),('FONTNAME',(0,r),(-1,r),'FSERefGothic')]
    t.setStyle(TableStyle(styles));_,th=t.wrap(sum(widths),1000)
    assert top+th<=775
    t.drawOn(c,x,H-top-th);AUDIT.append({'page':c.getPageNumber(),'type':'table','top':top,'bottom':top+th})
    return top+th+4



NONDATA=ROOT/'artifacts/day19_multimaterial_dataset_v3'
OLDOUT=ROOT/'artifacts/day19_multimaterial_gp40_recovered_20260909_v1'
NEWOUT=ROOT/'artifacts/day20_cake_recovery_gp40_20260909_v1'
NOODLE=ROOT/'artifacts/day18_swept_support_vace_gp40_20260908_v1/projected_final_hold.png'
REFPDF=ROOT/'output/pdf/FoodStateEdit_PreDefense_GUO_2530030_SeniorFormat_20260907.pdf'
USED_IMAGES=set()
OLD_IMAGE_CROP=image_crop
def image_crop(c,path,rect,x,y,w,h):
    USED_IMAGES.add(path)
    if rect is None:
        with Image.open(path) as im:
            assert abs((w/h)/(im.width/im.height)-1)<0.012, ('image aspect',path,w,h,im.size)
    OLD_IMAGE_CROP(c,path,rect,x,y,w,h)

def outcome(case,arm='relative3d'):
    root=NEWOUT if case=='cake' and arm=='relative3d' else OLDOUT
    return root/(case+'__'+arm)/'projected_final_hold.png'

def framework(c):
    box(c,8,19,180,408,colors.HexColor('#F8F8F8'),BLACK,15)
    txt(c,'Input',98,403,24,'ArialB',align='center')
    box(c,22,340,152,41,colors.HexColor('#FFF4CF'),ORANGE,3,1)
    txt(c,'Lift one bite',98,354,21,align='center')
    txt(c,'Action + material',98,316,18,align='center')
    image_crop(c,NONDATA/'cake/reference.png',None,22,173,152,152*512/688)
    txt(c,'Source image',98,151,18,align='center')
    image_crop(c,NONDATA/'cake/edit_alpha.png',None,22,56,100,100*512/688)
    txt(c,'Edit',151,102,17,align='center');txt(c,'mask',151,78,17,align='center')

    group(c,219,244,668,190,'Relative 3D action control',RED)
    # Explicit procedural control with projected contact, depth, and a lifted solid.
    origin=(248,274)
    for dx,dy,t in [(52,0,'x'),(-17,23,'z'),(0,71,'y')]:
        arrow(c,[origin,(origin[0]+dx,origin[1]+dy)],GRAY,lw=1)
        txt(c,t,origin[0]+dx+3,origin[1]+dy,14,col=GRAY)
    c.setStrokeColor(GRAY);c.ellipse(279,271,407,311)
    def poly(points,fill):
        c.setFillColor(fill);c.setStrokeColor(ORANGE);p=c.beginPath();p.moveTo(*points[0])
        for q in points[1:]:p.lineTo(*q)
        p.close();c.drawPath(p,fill=1,stroke=1)
    poly([(320,328),(355,319),(355,362),(320,370)],colors.HexColor('#E8C782'))
    poly([(355,319),(374,338),(374,381),(355,362)],colors.HexColor('#C8A65B'))
    poly([(320,370),(340,389),(374,381),(355,362)],colors.HexColor('#FFF1CA'))
    c.setStrokeColor(BLUE);c.setLineWidth(4);c.line(374,354,420,391)
    c.setFillColor(RED);c.circle(374,354,4,fill=1,stroke=0)
    txt(c,'contact / depth',351,251,17,align='center')
    box(c,454,291,155,78,BGRED,RED,7)
    txt(c,'Perspective +',531,340,20,align='center');txt(c,'visibility',531,313,20,align='center')
    arrow(c,[(420,333),(452,333)])
    for i,case in enumerate(['soup','rice','cake']):
        image_crop(c,NONDATA/case/'relative3d_final.png',None,633+i*82,298,78,78*512/688)
    txt(c,'Material-specific controls',755,385,18,'ArialB',align='center')
    txt(c,'hold frame / three materials',754,269,16,align='center')
    arrow(c,[(610,331),(631,331)])

    group(c,219,20,668,197,'Frozen diffusion + local compositing',BLUE)
    box(c,234,102,128,73,BGBLUE,BLUE,7)
    txt(c,'Condition',298,150,19,align='center');txt(c,'encoder',298,123,19,align='center')
    box(c,389,84,187,91,BGBLUE,BLUE,7)
    txt(c,'VACE backbone',481,150,19,'ArialB',align='center');cold(c,562,163)
    txt(c,'Wan2.2 Fun A14B',481,125,17,align='center')
    txt(c,'LoRA off / TTM off',481,101,16,align='center')
    box(c,602,103,110,71,BGGREEN,GREEN,6)
    txt(c,'Support',657,150,19,align='center');txt(c,'composite',657,122,18,align='center')
    image_crop(c,outcome('cake'),None,738,83,128,128*512/688)
    txt(c,'Edited output',802,61,17,align='center')
    txt(c,'Shared settings for the new soup / rice / cake pilot',485,47,16,col=GRAY,align='center')
    for a,b in [(363,389),(576,602),(712,736)]:arrow(c,[(a,138),(b,138)])
    arrow(c,[(188,361),(204,361),(204,363),(236,363)],RED)
    arrow(c,[(174,217),(207,217),(207,139),(233,139)],BLUE)
    arrow(c,[(760,244),(760,232),(298,232),(298,177)],BLUE)
    arrow(c,[(122,68),(205,68),(205,9),(656,9),(656,101)],GREEN)

def material_figure(c):
    for i,(case,title,a,b) in enumerate([
            ('noodle','Noodle','Continuity','Pinch / attachment'),
            ('soup','Soup','Containment','Spoon contact'),
            ('rice','Fried rice','Supported grains','Source reduction'),
            ('cake','Cake','One pre-cut bite','Contact / source gap')]):
        x=8+i*223
        box(c,x,58,211,221,colors.white,GRAY,7,1)
        txt(c,title,x+105,252,24,'ArialB',align='center')
        path=NOODLE if case=='noodle' else outcome(case)
        with Image.open(path) as im:
            scale=min(187/im.width,134/im.height);iw=im.width*scale;ih=im.height*scale
        image_crop(c,path,None,x+(211-iw)/2,102,iw,ih)
        txt(c,a,x+105,84,18,align='center')
        txt(c,b,x+105,62,17,align='center')
        arrow(c,[(x+105,57),(x+105,39)],GREEN,lw=1.4)
    box(c,8,2,880,33,BGGREEN,GREEN,5,1)
    txt(c,'Separate review: action / contact / material / source update / realism',448,12,19,align='center')

def wide_para(c,s,top):
    p=Paragraph(mixed(s,8.291,8.96),ParagraphStyle('wide',fontName='FSERefMincho',
        fontSize=8.291,leading=10.3,wordWrap='CJK'))
    _,height=p.wrap(RX+CW-LX,500)
    p.drawOn(c,LX,H-top-height)
    AUDIT.append(dict(page=c.getPageNumber(),type='caption',x=LX,top=top,bottom=top+height,text=s))
    BODYTEXT.append(s)
    return top+height+7

def paired_results(c,top):
    full=RX+CW-LX
    label=29;gap=5;pw=(full-label-3*gap)/4;ph=pw*512/688
    for i,title in enumerate(['Input','3D control','VACE raw','Final composite']):
        txt(c,title,LX+label+i*(pw+gap)+pw/2,H-top-8,7.5,'ArialB',align='center')
    y=top+15
    for case,label1,label2 in [('soup','Soup','real'),('rice','Rice','real'),('cake','Cake','synthetic')]:
        txt(c,label1,LX,H-y-17,8.5,'ArialB')
        txt(c,label2,LX,H-y-30,6.2,'Arial')
        base=ROOT/'artifacts/teacher_mechanism_audit_20260914_v1'/case
        paths=[base/'input.png',base/'relative3d_control_f20.png',base/'relative3d_raw_seed2_f20.png',base/'relative3d_final_seed2_f20.png']
        for i,p in enumerate(paths):
            image_crop(c,p,None,LX+label+i*(pw+gap),H-y-ph,pw,ph)
        y+=ph+8
    AUDIT.append(dict(page=c.getPageNumber(),type='wide_figure',top=top,bottom=y))
    return y

def first_page_previous(c):
    # Match the supplied senior manuscript header: one centered title, followed
    # by one author row and one supervisor row.  The former running header and
    # duplicated top-right student number are intentionally omitted.
    txt(c,TITLE,W/2,H-49.0-15.0*.88,15.0,'FSERefGothic',align='center')
    txt(c,'発表者： メディア情報学    学籍番号 2530030    GUO ZHENGPENG',W/2,H-91.0,10.4,'FSERefMincho',align='center')
    txt(c,'主指導教員： 柳井 啓司 教授    指導教員： 高橋 裕樹 教授',W/2,H-108.0,10.4,'FSERefMincho',align='center')
    y=head(c,'1  はじめに',LX,136.0)
    y=para(c,'料理の画像に「一匙すくう」「一口持ち上げる」といった動作を加えられれば，制作者は既存の写真から料理の見せ方を試すことができる．見る側にも，盛り付けだけでは伝えにくい食材の形や動作を，視覚的に示せると考える．',LX,y)
    y=para(c,'本研究の目的は，元の料理と周辺場面を保ちながら，指定した食物操作を表現する画像編集を支援することである．想定する用途は飲食店や食物コンテンツ制作者の表現支援であり，制作負担の軽減と見る楽しさの向上を目指す．これらの社会的効果は未評価であり，今後の利用者評価で検証する．',LX,y)
    y=para(c,'食物操作では，道具の移動に加えて，食材の変形，接触，運ばれる部分と元の領域の対応が必要となる．麺の連続性，スープの保持，炒飯の粒，ケーキの切り口では必要な条件が異なる．そこで，材料に応じた相対3D制御と既存の拡散モデルを組み合わせ，動作成立と自然さを分けて検証する．',LX,y)
    y=head(c,'2  関連研究',LX,y+6)
    y=para(c,'VACE [1]は画像・動画・マスク等を条件とする生成・編集基盤であり，本研究では外観生成に用いる．GeoEdit [2]は幾何条件を扱う画像編集法であるが，本研究は食物操作の時間系列をVACEへ与える点が異なる．',LX,y)
    y=para(c,'LoRA [3]は先行する麺のseen-sample診断だけでVACE制御分岐へrank 8として学習した．明瞭な意味改善が得られず，材料比較への麺バイアス混入も避けるため，主実験60条件と追加3材料では使用しない．SAM3 [4]は領域抽出の診断候補であり，現行生成経路には接続していない．',LX,y)
    y=head(c,'3  提案する編集フレームワーク',LX,y+6)
    y=head(c,'3.1  材料別の相対3D制御',LX,y,True)
    y=para(c,'図1では，画像上で手動指定した接近・接触・最終・器接続の正規化座標U，相対深度D，21フレーム5段階の状態Qから相対3D状態を作る．箸は2本の3D線分，麺は128点のベジエ曲線，他食材は局所載荷形状であり，CADモデル，場面深度推定，物理シミュレーションは用いない．',LX,y)
    y=para(c,'各時刻の3D点を正規化透視投影し，深度順に描画してRGB制御列C（21×512×688×3）とマスクM（21×512×688×1）を作る．VACEの学習済みVAEとCondition branchがこれを潜在特徴へ符号化し，Wanの各去雑音ブロックへhintとして加える．独自のCondition Encoderは学習せず，模式図の画素をコピーするのではなく，固定重みの生成器が外観を再生成する．',LX,y)

    yy=placed_figure(c,framework,RX,136.0,(900,450))
    yy=para(c,'図1  相対3D制御と凍結VACEによる編集．赤は制御構築，青は既存モデル，緑は局所合成を示す．',RX,yy+4,'caption',False,6)
    yy=placed_figure(c,material_figure,RX,yy,(900,290))
    yy=para(c,'図2  材料ごとの確認項目と出力例．麺は既存の高持上げ実験，ほかは今回の相対3D条件．',RX,yy+4,'caption',False,6)
    yy=head(c,'3.2  動作範囲を覆う局所合成',RX,yy,True)
    yy=para(c,'全動作の変化領域を覆うマスクSを作り，生成系列Jと元画像Iを合成する．各フレームfの出力は次式となる．',RX,yy)
    yy=equation(c,'I<sub>f</sub><super>out</super> = S J<sub>f</sub> + (1 - S) I',RX,yy,1)
    yy=para(c,'領域外の画素差0はこの合成の性質であり，モデルの生成能力ではない．麺の既存実験では，同一の生成系列に対し合成範囲だけを変え，既に生成されていた高い位置の箸が消える不具合を修復した．',RX,yy)
    yy=head(c,'4  実験',RX,yy+2)
    yy=head(c,'4.1  麺以外の3材料の対照実験',RX,yy,True)
    yy=para(c,'スープ，炒飯，予め切れ目のあるケーキ各1画像で，平面移動と相対3D制御を比較した．スープと炒飯は既使用の実画像（UECFOOD256由来），ケーキは合成入力であり，未見データ評価ではない．',RX,yy)
    yy=para(c,'図3では画像，プロンプト，編集範囲を揃え，Wan2.2-VACE-Fun-A14B，688×512画素，seed 1，21フレーム，20 steps，VACE scale 1，LoRA・TTM offとした．別の同条件消融は4例×3 seeds×5条件の60セルで実施し，全セルでLoRA・TTMを無効化した．',RX,yy)
    yy=para(c,'動作・接触・材料・元領域・自然度を個別に観察した．所見は固定6時点のエージェントによる確認であり，人間による独立ブラインド評価ではない．',RX,yy)

def first_page(c):
    import importlib.util
    module_path=ROOT/'tmp/paper_framework_v28/render_figure.py'
    spec=importlib.util.spec_from_file_location('framework_renderer',module_path)
    renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    diagram=json.loads((module_path.parent/'framework.json').read_text(encoding='utf-8'))
    USED_IMAGES.update(Path(a['file']) for a in diagram['items'] if a['kind']=='image')
    txt(c,TITLE,W/2,H-49.0-15.0*.88,15.0,'FSERefGothic',align='center')
    txt(c,'発表者： メディア情報学    学籍番号 2530030    GUO ZHENGPENG',W/2,H-91.0,10.4,'FSERefMincho',align='center')
    txt(c,'主指導教員： 柳井 啓司 教授    指導教員： 高橋 裕樹 教授',W/2,H-108.0,10.4,'FSERefMincho',align='center')
    full=RX+CW-LX
    renderer.render(c,diagram,LX,H-132-diagram['height']*full/1200,full)
    AUDIT.append(dict(page=1,type='wide_framework',top=125,bottom=132+diagram['height']*full/1200))
    start=wide_para(c,'図1  提案手法の全体像．赤：手動アンカーと相対深度による操作制御，青：凍結したVACE/Wanによる条件付き生成，緑：元画像との領域限定合成．画像は同じケーキ例の入力，制御列，出力を示す．',329)
    y=head(c,'1  はじめに',LX,start)
    y=para(c,'食物写真に「一匙すくう」「一口持ち上げる」動作を加え，見た人が料理を味わう場面を想像できる編集を目指す．既存の画像編集では，道具の追加と共に器や背景まで変わることがある．本研究は元の料理と周辺場面を保ちながら，食材の接触・変形・持上げを指定する．飲食店や制作者の表現支援を想定するが，見る楽しさや制作負担への効果は未評価である．',LX,y)
    y=head(c,'2  関連研究',LX,y+4)
    y=para(c,'VACE [1]は画像・動画・マスクを条件とする生成・編集基盤である．GeoEdit [2]は幾何条件による画像編集を扱う．本研究はVACEの前段に食物操作の制御列生成，後段に固定フレーム選択と領域限定合成を追加する．',LX,y)
    y=para(c,'SAM3 [3]は領域抽出の候補であり，現行生成経路には未接続である．現在の編集範囲は，手続き的に描画した道具・食物の全フレームの変更領域を合併し，膨張と境界重みを加えて作る．',LX,y)
    y=head(c,'3  相対3D制御による画像編集',LX,y+4)
    y=para(c,'画像座標(u,v)と手動の相対深度ZからX=(u-c_x)Z/f_x，Y=(v-c_y)Z/f_yを計算する．688×512画像ではf_x=f_y=825.6，(c_x,c_y)=(344,256)とする．接触点P（3成分）と食材形状V（ケーキは8×3頂点）を動かし，元画像の食材テクスチャを投影面に写す．麺は曲線，箸は2線分であり，CADメッシュや実深度復元は用いない．',LX,y)
    y=para(c,'2D平面制御（Planar）は形状を画像平面で平行移動する．相対3D制御は相対深度と回転を加えて投影する．両者の接触点軌跡は共通であり，比較対象は主に形状と前後関係である．麺の対照は固定描画順と深度に基づく可視性の違いである．',LX,y)

    yy=head(c,'3.1  透視投影と条件付き生成',RX,start,True)
    yy=para(c,'各時刻の3D点をu=f_xX/Z+c_x，v=f_yY/Z+c_yで画像へ戻し，面の深度順や麺の可視性を使って描画する．RGB制御列C（21×512×688×3）と重みM（21×512×688×1，0〜1）を与える．制御には持上げ形状が既に含まれるが，その外観は手続き的な近似である．',RX,yy)
    yy=para(c,'CをMで保持領域と変更領域に分け，それぞれを学習済みVAEで符号化する．マスクを再配列して結合し，VACE条件分岐のhintを対応するWan DiTブロックへ加える．VACE [1]のAdapter学習はDiTを凍結して条件分岐を学習する方式であるが，本研究は公開済み重みを利用し，全モデルに追加学習を行わない．独自のCondition Encoderはない．',RX,yy)
    yy=head(c,'3.2  最終画像の局所合成',RX,yy+3,True)
    yy=para(c,'全動作範囲のマスクαと固定フレームJ（図1のt*=20，0始まり）でIout=αJ+(1−α)Iを作る．領域外の画素差0は合成の性質であり，生成能力を意味しない．',RX,yy)
    yy=head(c,'4  実験',RX,yy+4)
    yy=para(c,'図2は既使用実画像のスープ・炒飯と合成ケーキを示す．PAI/Wan2.2-VACE-Fun-A14B，seed 2，688×512画素，21 frames，20 steps，scale 1を用い，主比較はLoRA・TTMなしで統一した．4例×3 seeds×5条件の60出力から，表2は3条件36出力を再解析した．',RX,yy)
    yy=para(c,'Actionは道具・接触・食物動作，Photoは写真自然度，Preservationは領域外保持を確認する．表1は選定スープ例の内部判定であり，独立したブラインド評価や統計的成功率ではない．',RX,yy)

def second_page(c):
    y=paired_results(c,90.82)
    y=wide_para(c,'図2  入力，RGB制御，VACEの原始生成，局所合成後の最終結果（第20フレーム，0始まり）．生成画像は保存MP4から復号し，追加の外観修復はしていない．持上げ配置は制御に含まれ，生成器は外観を再生成する．ケーキは合成の切り分け済みブロックである．',y+1)
    y=wide_para(c,'表1  別の合成スープ1入力・seed 1．PPTと同じ数値で，✓/×は内部判定．外保持は領域外のRGB完全一致画素率，PSNR/SSIMは入力保持の診断．Qwenのraw出力は入力サイズに調整し，共通の局所合成は適用していない．',y)
    y=table(c,[['手法','PSNR','SSIM','外MAE','外保持','Action','Photo','Pres.'],
        ['ChordEdit [5]','31.78','0.982','0.000','100.0%','×','×','✓'],
        ['Qwen Image [4]','15.73','0.628','18.192','0.3%','✓','✓','×'],
        ['FoodStateEdit','20.68','0.965','0.000','100.0%','✓','✓','✓']],LX,y,
        [76,52,48,60,59,57,57,57],highlight=(3,),size=6.5)
    audit=json.loads((ROOT/'results/teacher_mechanism_audit_20260914_v1.json').read_text(encoding='utf-8'))
    y=wide_para(c,'表2  4開発入力×3 seeds（各行12出力）．領域外はα=0，領域内はα>0，MAEはRGB絶対画素差の平均（0〜255）．独立入力数は4であり12ではない．画素変化量は動作・写真品質を表さない．',y+2)
    names=['原画像反復条件','2D平面制御','相対3D制御']
    rows=[['VACEへの入力','raw領域外MAE','合成後領域外MAE','合成後領域内MAE']]
    for name,row in zip(names,audit['summaries']):rows.append([name,f"{row['mean_raw_outside_mae']:.2f}",f"{row['mean_final_outside_mae']:.2f}",f"{row['mean_final_inside_mae']:.2f}"])
    y=table(c,rows,LX,y,[145,119,119,123],highlight=(3,),size=6.5)

    left=head(c,'4.2  結果と解釈',LX,y+2)
    left=para(c,'表1の選定例では三条件が成立した．表2では全36出力の合成を再現し，領域外差0を確認した．これは局所合成の効果であり，原始生成の背景保持性能ではない．',LX,left)
    left=para(c,'2Dと3Dで操作位置は共通であり，現状の開発4例で明瞭な3D改善は未確認である．図2は制御と生成の役割を示す例であり，写真自然度や汎化の優位を主張しない．',LX,left)

    right=head(c,'5  おわりに',RX,y+2)
    right=para(c,'食物操作の制御列生成と領域限定合成をVACEへ接続した．3D制御の追加効果と未使用画像への汎化は未実証である．',RX,right)
    right=para(c,'今後は独立画像と複数人のブラインド評価，困難場面，食物の元領域対応，失敗分類を追加し，生成中の接触・元領域検証を去雑音へ戻す誘導法を検討する．',RX,right)
    right=head(c,'参考文献',RX,right+4)
    for ref in ['[1] Z. Jiang et al. “VACE.” ICCV, 2025.',
                '[2] Y. He et al. “GeoEdit.” arXiv:2606.30003, 2026.',
                '[3] N. Carion et al. “SAM 3.” arXiv:2511.16719, 2025.',
                '[4] C. Wu et al. “Qwen-Image.” arXiv:2508.02324, 2025.',
                '[5] L. Lu et al. “ChordEdit.” CVPR, 2026.']:
        right=para(c,ref,RX,right,'ref',False,1)

def build():
    import shutil
    (HERE/'fonts').mkdir(parents=True,exist_ok=True)
    fonts();FIG.mkdir(parents=True,exist_ok=True);OUT.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/'paper/predefense_20260907_senior_match/fonts/LICENSE',HERE/'fonts/LICENSE')
    result=json.loads((ROOT/'results/day20_non_noodle_result_20260909.json').read_text(encoding='utf-8'))
    assert result['conditions_count']==6 and not result['claims']['stable_3d_advantage_established']
    # All shown non-noodle final outputs must match frozen verified evidence.
    for rec in result['technical_metrics']:
        path=outcome(rec['source_dataset_case'],rec['condition'].split('__')[1])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==rec['final_sha256']
    for name,draw,size in [('material_checks',material_figure,(900,290))]:
        f=canvas.Canvas(str(FIG/(name+'.pdf')),pagesize=size);draw(f);f.showPage();f.save()
    c=canvas.Canvas(str(OUT),pagesize=(W,H),pageCompression=1)
    c.setTitle(TITLE)
    c.setAuthor('GUO ZHENGPENG 2530030')
    first_page(c);c.showPage();second_page(c);c.showPage();c.save()
    reader=PdfReader(OUT);assert len(reader.pages)==2
    pdftext='\n'.join(p.extract_text() for p in reader.pages)
    assert TITLE in pdftext
    assert '柳井 啓司 教授' in pdftext and '高橋 裕樹 教授' in pdftext and 'メディア情報学' in pdftext and '要確認' not in pdftext
    assert '2026年度 修士論文中間発表会 資料' not in pdftext
    manuscript_header=(
        f'# {TITLE}\n\n'
        '発表者：メディア情報学　学籍番号 2530030　GUO ZHENGPENG\n\n'
        '主指導教員：柳井 啓司 教授　指導教員：高橋 裕樹 教授'
    )
    (HERE/'report_ja.md').write_text(manuscript_header+'\n\n'+'\n\n'.join(BODYTEXT),encoding='utf-8')
    sources=USED_IMAGES|{REFPDF,ROOT/'results/day20_non_noodle_result_20260909.json',
        ROOT/'results/day34_presentation_soup_same_input_metrics_v1.json',ROOT/'results/teacher_mechanism_audit_20260914_v1.json',
        ROOT/'results/DAY18_HIGH_LIFT_SUPPORT_REPAIR_20260908.md',
        ROOT/'results/DAY16_GEOMETRY_PROMPTED_SAM3_RESULT_20260907.md'}
    (HERE/'layout_audit.json').write_text(json.dumps(dict(reference=str(REFPDF),page_count=2,
        japanese_body_size_pt=9.212,body_leading_pt=12.752,title_size_pt=15.0,
        left_x_pt=LX,right_x_pt=RX,column_width_pt=CW,blocks=AUDIT,
        header_style='Single centered title, centered author row, centered supervisor row, matching the supplied manuscript header.',
        layout_deviation='Page 2 top is a wide three-material paired figure for legibility; other text remains two columns.',
        original_preserved=True,new_inference=False,retouched_results=False,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)},
        output_sha256=hashlib.sha256(OUT.read_bytes()).hexdigest()),ensure_ascii=False,indent=2),encoding='utf-8')
    print(OUT)
if __name__=='__main__':build()
