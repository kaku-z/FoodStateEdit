"""Render the presentation plan as a readable PDF without altering experiment images."""
from pathlib import Path
import re
from xml.sax.saxutils import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/FORMAL_PRESENTATION_PLAN_20260909.md'
OUTPUT=ROOT/'output/pdf/FoodStateEdit_presentation_plan_20260909_v2.pdf'
pdfmetrics.registerFont(TTFont('CJK','C:/Windows/Fonts/simsun.ttc',subfontIndex=0))
pdfmetrics.registerFont(TTFont('CJKbold','C:/Windows/Fonts/simhei.ttf'))
normal=ParagraphStyle('Body',fontName='CJK',fontSize=10.8,leading=16.5,spaceAfter=7,wordWrap='CJK')
title=ParagraphStyle('Title',parent=normal,fontName='CJKbold',fontSize=21,leading=29,spaceAfter=16)
h2=ParagraphStyle('Heading',parent=normal,fontName='CJKbold',fontSize=14.5,leading=21,spaceBefore=10,spaceAfter=9,keepWithNext=True)
h3=ParagraphStyle('Subheading',parent=normal,fontName='CJKbold',fontSize=11.6,leading=18,spaceBefore=9,spaceAfter=7,keepWithNext=True)
cell=ParagraphStyle('Cell',parent=normal,fontSize=9.7,leading=14,spaceAfter=0)
styles={1:title,2:h2,3:h3}
width=A4[0]-88
def para(text,style=normal):
    return Paragraph(escape(text),style)
def table(rows):
    count=len(rows[0])
    widths=([60,143,width-203] if count==3 else [62,126,143,width-331])
    cells=[[para(c,cell) for c in row] for row in rows]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8EDF1')),
                           ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#D9D9D9')),
                           ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                           ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
                           ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    return t
story=[];lines=SOURCE.read_text(encoding='utf-8').splitlines();i=0
while i<len(lines):
    line=lines[i].strip()
    if not line:i+=1;continue
    if line=='<!-- pagebreak -->':story.append(PageBreak());i+=1;continue
    if line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            row=[c.strip() for c in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r'[-: ]+',c) for c in row):rows.append(row)
            i+=1
        story.extend([table(rows),Spacer(1,10)]);continue
    match=re.match(r'^(#{1,3}) (.*)',line)
    if match:story.append(para(match[2],styles[len(match[1])]))
    elif line.startswith('- '):story.append(para('• '+line[2:]))
    else:story.append(para(line))
    i+=1
def footer(canvas,doc):
    canvas.saveState();canvas.setFont('CJK',9);canvas.setFillColor(colors.HexColor('#555555'))
    canvas.drawString(44,25,'FoodStateEdit   GUO ZHENGPENG   2026年9月9日')
    canvas.drawRightString(A4[0]-44,25,str(doc.page));canvas.restoreState()
OUTPUT.parent.mkdir(parents=True,exist_ok=True)
if OUTPUT.exists():raise FileExistsError(OUTPUT)
doc=SimpleDocTemplate(str(OUTPUT),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=36,bottomMargin=43,
                     title='FoodStateEdit 正式发表准备计划书',author='GUO ZHENGPENG')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUTPUT)
