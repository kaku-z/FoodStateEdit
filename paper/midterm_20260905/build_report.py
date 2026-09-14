"""Build a Japanese midterm report in the supplied senior-report layout."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name('report_ja.json')
OUT = ROOT / 'output/pdf/FoodStateEdit_Midterm_GUO_2530030_20260906.pdf'
QA = ROOT / 'tmp/pdfs/foodstateedit_midterm_20260906'
WIDTH, HEIGHT = A4
LEFT, RIGHT, BOTTOM = 88, 88, 58
TEXTWIDTH = WIDTH - LEFT - RIGHT

pdfmetrics.registerFont(TTFont('Mincho', 'C:/Windows/Fonts/yumin.ttf'))
pdfmetrics.registerFont(TTFont('Gothic', 'C:/Windows/Fonts/meiryob.ttc', subfontIndex=0))
pdfmetrics.registerFontFamily('Mincho', normal='Mincho', bold='Gothic', italic='Mincho', boldItalic='Gothic')

STYLES = {
    'body': ParagraphStyle('body', fontName='Mincho', fontSize=10.7, leading=17.7,
                           firstLineIndent=10.7, alignment=TA_JUSTIFY, wordWrap='CJK'),
    'heading': ParagraphStyle('heading', fontName='Gothic', fontSize=14.4, leading=22, wordWrap='CJK'),
    'caption': ParagraphStyle('caption', fontName='Mincho', fontSize=9.2, leading=14,
                              alignment=TA_LEFT, wordWrap='CJK'),
    'equation': ParagraphStyle('equation', fontName='Mincho', fontSize=11.4, leading=19,
                               alignment=TA_CENTER, wordWrap='CJK'),
    'cell': ParagraphStyle('cell', fontName='Mincho', fontSize=8.7, leading=13, wordWrap='CJK'),
    'cellhead': ParagraphStyle('cellhead', fontName='Gothic', fontSize=8.7, leading=13, wordWrap='CJK'),
    'reference': ParagraphStyle('reference', fontName='Mincho', fontSize=10.1, leading=16,
                                wordWrap='CJK', leftIndent=18, firstLineIndent=-18),
}


class Report:
    def __init__(self):
        OUT.parent.mkdir(parents=True, exist_ok=True)
        QA.mkdir(parents=True, exist_ok=True)
        self.c = canvas.Canvas(str(OUT), pagesize=A4, pageCompression=1)
        self.c.setTitle('FoodStateEdit: 3次元動作制御に基づく柔軟な食品の拡散補完')
        self.c.setAuthor('GUO ZHENGPENG (2530030)')
        self.c.setSubject('修士研究中間報告・草稿。2026-09-05までの研究証拠に基づく。')
        self.y = HEIGHT - 86
        self.page_records = []
        self.label = ''

    def start(self, label='', number=None):
        self.label = label
        self.y = HEIGHT - 91
        if number is not None:
            self.c.setFont('Gothic', 10.5)
            self.c.drawString(LEFT + 6, HEIGHT - 74, label)
            self.c.drawRightString(WIDTH - RIGHT - 6, HEIGHT - 74, str(number))
            self.c.setLineWidth(.5)
            self.c.line(LEFT, HEIGHT - 79, WIDTH - RIGHT, HEIGHT - 79)

    def check(self, height):
        if self.y - height < BOTTOM:
            raise RuntimeError(f'Layout overflow on {self.label}: y={self.y:.1f}, block={height:.1f}')

    def paragraph(self, text, kind='body', gap=None):
        if kind == 'heading':
            self.y -= 12
        p = Paragraph(text, STYLES[kind])
        _, height = p.wrap(TEXTWIDTH, HEIGHT)
        self.check(height)
        p.drawOn(self.c, LEFT, self.y - height)
        self.y -= height + (gap if gap is not None else (11 if kind == 'body' else 8))

    def chapter(self, number, title):
        top = HEIGHT - 99
        self.c.setFillColor(colors.black)
        self.c.rect(LEFT, top - 108, 7, 108, fill=1, stroke=0)
        self.c.setFont('Gothic', 23)
        self.c.drawString(LEFT + 44, top - 39, number)
        self.c.drawString(LEFT + 44, top - 92, title)
        self.y = top - 137

    def picture(self, relative, max_height, x=None, width=None):
        path = ROOT / relative
        with PILImage.open(path) as im:
            iw, ih = im.size
        width = width or TEXTWIDTH
        draw_w = min(width, max_height * iw / ih)
        draw_h = draw_w * ih / iw
        self.check(draw_h)
        origin = LEFT if x is None else x
        self.c.drawImage(str(path), origin + (width - draw_w) / 2, self.y - draw_h,
                         width=draw_w, height=draw_h, preserveAspectRatio=True, mask='auto')
        return draw_h

    def table(self, spec):
        self.paragraph(spec['caption'], 'caption', 3)
        rows = [[Paragraph(escape(cell), STYLES['cellhead' if i == 0 else 'cell'])
                 for cell in row] for i, row in enumerate(spec['rows'])]
        table = Table(rows, colWidths=[TEXTWIDTH * n for n in spec['widths']])
        table.setStyle(TableStyle([
            ('LINEABOVE', (0, 0), (-1, 0), .7, colors.black),
            ('LINEBELOW', (0, 0), (-1, 0), .4, colors.black),
            ('LINEBELOW', (0, -1), (-1, -1), .7, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        _, h = table.wrap(TEXTWIDTH, HEIGHT)
        self.check(h)
        table.drawOn(self.c, LEFT, self.y - h)
        self.y -= h + 14

    def diagram(self):
        height = 151
        self.check(height)
        top = self.y
        c = self.c
        def box(x, y, w, h, lines, fill):
            c.setFillColor(colors.HexColor(fill))
            c.setStrokeColor(colors.HexColor('#40515e'))
            c.roundRect(x, y, w, h, 4, fill=1, stroke=1)
            c.setFillColor(colors.black)
            c.setFont('Gothic', 8.8)
            for i, line in enumerate(lines):
                c.drawCentredString(x+w/2, y+h/2+(len(lines)-1)*6-i*12-3, line)
        def arrow(x1, y1, x2, y2):
            c.setStrokeColor(colors.black)
            c.setLineWidth(.75)
            c.line(x1, y1, x2, y2)
            if y1 == y2:
                c.line(x2, y2, x2-4, y2+2.5)
                c.line(x2, y2, x2-4, y2-2.5)
            else:
                sign = 1 if y2 < y1 else -1
                c.line(x2,y2,x2-2.5,y2+sign*4)
                c.line(x2,y2,x2+2.5,y2+sign*4)
        w = 87
        xs = [LEFT+i*110 for i in range(4)]
        y = top - 55
        box(xs[0],y,w,44,['相対3次元状態','道具・麺・接触'],'#e4edf3')
        box(xs[1],y,w,44,['深度感知投影','RGB制御系列'],'#e4edf3')
        box(xs[2],y,w,44,['VACE + LoRA','外観生成'],'#fff0cc')
        box(xs[3],y,w,44,['領域外の保護','編集結果'],'#edf2e8')
        for i in range(3): arrow(xs[i]+w,y+22,xs[i+1],y+22)
        box(LEFT+3,top-133,155,42,['同じ3次元状態からマスク生成','麺 / 挟持点 / 碗内の起点'],'#e4edf3')
        box(LEFT+207,top-133,192,42,['学習時：局所重み付き誤差','基盤モデル固定・LoRAのみ更新'],'#fff0cc')
        arrow(xs[0]+43,y,LEFT+46,top-91)
        arrow(LEFT+158,top-112,LEFT+207,top-112)
        arrow(xs[2]+43,top-91,xs[2]+43,y)
        self.y -= height

    def block(self, b):
        if 'h' in b: self.paragraph(b['h'], 'heading')
        elif 'p' in b: self.paragraph(b['p'])
        elif 'equation' in b: self.paragraph(b['equation'], 'equation')
        elif 'caption' in b: self.paragraph(b['caption'], 'caption', 13)
        elif 'table' in b: self.table(b['table'])
        elif 'diagram' in b: self.diagram()
        elif 'image' in b: self.y -= self.picture(b['image'], b['height']) + 7
        elif 'pair' in b:
            w = (TEXTWIDTH-12)/2
            self.c.setFont('Gothic', 9)
            for i, label in enumerate(b['labels']):
                self.c.drawCentredString(LEFT+i*(w+12)+w/2, self.y-10, label)
            self.y -= 17
            heights = [self.picture(p,b['height'],LEFT+i*(w+12),w) for i,p in enumerate(b['pair'])]
            self.y -= max(heights)+7
        elif 'ref' in b:
            self.paragraph(escape(b['ref']), 'reference', 3)
            link = escape(b['url'])
            label = '原文・実装へのリンク' if len(link)>85 else link
            self.paragraph(f'<link href="{link}" color="#163e64">{label}</link>', 'caption', 13)

    def finish_page(self):
        self.page_records.append({'pdf_page':len(self.page_records)+1,'label':self.label,'lowest_content_y':round(self.y,2)})
        self.c.showPage()


def main():
    data = json.loads(SOURCE.read_text(encoding='utf-8'))
    report = Report()
    c = report.c
    report.start('表紙')
    c.setFont('Mincho',19)
    c.drawCentredString(WIDTH/2,HEIGHT-212,'令和8年度　修士研究中間報告')
    c.setFont('Mincho',24)
    for i,line in enumerate(data['title']): c.drawCentredString(WIDTH/2,HEIGHT-281-i*39,line)
    c.setFont('Times-Roman',17)
    c.drawCentredString(WIDTH/2,HEIGHT-379,'FoodStateEdit')
    c.setFont('Mincho',14)
    c.drawCentredString(WIDTH/2,HEIGHT-568,f"{data['student_id']}　{data['author']}")
    c.setFont('Mincho',11)
    c.drawCentredString(WIDTH/2,HEIGHT-606,'所属・主指導教員・指導教員：確認後に追記')
    c.setFont('Mincho',14)
    c.drawCentredString(WIDTH/2,HEIGHT-678,data['date'])
    c.setFont('Mincho',10)
    c.drawCentredString(WIDTH/2,HEIGHT-713,'草稿 / 実験記録の基準日：2026年9月5日')
    report.finish_page()
    report.start('概要')
    c.setFont('Gothic',19)
    c.drawCentredString(WIDTH/2,HEIGHT-123,'概　要')
    report.y=HEIGHT-162
    for p in data['abstract']: report.paragraph(p,gap=18)
    report.finish_page()
    report.start('目次','i')
    report.paragraph('目　次','heading',22)
    for number,page in enumerate(data['pages'],1):
        if 'opening' in page or page.get('references'):
            report.y-=9
            c.setFont('Gothic',11)
            c.drawString(LEFT,report.y,page['chapter'])
            c.drawRightString(WIDTH-RIGHT,report.y,str(number))
            report.y-=24
        for block in page['blocks']:
            if 'h' not in block or page.get('references'): continue
            title=block['h']
            c.setFont('Mincho',9.5)
            c.drawString(LEFT+13,report.y,title)
            c.drawRightString(WIDTH-RIGHT,report.y,str(number))
            c.setLineWidth(.25)
            c.setDash(1,3)
            start=LEFT+17+pdfmetrics.stringWidth(title,'Mincho',9.5)
            if start<WIDTH-RIGHT-25: c.line(start,report.y-1,WIDTH-RIGHT-22,report.y-1)
            c.setDash()
            report.y-=17.2
    report.check(0)
    report.finish_page()
    for number,page in enumerate(data['pages'],1):
        report.start(page['chapter'],number)
        c.bookmarkPage(f'body-{number}')
        if 'opening' in page:
            c.addOutlineEntry(page['chapter'],f'body-{number}',level=0)
            report.chapter(*page['opening'])
        for block in page['blocks']: report.block(block)
        report.finish_page()
    c.save()
    # Editable prose companion; the JSON remains the authoritative layout source.
    md=['# '+''.join(data['title']),'',f"{data['author']} / {data['student_id']}",'','## 概要','',*data['abstract'],'']
    for page in data['pages']:
        if 'opening' in page: md += ['# '+page['chapter'],'']
        for b in page['blocks']:
            if 'h' in b: md += ['## '+b['h'],'']
            elif 'p' in b: md += [b['p'],'']
            elif 'equation' in b: md += [b['equation'],'']
            elif 'caption' in b: md += [b['caption'],'']
            elif 'image' in b: md += [f"![図](../../{b['image']})",'']
            elif 'pair' in b: md += [f'![{label}](../../{p})' for label,p in zip(b['labels'],b['pair'])]+['']
            elif 'table' in b:
                t=b['table']; md += [t['caption'],'','| '+' | '.join(t['rows'][0])+' |','| '+' | '.join(['---']*len(t['rows'][0]))+' |']
                md += ['| '+' | '.join(row)+' |' for row in t['rows'][1:]]+['']
            elif 'ref' in b: md += [f"{b['ref']} [{b['url']}]({b['url']})",'']
    SOURCE.with_suffix('.md').write_text('\n'.join(md),encoding='utf-8')
    evidence={
        'pdf':str(OUT),'pdf_sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),
        'page_count':len(report.page_records),'pages':report.page_records,
        'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'author':data['author'],'student_id':data['student_id'],
        'pending_cover_fields':['affiliation','principal_supervisor','supervisor'],
        'claim_status':'Day 13 effectiveness unverified; no new experiment performed for report',
    }
    (QA/'layout_audit.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(evidence,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
