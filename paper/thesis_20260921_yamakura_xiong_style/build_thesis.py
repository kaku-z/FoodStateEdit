"""Build the Japanese FoodStateEdit thesis-style interim report.

The visual language follows the two user-provided senior reports (minimal A4,
chapter bars, restrained typography), while every scientific statement is
bounded by the repository's frozen evidence.  The script performs no model
inference and does not alter experiment artifacts.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

DEP = Path(r"C:\Users\kaku\.cache\foodstateedit_pdf_dependencies")
if DEP.exists():
    sys.path.insert(0, str(DEP))

from PIL import Image as PILImage, ImageDraw, ImageFont
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont as FTFont
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "output" / "pdf" / "FoodStateEdit_Thesis_GUO_2530030_20260921.pdf"
ASSET = HERE / "generated_assets"
FONT_DIR = ROOT / "paper" / "formal_20260909_senior_update" / "fonts_v9"
SOURCE_FONT_DIR = ROOT / "paper" / "predefense_20260907_senior_match" / "fonts"

TITLE = "3次元動作制御に基づく食物画像編集"
SUBTITLE = "Food Image Editing Based on Three-Dimensional Action Control"

INK = colors.HexColor("#171717")
MID = colors.HexColor("#5B6573")
BLUE = colors.HexColor("#2B5C8A")
RED = colors.HexColor("#A83A32")
GREEN = colors.HexColor("#3F795F")
PALE_BLUE = colors.HexColor("#EAF1F7")
PALE_RED = colors.HexColor("#F7ECEA")
PALE_GREEN = colors.HexColor("#ECF4EF")
LIGHT = colors.HexColor("#F5F5F3")


def convert_font(src: Path, dest: Path, family: str) -> None:
    """Convert the source CFF OTF to a TTF subset containing this paper's glyphs."""
    text = Path(__file__).read_text(encoding="utf-8")
    chars = set(map(ord, text)) | set(range(32, 127))
    if dest.exists() and dest.stat().st_mtime >= Path(__file__).stat().st_mtime:
        return
    f = FTFont(src)
    cmap = {u: g for u, g in f.getBestCmap().items() if u in chars}
    order = [".notdef"] + sorted(set(cmap.values()) - {".notdef"})
    glyph_set = f.getGlyphSet()
    glyphs = {}
    for name in order:
        pen = TTGlyphPen(None)
        glyph_set[name].draw(Cu2QuPen(pen, max_err=0.5, reverse_direction=True))
        glyphs[name] = pen.glyph()
    b = FontBuilder(f["head"].unitsPerEm, isTTF=True)
    b.setupGlyphOrder(order)
    b.setupCharacterMap(cmap)
    b.setupGlyf(glyphs)
    b.setupHorizontalMetrics({g: f["hmtx"].metrics[g] for g in order})
    b.setupHorizontalHeader(ascent=f["hhea"].ascent, descent=f["hhea"].descent)
    b.setupNameTable({"familyName": family, "styleName": "Regular", "uniqueFontIdentifier": family,
                      "fullName": family, "psName": family,
                      "copyright": "Derived from Harano Aji Fonts; SIL OFL 1.1."})
    b.setupOS2(sTypoAscender=f["OS/2"].sTypoAscender, sTypoDescender=f["OS/2"].sTypoDescender,
               usWinAscent=f["OS/2"].usWinAscent, usWinDescent=f["OS/2"].usWinDescent)
    b.setupPost(); b.setupMaxp(); b.save(dest)


def register_fonts() -> None:
    ASSET.mkdir(parents=True, exist_ok=True)
    mincho = ASSET / "FSEThesisMincho.ttf"
    gothic = ASSET / "FSEThesisGothic.ttf"
    convert_font(SOURCE_FONT_DIR / "HaranoAjiMincho-Regular.otf", mincho, "FSEThesisMincho")
    convert_font(SOURCE_FONT_DIR / "HaranoAjiGothic-Medium.otf", gothic, "FSEThesisGothic")
    pdfmetrics.registerFont(TTFont("JPMincho", str(mincho)))
    pdfmetrics.registerFont(TTFont("JPGothic", str(gothic)))
    pdfmetrics.registerFontFamily("JPMincho", normal="JPMincho", bold="JPGothic")


class ThesisDocTemplate(BaseDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, ChapterHeading):
            text = flowable.title
            key = f"chapter_{self.page}_{flowable.number}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=0, closed=False)
            self.notify("TOCEntry", (0, text, max(1, self.page - 3), key))
            return
        if isinstance(flowable, Paragraph):
            style = flowable.style.name
            if style == "Section":
                level = 1
                text = flowable.getPlainText()
                key = f"h{self.page}_{level}_{len(text)}"
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(text, key, level=level, closed=False)
                self.notify("TOCEntry", (level, text, max(1, self.page - 3), key))


class ChapterRule(Flowable):
    def __init__(self, height=7 * mm):
        super().__init__()
        self.height = height

    def draw(self):
        self.canv.setFillColor(INK)
        self.canv.rect(0, 0, 2.2 * mm, self.height, fill=1, stroke=0)


class ChapterHeading(Flowable):
    def __init__(self, number: int, title: str):
        super().__init__()
        self.number = number
        self.title = f"第{number}章　{title}"
        self.height = 15 * mm

    def wrap(self, availWidth, availHeight):
        self.width = availWidth
        return availWidth, self.height

    def draw(self):
        self.canv.setFillColor(INK)
        self.canv.rect(0, self.height - 10 * mm, 2.2 * mm, 8.2 * mm, fill=1, stroke=0)
        self.canv.setFont("JPGothic", 17.5)
        self.canv.drawString(6 * mm, self.height - 8.1 * mm, self.title)


def page_frame(canvas, doc):
    page = canvas.getPageNumber()
    if page == 1:
        return
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#B9BDC2"))
    canvas.setLineWidth(0.45)
    canvas.line(24 * mm, 282 * mm, 186 * mm, 282 * mm)
    canvas.setFont("JPMincho", 7.5)
    canvas.setFillColor(MID)
    canvas.drawString(24 * mm, 285.2 * mm, TITLE)
    canvas.drawRightString(186 * mm, 16 * mm, str(page - 3) if page >= 4 else "")
    canvas.restoreState()


def styles():
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontName="JPMincho",
        fontSize=9.4,
        leading=15.2,
        alignment=TA_JUSTIFY,
        wordWrap="CJK",
        firstLineIndent=9.4,
        textColor=INK,
        spaceAfter=3.2 * mm,
        splitLongWords=False,
    )
    noindent = ParagraphStyle("NoIndent", parent=body, firstLineIndent=0)
    chapter = ParagraphStyle(
        "Chapter",
        parent=base["Heading1"],
        fontName="JPGothic",
        fontSize=17.5,
        leading=23,
        textColor=INK,
        spaceBefore=0,
        spaceAfter=8 * mm,
        keepWithNext=True,
    )
    front = ParagraphStyle("Front", parent=chapter)
    section = ParagraphStyle(
        "Section",
        parent=base["Heading2"],
        fontName="JPGothic",
        fontSize=12.2,
        leading=17,
        textColor=INK,
        spaceBefore=5 * mm,
        spaceAfter=2.5 * mm,
        keepWithNext=True,
    )
    sub = ParagraphStyle(
        "Subsection",
        parent=base["Heading3"],
        fontName="JPGothic",
        fontSize=10.3,
        leading=14,
        textColor=INK,
        spaceBefore=3 * mm,
        spaceAfter=1.5 * mm,
        keepWithNext=True,
    )
    caption = ParagraphStyle(
        "Caption",
        parent=base["BodyText"],
        fontName="JPMincho",
        fontSize=7.8,
        leading=10.8,
        alignment=TA_LEFT,
        wordWrap="CJK",
        textColor=INK,
        spaceBefore=1.3 * mm,
        spaceAfter=4 * mm,
    )
    note = ParagraphStyle(
        "Note",
        parent=noindent,
        fontSize=8.2,
        leading=12.2,
        textColor=MID,
        backColor=LIGHT,
        borderPadding=7,
        borderColor=colors.HexColor("#D8D8D4"),
        borderWidth=0.4,
        borderRadius=2,
        spaceBefore=2 * mm,
        spaceAfter=4 * mm,
    )
    ref = ParagraphStyle(
        "Reference",
        parent=noindent,
        fontSize=8.2,
        leading=12.1,
        leftIndent=6 * mm,
        firstLineIndent=-6 * mm,
        spaceAfter=2.2 * mm,
    )
    return {
        "body": body,
        "noindent": noindent,
        "chapter": chapter,
        "front": front,
        "section": section,
        "sub": sub,
        "caption": caption,
        "note": note,
        "ref": ref,
    }


def crop_fit(src: Path, size: tuple[int, int]) -> PILImage.Image:
    im = PILImage.open(src).convert("RGB")
    tw, th = size
    ratio = max(tw / im.width, th / im.height)
    im = im.resize((round(im.width * ratio), round(im.height * ratio)), PILImage.Resampling.LANCZOS)
    left = (im.width - tw) // 2
    top = (im.height - th) // 2
    return im.crop((left, top, left + tw, top + th))


def make_framework(path: Path) -> None:
    """Make a paper-style framework figure from verified local artifacts."""
    W, H = 2200, 710
    im = PILImage.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(str(ASSET / "FSEThesisGothic.ttf"), 42)
    small = ImageFont.truetype(str(ASSET / "FSEThesisMincho.ttf"), 30)
    tiny = ImageFont.truetype(str(ASSET / "FSEThesisMincho.ttf"), 25)
    inp = ROOT / "artifacts" / "day19_multimaterial_dataset_v3" / "cake" / "reference.png"
    ctl = ROOT / "artifacts" / "day19_multimaterial_dataset_v3" / "cake" / "relative3d_review.png"
    out = ROOT / "artifacts" / "day20_cake_recovery_gp40_20260909_v1" / "cake__relative3d" / "projected_final_hold.png"
    thumbs = [crop_fit(inp, (360, 270)), crop_fit(ctl, (510, 270)), crop_fit(out, (360, 270))]
    xs = [55, 700, 1785]
    im.paste(thumbs[0], (xs[0], 205)); im.paste(thumbs[1], (xs[1], 205)); im.paste(thumbs[2], (xs[2], 205))
    for x, w in [(xs[0], 360), (xs[1], 510), (xs[2], 360)]:
        d.rectangle((x, 205, x + w, 475), outline=(35, 35, 35), width=3)
    d.text((118, 120), "入力画像", font=font, fill=(20, 20, 20))
    d.text((760, 120), "21フレームRGB制御列", font=font, fill=(158, 48, 45))
    d.text((1870, 120), "最終画像", font=font, fill=(34, 96, 67))

    # Relative-3D state and projection, drawn as a continuous annotated path.
    x0, y0 = 465, 340
    d.line((x0, y0, x0 + 150, y0 - 85), fill=(50, 88, 130), width=6)
    d.line((x0, y0, x0 + 20, y0 - 150), fill=(50, 88, 130), width=6)
    d.line((x0, y0, x0 - 85, y0 + 75), fill=(50, 88, 130), width=6)
    d.ellipse((x0 - 10, y0 - 10, x0 + 10, y0 + 10), fill=(168, 58, 50))
    d.text((410, 505), "相対3D状態 (X,Y,Z)", font=small, fill=(158, 48, 45))
    d.text((430, 548), "接触点・奥行・食材形状", font=tiny, fill=(75, 75, 75))

    def arrow(a, b, color=(45, 45, 45)):
        d.line((a[0], a[1], b[0], b[1]), fill=color, width=7)
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        q1 = (b[0] - 24 * math.cos(ang - .45), b[1] - 24 * math.sin(ang - .45))
        q2 = (b[0] - 24 * math.cos(ang + .45), b[1] - 24 * math.sin(ang + .45))
        d.polygon([b, q1, q2], fill=color)

    arrow((420, 340), (465, 340)); arrow((630, 340), (690, 340))
    arrow((1225, 340), (1285, 340)); arrow((1710, 340), (1775, 340))
    # VACE/Wan core as one integrated diffusion module, not a stack of disconnected cards.
    d.rounded_rectangle((1290, 210, 1705, 470), radius=36, fill=(234, 241, 247), outline=(43, 92, 138), width=5)
    d.text((1390, 252), "VACE", font=font, fill=(43, 92, 138))
    d.text((1335, 322), "VCU / Context Adapter", font=small, fill=(43, 92, 138))
    d.text((1360, 372), "Wan DiT (凍結)", font=small, fill=(43, 92, 138))
    d.text((1318, 510), "学習済み動画拡散モデルによる外観生成", font=tiny, fill=(75, 75, 75))
    # Preservation path.
    d.line((235, 490, 235, 630, 1980, 630, 1980, 492), fill=(63, 121, 95), width=6)
    arrow((1980, 630), (1980, 492), color=(63, 121, 95))
    d.text((760, 646), "元画像を領域外で保持する局所合成", font=small, fill=(63, 121, 95))
    d.text((60, 55), "本研究の追加部分", font=small, fill=(158, 48, 45))
    d.line((330, 78, 1210, 78), fill=(158, 48, 45), width=5)
    d.text((1300, 55), "公開済みVACE/Wan", font=small, fill=(43, 92, 138))
    d.line((1625, 78, 1710, 78), fill=(43, 92, 138), width=5)
    im.save(path, quality=95)


def make_baseline_chart(path: Path) -> None:
    from reportlab.pdfgen import canvas

    # A temporary vector PDF is not convenient for Platypus embedding; draw a high-DPI PNG with PIL.
    W, H = 1700, 820
    im = PILImage.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    title = ImageFont.truetype(str(ASSET / "FSEThesisGothic.ttf"), 44)
    font = ImageFont.truetype(str(ASSET / "FSEThesisMincho.ttf"), 31)
    small = ImageFont.truetype(str(ASSET / "FSEThesisMincho.ttf"), 25)
    d.text((65, 35), "選定開発4例×3 seedにおける内部判定", font=title, fill=(25, 25, 25))
    labels = ["Action", "Photo", "Preservation", "Strict"]
    chord = [0.0, 1 / 12, 1.0, 0.0]
    qwen = [3 / 12, 1.0, 0.0, 0.0]
    left, top, right, bottom = 160, 130, 1620, 680
    d.line((left, bottom, right, bottom), fill=(20, 20, 20), width=3)
    d.line((left, top, left, bottom), fill=(20, 20, 20), width=3)
    for t in range(0, 101, 25):
        y = bottom - (bottom - top) * t / 100
        d.line((left, y, right, y), fill=(225, 225, 225), width=2)
        d.text((55, y - 15), f"{t}%", font=small, fill=(80, 80, 80))
    group = (right - left) / 4
    bw = 105
    for i, lab in enumerate(labels):
        cx = left + group * (i + 0.5)
        for j, (val, color) in enumerate([(chord[i], (99, 130, 161)), (qwen[i], (188, 92, 78))]):
            x = cx - 115 + j * 125
            h = (bottom - top) * val
            d.rectangle((x, bottom - h, x + bw, bottom), fill=color)
            d.text((x + 24, bottom - h - 42), f"{round(val * 100)}", font=small, fill=color)
        d.text((cx - 70, 705), lab, font=font, fill=(25, 25, 25))
    d.rectangle((1160, 70, 1198, 100), fill=(99, 130, 161)); d.text((1210, 68), "ChordEdit", font=small, fill=(40, 40, 40))
    d.rectangle((1400, 70, 1438, 100), fill=(188, 92, 78)); d.text((1450, 68), "Qwen Image", font=small, fill=(40, 40, 40))
    d.text((60, 780), "注：非盲検・選定開発例。Oursを含むheld-out比較ではない。", font=small, fill=(130, 55, 48))
    im.save(path, quality=95)


def fig(path: Path, width: float, caption: str, S, max_height: float | None = None):
    with PILImage.open(path) as im:
        w, h = im.size
    height = width * h / w
    if max_height and height > max_height:
        height = max_height
        width = height * w / h
    return KeepTogether([Image(str(path), width=width, height=height), Paragraph(caption, S["caption"])])


def table(data, widths, font_size=7.8, header=True, highlights=()):
    rows = []
    for ri, row in enumerate(data):
        rows.append([
            Paragraph(str(cell), ParagraphStyle(
                f"cell_{ri}_{ci}", fontName="JPGothic" if ri == 0 else "JPMincho",
                fontSize=font_size, leading=font_size + 3, wordWrap="CJK",
                alignment=TA_LEFT if ci == 0 else TA_CENTER,
            ))
            for ci, cell in enumerate(row)
        ])
    t = LongTable(rows, colWidths=widths, repeatRows=1 if header else 0)
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, INK),
    ]
    for row in highlights:
        cmds += [("BACKGROUND", (0, row), (-1, row), PALE_RED), ("TEXTCOLOR", (0, row), (-1, row), RED)]
    t.setStyle(TableStyle(cmds))
    return t


def chapter(story, number: int, title: str, S):
    story.append(PageBreak())
    story.append(ChapterHeading(number, title))


def p(story, text: str, S, style="body"):
    story.append(Paragraph(text, S[style]))


def build() -> None:
    ASSET.mkdir(parents=True, exist_ok=True)
    register_fonts()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    framework = ASSET / "framework.png"
    baseline_chart = ASSET / "baseline_rates.png"
    make_framework(framework)
    make_baseline_chart(baseline_chart)
    S = styles()

    doc = ThesisDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=24 * mm, rightMargin=24 * mm,
        topMargin=22 * mm, bottomMargin=22 * mm,
        title=TITLE, author="GUO ZHENGPENG",
    )
    from reportlab.platypus import Frame
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=page_frame)])

    story = []
    # Cover
    story += [Spacer(1, 43 * mm)]
    p(story, "令和8年度　修士論文中間報告", S, "noindent")
    story[-1].style = ParagraphStyle("CoverKicker", parent=S["noindent"], fontName="JPMincho", fontSize=14, leading=20, alignment=TA_CENTER)
    story += [Spacer(1, 20 * mm)]
    p(story, TITLE, S, "noindent")
    story[-1].style = ParagraphStyle("CoverTitle", parent=S["noindent"], fontName="JPGothic", fontSize=23, leading=34, alignment=TA_CENTER)
    p(story, SUBTITLE, S, "noindent")
    story[-1].style = ParagraphStyle("CoverSub", parent=S["noindent"], fontName="JPMincho", fontSize=10.5, leading=15, alignment=TA_CENTER, textColor=MID)
    story += [Spacer(1, 52 * mm)]
    cover_lines = [
        "電気通信大学　大学院情報理工学研究科",
        "情報学専攻　メディア情報学プログラム",
        "学籍番号 2530030　GUO ZHENGPENG",
        "主指導教員　柳井 啓司 教授",
        "副指導教員　高橋 裕樹 教授",
        "令和8年9月21日",
    ]
    for line in cover_lines:
        q = Paragraph(line, ParagraphStyle("CoverMeta", parent=S["noindent"], fontName="JPMincho", fontSize=11.5, leading=21, alignment=TA_CENTER))
        story.append(q)

    # Abstracts
    story.append(PageBreak())
    p(story, "概要", S, "front")
    p(story, "本研究は，一枚の食物写真に「すくう」「つまむ」「一口を持ち上げる」といった取食動作を付加し，元画像の容器・背景・料理全体を保持した最終画像を生成する課題を扱う．一般的な画像編集では，写真的な外観が得られても非編集領域が変化しやすく，動画生成では時間的な動きが得られても接触位置や食物の移動元が曖昧になりやすい．そこで，画像上の接触点，食物形状，相対深度，動作段階から21フレームのRGB制御列を構築し，公開済みVACE/Wanへ入力した後，動的支持領域だけを元画像へ合成するFoodStateEditを実装した．", S)
    p(story, "4種類の選定開発例を用いた同条件消融では，明示的な動作制御がないNative入力に比べ，道具と食物の相互作用を生成しやすいことを確認した．一方，相対3D制御の2D制御に対する明瞭な改善は0/4例であり，現段階では3D表現の優越性を主張できない．外部ベースラインでは，Qwen Imageは写真自然度が高いが領域外保持に失敗し，ChordEditは領域外保持を満たすが動作生成に失敗した．また，食物を取り去った後の残存部を扱う状態遷移実験では，持上げ部分と欠損部を同じ除去体積から生成し，参照状態の矛盾と材質供給領域の混入が主要な失敗原因であることを特定した．純粋な材質供給領域への修正により特定の白色三角欠陥を40.30%から0%へ低減したが，床面・影・接触の写真自然度は未解決である．", S)
    p(story, "本稿の貢献は，(1) 食物操作を段階化した相対3D制御列，(2) 元画像を画素単位で保持する局所合成，(3) 持上げ部分と残存部分を対応付ける状態遷移設計，(4) 成功だけでなく負の結果を含む再現可能な評価手順にある．ただし，評価は選定開発例と合成ケーキを含み，独立盲検・held-out汎化・物理正確性は未達である．", S)
    p(story, "キーワード：食物画像編集，動画拡散モデル，3次元動作制御，状態遷移，背景保持", S, "noindent")
    story.append(Spacer(1, 7 * mm))
    p(story, "Abstract", S, "sub")
    p(story, "This study addresses interaction-aware food image editing: adding utensil contact and bite-lifting actions to a single food photograph while preserving the original container and background. FoodStateEdit converts manually specified contact, relative depth, food geometry, and action phase into a 21-frame RGB control sequence for a frozen VACE/Wan generator, then composites only the dynamic support back into the source image. Selected development experiments show that explicit motion control is useful, but do not establish a clear advantage of relative-3D control over the matched 2-D control. A state-transfer study further links the carried bite and the remaining cavity through one shared removal volume. Correcting a contaminated material donor eliminated a specific white-wall artifact, although photographic quality remains limited. The present evidence supports a reproducible control-and-diagnosis framework, not held-out generalization or physical correctness.", S, "noindent")
    p(story, "Keywords: food image editing, video diffusion, 3D action control, state transfer, background preservation", S, "noindent")

    # TOC
    story.append(PageBreak())
    p(story, "目次", S, "front")
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("TOC1", fontName="JPGothic", fontSize=10.5, leading=18, leftIndent=0, firstLineIndent=0),
        ParagraphStyle("TOC2", fontName="JPMincho", fontSize=9, leading=15, leftIndent=8 * mm, firstLineIndent=0, textColor=MID),
    ]
    story.append(toc)

    # Chapter 1
    chapter(story, 1, "はじめに", S)
    p(story, "食物写真は味や質感だけでなく，食べる行為を想像させる媒体である．しかし，一枚の静止画に写る料理へ後から「スプーンですくう」「箸で麺を持ち上げる」「フォークで一口を取る」といった動作を加えるには，道具の形状，食物との接触，動いた食物の出所，露出した残存部，さらに元の器や背景の保持を同時に扱う必要がある．本研究は，普通の料理写真に取食の瞬間を加え，見る人が味わう場面を想像しやすくする画像編集を目標とする．飲食店やコンテンツ制作者が既存写真から新しい表現を作る支援も想定するが，楽しさや制作時間短縮の効果は本稿では評価していない．", S)
    p(story, "既存のテキスト指示型画像編集は，写真的な画像を生成できる一方，要求した場所以外の器，画角，背景まで変えることがある．動画拡散モデルは時間連続な変化を作れるが，食物操作に固有の接触位置や一口の出所は自動では定まらない．この二つの問題を分離するため，本研究では動作を前処理で明示し，外観生成をVACE [1]とWan [2]へ委ね，領域外保持を後処理で保証する．", S)
    p(story, "研究課題は三つである．第一に，食物と道具の接触を21フレームの制御列へ変換できるか．第二に，生成画像の非編集領域を元画像と一致させられるか．第三に，持ち上げた一口と元位置の欠損を同一状態として扱い，食物の複製や残留を減らせるか．現在の到達点は一，二，および第三の部分的な機構検証であり，汎化性能と最終的な写真品質は未確立である．", S)
    p(story, "本稿は，山倉の中間報告に見られる簡潔な章構成と，熊の修士論文に見られる方法・実験・消融を分離した説明を参考に再構成した．両文書は構成と視覚様式の参考資料であり，本研究の科学的根拠には用いていない．", S, "note")
    p(story, "1.1　本研究の貢献", S, "section")
    contrib = [
        ["項目", "本稿で確認した内容", "主張しない内容"],
        ["動作制御", "接近・接触・持上げ・停留を21フレームへ変換", "実測3D復元・物理シミュレーション"],
        ["画像保持", "局所合成により支持領域外を画素一致", "VACE単体の背景保持性能"],
        ["状態遷移", "一口と欠損を共有除去体積から構成", "質量保存・材質の一般化"],
        ["検証", "失敗を含むハッシュ・設定・出力の保存", "held-out統計・独立盲検"],
    ]
    story.append(table(contrib, [25 * mm, 73 * mm, 62 * mm], font_size=7.7))

    # Chapter 2
    chapter(story, 2, "関連研究", S)
    p(story, "2.1　動画生成・編集", S, "section")
    p(story, "VACEは参照画像，動画，マスクをVideo Condition Unitで統一し，Context Adapterを通じて動画拡散Transformerへ条件を注入する生成・編集基盤である [1]．WanはVACEの基盤となる大規模動画生成モデル群であり，時空間VAEとDiffusion Transformerを用いる [2]．本研究はVACEの重みやCondition Encoderを新たに学習するのではなく，VACEへ渡すRGB制御列とマスクを食物操作向けに構成する．したがって差分は生成器そのものではなく，操作状態の設計と出力の局所合成にある．", S)
    p(story, "2.2　静止画編集と幾何制御", S, "section")
    p(story, "Qwen-Imageは視覚言語表現とVAE表現を組み合わせ，精密な画像生成・編集を扱う [3]．ChordEditは一段編集を低エネルギー輸送として定式化し，高速なテキスト編集を行う [4]．GeoEditは対象を3Dへ持上げ，操作，レンダリングした幾何proxyを拡散過程へ注入する [5]．これらは有力な比較対象であるが，FoodStateEditは食物の接触段階と残存部を明示し，領域外を元画像で置換する点を重視する．", S)
    p(story, "2.3　領域抽出", S, "section")
    p(story, "SAM 3はテキスト，例示，視覚プロンプトを用いて画像・動画中の概念を検出，分割，追跡できる [6]．本研究ではSAM 3を観察器として試験したが，二本の箸の分離や細い麺の独立マスクに失敗する例があり，現行の主生成経路には接続していない．現在の編集マスクは，手続き的に描画した道具・食物の全フレーム支持領域を合併し，膨張と境界重みを加えて作る．", S)
    rel = [
        ["手法", "主な入力", "担当範囲", "本研究での位置付け"],
        ["VACE [1]", "画像・動画・マスク・テキスト", "時間連続な生成・編集", "凍結した生成backend"],
        ["Qwen Image [3]", "画像・テキスト", "写真的な静止画編集", "外部baseline"],
        ["ChordEdit [4]", "画像・source/target text", "高速局所編集", "外部baseline"],
        ["GeoEdit [5]", "画像・幾何proxy", "剛体幾何編集", "近接する関連研究"],
        ["SAM 3 [6]", "画像/動画・概念prompt", "領域抽出・追跡", "補助観察器（未接続）"],
    ]
    story.append(Spacer(1, 3 * mm)); story.append(table(rel, [27 * mm, 44 * mm, 43 * mm, 46 * mm], font_size=7.5))

    # Chapter 3
    chapter(story, 3, "提案手法", S)
    p(story, "3.1　全体構成", S, "section")
    story.append(fig(framework, 160 * mm, "図3.1　FoodStateEditの全体構成．赤は本研究の動作制御，青は公開済みVACE/Wan，緑は元画像による領域外保持を示す．制御画像は外観の正解ではなく，接触・奥行・動作段階を伝えるproxyである．", S, max_height=54 * mm))
    p(story, "入力は食物写真I，取食要求（道具，対象食物，動作）および手動アンカーである．前処理は相対3D状態S<sub>t</sub>からRGB制御C<sub>t</sub>と重みM<sub>t</sub>を作る．VACEはC，M，I，テキストから21フレームの動画Jを生成する．最後に固定した選択フレームt*を用い，I<sub>out</sub>=αJ<sub>t*</sub>+(1−α)Iで局所合成する．ここでα=0の領域は構造上Iと完全一致する．", S)
    p(story, "VACEとの差は三点である．第一に，食物と道具を同じ接触点で動かす状態設計をVACEの前段に追加する．第二に，接近・接触・持上げ・停留を固定時系列へ変換する．第三に，生成後の全画面を採用せず，操作支持領域のみを元画像へ戻す．VACEのVAE，Context Adapter，Wan DiTは公開済み重みを使用し，本稿の主比較ではLoRAを用いない．", S)
    p(story, "3.2　相対3D状態と透視投影", S, "section")
    p(story, "画像座標(u,v)と手動の相対深度Zから，仮想カメラの内部パラメータを用いてX=(u−c<sub>x</sub>)Z/f<sub>x</sub>，Y=(v−c<sub>y</sub>)Z/f<sub>y</sub>を計算する．688×512画素ではf<sub>x</sub>=f<sub>y</sub>=825.6，(c<sub>x</sub>,c<sub>y</sub>)=(344,256)とする．接触点P<sub>t</sub>，道具状態U<sub>t</sub>，食材形状V<sub>t</sub>を相対座標で更新し，u=f<sub>x</sub>X/Z+c<sub>x</sub>，v=f<sub>y</sub>Y/Z+c<sub>y</sub>で画像へ再投影する．これは実測深度やCADモデルではなく，単眼画像上で奥行順序と可視性を制御するための手続き的表現である．", S)
    p(story, "固体食材は頂点集合，麺は曲線，道具は線分または面として表す．状態ベクトルの概念形はS<sub>t</sub>={P<sub>t</sub>,U<sub>t</sub>,V<sub>t</sub>,Z<sub>t</sub>,R<sub>t</sub>}であり，出力Cは21×512×688×3，Mは21×512×688×1である．VACE内部では条件画像がVAEで符号化され，既存Context AdapterからWan DiTへhintとして加えられる．本研究固有の学習済みCondition Encoderは存在しない．", S)
    story.append(fig(ROOT / "artifacts" / "day19_multimaterial_dataset_v3" / "cake" / "relative3d_review.png", 160 * mm, "図3.2　ケーキ例の相対3D制御列（フレーム0, 3, 6, 10, 15, 20）．接近，接触，持上げ，停留を同一の接触状態から描画する．", S, max_height=79 * mm))
    p(story, "3.3　動作段階とマスク", S, "section")
    stages = [
        ["段階", "frame", "状態更新", "目的"],
        ["接近", "0–5", "道具を接触点へ移動", "背景を変えずに動作を開始"],
        ["接触", "6–8", "道具と食材の位置を固定", "接触関係を形成"],
        ["持上げ", "9–15", "食材を道具と同時に移動", "追従と元位置変化を表現"],
        ["停留", "16–20", "位置・姿勢を安定", "最終画像候補を得る"],
    ]
    story.append(table(stages, [25 * mm, 22 * mm, 61 * mm, 52 * mm], font_size=7.8))
    p(story, "編集範囲αは全フレームの操作支持領域の和集合から作る．道具，移動食物，露出部の領域を膨張し，境界にfeatherを加える．固定フレームの選択と合成は再現可能であるが，領域外完全一致はこの合成規則による保証であり，生成モデルが背景を保持できたことを意味しない．", S)

    # Chapter 4
    chapter(story, 4, "食物状態遷移", S)
    p(story, "4.1　持上げ部分と残存部分の対応", S, "section")
    p(story, "通常の画像編集では，餐具上に一口が現れても元位置に同じ食物が残り，複製に見えることがある．そこで，移動前のsource state，餐具と共に動くcarried state，元位置に残るremain stateを分け，同じ除去体積Ωから移動maskと欠損maskを生成する．概念的にはS<sub>t+1</sub>=F(S<sub>t</sub>,A<sub>t</sub>)であり，A<sub>t</sub>は道具動作である．現在のΩは手動直方体の画像投影で，物理的な質量保存を保証しない．", S)
    story.append(fig(ROOT / "artifacts" / "e5_bite_remain_review_v1_20260917" / "four_way_source_crop.png", 160 * mm, "図4.1　共有除去体積を用いた状態遷移診断．参照画像が元状態のままでは食物が再生成されるが，欠損状態を参照にすると元位置の空隙が現れる．ただし壁面は平面的で，持上げ部分が空隙を遮蔽する．", S, max_height=53 * mm))
    p(story, "4.2　材質供給領域と投影", S, "section")
    p(story, "欠損部の新しい切断面を作るため，元画像の食材表面から材質donorを取り出し，二つの壁面へ写像する．E6ではdonorの37.55%が皿の低彩度画素を含み，hard latent projectionがその誤りを強制した．E7では食材だけを含む最大正方領域をfail-closedで選び，壁面材質を補正した．soft variantでは初期8 denoising stepだけ0.35(1−k/8)<sup>2</sup>の重みで参照を投影する．", S)
    story.append(fig(ROOT / "artifacts" / "e7_cavity_review_v1_20260919" / "final_cavity_comparison.png", 160 * mm, "図4.2　欠損部材質の診断．E6の白色三角欠陥は，皿画素を含むdonorとhard投影の組合せで生じた．E7はdonor修正により欠陥を除去したが，床面と影は不自然なままである．", S, max_height=50 * mm))

    # Chapter 5
    chapter(story, 5, "実験", S)
    p(story, "5.1　実験設定", S, "section")
    setup = [
        ["項目", "設定"],
        ["生成backend", "PAI/Wan2.2-VACE-Fun-A14B（公開済み重み）"],
        ["解像度・長さ", "688×512，21 frames"],
        ["推論", "20 steps，VACE scale 1（scale診断では0.6/0.8を追加）"],
        ["学習・高速化", "主比較はLoRA off，TTM off"],
        ["seed", "1–3（実験ごとに固定）"],
        ["食物", "ラーメン，スープ，炒飯，合成ケーキ"],
        ["評価", "Action，Photo，Preservation，Strict End-to-End"],
    ]
    story.append(table(setup, [45 * mm, 115 * mm], font_size=8.0))
    p(story, "Actionは道具種類・接触位置・食物追従，Photoは変形や生成破綻を含む写真自然度，Preservationは領域外保持を表す．Strictは三条件を同時に満たす場合だけ成功とする．PSNR，SSIM，MAEは入力保持や変更量の診断であり，動作成功や自然度の代替ではない．独立盲検が未完了のため，本稿の✓/×と成功数は内部評価として明示する．", S)
    p(story, "5.2　多材料生成例", S, "section")
    story.append(fig(ROOT / "artifacts" / "day20_non_noodle_presentation_v1" / "non_noodle_comparison.png", 151 * mm, "図5.1　スープ，炒飯，合成ケーキの選定開発例．固定frame 20，seed 1，20 steps，LoRAなし．相対3D制御で道具と食物の持上げを生成できるが，道具境界，影，接触に写真上の不自然さが残る．", S, max_height=135 * mm))
    p(story, "6条件は技術的に完了し，すべて21フレームを持つ．スープと炒飯は実画像，ケーキは合成入力である．ケーキは比較的明瞭な持上げを示す一方，フォークは棒状で，欠損部と影が手続き的に見える．したがって多材料への適用可能性を示す原型であり，写真品質や一般化の成功例ではない．", S)
    p(story, "5.3　外部手法との比較", S, "section")
    story.append(fig(ROOT / "artifacts" / "day21_cross_method_pilot_v1" / "cross_method_pilot_grid.png", 160 * mm, "図5.2　同じ4入力による初期cross-method pilot（seed 1）．複数の列が同じWan/VACE backendを共有するため，foundation modelの順位表ではなく，制御と投影の開発比較である．", S, max_height=100 * mm))
    p(story, "別途，ChordEditとQwen Imageを同じ4入力×3 seedで実行した．表5.1は失敗seedを除外しない内部評価である．Qwenは12/12で写真自然度を満たしたが領域外保持は0/12であり，ChordEditは12/12で領域外を保持したがActionは0/12であった．いずれもStrictは0/12である．これは4つの選定開発clusterに限る記述統計であり，母集団への推論ではない．", S)
    ext = [
        ["方法", "Action", "Photo", "Preservation", "Strict"],
        ["ChordEdit [4]", "0/12", "1/12", "12/12", "0/12"],
        ["Qwen Image [3]", "3/12", "12/12", "0/12", "0/12"],
    ]
    story.append(table(ext, [48 * mm, 28 * mm, 28 * mm, 31 * mm, 25 * mm], font_size=8.0))
    story.append(fig(baseline_chart, 155 * mm, "図5.3　外部baselineの内部判定率．4入力をclusterとしてbootstrapした区間は結果記録に保存したが，独立評価者がいないため確認的検定は行わない．", S, max_height=74 * mm))
    p(story, "5.4　同条件消融", S, "section")
    ablation = [
        ["比較", "観察", "判定"],
        ["Native入力 vs 明示制御", "4例でNativeは要求動作を作れず，明示制御は道具と食物を生成", "制御proxyの必要性を支持"],
        ["2D制御 vs 相対3D制御", "明瞭な相対3D改善 0/4例", "3D優位は未成立"],
        ["固定scale vs 材質適応", "明瞭な改善 0/4例", "材質適応優位は未成立"],
        ["支持領域外", "全60セルで最大画素差0", "合成実装の完全性"],
    ]
    story.append(table(ablation, [43 * mm, 78 * mm, 39 * mm], font_size=7.7))
    p(story, "60セル（4例×3 seed×5条件）は技術的に完了した．明示制御の必要性は支持されたが，相対3Dおよび材質適応の意図した追加効果は通過しなかった．この負の結果は，現在の2D対照と相対3Dが画面上でほぼ同じ軌跡を共有し，奥行差が最終生成へ十分伝わらないことを示唆する．", S)
    p(story, "5.5　欠損部修正とSCFST予備実験", S, "section")
    cavity = [
        ["出力", "壁面低彩度率", "解釈"],
        ["E5b 投影なし", "0.00%", "欠損は出るが壁面・影が角張る"],
        ["E6 old donor / hard", "40.30%", "白色三角欠陥"],
        ["E7 pure donor / hard", "0.00%", "混入欠陥を除去"],
        ["E7 pure donor / soft", "0.00%", "境界は軟化，床面は未解決"],
    ]
    story.append(table(cavity, [50 * mm, 35 * mm, 75 * mm], font_size=7.8, highlights=(2,)))
    p(story, "E7は特定欠陥の原因と修正を検証したが，Photo Successを確立しない．さらに，Source-Conditioned Food State Transfer（SCFST）の配線と小規模adapter学習を行った．1つのseen synthetic cake pseudo-targetで16 step学習し，lossは0.03988から0.01585へ低下した．しかしstep16出力とzero-init baselineのMAEは0.262/255，PSNRは42.58 dBで，差は局所的であり明瞭な自然度改善は確認できなかった．これは容量診断であって汎化結果ではない．", S)
    story.append(fig(ROOT / "output" / "scfst_step16_visual_review_20260920" / "07_final_comparison.png", 160 * mm, "図5.4　SCFST adapterのzero-initとstep16の比較．差分は接触・欠損部に局在するが，写真的改善は明確でない．", S, max_height=49 * mm))

    # Chapter 6
    chapter(story, 6, "考察", S)
    p(story, "6.1　3次元制御の役割", S, "section")
    p(story, "相対3D制御の理論的利点は，2Dの終点指定よりも，前後関係，遮蔽，接触点，複数部位の同期を一つの状態で記述できることにある．特に，箸の二本構造，麺の前後関係，持上げ部分と元位置の空隙を同時に更新する場合，奥行を持つ状態表現は有用である．ただし，本実装では制御列がRGBへ投影された後にVACEへ入るため，3D構造そのものがモデル内部へ保存される保証はない．現状の0/4結果は，3Dが不要であることではなく，3D情報の注入方式が弱いことを示す．", S)
    p(story, "今後3Dの追加価値を検証するには，2Dと3Dで画面上の接触軌跡を同じにしつつ，遮蔽順序，回転，見えない面の出現だけを変えた反実仮想を作る必要がある．その上で，接触持続，二本箸分離，source residual，payload-cavity overlapを時系列で計測する．終点画像だけでは3Dの主な利点である動作連続性を評価できないため，動画指標と最終画像指標を分離する．", S)
    p(story, "6.2　成功した点と失敗した点", S, "section")
    findings = [
        ["確認できた点", "未確認・失敗"],
        ["明示制御により要求道具と取食動作を出現させられる", "相対3Dが2Dより良いという統計的証拠"],
        ["局所合成で領域外を画素単位に保持できる", "生成器単体での背景保持"],
        ["一口と欠損部を同じ除去体積から作れる", "自然な破断面・影・物理的質量保存"],
        ["donor混入という負最適化原因を特定・修正", "held-out実画像への汎化"],
        ["設定・ハッシュ・技術失敗を保存", "独立複数人の盲検評価"],
    ]
    story.append(table(findings, [80 * mm, 80 * mm], font_size=7.8))
    p(story, "6.3　研究上の限界", S, "section")
    p(story, "第一に，選定した4例と合成ケーキへの依存が強い．第二に，主観評価は開発者による非盲検で，評価者間一致度がない．第三に，入力画像の一口領域と奥行は手動で，SAM 3は主経路に未接続である．第四に，領域外保持は局所合成で保証されるが，境界内部の自然さを保証しない．第五に，LoRAおよびSCFSTの結果はseen-sample容量診断で，異なる食品へ一般化した証拠ではない．第六に，食物の物理法則，質量保存，安全性，栄養的正確性は扱っていない．", S)
    p(story, "これらの制限から，本稿の中心結論は「FoodStateEditが既存手法より優れる」ではなく，「食物操作を制御・保持・状態遷移へ分解した再現可能な研究基盤を構築し，現在の失敗箇所を特定した」である．", S, "note")

    # Chapter 7
    chapter(story, 7, "おわりに", S)
    p(story, "本研究は，単一の食物画像に取食動作を加えるFoodStateEditを構築した．接触点，相対深度，食材形状，動作段階から21フレームのRGB制御列を生成し，VACE/Wanで外観を生成した後，動的支持領域だけを元画像へ合成する．これにより，明示制御がない条件より道具と食物の相互作用を作りやすく，領域外を画素単位で保持できる．さらに，持上げた一口と残存部を一つの除去体積から構成し，参照状態と材質donorの矛盾が欠損部失敗の原因となることを確認した．", S)
    p(story, "一方，相対3D制御の2D制御に対する明瞭な改善は確認できず，Photo Success，未使用画像への汎化，独立盲検は未達である．次段階では，(1) 未使用の寿司・肉・サラダ等と複数道具を含むheld-out評価，(2) 遮蔽・細い麺・大回転等のstress test，(3) source duplication，wrong-source，source residual，material correspondenceの評価，(4) 失敗分類，(5) 生成後の選択ではなく去雑音中に接触・元領域整合性を返すonline guidanceを実施する．", S)
    p(story, "最終的には，食物画像編集を単なる物体追加ではなく，source，carried，remainの状態遷移として扱い，動作の連続性と元画像の同一性を両立させることを目指す．", S)
    p(story, "データと再現性", S, "section")
    p(story, "設定，runner，SHA-256，失敗記録，評価JSON，選定図はローカル研究リポジトリに保存した．第三者画像を含むため，画像自体の公開範囲は権利確認後に決定する．", S)
    p(story, "倫理・利益相反・資金", S, "section")
    p(story, "本研究は人を対象とする実験を含まない．独立盲検評価を実施する場合は，参加者説明とデータ管理を別途定める．開示すべき利益相反はない．本稿に記載した範囲で特定の外部資金はない．", S)
    p(story, "著者貢献", S, "section")
    p(story, "GUO ZHENGPENG：Conceptualization，Methodology，Software，Investigation，Validation，Visualization，Writing – original draft．柳井啓司教授および高橋裕樹教授：Supervision．", S)
    p(story, "AI利用開示", S, "section")
    p(story, "実験整理，文章構成，コード補助および版面作成に生成AI支援を用いた．実験条件，数値，図，引用および結論の責任は著者が負う．", S)

    # References
    chapter(story, 8, "参考文献", S)
    refs = [
        "[1] Z. Jiang, Z. Han, C. Mao, J. Zhang, Y. Pan, and Y. Liu, “VACE: All-in-One Video Creation and Editing,” Proc. ICCV, 2025. arXiv:2503.07598.",
        "[2] Wan Team, “Wan: Open and Advanced Large-Scale Video Generative Models,” arXiv:2503.20314, 2025.",
        "[3] C. Wu et al., “Qwen-Image Technical Report,” arXiv:2508.02324, 2025.",
        "[4] L. Lu, X. Chen, M. Guo, S. Li, J. Wang, and Y. Shi, “ChordEdit: One-Step Low-Energy Transport for Image Editing,” Proc. CVPR, pp. 14398–14407, 2026.",
        "[5] Y. He, J. Wang, X. Wang, M. Fong, S. Zhang, Y. Xue, H.-T. Zheng, and Y. Ma, “GeoEdit: Geometry-Aware Object Editing via Dual-Branch Denoising,” arXiv:2606.30003, 2026.",
        "[6] N. Carion et al., “SAM 3: Segment Anything with Concepts,” arXiv:2511.16719, 2025; ICLR, 2026.",
        "[7] A. Kirillov et al., “Segment Anything,” Proc. ICCV, pp. 4015–4026, 2023.",
        "[8] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The Unreasonable Effectiveness of Deep Features as a Perceptual Metric,” Proc. CVPR, pp. 586–595, 2018.",
    ]
    for ref in refs:
        p(story, ref, S, "ref")
    story.append(Spacer(1, 5 * mm))
    p(story, "参考資料（構成・版面のみ）", S, "section")
    p(story, "山倉隆太，『微分可能レンダラーを用いたロゴ画像生成』令和5年度卒業研究論文中間レポート，電気通信大学，2023．", S, "ref")
    p(story, "Peilin Xiong, “Multi-View Positional Embedding Transplant for Identity-Aware Image Editing,” Master’s Thesis, The University of Electro-Communications, 2025.", S, "ref")

    doc.multiBuild(story)
    print(json.dumps({"output": str(OUT), "pages": doc.page, "bytes": OUT.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    build()
