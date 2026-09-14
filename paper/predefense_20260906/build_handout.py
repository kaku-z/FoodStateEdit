from pathlib import Path
import math

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf" / "FoodStateEdit_PreDefense_Handout_GUO_2530030_20260906.pdf"
RESULTS = ROOT / "results"

PAGE_W, PAGE_H = A4
MARGIN_X = 43
TOP_Y = 812
BOTTOM_Y = 34
COL_GAP = 15
COL_W = (PAGE_W - 2 * MARGIN_X - COL_GAP) / 2
LEFT_X = MARGIN_X
RIGHT_X = MARGIN_X + COL_W + COL_GAP

INK = colors.HexColor("#17202A")
MUTED = colors.HexColor("#59636E")
RULE = colors.HexColor("#87929D")
LIGHT = colors.HexColor("#F4F6F7")
BLUE = colors.HexColor("#DCEBFA")
BLUE_DARK = colors.HexColor("#2E6E9E")
ORANGE = colors.HexColor("#FCE8CC")
ORANGE_DARK = colors.HexColor("#B76517")
GREEN = colors.HexColor("#DFF1E5")
GREEN_DARK = colors.HexColor("#2C7A4B")
RED = colors.HexColor("#B33A3A")
RED_LIGHT = colors.HexColor("#F8E2E2")
WHITE = colors.white


def register_fonts():
    pdfmetrics.registerFont(TTFont("JPMincho", "C:/Windows/Fonts/yumin.ttf"))
    pdfmetrics.registerFont(TTFont("JPGothic", "C:/Windows/Fonts/meiryo.ttc", subfontIndex=0))
    pdfmetrics.registerFont(TTFont("JPGothicB", "C:/Windows/Fonts/meiryob.ttc", subfontIndex=0))


register_fonts()

STYLES = {
    "body": ParagraphStyle(
        "body", fontName="JPMincho", fontSize=7.35, leading=10.25,
        textColor=INK, alignment=TA_JUSTIFY, wordWrap="CJK", spaceAfter=2.2,
    ),
    "small": ParagraphStyle(
        "small", fontName="JPMincho", fontSize=6.45, leading=8.25,
        textColor=INK, alignment=TA_LEFT, wordWrap="CJK",
    ),
    "caption": ParagraphStyle(
        "caption", fontName="JPMincho", fontSize=6.3, leading=8.0,
        textColor=INK, alignment=TA_CENTER, wordWrap="CJK",
    ),
    "table": ParagraphStyle(
        "table", fontName="JPMincho", fontSize=6.0, leading=7.2,
        textColor=INK, alignment=TA_CENTER, wordWrap="CJK",
    ),
    "table_left": ParagraphStyle(
        "table_left", fontName="JPMincho", fontSize=6.0, leading=7.2,
        textColor=INK, alignment=TA_LEFT, wordWrap="CJK",
    ),
    "note": ParagraphStyle(
        "note", fontName="JPGothic", fontSize=6.2, leading=8.1,
        textColor=MUTED, alignment=TA_LEFT, wordWrap="CJK",
    ),
}


def p(text, style="body"):
    return Paragraph(text, STYLES[style])


def draw_paragraph(c, text, x, y_top, width, style="body"):
    para = p(text, style)
    _, h = para.wrap(width, PAGE_H)
    para.drawOn(c, x, y_top - h)
    return y_top - h


def draw_heading(c, number, title, x, y_top, width=COL_W):
    c.setFont("JPGothicB", 9.4)
    c.setFillColor(INK)
    c.drawString(x, y_top - 10, f"{number}  {title}")
    c.setStrokeColor(BLUE_DARK)
    c.setLineWidth(0.8)
    c.line(x, y_top - 13.2, x + width, y_top - 13.2)
    return y_top - 18


def draw_subheading(c, number, title, x, y_top):
    c.setFont("JPGothicB", 7.8)
    c.setFillColor(INK)
    c.drawString(x, y_top - 8.5, f"{number}  {title}")
    return y_top - 12


def draw_header(c, page_no):
    c.setFillColor(MUTED)
    c.setFont("JPGothic", 6.7)
    c.drawString(MARGIN_X, PAGE_H - 24, "2026年度 修士論文中間発表資料")
    c.drawRightString(PAGE_W - MARGIN_X, PAGE_H - 24, "2530030  GUO ZHENGPENG")
    c.setStrokeColor(RULE)
    c.setLineWidth(0.45)
    c.line(MARGIN_X, PAGE_H - 29, PAGE_W - MARGIN_X, PAGE_H - 29)
    c.setFont("JPGothic", 6.8)
    c.drawCentredString(PAGE_W / 2, 19, str(page_no))


def draw_title(c):
    c.setFillColor(INK)
    c.setFont("JPGothicB", 16.0)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 53, "相対3次元動作制御と局所構造重み付き拡散学習による")
    c.drawCentredString(PAGE_W / 2, PAGE_H - 75, "柔軟食品編集 - FoodStateEdit -")
    c.setFont("JPGothic", 8.2)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 92, "発表者: 2530030  GUO ZHENGPENG")
    c.setStrokeColor(INK)
    c.setLineWidth(0.7)
    c.line(MARGIN_X, PAGE_H - 101, PAGE_W - MARGIN_X, PAGE_H - 101)


def rounded_box(c, x, y, w, h, fill, stroke, title, subtitle=None, title_size=7.2):
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(0.75)
    c.roundRect(x, y, w, h, 4, fill=1, stroke=1)
    c.setFillColor(INK)
    c.setFont("JPGothicB", title_size)
    c.drawCentredString(x + w / 2, y + h / 2 + (2.1 if subtitle else -2.3), title)
    if subtitle:
        c.setFont("JPGothic", 5.4)
        c.setFillColor(MUTED)
        c.drawCentredString(x + w / 2, y + h / 2 - 7.0, subtitle)


def arrow(c, x1, y1, x2, y2, color=RULE, width=0.8):
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(width)
    c.line(x1, y1, x2, y2)
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 4.0
    for delta in (2.55, -2.55):
        c.line(x2, y2, x2 + size * math.cos(angle + delta), y2 + size * math.sin(angle + delta))


def draw_framework(c, x, y_top, w):
    h = 348
    y0 = y_top - h
    c.setFillColor(WHITE)
    c.setStrokeColor(RULE)
    c.setLineWidth(0.6)
    c.roundRect(x, y0, w, h, 5, fill=1, stroke=1)
    c.setFillColor(INK)
    c.setFont("JPGothicB", 8.4)
    c.drawString(x + 8, y_top - 15, "FoodStateEdit フレームワーク")

    bw = w - 28
    bx = x + 14
    rounded_box(c, bx, y_top - 51, bw, 25, LIGHT, RULE,
                "入力 I, 動作 a, 道具 u, 編集領域 S", "画像・行動・支持領域")
    arrow(c, x + w / 2, y_top - 51, x + w / 2, y_top - 64)
    rounded_box(c, bx, y_top - 101, bw, 36, BLUE, BLUE_DARK,
                "相対3次元動作状態", "rigid utensil + flexible strand + contact/source")

    branch_top = y_top - 120
    mid = x + w / 2
    c.setFont("JPGothicB", 6.5)
    c.setFillColor(BLUE_DARK)
    c.drawCentredString(x + w * 0.28, branch_top, "推論経路")
    c.setFillColor(ORANGE_DARK)
    c.drawCentredString(x + w * 0.72, branch_top, "学習経路")
    arrow(c, mid, y_top - 101, x + w * 0.28, branch_top - 12, BLUE_DARK)
    arrow(c, mid, y_top - 101, x + w * 0.72, branch_top - 12, ORANGE_DARK)

    branch_w = (w - 36) / 2
    lx = x + 12
    rx = lx + branch_w + 12
    rounded_box(c, lx, y_top - 166, branch_w, 31, BLUE, BLUE_DARK,
                "深度対応投影", "21-frame RGB control", 6.7)
    rounded_box(c, rx, y_top - 166, branch_w, 31, ORANGE, ORANGE_DARK,
                "局所構造マスク", "strand / pinch / source", 6.7)
    arrow(c, lx + branch_w / 2, y_top - 166, lx + branch_w / 2, y_top - 179, BLUE_DARK)
    arrow(c, rx + branch_w / 2, y_top - 166, rx + branch_w / 2, y_top - 179, ORANGE_DARK)
    rounded_box(c, lx, y_top - 218, branch_w, 38, LIGHT, BLUE_DARK,
                "VACE 条件系列", "seed 1 / 21 frames / scale 1", 6.7)
    rounded_box(c, rx, y_top - 218, branch_w, 38, ORANGE, ORANGE_DARK,
                "正規化局所重み W", "weighted FlowMatch", 6.7)
    arrow(c, rx + branch_w / 2, y_top - 218, rx + branch_w / 2, y_top - 231, ORANGE_DARK)
    rounded_box(c, rx, y_top - 264, branch_w, 32, RED_LIGHT, RED,
                "rank-8 VACE LoRA", "学習対象", 6.7)

    merge_y = y_top - 285
    arrow(c, lx + branch_w / 2, y_top - 218, mid - 21, merge_y, BLUE_DARK)
    arrow(c, rx + branch_w / 2, y_top - 264, mid + 21, merge_y, RED)
    rounded_box(c, bx, y_top - 316, bw, 31, GREEN, GREEN_DARK,
                "VACE 拡散推論 + 厳密な領域外合成", "outside-support change = 0")
    arrow(c, mid, y_top - 316, mid, y_top - 329, GREEN_DARK)
    c.setFillColor(GREEN_DARK)
    c.setFont("JPGothicB", 7.0)
    c.drawCentredString(mid, y_top - 341, "出力: 道具接触と麺変形を伴う編集画像")
    return y0


def draw_table(c, data, x, y_top, widths, row_heights=None, header_rows=1,
               highlight_rows=None, font_size=6.0, aligns=None):
    cells = []
    for r, row in enumerate(data):
        current = []
        for col, value in enumerate(row):
            style = "table_left" if aligns and aligns[col] == "left" else "table"
            current.append(Paragraph(str(value), ParagraphStyle(
                f"tbl_{r}_{col}", parent=STYLES[style], fontSize=font_size,
                leading=font_size + 1.2,
            )))
        cells.append(current)
    table = Table(cells, colWidths=widths, rowHeights=row_heights, repeatRows=header_rows)
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.35, RULE),
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), BLUE),
        ("FONTNAME", (0, 0), (-1, header_rows - 1), "JPGothicB"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
        ("TOPPADDING", (0, 0), (-1, -1), 2.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
    ]
    for row in highlight_rows or []:
        commands.append(("BACKGROUND", (0, row), (-1, row), GREEN))
    table.setStyle(TableStyle(commands))
    _, h = table.wrap(sum(widths), PAGE_H)
    table.drawOn(c, x, y_top - h)
    return y_top - h


def draw_formula_box(c, x, y_top, w):
    h = 69
    c.setFillColor(LIGHT)
    c.setStrokeColor(RULE)
    c.roundRect(x, y_top - h, w, h, 4, fill=1, stroke=1)
    c.setFillColor(INK)
    c.setFont("Helvetica", 7.4)
    c.drawString(x + 9, y_top - 18, "W_raw = 1 + 3 M_strand + 7 M_pinch + 5 M_source")
    c.drawString(x + 9, y_top - 35, "W = W_raw / mean(W_raw)")
    c.drawString(x + 9, y_top - 52, "L = w(t) sum_i W_i ||v_theta,i - v_target,i||^2 / sum_i W_i")
    c.setFont("JPGothic", 5.5)
    c.setFillColor(MUTED)
    c.drawRightString(x + w - 7, y_top - 64, "平均重みを1に保ち、学習率との交絡を抑制")
    return y_top - h


def draw_bar_chart(c, x, y_top, w):
    h = 104
    c.setFillColor(WHITE)
    c.setStrokeColor(RULE)
    c.roundRect(x, y_top - h, w, h, 4, fill=1, stroke=1)
    c.setFont("JPGothicB", 7.0)
    c.setFillColor(INK)
    c.drawString(x + 8, y_top - 13, "提案法の改善率 (対 relative3D-uniform)")
    chart_x = x + 54
    chart_y = y_top - 84
    chart_w = w - 70
    scale = chart_w / 6.0
    c.setStrokeColor(RULE)
    c.setLineWidth(0.4)
    c.line(chart_x, chart_y, chart_x + chart_w, chart_y)
    for tick in range(0, 7):
        tx = chart_x + tick * scale
        c.line(tx, chart_y - 2, tx, chart_y + 2)
        c.setFont("Helvetica", 5.2)
        c.setFillColor(MUTED)
        c.drawCentredString(tx, chart_y - 9, str(tick))
    gate_x = chart_x + 5 * scale
    c.setStrokeColor(RED)
    c.setDash(2, 2)
    c.line(gate_x, chart_y + 2, gate_x, chart_y + 52)
    c.setDash()
    c.setFont("JPGothic", 5.2)
    c.setFillColor(RED)
    c.drawCentredString(gate_x, chart_y + 56, "5% gate")
    bars = [("topology", 2.22296, chart_y + 34), ("pinch", 2.83497, chart_y + 12)]
    for label, value, by in bars:
        c.setFont("Helvetica", 6.0)
        c.setFillColor(INK)
        c.drawRightString(chart_x - 5, by + 3, label)
        c.setFillColor(BLUE_DARK if label == "topology" else ORANGE_DARK)
        c.rect(chart_x, by, value * scale, 9, fill=1, stroke=0)
        c.setFont("Helvetica-Bold", 6.1)
        c.drawString(chart_x + value * scale + 4, by + 2, f"{value:.2f}%")
    return y_top - h


def draw_image_fit(c, path, x, y_top, w, h):
    img = ImageReader(str(path))
    iw, ih = img.getSize()
    scale = min(w / iw, h / ih)
    dw, dh = iw * scale, ih * scale
    dx = x + (w - dw) / 2
    dy = y_top - h + (h - dh) / 2
    c.setFillColor(WHITE)
    c.setStrokeColor(RULE)
    c.rect(x, y_top - h, w, h, fill=1, stroke=1)
    c.drawImage(img, dx, dy, dw, dh, preserveAspectRatio=True, mask="auto")
    return y_top - h


def first_page(c):
    draw_header(c, 1)
    draw_title(c)
    y = PAGE_H - 113

    y = draw_heading(c, "1", "はじめに", LEFT_X, y)
    y = draw_paragraph(c,
        "調理画像編集では、食材の追加・除去だけでなく、箸やスプーンの動作に応じた麺の持ち上げ、接触、変形を生成する必要がある。"
        "しかし麺は細長く柔軟であり、2次元制御だけでは奥行き関係、連結性、把持点を同時に保つことが難しい。", LEFT_X, y, COL_W)
    y -= 3
    y = draw_paragraph(c,
        "本研究 FoodStateEdit は、相対3次元動作状態を共通表現として、推論時の深度対応制御と学習時の局所構造重みを一体化する。"
        "目的は、編集領域外を厳密に保存しながら、柔軟食品と道具の相互作用を拡散モデルで補完することである。", LEFT_X, y, COL_W)

    y -= 5
    y = draw_heading(c, "2", "関連研究と課題", LEFT_X, y)
    y = draw_paragraph(c,
        "ControlNet [1] と VACE [2] は空間・時系列条件による生成制御を可能にし、LoRA [3] は少数パラメータでモデルを適応できる。"
        "一方、一般的な画像・動画編集は、柔軟物体の連結性や接触点に対する局所的な学習配分を明示しない。", LEFT_X, y, COL_W)
    y -= 2
    y = draw_paragraph(c,
        "先行実験では、相対3次元投影により道具軌道の幾何整合性は改善したが、麺の消失や接触の曖昧さが残った。"
        "そこで、幾何制御だけでなく、麺・把持点・器内の供給元に学習を集中させる必要がある。", LEFT_X, y, COL_W)

    y -= 5
    y = draw_heading(c, "3", "提案手法", LEFT_X, y)
    y = draw_subheading(c, "3.1", "相対3次元動作状態", LEFT_X, y)
    y = draw_paragraph(c,
        "入力画像 I、動作 a、道具 u、編集領域 S から、道具を剛体、麺を可変形ストランドとして表現する。"
        "カメラに対する相対深度、道具先端、麺の端点、接触点、器内の供給元を同一座標系で管理し、21フレームのRGB制御列へ投影する。", LEFT_X, y, COL_W)

    y -= 3
    y = draw_subheading(c, "3.2", "局所構造重み付き FlowMatch", LEFT_X, y)
    y = draw_paragraph(c,
        "3次元状態から strand、pinch、source の3マスクを生成し、画素平均が1となるよう正規化した重み W を FlowMatch [4] の速度予測誤差へ適用する。", LEFT_X, y, COL_W)
    y -= 2
    y = draw_formula_box(c, LEFT_X, y, COL_W)
    y -= 5
    y = draw_paragraph(c,
        "学習対象は VACE transformer の rank-8 LoRA である。推論後は領域 S の外側を原画像と厳密に合成するため、領域外変化は理論上および実測上0となる。", LEFT_X, y, COL_W)
    y -= 5
    y = draw_subheading(c, "3.3", "本研究の核心", LEFT_X, y)
    core = [
        ["観点", "提案内容"],
        ["表現", "剛体道具と柔軟ストランドを同一の相対3D状態で記述"],
        ["結合", "同じ3D状態から推論制御と学習重みを同時に導出"],
        ["保全", "編集領域外を画素単位で厳密に保存"],
    ]
    y = draw_table(c, core, LEFT_X, y, [43, COL_W - 43], highlight_rows=[1, 2, 3],
                   font_size=5.65, aligns=["center", "left"])

    fy = PAGE_H - 113
    fy = draw_framework(c, RIGHT_X, fy, COL_W)
    fy -= 3
    fy = draw_paragraph(c, "図1  提案フレームワーク。青は推論制御、橙は局所重み、赤は学習対象、緑は出力保全を示す。",
                        RIGHT_X, fy, COL_W, "caption")
    fy -= 7
    fy = draw_subheading(c, "3.4", "事前登録した3群比較", RIGHT_X, fy)
    fy = draw_paragraph(c,
        "幾何表現と学習重みの効果を分離するため、同一初期値・同一乱数・32更新で次の3群を比較した。", RIGHT_X, fy, COL_W)
    data = [
        ["群", "制御", "学習重み", "目的"],
        ["A", "planar", "uniform", "2D基準"],
        ["B", "relative3D", "uniform", "3D効果"],
        ["C", "relative3D", "topology", "提案法"],
    ]
    fy -= 2
    fy = draw_table(c, data, RIGHT_X, fy, [22, 61, 68, COL_W - 151],
                    highlight_rows=[3], font_size=5.8)
    fy -= 6
    c.setFillColor(RED_LIGHT)
    c.setStrokeColor(RED)
    c.roundRect(RIGHT_X, fy - 42, COL_W, 42, 4, fill=1, stroke=1)
    draw_paragraph(c,
        "<b>判定条件:</b> C が B に対し topology と pinch の双方を5%以上改善し、2名の独立評価で明瞭な意味的改善を確認する。", RIGHT_X + 7, fy - 6, COL_W - 14, "note")


def second_page(c):
    draw_header(c, 2)
    top = PAGE_H - 42

    y = draw_heading(c, "4", "実験", LEFT_X, top)
    y = draw_subheading(c, "4.1", "設定と監査", LEFT_X, y)
    y = draw_paragraph(c,
        "RTX A6000 上で、seed 1、21フレーム、推論20 steps、VACE scale 1、TTM off を固定した。"
        "3群は同一初期値ハッシュと同一乱数列を共有し、全6チェックポイントで160個の有限LoRAテンソルと80層の公式ローダを検証した。", LEFT_X, y, COL_W)
    y -= 3
    y = draw_subheading(c, "4.2", "定量結果", LEFT_X, y)
    data = [
        ["条件", "support", "topology", "pinch", "connect."],
        ["planar-uniform", "12.971", "21.745", "29.187", "14.626"],
        ["relative3D-off", "15.272", "33.004", "38.968", "24.430"],
        ["relative3D-uniform", "15.077", "31.063", "38.053", "22.146"],
        ["proposed", "15.008", "30.373", "36.974", "21.931"],
    ]
    y -= 1
    y = draw_table(c, data, LEFT_X, y, [71, 43, 47, 42, COL_W - 203],
                   highlight_rows=[4], font_size=5.35, aligns=["left", "center", "center", "center", "center"])
    y -= 3
    y = draw_paragraph(c,
        "表1  低いほど良い局所誤差。proposed は relative3D-uniform より全4指標で低いが、主要指標の改善は小さい。", LEFT_X, y, COL_W, "caption")
    y -= 7
    y = draw_bar_chart(c, LEFT_X, y, COL_W)
    y -= 8
    y = draw_paragraph(c,
        "topology は2.22%、pinch は2.83%の改善に留まり、事前登録した5%閾値を満たさなかった。"
        "また relative3D-uniform は planar-uniform より global support が16.23%悪化した。"
        "したがって、現時点で相対3次元制御または局所構造重みの有効性を確立したとは言えない。", LEFT_X, y, COL_W)
    y -= 6
    y = draw_subheading(c, "4.3", "実験の再現性", LEFT_X, y)
    audit = [
        ["監査項目", "結果"],
        ["初期値 / 乱数", "3群で一致"],
        ["pipeline load", "各試行 1回"],
        ["条件数 / フレーム", "5 / 21"],
        ["領域外変化", "0"],
    ]
    y = draw_table(c, audit, LEFT_X, y, [112, COL_W - 112], highlight_rows=[1, 2, 3, 4], font_size=5.8)
    y -= 8
    y = draw_subheading(c, "4.4", "対照実験の解釈", LEFT_X, y)
    interpretation = [
        ["仮説", "観測", "判定"],
        ["3D制御", "planarよりsupport悪化", "未確認"],
        ["局所重み", "主要誤差2.22/2.83%改善", "閾値未達"],
        ["領域外保全", "全試行で変化0", "確認"],
    ]
    y = draw_table(c, interpretation, LEFT_X, y, [54, 113, COL_W - 167],
                   highlight_rows=[3], font_size=5.45)
    y -= 6
    c.setFillColor(BLUE)
    c.setStrokeColor(BLUE_DARK)
    c.roundRect(LEFT_X, y - 93, COL_W, 93, 4, fill=1, stroke=1)
    draw_paragraph(c,
        "<b>現時点で成立する正向貢献</b><br/>"
        "1) 柔軟食品編集を、相対3D幾何・接触・局所学習の統一問題として定式化した。<br/>"
        "2) 同一乱数・同一初期値の3群比較と失敗判定を含む監査可能な評価系を構築した。<br/>"
        "3) 局所重みは全指標で方向的改善を示し、次の意味的損失設計に検証可能な根拠を与えた。",
        LEFT_X + 8, y - 7, COL_W - 16, "note")

    ry = draw_heading(c, "", "定性比較", RIGHT_X, top)
    img1 = RESULTS / "day13_flexible_completion_20260906" / "relative3d_uniform_step32_review.png"
    img2 = RESULTS / "day13_flexible_completion_20260906" / "relative3d_topology_weighted_step32_review.png"
    ry = draw_image_fit(c, img1, RIGHT_X, ry, COL_W, 121)
    ry -= 2
    ry = draw_paragraph(c, "(a) relative3D-uniform, step32", RIGHT_X, ry, COL_W, "caption")
    ry -= 4
    ry = draw_image_fit(c, img2, RIGHT_X, ry, COL_W, 121)
    ry -= 2
    ry = draw_paragraph(c, "(b) proposed: relative3D-topology-weighted, step32", RIGHT_X, ry, COL_W, "caption")
    ry -= 4
    ry = draw_paragraph(c,
        "図2  同一seedによる21フレーム比較。提案法では箸付近の麺形状に小さな変化が見られるが、把持、連続した持ち上げ、最終保持の明瞭な改善は確認できない。", RIGHT_X, ry, COL_W, "caption")

    ry -= 8
    ry = draw_heading(c, "5", "考察", RIGHT_X, ry)
    ry = draw_paragraph(c,
        "局所構造重みは topology、pinch、connection を一貫して低下させたため、学習信号の配分方向は妥当と考えられる。"
        "ただし均一学習との差は小さく、画素速度誤差だけでは、端点の連続性や道具との接触を十分に表現できない可能性が高い。", RIGHT_X, ry, COL_W)
    ry -= 2
    ry = draw_paragraph(c,
        "また relative3D 制御の global support 悪化は、深度投影の情報が生成器にそのまま有効な意味制約とはならないことを示す。"
        "次段階では接触・端点・最終保持を直接監督し、学習長を増やした新しい事前登録比較を行う。", RIGHT_X, ry, COL_W)

    ry -= 6
    ry = draw_heading(c, "6", "結論", RIGHT_X, ry)
    ry = draw_paragraph(c,
        "相対3次元状態から制御列と局所重みを導く FoodStateEdit を構築し、完全に対応した比較実験を実施した。"
        "提案重みは方向的な数値改善を示したが主要判定は未達であり、有効性は未確立である。"
        "本結果は seen synthetic sample の能力診断であり、一般化、実画像性能、物理的正しさ、写真品質を主張しない。", RIGHT_X, ry, COL_W)

    ry -= 6
    ry = draw_heading(c, "参考文献", "", RIGHT_X, ry)
    refs = [
        "[1] Zhang et al., Adding Conditional Control to Text-to-Image Diffusion Models, ICCV, 2023.",
        "[2] Jiang et al., VACE: All-in-One Video Creation and Editing, ICCV, 2025.",
        "[3] Hu et al., LoRA: Low-Rank Adaptation of Large Language Models, 2021.",
        "[4] Lipman et al., Flow Matching for Generative Modeling, 2022.",
        "[5] DiffSynth-Studio, commit 899dca2, reproducibility anchor.",
    ]
    for ref in refs:
        ry = draw_paragraph(c, ref, RIGHT_X, ry, COL_W, "small") - 1


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=A4, pageCompression=1)
    c.setTitle("FoodStateEdit Pre-Defense Handout")
    c.setAuthor("GUO ZHENGPENG 2530030")
    c.setSubject("Relative 3D action control and topology-weighted diffusion for flexible food editing")
    first_page(c)
    c.showPage()
    second_page(c)
    c.showPage()
    c.save()
    print(OUTPUT)


if __name__ == "__main__":
    build()
