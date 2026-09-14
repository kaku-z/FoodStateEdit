from pathlib import Path
import argparse
import html

W, H = 1280, 720
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "svg_output"

BG = "#FFFFFF"
PALE = "#EAF6FC"
BLUE = "#0070C0"
RED = "#E31B23"
CYAN = "#5BC0EB"
TEXT = "#222222"
MUTED = "#666666"
RULE = "#D8DDE3"
GREEN = "#2E8B57"
FONT = "Yu Gothic, Arial, Meiryo, sans-serif"
DATA = "Arial, Yu Gothic, sans-serif"


def esc(s):
    return html.escape(str(s), quote=True)


def tx(x, y, text, size=26, fill=TEXT, weight=400, anchor="start", family=FONT,
       letter_spacing=None, opacity=None):
    attrs = [f'x="{x}"', f'y="{y}"', f'font-family="{family}"',
             f'font-size="{size}"', f'fill="{fill}"', f'font-weight="{weight}"',
             f'text-anchor="{anchor}"']
    if letter_spacing is not None:
        attrs.append(f'letter-spacing="{letter_spacing}"')
    if opacity is not None:
        attrs.append(f'opacity="{opacity}"')
    return f'<text {" ".join(attrs)}>{esc(text)}</text>'


def multiline(x, y, lines, size=26, fill=TEXT, weight=400, leading=1.35,
              anchor="start", family=FONT):
    return "\n".join(
        tx(x, y + i * size * leading, line, size, fill, weight, anchor, family)
        for i, line in enumerate(lines)
    )


def rect(x, y, w, h, fill="none", stroke="none", sw=1, rx=0, opacity=None):
    attrs = [f'x="{x}"', f'y="{y}"', f'width="{w}"', f'height="{h}"',
             f'fill="{fill}"', f'stroke="{stroke}"', f'stroke-width="{sw}"']
    if rx:
        attrs.append(f'rx="{rx}"')
    if opacity is not None:
        attrs.append(f'opacity="{opacity}"')
    return f'<rect {" ".join(attrs)}/>'


def line(x1, y1, x2, y2, stroke=RULE, sw=2, dash=None):
    attrs = [f'x1="{x1}"', f'y1="{y1}"', f'x2="{x2}"', f'y2="{y2}"',
             f'stroke="{stroke}"', f'stroke-width="{sw}"']
    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')
    return f'<line {" ".join(attrs)}/>'


def circle(cx, cy, r, fill="none", stroke="none", sw=1):
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def polygon(points, fill=BLUE, stroke="none", sw=1):
    pts = " ".join(f"{x},{y}" for x, y in points)
    return f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def img(path, x, y, w, h, mode="meet"):
    return (f'<image href="../images/{esc(path)}" x="{x}" y="{y}" width="{w}" height="{h}" '
            f'preserveAspectRatio="xMidYMid {mode}"/>')


def group(gid, body, bounds="72 160 1136 500"):
    return f'<g id="{gid}" data-pptx-bounds="{bounds}">\n{body}\n</g>'


def header(title, num, section=None, pale=False):
    parts = []
    if pale:
        parts.append(rect(0, 0, W, H, PALE))
    else:
        parts.append(rect(0, 0, W, H, BG))
    parts.append(line(72, 42, 1208, 42, RULE, 2))
    if section:
        parts.append(tx(72, 76, section.upper(), 16, BLUE, 700, letter_spacing=1.8))
    parts.append(tx(72, 123, title, 42, TEXT, 700))
    parts.append(rect(72, 142, 74, 4, BLUE))
    parts.append(tx(72, 686, "FoodStateEdit｜メディア情報学", 14, MUTED, 400))
    parts.append(tx(1208, 686, f"{num:02d}", 16, BLUE, 700, "end", DATA))
    return "\n".join(parts)


def arrow(x1, y, x2, color=BLUE, sw=4):
    return "\n".join([
        line(x1, y, x2 - 12, y, color, sw),
        polygon([(x2 - 12, y - 8), (x2, y), (x2 - 12, y + 8)], color)
    ])


def frame_image(path, x, y, w, h, label, label_fill=BLUE, mode="meet"):
    return "\n".join([
        rect(x, y, w, h, "#F7F9FB", RULE, 2, 6),
        img(path, x + 6, y + 6, w - 12, h - 12, mode),
        rect(x, y, w, 34, label_fill, "none", 0, 6),
        tx(x + 14, y + 24, label, 17, "#FFFFFF", 700),
    ])


def slide01():
    body = [rect(0, 0, W, H, BG), rect(0, 0, 28, H, BLUE),
            line(90, 84, 1192, 84, RULE, 2),
            tx(90, 137, "修士研究 予備発表", 18, BLUE, 700, letter_spacing=2.0),
            multiline(90, 242, ["3次元動作制御に基づく", "食物画像編集"], 54, TEXT, 700, 1.28),
            tx(92, 398, "Food Image Editing Based on 3D Motion Control", 24, MUTED, 400, family=DATA),
            rect(90, 440, 178, 6, RED),
            tx(90, 495, "FoodStateEdit", 32, BLUE, 700, family=DATA),
            multiline(846, 522, ["メディア情報学", "GUO ZHENGPENG　2530030",
                                 "主指導：柳井 啓司　　副指導：高橋 裕樹"], 20, TEXT, 400, 1.55),
            tx(1192, 676, "2026.09.13", 16, MUTED, 400, "end", DATA)]
    return svg("cover", group("cover-main", "\n".join(body), "0 0 1280 720"))


def slide02():
    parts = [header("研究目的：食べる瞬間を感じる食物画像", 2, "PURPOSE")]
    left = [tx(76, 196, "研究目的", 23, BLUE, 700),
            multiline(76, 242, ["普通の食物写真に、", "「食べる直前」の臨場感を加える"], 31, TEXT, 700, 1.35),
            multiline(76, 340, ["道具で食物を動かす状態を生成し、", "見る人が味・食感・動きを想像できる", "視覚表現を目指す"], 20, TEXT, 500, 1.6),
            line(76, 458, 522, 458, RULE, 2),
            tx(76, 500, "利用場面", 23, BLUE, 700),
            multiline(92, 540, ["食品メニュー・広告", "レシピ・食育・デジタル創作"], 20, MUTED, 600, 1.6)]
    right = [frame_image("p02_cake_input.png", 594, 192, 276, 278, "普通の食物写真", BLUE),
             arrow(879, 331, 924, RED, 4),
             frame_image("p02_cake_fork_scale0p6.png", 936, 192, 276, 278, "操作状態を生成", RED),
             rect(594, 500, 618, 78, PALE, "none", 0, 6),
             tx(903, 534, "静止した料理写真から、", 25, TEXT, 700, "middle"),
             tx(903, 565, "食べる場面を想像できる画像へ", 25, BLUE, 700, "middle"),
             tx(903, 603, "※ 視覚的な臨場感を対象とし、味覚刺激そのものは扱わない", 14, MUTED, 400, "middle")]
    parts.append(group("slide-content", "\n".join(left + right)))
    return svg("content", "\n".join(parts))


def slide03():
    parts = [header("既存画像編集の問題：不要な領域まで変化する", 3, "PROBLEM")]
    body = [
        frame_image("p09_ramen_input.png", 72, 182, 500, 318, "入力画像", BLUE),
        frame_image("p09_ramen_qwen_seed1.png", 708, 182, 500, 318, "一般的な生成編集例", RED),
        arrow(588, 340, 690, RED, 5),
        tx(322, 536, "編集したい部分", 20, BLUE, 700, "middle"),
        tx(322, 568, "箸と持ち上げる麺", 24, TEXT, 700, "middle"),
        tx(958, 536, "実際に変化した部分", 20, RED, 700, "middle"),
        tx(958, 568, "画角・背景・器・麺全体", 24, TEXT, 700, "middle"),
        rect(104, 592, 1072, 52, "#F1FAF4", GREEN, 2, 5),
        tx(640, 626, "本研究：必要な操作領域だけを変え、その他は原画像のまま保持する", 25, GREEN, 700, "middle"),
        tx(72, 661, "※ 同一入力・seed 1 の例。右画像は写真らしいが、入力との局所対応が失われている。", 14, MUTED, 500),
    ]
    parts.append(group("unwanted-global-edit", "\n".join(body), "60 165 1160 515"))
    return svg("content", "\n".join(parts))


def slide04():
    parts = [header("課題：最終画像だけでは操作の正しさを保証できない", 4, "PROBLEM")]
    labels = [("1", "道具", "適切な種類・姿勢"), ("2", "接触", "食物との接点"),
              ("3", "持ち上げ", "食物の追従・移動"), ("4", "保持", "背景と容器の保存")]
    xs = [86, 370, 654, 938]
    items = []
    for i, (n, t, d) in enumerate(labels):
        x = xs[i]
        items += [circle(x + 98, 272, 42, PALE, BLUE, 3), tx(x + 98, 283, n, 28, BLUE, 700, "middle", DATA),
                  tx(x + 98, 347, t, 28, TEXT, 700, "middle"), tx(x + 98, 382, d, 19, MUTED, 400, "middle")]
        if i < 3:
            items.append(arrow(x + 150, 272, xs[i + 1] + 44, RULE, 3))
    eq = [rect(104, 452, 1072, 110, PALE, BLUE, 2, 8),
          tx(640, 494, "Strict End-to-End Success", 31, BLUE, 700, "middle", DATA),
          tx(640, 534, "= Action × Photo × Preservation", 28, TEXT, 600, "middle", DATA)]
    goal = [rect(196, 596, 888, 46, "#FFFFFF", RED, 2, 4),
            tx(640, 627, "目標：接近 → 接触 → 持ち上げ → 停留を明示的に制御する", 25, RED, 700, "middle")]
    parts.append(group("slide-content", "\n".join(items + eq + goal)))
    return svg("content", "\n".join(parts))


def slide05():
    parts = [header("関連研究と位置付け", 5, "RELATED WORK")]
    cols = [
        ("Qwen Image", "静止画編集", "写真品質に強い", BLUE),
        ("VACE", "動画拡散", "時間連続生成", CYAN),
        ("GeoEdit 系", "幾何編集", "位置・形状制御", "#5D83A6"),
        ("SAM", "領域抽出", "食物・容器の分離", "#69A87A"),
    ]
    cards = []
    for i, (name, field, strength, color) in enumerate(cols):
        x = 72 + i * 286
        cards += [rect(x, 190, 258, 176, "#FFFFFF", RULE, 2, 6), rect(x, 190, 258, 8, color),
                  tx(x + 20, 235, name, 26, color, 700, family=DATA),
                  tx(x + 20, 278, field, 20, TEXT, 700), tx(x + 20, 320, strength, 20, MUTED, 400)]
    bridge = [arrow(214, 397, 640, RULE, 3), arrow(498, 397, 640, RULE, 3),
              arrow(782, 397, 640, RULE, 3), arrow(1066, 397, 640, RULE, 3),
              rect(270, 432, 740, 154, PALE, BLUE, 3, 8),
              tx(640, 478, "FoodStateEdit（本研究）", 31, BLUE, 700, "middle", DATA),
              tx(640, 522, "相対3D代理 ＋ 段階動作 ＋ 保持評価", 30, TEXT, 700, "middle"),
              tx(640, 558, "既存基盤モデルに柔軟物の操作制御と検証を追加", 20, MUTED, 400, "middle")]
    parts.append(group("slide-content", "\n".join(cards + bridge)))
    return svg("content", "\n".join(parts))


def slide06():
    parts = [header("提案手法：相対3D制御と凍結VACE", 6, "METHOD")]
    body = [
        rect(142, 170, 996, 502, "#FFFFFF", RULE, 1, 4),
        img("p05_framework_report_exact.png", 152, 178, 976, 486, "meet"),
    ]
    parts.append(group("method-framework-from-report", "\n".join(body), "142 170 996 502"))
    return svg("content", "\n".join(parts))


def slide07():
    parts = [header("相対3D代理とVACEの接続", 7, "CONTROL MECHANISM")]
    body = [
        rect(72, 164, 1136, 64, PALE, BLUE, 2, 5),
        tx(640, 190, "相対3D代理は生成モデルではなく、VACEの前段に置く制御信号生成部", 25, BLUE, 700, "middle"),
        tx(640, 217, "FoodStateEdit = 相対3D制御器 ＋ 凍結VACE ＋ 局所合成", 20, TEXT, 700, "middle", DATA),
    ]
    stages = [
        (72, "01", "入力と編集領域", ["入力画像", "edit_alpha", "代理軌跡から生成"]),
        (306, "02", "相対3D代理", ["接触点 Pcontact", "深度順序 z", "接近・接触・持上げ"]),
        (540, "03", "VACE条件へ変換", ["21-frame control", "透視投影", "支持領域内だけ描画"]),
        (774, "04", "凍結VACE", ["Wan2.2-VACE-Fun-A14B", "重みは変更しない", "条件付き拡散生成"]),
        (1008, "05", "出力", ["21フレーム動画", "安定フレーム選択", "局所投影・合成"]),
    ]
    for i, (x, num, title, lines) in enumerate(stages):
        fill = "#F1FAF4" if i in (1, 2) else "#FFFFFF"
        stroke = GREEN if i in (1, 2) else RULE
        title_color = GREEN if i in (1, 2) else BLUE
        body += [
            rect(x, 258, 200, 210, fill, stroke, 2, 5),
            tx(x + 18, 291, num, 17, title_color, 700, family=DATA),
            tx(x + 100, 331, title, 24, title_color, 700, "middle"),
            line(x + 20, 348, x + 180, 348, stroke, 2),
        ]
        for j, label in enumerate(lines):
            body.append(tx(x + 100, 382 + j * 34, label, 17, TEXT, 500, "middle", DATA if j < 2 else FONT))
        if i < len(stages) - 1:
            body += [line(x + 202, 363, x + 230, 363, BLUE, 3),
                     polygon([(x + 230, 356), (x + 238, 363), (x + 230, 370)], BLUE)]
    body += [
        tx(72, 515, "VACEへ渡す三つの信号", 20, BLUE, 700),
        line(72, 530, 1208, 530, RULE, 2),
        tx(92, 567, "vace_video", 18, GREEN, 700, family=DATA),
        tx(380, 567, "道具と食物を、どの位置で、どの時刻に動かすか", 20, TEXT, 700),
        tx(92, 601, "vace_video_mask", 18, GREEN, 700, family=DATA),
        tx(380, 601, "21フレームの変化領域を和集合し、膨張・羽化して生成", 20, TEXT, 700),
        tx(92, 635, "vace_reference_image + prompt", 18, GREEN, 700, family=DATA),
        tx(380, 635, "元画像の内容と、写真的な道具・食物外観をVACEの事前分布から生成", 20, TEXT, 700),
        tx(72, 666, "※ maskはSAM出力ではなく代理軌跡から生成。相対深度は正規化した制御用scaffold。", 14, RED, 600),
    ]
    parts.append(group("relative3d-vace-control", "\n".join(body), "60 150 1160 525"))
    return svg("content", "\n".join(parts))


def slide08():
    parts = [header("同一入力での生成結果比較", 8, "QUALITATIVE RESULT")]
    columns = [
        (72, "p10_success_soup_input.png", "Input（参照画像）", "#365F7D", "元のスープ画像"),
        (460, "p09_soup_qwen_day34_seed1.png", "Qwen Image", RED, "写真らしいが画角・背景も再構成"),
        (848, "p10_success_soup_output.png", "FoodStateEdit（提案手法）", GREEN, "スプーン動作を追加し領域外を保持"),
    ]
    body = []
    for x, path, label, color, caption in columns:
        fill = "#F1FAF4" if color == GREEN else "#F7F9FB"
        body += [
            rect(x, 174, 360, 58, color, color, 1, 5),
            tx(x + 180, 212, label, 25, "#FFFFFF", 700, "middle"),
            rect(x, 244, 360, 310, fill, color, 3 if color == GREEN else 1, 5),
            img(path, x + 8, 252, 344, 294, "meet"),
            tx(x + 180, 586, caption, 18, color, 700, "middle"),
        ]
    body += [
        rect(72, 612, 1136, 44, PALE, BLUE, 1, 4),
        tx(640, 640, "同一入力・seed 1｜目標動作の実画像GTは存在しないため、Inputを参照画像として表示", 17, BLUE, 700, "middle"),
        tx(640, 668, "選択済み合成1例の内部比較。統計的優位性・実画像への一般化は未証明。", 13, MUTED, 400, "middle"),
    ]
    parts.append(group("same-input-triptych", "\n".join(body), "72 174 1136 500"))
    return svg("content", "\n".join(parts))


def slide09():
    parts = [header("複数食品での多手法比較", 9, "MULTI-CASE RESULT")]
    body = [
        img("p09comp_noodle.png", 72, 154, 1136, 130, "meet"),
        img("p09comp_soup.png", 72, 304, 1136, 130, "meet"),
        img("p09comp_cake.png", 72, 454, 1136, 130, "meet"),
        rect(72, 612, 1136, 42, PALE, BLUE, 1, 4),
        tx(640, 639, "Input｜ChordEdit｜Qwen Image｜FoodStateEdit（Ours）を同一入力で比較", 18, BLUE, 700, "middle"),
        tx(640, 672, "選択済みseed 1の内部比較。実画像GTなし。統計的優位性・一般化は未証明。", 13, MUTED, 400, "middle"),
    ]
    parts.append(group("multi-case-comparison", "\n".join(body), "72 154 1136 530"))
    return svg("content", "\n".join(parts))


def slide10():
    parts = [header("実験設定", 10, "EXPERIMENT")]
    left = [rect(72, 184, 552, 366, "#FFFFFF", RULE, 2, 6), rect(72, 184, 552, 48, BLUE),
            tx(96, 217, "同条件消融", 24, "#FFFFFF", 700),
            tx(102, 294, "4", 54, BLUE, 700, family=DATA), tx(160, 289, "food cases", 24, TEXT, 700, family=DATA),
            tx(102, 371, "3", 54, BLUE, 700, family=DATA), tx(160, 366, "seeds", 24, TEXT, 700, family=DATA),
            tx(102, 448, "5", 54, BLUE, 700, family=DATA), tx(160, 443, "conditions", 24, TEXT, 700, family=DATA),
            line(328, 260, 328, 488, RULE, 2),
            multiline(356, 285, ["native input", "2D planar proxy", "fixed relative3D", "material-adaptive relative3D", "rollback-related condition"], 20, TEXT, 500, 1.6, family=DATA)]
    right = [rect(656, 184, 552, 366, PALE, "none", 0, 6), tx(684, 223, "評価項目", 24, BLUE, 700)]
    metrics = [("Action", "道具・接触・動作"), ("Photo", "写真として自然"),
               ("Preservation", "非編集領域を保持"), ("Strict", "三条件を同時達成")]
    for i, (name, desc) in enumerate(metrics):
        y = 266 + i * 64
        right += [circle(691, y - 7, 8, BLUE), tx(716, y, name, 24, TEXT, 700, family=DATA),
                  tx(946, y, desc, 20, MUTED, 400)]
    foot = [rect(126, 585, 1028, 50, "#FFFFFF", BLUE, 2, 5),
            tx(640, 618, "4 × 3 × 5 = 60 cells　｜　21 frames　｜　20 steps　｜　同一 seed", 25, BLUE, 700, "middle", DATA)]
    parts.append(group("slide-content", "\n".join(left + right + foot)))
    return svg("content", "\n".join(parts))


def slide11():
    parts = [header("評価結果：入力保持と編集量", 11, "RESULTS")]
    metric_widths = [206, 100, 140, 140, 185, 175, 190]
    metric_headers = ["手法", "区分", "PSNR ↑", "SSIM ↑", "領域外 MAE ↓", "領域外完全保持 ↑", "編集領域内変化 MAE"]
    metric_rows = [
        ("ChordEdit", "Baseline", "26.29", "0.9438", "0.000", "100.0%", "20.99",
         "#F2F5F8", "#5D83A6", "#5D83A6", 400),
        ("Qwen Image", "Baseline", "13.90", "0.4506", "35.468", "0.17%", "51.12",
         "#FFF7F7", RED, "#B04A4A", 400),
        ("Native VACE", "Ours（条件）", "34.60", "0.9823", "0.000", "100.0%", "8.33",
         "#FFFFFF", BLUE, GREEN, 500),
        ("2D planar proxy", "Ours（条件）", "24.86", "0.9473", "0.000", "100.0%", "20.18",
         "#FFFFFF", BLUE, GREEN, 500),
        ("fixed relative3D", "Ours（条件）", "24.60", "0.9452", "0.000", "100.0%", "20.73",
         "#FFFFFF", BLUE, GREEN, 500),
        ("★ FoodStateEdit", "提案手法", "23.29", "0.9384", "0.000", "100.0%", "25.86",
         "#F1FAF4", GREEN, GREEN, 700),
    ]
    metric_y = 182
    body = [tx(72, 165, "従来画像指標（入力との類似度、各手法 n=12）｜緑の ★ 行＝提案手法", 20, BLUE, 700),
            rect(72, metric_y, sum(metric_widths), 357, "#FFFFFF", RULE, 2, 4)]
    x = 72
    for i, (label, w) in enumerate(zip(metric_headers, metric_widths)):
        body += [rect(x, metric_y, w, 45, "#365F7D", "#365F7D", 1),
                 tx(x + (16 if i in (0, 1) else w / 2), metric_y + 30, label, 16, "#FFFFFF", 700,
                    "start" if i in (0, 1) else "middle", DATA)]
        x += w
    for r, row in enumerate(metric_rows):
        values, fill, accent, tag, weight = row[:7], row[7], row[8], row[9], row[10]
        y = metric_y + 45 + r * 52
        x = 72
        for c, (value, w) in enumerate(zip(values, metric_widths)):
            body += [rect(x, y, w, 52, fill, RULE, 1)]
            if c == 0:
                body.append(tx(x + 16, y + 34, value, 17, accent, 700, "start", DATA))
            elif c == 1:
                body.append(tx(x + 10, y + 34, value, 15, tag, weight, "start", DATA))
            else:
                body.append(tx(x + w / 2, y + 34, value, 17, TEXT, weight, "middle", DATA))
            x += w
    body += [
        rect(72, 562, 1136, 62, "#F1FAF4", GREEN, 2, 5),
        tx(640, 588, "FoodStateEditは非編集領域を完全に保持しながら、", 20, GREEN, 700, "middle"),
        tx(640, 614, "比較したVACE系条件の中で操作対象領域に最大の変化を与えた。", 20, GREEN, 700, "middle"),
        tx(72, 650, "※ 変化量は動作成功率ではない。4例×3 seedsの選択済み開発データ。Action・Photo・Strictは統一採点中。", 14, RED, 600),
    ]
    parts.append(group("evaluation-table", "\n".join(body), "60 145 1160 525"))
    return svg("content", "\n".join(parts))


def slide12():
    parts = [header("消融実験：制御条件の定量比較", 12, "ABLATION")]
    widths = [350, 90, 150, 150, 220, 176]
    headers = ["条件", "n", "PSNR ↑", "SSIM ↑", "編集領域内変化 MAE", "領域外完全保持 ↑"]
    rows = [
        ("Native VACE", "12", "34.60", "0.9823", "8.33", "100.0%"),
        ("2D planar proxy", "12", "24.86", "0.9473", "20.18", "100.0%"),
        ("fixed relative3D", "12", "24.60", "0.9452", "20.73", "100.0%"),
        ("★ material-adaptive relative3D（Ours）", "12", "23.29", "0.9384", "25.86", "100.0%"),
    ]
    body = [
        tx(72, 170, "4 food cases × 3 seeds｜同一入力・同一生成条件", 20, BLUE, 700),
        rect(72, 192, 1136, 294, "#FFFFFF", RULE, 2, 4),
    ]
    x = 72
    for c, (label, w) in enumerate(zip(headers, widths)):
        body += [rect(x, 192, w, 54, "#365F7D", "#365F7D", 1),
                 tx(x + (16 if c == 0 else w / 2), 227, label, 17, "#FFFFFF", 700,
                    "start" if c == 0 else "middle", DATA)]
        x += w
    for r, row in enumerate(rows):
        y = 246 + r * 60
        fill = "#F1FAF4" if r == 3 else ("#FFFFFF" if r % 2 == 0 else "#F4F7FA")
        accent = GREEN if r == 3 else TEXT
        weight = 700 if r == 3 else 500
        x = 72
        for c, (value, w) in enumerate(zip(row, widths)):
            body += [rect(x, y, w, 60, fill, GREEN if r == 3 else RULE, 2 if r == 3 else 1)]
            body.append(tx(x + (16 if c == 0 else w / 2), y + 38, value, 18, accent,
                           700 if c == 0 or r == 3 else weight,
                           "start" if c == 0 else "middle", DATA))
            x += w
    body += [
        rect(72, 520, 548, 88, "#F1FAF4", GREEN, 2, 5),
        tx(96, 552, "確認できたこと", 20, GREEN, 700),
        tx(96, 584, "Oursは領域外を保持しつつ、編集領域の変化量が最大", 19, TEXT, 700),
        rect(660, 520, 548, 88, "#FFF5F5", RED, 2, 5),
        tx(684, 552, "まだ確認できていないこと", 20, RED, 700),
        tx(684, 584, "relative3D・材質適応による動作成功率の優位性", 19, TEXT, 700),
        tx(72, 642, "※ PSNR/SSIMは入力保持、MAEは変更量の診断。動作正解率や写真品質そのものではない。選択済み開発データ。", 15, MUTED, 500),
    ]
    parts.append(group("ablation-table", "\n".join(body), "60 160 1160 540"))
    return svg("content", "\n".join(parts))


def slide13():
    parts = [header("現在の課題と今後の予定", 13, "NEXT STEP", pale=True)]
    body = [
        rect(72, 176, 388, 344, "#FFFFFF", RULE, 1, 4),
        rect(72, 176, 8, 344, RED, "none", 0),
        tx(106, 224, "現在の課題", 27, RED, 700),
        tx(106, 282, "成功例が選択済み合成画像に偏る", 24, TEXT, 700),
        tx(106, 330, "実画像への一般化を未検証", 24, TEXT, 500),
        tx(106, 375, "動作・写真評価は内部非盲検", 24, TEXT, 500),
        tx(106, 420, "時系列の接触・持ち上げも未確立", 24, TEXT, 500),

        tx(520, 218, "次の目標", 27, BLUE, 700),
        tx(520, 258, "実画像でも動作・写真品質・背景保持を同時に満たす", 24, TEXT, 700),
        line(520, 282, 1188, 282, RULE, 2),

        tx(520, 329, "01", 20, BLUE, 700, family=DATA),
        tx(574, 329, "held-out 実画像40件を同条件・複数seedで生成", 24, TEXT, 700),
        tx(520, 392, "02", 20, BLUE, 700, family=DATA),
        tx(574, 392, "2–3名で Action・Photo・Preservation を盲評", 24, TEXT, 700),
        tx(520, 455, "03", 20, BLUE, 700, family=DATA),
        tx(574, 455, "bootstrap CI と失敗分類から材質制御を修正", 24, TEXT, 700),

        rect(116, 566, 1048, 72, BLUE, "none", 0, 4),
        tx(640, 610, "目標：柔軟・流動・粒状食品で再現可能な改善を示す", 28, "#FFFFFF", 700, "middle"),
    ]
    parts.append(group("next-step-summary", "\n".join(body), "72 176 1116 462"))
    return svg("ending", "\n".join(parts))


def svg(role, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
            f'data-pptx-page-role="{role}">\n{body}\n</svg>\n')


SLIDES = {
    1: slide01, 2: slide02, 3: slide03, 4: slide04, 5: slide05, 6: slide06,
    7: slide07, 8: slide08, 9: slide09, 10: slide10, 11: slide11, 12: slide12,
    13: slide13,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("slides", nargs="*", type=int)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    todo = args.slides or list(range(1, 14))
    for n in todo:
        if n not in SLIDES:
            raise SystemExit(f"unknown slide: {n}")
        path = OUT / f"slide_{n:02d}.svg"
        path.write_text(SLIDES[n](), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
