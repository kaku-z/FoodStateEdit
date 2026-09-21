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
    parts = [header("食物操作編集の成功条件", 4, "SUCCESS CRITERIA")]
    left = [frame_image("p10_success_soup_output.png", 72, 184, 520, 340,
                        "成功例：スプーンですくう状態", GREEN),
            tx(332, 556, "道具と食物の接触・移動を同時に確認", 19, MUTED, 500, "middle")]
    conditions = [
        ("01", "動作が正しい", "道具の種類、接触位置、食物の追従"),
        ("02", "写真として自然", "道具・食物に明らかな変形や破綻がない"),
        ("03", "背景を保持", "容器・机・背景など非編集領域が変わらない"),
    ]
    right = [tx(656, 198, "三つの判定", 25, BLUE, 700)]
    for i, (num, title, desc) in enumerate(conditions):
        y = 244 + i * 102
        right += [rect(656, y, 520, 78, "#FFFFFF", GREEN if i == 2 else RULE, 2, 6),
                  circle(690, y + 39, 21, PALE, BLUE, 2),
                  tx(690, y + 46, num, 14, BLUE, 700, "middle", DATA),
                  tx(728, y + 33, title, 22, TEXT, 700),
                  tx(728, y + 60, desc, 17, MUTED, 500)]
    right += [rect(656, 572, 520, 54, "#FFF5F5", RED, 2, 5),
              tx(916, 606, "Strict成功 = 三条件をすべて満たす", 23, RED, 700, "middle"),
              tx(656, 655, "評価は平均点ではなく、Action × Photo × Preservation の同時成立", 14, MUTED, 500)]
    parts.append(group("success-criteria", "\n".join(left + right), "60 160 1160 525"))
    return svg("content", "\n".join(parts))


def slide05():
    parts = [header("関連手法の役割と本研究", 5, "RELATED WORK")]
    cols = [
        ("Qwen Image", "写真らしい静止画編集", "外観・質感の生成", "p09_soup_qwen_seed1.png", BLUE, "[4]"),
        ("VACE", "時間連続な動画生成", "画像・動画・マスク条件", "p10_success_soup_output.png", CYAN, "[1]"),
        ("GeoEdit 系", "位置と形状の幾何制御", "Wan/VACE系 baseline", "p05_ramen_control.png", "#5D83A6", "[3]"),
        ("SAM 3", "対象領域の抽出", "現行生成経路外の候補観測", "p06_family_controls.png", "#69A87A", "[2]"),
    ]
    cards = []
    for i, (name, role, detail, image_path, color, ref) in enumerate(cols):
        x = 72 + i * 286
        cards += [rect(x, 178, 258, 254, "#FFFFFF", RULE, 2, 6),
                  rect(x, 178, 258, 8, color),
                  tx(x + 18, 215, name, 23, color, 700, family=DATA),
                  tx(x + 232, 215, ref, 16, color, 700, "end", DATA),
                  rect(x + 16, 232, 226, 92, "#F7F9FB", RULE, 1, 4),
                  img(image_path, x + 20, 236, 218, 84, "meet"),
                  tx(x + 18, 354, role, 18, TEXT, 700),
                  tx(x + 18, 386, detail, 16, MUTED, 500)]
    bridge = [
        arrow(214, 458, 640, RULE, 3), arrow(498, 458, 640, RULE, 3),
        arrow(782, 458, 640, RULE, 3), arrow(1066, 458, 640, RULE, 3),
        rect(176, 486, 928, 112, PALE, BLUE, 3, 8),
        tx(640, 524, "FoodStateEdit（本研究）", 29, BLUE, 700, "middle", DATA),
        tx(640, 558, "相対3D代理で動作を指定し、VACE生成後も非編集領域を保持", 23, TEXT, 700, "middle"),
        tx(640, 586, "既存手法を置き換えず、柔軟物操作の制御と評価を追加", 17, MUTED, 500, "middle"),
        tx(72, 632, "参考文献： [1] Jiang et al., VACE, ICCV 2025　 [2] Carion et al., SAM 3, arXiv:2511.16719", 12, MUTED, 400),
        tx(72, 650, "[3] GeoEdit official repository　 [4] Qwen-Image official model card", 12, MUTED, 400),
        tx(72, 670, "※ SAM 3は候補領域抽出用で、現行の生成経路には接続していない。", 12, RED, 500),
    ]
    parts.append(group("related-method-roles", "\n".join(cards + bridge), "60 150 1160 525"))
    return svg("content", "\n".join(parts))


def slide06():
    parts = [header("提案手法：相対3D制御と凍結VACE", 6, "METHOD")]
    stages = [
        (72, "入力画像", "I_ref", ["688 × 512 × 3", "食物・容器・背景", "動作と材料を指定"], "p05_ramen_input.png", BLUE),
        (356, "相対3D代理", "S_t", ["K control points × 3", "接触点・相対深度", "21段階の動作状態"], "p05_ramen_control.png", RED),
        (640, "凍結VACE", "C, M, prompt", ["C: 21 × 512 × 688 × 3", "M: 21 × 512 × 688 × 1", "Wan2.2-VACE-Fun-A14B"], "p06_family_controls.png", CYAN),
        (924, "局所合成", "I_out", ["支持領域内を更新", "領域外はI_refに固定", "最終静止画を保存"], "p05_ramen_output.png", GREEN),
    ]
    body = []
    for i, (x, title, symbol, lines, image_path, color) in enumerate(stages):
        body += [rect(x, 180, 250, 282, "#FFFFFF", color, 3, 7),
                 rect(x, 180, 250, 45, color, color, 1, 7),
                 tx(x + 125, 210, title, 22, "#FFFFFF", 700, "middle"),
                 rect(x + 14, 240, 222, 92, "#F7F9FB", RULE, 1, 4),
                 img(image_path, x + 18, 244, 214, 84, "meet"),
                 tx(x + 125, 360, symbol, 19, color, 700, "middle", DATA)]
        for j, label in enumerate(lines):
            body.append(tx(x + 125, 387 + j * 22, label, 16, TEXT if j == 0 else MUTED, 700 if j == 0 else 500, "middle", DATA if "×" in label or "K " in label or "Wan" in label else FONT))
        if i < 3:
            body += [line(x + 253, 320, x + 274, 320, BLUE, 3),
                     polygon([(x + 274, 313), (x + 282, 320), (x + 274, 327)], BLUE)]
    body += [
        rect(72, 500, 1136, 122, PALE, BLUE, 2, 7),
        tx(96, 532, "データの流れ", 22, BLUE, 700),
        tx(96, 568, "I_ref + 操作指定 + 材料指定", 19, TEXT, 700, family=DATA),
        arrow(390, 558, 462, BLUE, 3),
        tx(478, 568, "S_t → C, M", 19, TEXT, 700, family=DATA),
        arrow(630, 558, 702, BLUE, 3),
        tx(718, 568, "凍結VACE生成", 19, TEXT, 700),
        arrow(904, 558, 976, BLUE, 3),
        tx(992, 568, "I_out", 19, GREEN, 700, family=DATA),
        tx(96, 600, "本研究の追加部分は相対3D制御器と局所合成であり、VACEの重みは変更しない。", 17, MUTED, 500),
    ]
    parts.append(group("method-framework-detailed", "\n".join(body), "60 160 1160 470"))
    return svg("content", "\n".join(parts))


def slide07():
    parts = [header("3D操作代理がVACEへ渡す制御", 7, "CONTROL MECHANISM")]
    body = [rect(72, 164, 1136, 52, PALE, BLUE, 2, 5),
            tx(640, 196, "相対3D代理は生成モデルではなく、VACE前段の制御信号生成部", 24, BLUE, 700, "middle"),
            rect(72, 244, 332, 242, "#FFFFFF", RULE, 2, 7),
            tx(238, 278, "入力画像 + 操作指定", 22, BLUE, 700, "middle"),
            img("p05_ramen_input.png", 98, 294, 280, 112, "meet"),
            tx(238, 438, "I_ref ∈ R^(512×688×3)", 18, TEXT, 700, "middle", DATA),
            tx(238, 467, "material ∈ R^4（one-hot）", 15, MUTED, 500, "middle", DATA),
            rect(474, 244, 332, 242, "#F1FAF4", GREEN, 3, 7),
            tx(640, 278, "相対3D操作代理 S_t", 22, GREEN, 700, "middle"),
            img("p05_ramen_control.png", 500, 294, 280, 92, "meet"),
            tx(640, 416, "P_tool, P_food ∈ R^(K×3)", 16, TEXT, 700, "middle", DATA),
            tx(640, 444, "z_rel ∈ R、phase ∈ R^4", 15, MUTED, 500, "middle", DATA),
            tx(640, 472, "透視投影で 21 frames", 17, GREEN, 700, "middle"),
            rect(876, 244, 332, 242, "#FFFFFF", CYAN, 3, 7),
            tx(1042, 278, "VACE生成 → 局所合成", 22, BLUE, 700, "middle"),
            img("p05_ramen_output.png", 902, 294, 280, 92, "meet"),
            tx(1042, 416, "C ∈ R^(21×512×688×3)", 17, TEXT, 700, "middle", DATA),
            tx(1042, 444, "M ∈ R^(21×512×688×1)", 17, TEXT, 700, "middle", DATA),
            tx(1042, 472, "領域内のみ更新、外側は原画像", 14, MUTED, 500, "middle"),
            line(406, 364, 466, 364, BLUE, 4), polygon([(466, 356), (478, 364), (466, 372)], BLUE),
            line(808, 364, 868, 364, BLUE, 4), polygon([(868, 356), (880, 364), (868, 372)], BLUE),
            rect(72, 526, 1136, 106, PALE, BLUE, 2, 6),
            tx(96, 558, "VACEの担当", 20, BLUE, 700),
            tx(96, 590, "制御動画 C が「どこを・いつ・どう動かすか」を伝え、mask M が変更可能領域を指定する。", 18, TEXT, 500),
            tx(96, 617, "3D代理は外観を描く生成モデルではなく、接近 → 接触 → 持上げ → 停留の構造を符号化する前段制御器。", 18, TEXT, 500),
            tx(72, 659, "※ 現行maskはSAM出力ではなく、21フレームの代理軌跡の和集合・膨張・羽化で生成。", 14, RED, 600),
    ]
    parts.append(group("relative3d-vace-dimensions", "\n".join(body), "60 150 1160 525"))
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
    parts = [header("複数食品での多手法比較：入力・Prompt・出力", 9, "MULTI-CASE RESULT")]
    body = [
        rect(72, 158, 1136, 58, "#F7F9FB", RULE, 1, 5),
        tx(88, 181, "共通Prompt", 17, BLUE, 700),
        tx(220, 181, "realistic food photo; add the utensil action; preserve bowl and background.", 14, TEXT, 500, family=DATA),
        tx(220, 205, "材料・道具・動作語のみ置換。seed 1、21 frames、20 steps。", 13, MUTED, 500),
    ]
    rows = [
        ("Noodle", "p09comp_noodle.png", "箸で麺を持ち上げる", "細い柔軟物の連続性・把持"),
        ("Soup", "p09comp_soup.png", "スプーンですくう", "液体の追従・器の保持"),
        ("Cake", "p09comp_cake.png", "フォークで取り分ける", "切り口・支持領域の保持"),
    ]
    for i, (case, path, action, focus) in enumerate(rows):
        y = 226 + i * 135
        body += [rect(72, y, 828, 122, "#FFFFFF", RULE, 1, 4),
                 img(path, 80, y + 5, 812, 112, "meet"),
                 rect(918, y, 290, 122, "#FFFFFF", GREEN if i == 1 else RULE, 2, 5),
                 tx(938, y + 28, case, 21, GREEN if i == 1 else BLUE, 700, family=DATA),
                 tx(938, y + 57, "編集目標：" + action, 16, TEXT, 700),
                 tx(938, y + 84, "確認点：" + focus, 15, MUTED, 500),
                 tx(938, y + 108, "列順：Input / Chord / Qwen / Ours", 12, MUTED, 500, family=DATA)]
    body += [
        rect(72, 640, 1136, 37, PALE, BLUE, 1, 4),
        tx(640, 665, "選択済みseed 1の内部定性比較。目標動作の実画像Ground Truthなし。強モデルの未実行を結果として扱わない。", 14, BLUE, 700, "middle"),
    ]
    parts.append(group("multi-case-comparison", "\n".join(body), "72 154 1136 530"))
    return svg("content", "\n".join(parts))


def slide10():
    parts = [header("実験設定と評価項目", 10, "EXPERIMENT")]
    left = [rect(72, 184, 552, 366, "#FFFFFF", RULE, 2, 6), rect(72, 184, 552, 48, BLUE),
            tx(96, 217, "同条件消融", 24, "#FFFFFF", 700),
            tx(102, 294, "4", 54, BLUE, 700, family=DATA), tx(160, 289, "food cases", 24, TEXT, 700, family=DATA),
            tx(102, 371, "3", 54, BLUE, 700, family=DATA), tx(160, 366, "seeds", 24, TEXT, 700, family=DATA),
            tx(102, 448, "5", 54, BLUE, 700, family=DATA), tx(160, 443, "conditions", 24, TEXT, 700, family=DATA),
            line(328, 260, 328, 488, RULE, 2),
            multiline(356, 285, ["native input", "2D planar proxy", "fixed relative3D", "material-adaptive relative3D", "rollback-related condition"], 20, TEXT, 500, 1.6, family=DATA)]
    right = [rect(656, 184, 552, 366, PALE, "none", 0, 6), tx(684, 223, "評価項目と判定単位", 24, BLUE, 700)]
    metrics = [("Action", "道具・接触・動作が正しい"), ("Photo", "変形や破綻がなく自然"),
               ("Preservation", "非編集領域の画素を保持"), ("Strict", "3項目すべてを同時達成")]
    for i, (name, desc) in enumerate(metrics):
        y = 266 + i * 64
        right += [circle(691, y - 7, 8, BLUE), tx(716, y, name, 24, TEXT, 700, family=DATA),
                  tx(946, y, desc, 20, MUTED, 400)]
    right += [tx(684, 502, "Strict = Action ∧ Photo ∧ Preservation", 17, RED, 700, family=DATA),
              tx(684, 530, "平均値ではなく、同一出力が三条件を全て満たすかを判定", 15, MUTED, 500)]
    foot = [rect(126, 585, 1028, 50, "#FFFFFF", BLUE, 2, 5),
            tx(640, 618, "4 × 3 × 5 = 60 cells　｜　21 frames　｜　20 steps　｜　同一 seed", 25, BLUE, 700, "middle", DATA),
            tx(72, 662, "PSNR/SSIMは入力保持、MAEは変更量の診断。Action・Photo・Strictは画像と動画の内部判定。", 14, MUTED, 500)]
    parts.append(group("slide-content", "\n".join(left + right + foot)))
    return svg("content", "\n".join(parts))


def slide11():
    parts = [header("主結果：保持と操作の同時成立", 11, "RESULTS")]
    metric_widths = [190, 55, 105, 105, 130, 145, 100, 100, 106]
    metric_headers = ["手法", "n", "PSNR ↑", "SSIM ↑", "領域外 MAE ↓", "領域外保持 ↑", "Action", "Photo", "Strict"]
    metric_rows = [
        ("ChordEdit", "12", "31.78", "0.982", "0.000", "100.0%", "✗", "✗", "✗", "#F2F5F8", "#5D83A6", 400),
        ("Qwen Image", "12", "15.73", "0.628", "18.192", "0.3%", "✓", "✓", "✗", "#FFF7F7", RED, 400),
        ("★ FoodStateEdit", "12", "20.68", "0.965", "0.000", "100.0%", "✓", "✓", "✓", "#F1FAF4", GREEN, 700),
    ]
    metric_y = 182
    body = [tx(72, 165, "各手法 n=12（4入力 × 3 seeds）｜緑の ★ 行＝提案手法", 20, BLUE, 700),
            rect(72, metric_y, sum(metric_widths), 246, "#FFFFFF", RULE, 2, 4)]
    x = 72
    for i, (label, w) in enumerate(zip(metric_headers, metric_widths)):
        body += [rect(x, metric_y, w, 45, "#365F7D", "#365F7D", 1),
                 tx(x + (16 if i in (0, 1) else w / 2), metric_y + 30, label, 16, "#FFFFFF", 700,
                    "start" if i in (0, 1) else "middle", DATA)]
        x += w
    for r, row in enumerate(metric_rows):
        values, fill, accent, weight = row[:9], row[9], row[10], row[11]
        y = metric_y + 45 + r * 52
        x = 72
        for c, (value, w) in enumerate(zip(values, metric_widths)):
            body += [rect(x, y, w, 52, fill, RULE, 1)]
            if c == 0:
                body.append(tx(x + 16, y + 34, value, 17, accent, 700, "start", DATA))
            elif c >= 6:
                body.append(tx(x + w / 2, y + 34, value, 22, GREEN if value == "✓" else RED, 700, "middle", DATA))
            else:
                body.append(tx(x + w / 2, y + 34, value, 17, TEXT, weight, "middle", DATA))
            x += w
    body += [
        rect(72, 468, 1136, 78, "#F1FAF4", GREEN, 2, 5),
        tx(640, 500, "FoodStateEditは領域外を完全に保持し、Action・Photo・Strictを同時に満たした。", 21, GREEN, 700, "middle"),
        tx(72, 574, "PSNR/SSIMは入力保持、MAEは変更量の診断であり、動作成功率や写真品質そのものではない。", 15, MUTED, 500),
        tx(72, 604, "※ 選択済み開発データの内部評価。統計的有意差や実画像への一般化は未検証。", 15, RED, 600),
        tx(72, 646, "比較範囲：ChordEdit / Qwen Image / FoodStateEdit。2D planar と fixed relative3D は次頁の内部消融へ分離。", 14, MUTED, 500),
    ]
    parts.append(group("evaluation-table", "\n".join(body), "60 145 1160 525"))
    return svg("content", "\n".join(parts))


def slide12():
    parts = [header("消融実験：制御条件の定量比較", 12, "ABLATION")]
    widths = [350, 90, 150, 150, 220, 176]
    headers = ["条件", "n", "PSNR ↑", "SSIM ↑", "領域内 MAE", "領域外保持 ↑"]
    rows = [
        ("Native VACE", "12", "34.60", "0.9823", "8.33", "100.0%"),
        ("2D planar proxy", "12", "24.86", "0.9473", "20.18", "100.0%"),
        ("fixed relative3D", "12", "24.60", "0.9452", "20.73", "100.0%"),
        ("★ adaptive relative3D", "12", "23.29", "0.9384", "25.86", "100.0%"),
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


def slide14():
    parts = [header("評価指標の定義と計算", 14, "EVALUATION")]
    body = [
        rect(72, 174, 548, 430, "#FFFFFF", RULE, 2, 7),
        tx(98, 214, "画像指標", 25, BLUE, 700),
        tx(98, 258, "PSNR", 19, TEXT, 700, family=DATA),
        tx(310, 258, "画素誤差の尺度（高いほど保持）", 15, MUTED, 500),
        tx(98, 304, "SSIM", 19, TEXT, 700, family=DATA),
        tx(310, 304, "構造類似度（高いほど保持）", 15, MUTED, 500),
        tx(98, 350, "outside MAE", 18, TEXT, 700, family=DATA),
        tx(310, 350, "Ω_out の平均絶対画素差", 15, MUTED, 500),
        tx(98, 396, "exact preserve", 18, TEXT, 700, family=DATA),
        tx(310, 396, "差が0の領域外画素割合", 15, MUTED, 500),
        rect(98, 444, 474, 112, PALE, BLUE, 1, 5),
        tx(335, 478, "MAE_out = 1/|Ω_out|", 22, BLUE, 700, "middle", DATA),
        tx(335, 514, "Σ_(p∈Ω_out) |I_out(p) − I_ref(p)|", 19, TEXT, 600, "middle", DATA),
        tx(335, 542, "低いほど非編集領域を保持", 16, MUTED, 500, "middle"),

        rect(660, 174, 548, 430, "#FFFFFF", RULE, 2, 7),
        tx(686, 214, "意味判定", 25, GREEN, 700),
        tx(686, 258, "Action", 21, TEXT, 700, family=DATA),
        tx(846, 258, "道具・接触・食物追従が正しい", 17, MUTED, 500),
        tx(686, 304, "Photo", 21, TEXT, 700, family=DATA),
        tx(846, 304, "変形・複製・破綻がなく写真として自然", 17, MUTED, 500),
        tx(686, 350, "Preservation", 21, TEXT, 700, family=DATA),
        tx(846, 350, "容器・机・背景などを保持", 17, MUTED, 500),
        tx(686, 396, "Strict", 21, RED, 700, family=DATA),
        tx(846, 396, "Action ∧ Photo ∧ Preservation", 17, RED, 700, family=DATA),
        rect(686, 444, 474, 112, "#FFF5F5", RED, 1, 5),
        tx(923, 478, "Strict = 1", 22, RED, 700, "middle", DATA),
        tx(923, 514, "三つの判定をすべて満たす場合だけ", 18, TEXT, 600, "middle"),
        tx(923, 542, "平均スコアではない", 16, MUTED, 500, "middle"),
        tx(72, 646, "現状：Action・Photoは内部非盲検の開発判定。今後は2〜3名の盲評と画像単位bootstrap CIを予定。", 15, RED, 600),
    ]
    parts.append(group("metric-definitions", "\n".join(body), "60 150 1160 525"))
    return svg("content", "\n".join(parts))


def slide15():
    parts = [header("追加比較：4入力 pilot の動作・写真判定", 15, "PILOT COMPARISON")]
    widths = [300, 116, 116, 116, 116, 372]
    headers = ["条件", "完成", "領域外保持", "Action", "Photo", "解釈"]
    rows = [
        ("Input / no edit", "4/4", "4/4", "—", "—", "負対照：背景は保持するが操作なし"),
        ("Vanilla GeoEdit", "4/4", "4/4", "1/4", "0/4", "外部baseline。スープのみ暫定成功"),
        ("GeoEdit + union mask", "4/4", "4/4", "0/4", "0/4", "同じバックエンドによるマスク対照"),
        ("FoodStateEdit staged", "4/4", "4/4", "0/4", "0/4", "早期の段階投影版"),
        ("Native VACE static", "4/4", "4/4", "0/4", "0/4", "GeoEdit TTMを外しても動作は未成立"),
        ("FoodStateEdit dynamic 2-D", "4/4", "4/4", "0/4", "0/4", "制御信号は現れるが、投影結果は未通過"),
    ]
    body = [tx(72, 164, "4入力、seed 1、21 frames、20 steps。内部非盲検の開発診断", 18, BLUE, 700),
            rect(72, 184, sum(widths), 366, "#FFFFFF", RULE, 2, 4)]
    x = 72
    for c, (h, w) in enumerate(zip(headers, widths)):
        body += [rect(x, 184, w, 46, "#365F7D", "#365F7D", 1),
                 tx(x + (14 if c in (0, 5) else w / 2), 214, h, 16, "#FFFFFF", 700,
                    "start" if c in (0, 5) else "middle", DATA)]
        x += w
    for r, row in enumerate(rows):
        y = 230 + r * 53
        fill = "#F1FAF4" if r == 1 else ("#FFFFFF" if r % 2 == 0 else "#F4F7FA")
        x = 72
        for c, (value, w) in enumerate(zip(row, widths)):
            body += [rect(x, y, w, 53, fill, RULE, 1),
                     tx(x + (14 if c in (0, 5) else w / 2), y + 33, value, 15,
                        GREEN if (r == 1 and c in (0, 3)) else TEXT,
                        700 if r == 1 else 500,
                        "start" if c in (0, 5) else "middle", DATA)]
            x += w
    body += [
        rect(72, 574, 1136, 56, PALE, BLUE, 2, 5),
        tx(640, 608, "明示的な操作代理の必要性を示すが、3D優位性や全baselineへの優位性は未検証", 18, BLUE, 700, "middle"),
        tx(72, 656, "出典：DAY21_CROSS_METHOD_EVALUATION_STATUS_20260912.md。held-out評価と盲評は未実施。", 13, MUTED, 500),
    ]
    parts.append(group("pilot-comparison-table", "\n".join(body), "60 150 1160 525"))
    return svg("content", "\n".join(parts))


def svg(role, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
            f'data-pptx-page-role="{role}">\n{body}\n</svg>\n')


SLIDES = {
    1: slide01, 2: slide02, 3: slide03, 4: slide04, 5: slide05, 6: slide06,
    7: slide07, 8: slide08, 9: slide09, 10: slide10, 11: slide11, 12: slide12,
    13: slide13, 14: slide14, 15: slide15,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("slides", nargs="*", type=int)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    todo = args.slides or list(range(1, 16))
    for n in todo:
        if n not in SLIDES:
            raise SystemExit(f"unknown slide: {n}")
        path = OUT / f"slide_{n:02d}.svg"
        path.write_text(SLIDES[n](), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
