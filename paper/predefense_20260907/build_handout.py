"""Two-page Japanese progress report. Existing evidence only; no generated imagery.

Run with the bundled Python (ReportLab, Pillow and pypdf installed).
Image crops are uniform layout crops of preserved experiment evidence.
The 2026-09-06 handout and all underlying experiment artifacts are preserved.
"""
from pathlib import Path
import hashlib
import json
import math

from PIL import Image
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
HERE = Path(__file__).resolve().parent
OUT = ROOT / "output/pdf/FoodStateEdit_PreDefense_Handout_GUO_2530030_20260907.pdf"
R14 = ROOT / "results/day14_contact_guidance_observer_20260907"
R13 = ROOT / "results/day13_flexible_completion_20260906"
E13 = ROOT / "artifacts/day13_recovery_results_20260906T0115Z/evaluation_v1"
BOARD = ROOT / "artifacts/day14_contact_guidance_observer_20260907_v3/rgb_observer_counterfactuals.png"
P_W, P_H = A4
M, GAP = 44, 16
CW = (P_W - 2 * M - GAP) / 2
LX, RX = M, M + CW + GAP
BOTTOM = 40
INK = colors.HexColor("#17202A")
GRAY = colors.HexColor("#666D74")
RULE = colors.HexColor("#A0A6AC")
BLUE = colors.HexColor("#EAF1F8")
BLUE_D = colors.HexColor("#346483")
ORANGE = colors.HexColor("#FFF1DE")
ORANGE_D = colors.HexColor("#A46725")
GREEN = colors.HexColor("#EAF4EC")
GREEN_D = colors.HexColor("#41744B")
RED = colors.HexColor("#AD3C3C")
PALE = colors.HexColor("#F3F4F5")

for name, file, index in [
    ("Mincho", "yumin.ttf", 0), ("Gothic", "meiryo.ttc", 0),
    ("GothicB", "meiryob.ttc", 0),
]:
    pdfmetrics.registerFont(TTFont(name, "C:/Windows/Fonts/" + file, subfontIndex=index))
pdfmetrics.registerFontFamily("Mincho", normal="Mincho", bold="GothicB", italic="Mincho", boldItalic="GothicB")

STYLES = {
    "body": ParagraphStyle("body", fontName="Mincho", fontSize=8.55, leading=11.7,
        textColor=INK, alignment=TA_JUSTIFY, wordWrap="CJK"),
    "caption": ParagraphStyle("caption", fontName="Mincho", fontSize=7.0, leading=9.0,
        textColor=INK, alignment=TA_LEFT, wordWrap="CJK"),
    "note": ParagraphStyle("note", fontName="Gothic", fontSize=7.1, leading=9.5,
        textColor=GRAY, alignment=TA_LEFT, wordWrap="CJK"),
    "ref": ParagraphStyle("ref", fontName="Mincho", fontSize=6.5, leading=8.0,
        textColor=INK, alignment=TA_LEFT, wordWrap="CJK"),
}
AUDIT, TEXT = [], []


def para(c, s, x, top, w=CW, style="body", gap=4):
    item = Paragraph(s, STYLES[style])
    _, h = item.wrap(w, 1000)
    assert top - h >= BOTTOM, (s[:45], top - h)
    item.drawOn(c, x, top - h)
    AUDIT.append({"page": c.getPageNumber(), "type": style, "top": top, "bottom": top-h, "text": s})
    TEXT.append(s)
    return top - h - gap


def heading(c, label, x, top, sub=False):
    size = 8.8 if sub else 10.0
    c.setFillColor(INK)
    c.setFont("Gothic" if sub else "GothicB", size)
    c.drawString(x, top - size, label)
    TEXT.append(label)
    return top - size - (6 if sub else 8)


def label(c, s, x, y, size=6.8, color=INK, center=False):
    c.setFont("Gothic", size)
    c.setFillColor(color)
    if center:
        c.drawCentredString(x, y, s)
    else:
        c.drawString(x, y, s)


def header(c, page):
    label(c, "2026年度 修士論文中間発表資料", M, P_H-42, 8.0)
    c.setFont("Mincho", 8)
    c.drawRightString(P_W-M, P_H-42, "2530030")
    label(c, str(page), P_W/2, 24, 7.0, GRAY, True)


def arrow(c, x1, y1, x2, y2, color=GRAY, dashed=False):
    c.saveState()
    c.setStrokeColor(color)
    c.setLineWidth(.7)
    if dashed:
        c.setDash(3, 2)
    c.line(x1, y1, x2, y2)
    c.setDash()
    angle = math.atan2(y2-y1, x2-x1)
    for delta in (2.6, -2.6):
        c.line(x2, y2, x2+3.5*math.cos(angle+delta), y2+3.5*math.sin(angle+delta))
    c.restoreState()


def box(c, x, top, w, h, title, detail, fill=PALE, stroke=RULE, dashed=False):
    c.saveState()
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(.65)
    if dashed:
        c.setDash(3, 2)
    c.roundRect(x, top-h, w, h, 4, fill=1, stroke=1)
    c.restoreState()
    label(c, title, x+w/2, top-13, 7.0, INK, True)
    label(c, detail, x+w/2, top-24, 5.6, GRAY, True)


def framework(c, x, top):
    """Expected geometry and measured output are separate inputs to constraints."""
    w = CW
    c.setLineWidth(.6)
    c.setStrokeColor(RULE)
    c.roundRect(x, top-270, w, 270, 5, fill=0, stroke=1)
    label(c, "FoodStateEdit: 制御と観測を分離", x+9, top-14, 8.1)
    box(c, x+12, top-24, w-24, 32, "画像 I・編集領域 S・持ち上げ動作", "入力画像と指定した動作・支持領域")
    cx = x+w/2
    bw, bx1, bx2 = (w-34)/2, x+10, x+24+(w-34)/2
    arrow(c, cx, top-56, bx1+bw/2, top-66)
    box(c, bx1, top-68, bw, 34, "相対3D状態", "麺曲線・箸先・深度・段階", BLUE, BLUE_D)
    box(c, bx2, top-68, bw, 34, "局所重み付き LoRA", "既実験: strand / pinch / source", ORANGE, ORANGE_D)
    arrow(c, bx1+bw, top-85, bx2, top-85, ORANGE_D)
    arrow(c, bx1+bw/2, top-102, bx1+bw/2, top-114, BLUE_D)
    arrow(c, bx2+bw/2, top-102, bx2+bw/2, top-114, ORANGE_D)
    box(c, x+12, top-116, w-24, 34, "VACE 拡散生成 + 領域外の原画像合成", "既実験の出力系列 Y (21 frames)", BLUE, BLUE_D)
    arrow(c, bx2+bw/2, top-150, bx2+bw/2, top-162)
    # Independent 3D prior goes around the generator, not through its RGB output.
    c.setStrokeColor(BLUE_D)
    c.setLineWidth(.6)
    c.lines([(bx1,top-88,x+4,top-88), (x+4,top-88,x+4,top-181)])
    arrow(c,x+4,top-181,bx1,top-181,BLUE_D)
    box(c, bx1, top-164, bw, 35, "期待構造", "投影曲線・接触段階・可視性", GREEN, GREEN_D)
    box(c, bx2, top-164, bw, 35, "画像から独立に観測", "現行: 色尤度 / 次段階: SAM3", ORANGE, ORANGE_D)
    arrow(c, bx1+bw/2, top-199, cx-20, top-211, GREEN_D)
    arrow(c, bx2+bw/2, top-199, cx+20, top-211, ORANGE_D)
    box(c, x+12, top-214, w-24, 34, "段階・遮蔽対応の構造エネルギー", "合成マスクで検証済 / VACEへの導入は未実施", GREEN, GREEN_D)
    label(c, "検証ゲート通過後に潜在変数への誘導を検討", cx, top-261, 6.4, ORANGE_D, True)
    # Dashed feedback is explicitly a future path, not an implemented model.
    c.saveState()
    c.setStrokeColor(ORANGE_D)
    c.setLineWidth(.7)
    c.setDash(3,2)
    c.lines([(x+w-12,top-231,x+w-5,top-231), (x+w-5,top-231,x+w-5,top-132)])
    c.restoreState()
    arrow(c, x+w-5, top-132, x+w-12, top-132, ORANGE_D, True)
    return top-274


def table(c, rows, x, top, widths, highlights=(), size=6.7):
    styled = []
    for r, row in enumerate(rows):
        styled.append([Paragraph(str(v), ParagraphStyle(
            f"t{len(AUDIT)}_{r}_{i}", fontName="GothicB" if r == 0 else "Mincho",
            fontSize=size, leading=size+2.1, wordWrap="CJK",
            alignment=TA_LEFT if i == 0 else TA_CENTER, textColor=INK)) for i,v in enumerate(row)])
    t = Table(styled, colWidths=widths, hAlign="LEFT")
    commands = [
        ("LINEABOVE", (0,0),(-1,0),.8,INK),
        ("LINEBELOW", (0,0),(-1,0),.45,INK),
        ("LINEBELOW", (0,-1),(-1,-1),.8,INK),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),3), ("RIGHTPADDING",(0,0),(-1,-1),3),
        ("TOPPADDING",(0,0),(-1,-1),4), ("BOTTOMPADDING",(0,0),(-1,-1),4),
    ]
    for row in highlights:
        commands.append(("BACKGROUND",(0,row),(-1,row),PALE))
    t.setStyle(TableStyle(commands))
    _, h = t.wrap(sum(widths), 1000)
    assert top-h >= BOTTOM
    t.drawOn(c,x,top-h)
    AUDIT.append({"page": c.getPageNumber(),"type":"table","top":top,"bottom":top-h,"rows":rows})
    TEXT.append("\n".join(" | ".join(map(str,r)) for r in rows))
    return top-h-5


def formula(c, x, top, lines, number):
    h = len(lines)*13+10
    c.setFillColor(PALE)
    c.roundRect(x, top-h, CW, h, 3, fill=1, stroke=0)
    for i, line in enumerate(lines):
        item=Paragraph(line,ParagraphStyle("eq",fontName="Mincho",fontSize=8.0,leading=12))
        _,ph=item.wrap(CW-34,100)
        assert ph <= 12, ("Equation overflow",line)
        item.drawOn(c,x+8,top-5-13*i-ph)
    c.setFont("Mincho", 7.5)
    c.setFillColor(INK)
    c.drawRightString(x+CW-6, top-h+6, f"({number})")
    TEXT.extend(lines)
    return top-h-6


def crop(c, path, rect, x, top, w, h):
    """Only crop for page layout; never retouch or improve experimental pixels."""
    with Image.open(path) as im:
        panel = im.crop(rect)
        iw, ih = panel.size
        scale = min(w/iw, h/ih)
        dw, dh = iw*scale, ih*scale
        c.drawImage(ImageReader(panel), x+(w-dw)/2, top-h+(h-dh)/2, dw, dh)


def mask_figure(c, x, top):
    path = R14 / "synthetic_mask_optimization.png"
    pw, gap = (CW-8)/3, 4
    for i, text in enumerate(["補正なし", "固定制約", "段階・遮蔽対応"]):
        label(c, text, x+i*(pw+gap)+pw/2, top-7, 6.8, INK, True)
        crop(c,path,(480*i,60,480*(i+1),348),x+i*(pw+gap),top-13,pw,54)
    return top-72


def generation_figure(c, x, top):
    pw, gap = (CW-8)/3, 4
    for i, text in enumerate(["接触 f=6", "持上げ f=10", "保持 f=20"]):
        label(c,text,x+i*(pw+gap)+pw/2,top-7,6.7,INK,True)
    # Frozen review frames are [0,3,6,10,15,20], laid out 3 x 2.
    evidence = [(R13/"relative3d_uniform_step32_review.png", "(a) 3D + 均一重み"),
                (R13/"relative3d_topology_weighted_step32_review.png", "(b) 3D + 局所重み")]
    yt = top-12
    for path, name in evidence:
        for i, (cc, rr) in enumerate([(2,0),(0,1),(2,1)]):
            # Common ROI in all six panels; preserves raw evidence pixel values.
            rect = (cc*688+340,rr*576+125,cc*688+620,rr*576+405)
            crop(c,path,rect,x+i*(pw+gap),yt,pw,pw)
        label(c,name,x,yt-pw-10,6.6)
        yt -= pw+20
    return yt-1


def observer_figure(c, x, top):
    pw, gap = (CW-8)/3, 4
    panels = [((448,95,723,435),"合成教師画像"),
              ((448,495,723,835),"単色ブロック"),
              ((888,495,1163,835),"期待曲線の重畳")]
    for i,(rect,title) in enumerate(panels):
        label(c,title,x+i*(pw+gap)+pw/2,top-7,6.1,INK,True)
        crop(c,BOARD,rect,x+i*(pw+gap),top-12,pw,93)
    return top-110


def first_page(c, r14):
    header(c,1)
    label(c,"相対3次元状態と接触・遮蔽制約に基づく",P_W/2,766,14,INK,True)
    label(c,"柔軟食品の拡散補完に向けた検討",P_W/2,746,14,INK,True)
    label(c,"FoodStateEdit",P_W/2,729,9,GRAY,True)
    label(c,"発表者：学籍番号 2530030  GUO ZHENGPENG",P_W/2,709,9,INK,True)
    top=687
    y=heading(c,"1  はじめに",LX,top)
    y=para(c,"麺を箸で持ち上げる画像編集では、道具の移動だけでなく、麺の連続性、把持点、遮蔽関係を同時に保つ必要がある。本研究は、剛体の箸と柔軟な麺を相対3次元状態で記述し、幾何制御だけでは表せない外観を拡散モデルで補完する FoodStateEdit を検討する。",LX,y)
    y=para(c,"研究の焦点は、3D化そのものではなく、動作段階と可視性に応じて「どの構造を、いつ補完すべきか」を指定する点にある。本稿では、既存の局所重み付き学習と、新たな接触・遮蔽制約の予備検証を報告する。",LX,y)
    y=heading(c,"2  関連研究",LX,y-3)
    y=para(c,"VACE [1] は画像・動画の条件付き生成・編集を扱う。SAM3 [2] は概念プロンプトによる物体領域の検出・分割・追跡を扱う。一方、領域の検出だけでは、麺の連結や箸との接触が正しいとは限らない。clDice [3] のような中心線に基づく評価も参考に、幾何的な期待と画像からの観測を分離する必要がある。",LX,y)
    y=heading(c,"3  提案手法",LX,y-3)
    y=heading(c,"3.1  相対3D状態と局所重み",LX,y,True)
    y=para(c,"麺を3D曲線、箸を剛体として表し、先端、器内の接続点、深度を管理する。同じ状態から制御系列と麺・把持・接続領域のマスクを投影する。既実験では次の正規化重みを速度予測誤差に適用し、高ノイズ側のVACE部のみrank-8 LoRAで学習した。",LX,y)
    y=formula(c,LX,y,["W = (1 + 3M<sub>strand</sub> + 7M<sub>pinch</sub> + 5M<sub>source</sub>) / Z","Z = mean(1 + 3M<sub>strand</sub> + 7M<sub>pinch</sub> + 5M<sub>source</sub>)"],1)
    y=heading(c,"3.2  動作段階・遮蔽対応の構造制約",LX,y,True)
    y=para(c,"新たに、予測した麺の確率を投影曲線上でサンプリングし、弱い箇所を滑らかに強調する経路スコアを用いる。接触項は期待位置の麺端と2本の箸先の証拠を測る。接近中は無効化し、接触・持上げ・保持でのみ有効化する。",LX,y)
    y=formula(c,LX,y,["E<sub>f</sub> = g<sub>f</sub> [ E<sub>path</sub>(P<sub>n</sub>, C<sub>f</sub>, V<sub>f</sub>) + 0.5 E<sub>tip</sub>(P<sub>n</sub>, P<sub>u</sub>) ]"],2)
    y=para(c,"g は段階ゲート、C は投影曲線、V は可視点集合、P は画像から得る確率場を表す。前方の箸と画面上で重なる麺だけを遮蔽として除外する。本制約は固定経路上の代理指標であり、物体識別や任意の連結性を保証しない。",LX,y)
    y=para(c,"領域外は原画像と厳密に合成する。これは保全処理の性質であり、生成能力の改善とは区別する。",LX,y,style="note")

    y=framework(c,RX,top)
    y=para(c,"図1  全体構成。実線は既実験または個別実装済みの経路、破線は未実施の拡散誘導を示す。SAM3による新しい観測器の精度検証は今後の課題である。",RX,y,style="caption",gap=8)
    y=heading(c,"4  予備実験",RX,y)
    y=heading(c,"4.1  合成マスク上での機構検証",RX,y,True)
    y=para(c,"同一の断線・箸欠落確率場から、補正なし、固定制約、段階・遮蔽対応制約を比較した。補正条件ではCPU上で確率場を60回更新した。結果を図2・表1に示す。これはVACE生成画像ではない。",RX,y)
    y=mask_figure(c,RX,y)
    y=para(c,"図2  合成確率場の比較。緑は麺、赤は箸を表す。固定制約は遮蔽区間も埋めるが、遮蔽対応制約はその区間を保持し、可視の断線を補正する。",RX,y,style="caption",gap=6)
    y=para(c,"表1  最終エネルギーと既知の遮蔽領域内の麺確率。Eは各条件の目的関数値であり、条件間の優劣を直接示さない。",RX,y,style="caption",gap=3)
    opt=r14["synthetic_mask_experiment"]["optimization"]
    rows=[["条件","E (低いほど良い)","遮蔽部の麺確率"]]
    for key,title in [("none","補正なし"),("fixed","固定制約"),("phase_visibility","段階・遮蔽対応")]:
        rows.append([title,f'{opt[key]["total"]:.6f}',f'{opt[key]["hidden_region_predicted_noodle_mean"]:.4f}'])
    y=table(c,rows,RX,y,[85,78,CW-163],highlights=(3,),size=6.5)
    y=para(c,"遮蔽部の確率は固定制約の0.4492に対し、提案制約では初期値0.0200を維持した。4つの合成マスク検証を通過したが、これは目的関数の限定的な挙動確認であり、映像の自然さを示す結果ではない。",RX,y)
    AUDIT.append({"page":1,"type":"column_end","right_bottom":y})


def second_page(c,r14):
    header(c,2)
    top=780
    y=heading(c,"4.2  VACEによる既存の3群比較",LX,top,True)
    y=para(c,"同一の既知合成うどん1例を2行に複製し、平面制御・均一重み、相対3D・均一重み、相対3D・局所重みの3群を32更新で学習した。初期値とサンプル・ノイズ系列を揃え、LoRAなしを含む5条件を評価した。",LX,y)
    y=para(c,"推論はRTX A6000、seed 1、21フレーム、20 steps、VACE scale 1、TTM offで固定した。1回のpipeline読込で5条件を実行し、全出力で領域外の画素差は0であった。",LX,y)
    y=para(c,"表2  合成教師系列に対するRGB MAE (0-255、低いほど良い)。Supportは編集領域、Topo.は局所重み付き領域、Pinchは把持領域の誤差であり、意味的な成功率ではない。",LX,y,style="caption",gap=4)
    rows=[
        ["制御 / LoRA","Support","Topo.","Pinch"],
        ["平面 / なし","13.651","21.797","28.235"],
        ["平面 / 均一","12.971","21.745","29.187"],
        ["相対3D / なし","15.272","33.004","38.968"],
        ["相対3D / 均一","15.077","31.063","38.053"],
        ["相対3D / 局所重み","15.008","30.373","36.974"],
    ]
    y=table(c,rows,LX,y,[100,48,48,CW-196],highlights=(5,),size=7.0)
    y=para(c,"同じ相対3D制御の均一重みに対し、局所重みはTopo.を2.22%、Pinchを2.83%低減した。しかし事前設定した双方5%以上の改善条件を満たさず、図3でも明瞭な意味的改善は確認できない。独立した2名による盲検評価も未完了である。",LX,y)
    y=generation_figure(c,LX,y-2)
    y=para(c,"図3  均一重みと局所重みの生成結果。同じ位置・倍率の領域を切り出した接触・持上げ・保持フレーム。両条件の差は小さく、自然な持上げの改善を示す証拠は得られていない。",LX,y,style="caption",gap=7)
    y=para(c,"相対3D・均一重みのSupport誤差は平面・均一重みより16.23%大きかった。さらに教師画像と投影曲線の位置ずれを観察した。したがって、失敗をVACEだけに帰因せず、幾何と教師の対応も再点検する。",LX,y)

    y=heading(c,"4.3  画像観測器の反実仮想検証",RX,top,True)
    y=para(c,"構造制約のRGB入力に対する実現可能性を調べるため、凍結した色尤度観測器を同一画像の5条件に適用した。表3では単色ブロックが合成教師画像より低いエネルギーとなり、4つの意味的検証を全て満たさなかった。",RX,y)
    y=para(c,"表3  色尤度観測器のエネルギー。低い値を好む設計だが、不適切な入力を高く評価してしまう。",RX,y,style="caption",gap=3)
    cases=r14["rgb_observer_experiment"]["cases"]
    rows=[["RGB入力","E (低いほど良い)"]]
    for key,title in [("no_edit","未編集画像"),("synthetic_target","既存の合成教師画像"),
                      ("strand_restored_to_source","予定の麺領域を元画像に戻す"),
                      ("utensil_restored_to_source","予定の箸領域を元画像に戻す"),
                      ("solid_color_blob","麺色の単色ブロック")]:
        rows.append([title,f'{cases[key]["total"]:.6f}'])
    y=table(c,rows,RX,y,[165,CW-165],highlights=(5,),size=6.7)
    y=observer_figure(c,RX,y-2)
    y=para(c,"図4  観測器の診断画像。中央は反実仮想の色ブロックであり、生成結果ではない。右の緑線は期待曲線、シアンは箸先で、見えている持上げ麺との位置ずれがある。",RX,y,style="caption",gap=5)
    y=para(c,"有限で非零のRGB勾配は得られたが、誤った評価を最適化する危険がある。このため接触誘導付きVACEの実行には進めていない。",RX,y)
    y=heading(c,"5  SAM3による次段階の検証",RX,y-3)
    y=para(c,"次に、サーバー上の既存SAM3を凍結した候補分割器として再利用し、麺と2本の箸を別々に観測する。6つの動作フレームを抽出し、可視の麺中心線、箸インスタンスと先端、接触の有無・判定不能を人手で確認する。",RX,y)
    y=para(c,"評価はマスクIoU、中心線一致、箸先誤差、接触判定を分け、未編集・単色・断線・空把持・箸融合・真の遮蔽を対照に含める。設計用と検証用を分離し、投影位置も再監査する。SAM3の採用自体は新規性や有効性の証明ではなく、硬いマスクをそのまま微分可能な誘導器とはしない。",RX,y)
    y=heading(c,"6  おわりに",RX,y-3)
    y=para(c,"本研究は3D状態から導く段階・遮蔽対応制約を提案し、合成マスク上で遮蔽を欠損と区別する挙動を確認した。一方、実画像観測器と拡散生成への有効性は未確立である。結果は既知合成例の能力診断に限定され、汎化性能、実データ性能、物理的正しさ、写真品質は主張しない。",RX,y)
    y=heading(c,"参考文献",RX,y-2)
    refs=[
        '[1] Z. Jiang et al. <a href="https://arxiv.org/abs/2503.07598" color="#17202A">VACE: All-in-One Video Creation and Editing.</a> ICCV, 2025.',
        '[2] N. Carion et al. <a href="https://arxiv.org/abs/2511.16719" color="#17202A">SAM 3: Segment Anything with Concepts.</a> arXiv:2511.16719, 2025.',
        '[3] S. Shit et al. <a href="https://arxiv.org/abs/2003.07311" color="#17202A">clDice - A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation.</a> CVPR, 2021.',
    ]
    for ref in refs:
        y=para(c,ref,RX,y,style="ref",gap=2)
    label(c,"実験記録: Day 13 (2026-09-06) / Day 14 (2026-09-07)",M,46,6.3,GRAY)


def build():
    r14=json.loads((R14/"result.json").read_text(encoding="utf-8"))
    m13=json.loads((E13/"run_manifest.json").read_text(encoding="utf-8"))
    assert r14["new_vace_inference_count"] == 0
    assert sum(v for k,v in r14["gates"].items() if k.startswith("rgb_") and "gradient" not in k) == 0
    assert m13["pipeline_load_count"] == 1
    assert len(m13["conditions"]) == 5
    assert m13["all_outside_support_exact"]
    assert all(v["decoded_frames"]==21 for v in m13["conditions"].values())
    cfg=json.loads((ROOT/"configs/flexible_completion_execution_recovery_20260906_v1.json").read_text(encoding="utf-8"))
    assert cfg["evaluation"]["review_frame_indices"] == [0,3,6,10,15,20]
    for key,support,topology,pinch in [
        ("planar_lora_off","13.651","21.797","28.235"),
        ("planar_uniform_step32","12.971","21.745","29.187"),
        ("relative3d_lora_off","15.272","33.004","38.968"),
        ("relative3d_uniform_step32","15.077","31.063","38.053"),
        ("relative3d_topology_weighted_step32","15.008","30.373","36.974"),
    ]:
        record=m13["conditions"][key]
        assert f'{record["target_rgb_mae_inside_support_mean"]:.3f}'==support
        assert f'{record["target_rgb_mae_inside_3d_topology_weight_volume"]:.3f}'==topology
        assert f'{record["target_rgb_mae_inside_pinch_contact_roi"]:.3f}'==pinch
    OUT.parent.mkdir(parents=True,exist_ok=True)
    c=canvas.Canvas(str(OUT),pagesize=A4,pageCompression=1)
    c.setTitle("FoodStateEdit: Relative 3D, Contact and Occlusion - Interim Report")
    c.setAuthor("GUO ZHENGPENG (2530030)")
    c.setSubject("Two-page Japanese report; Day 13 negative result, Day 14 synthetic mechanism diagnostic and proposed SAM3 validation")
    first_page(c,r14)
    c.showPage()
    second_page(c,r14)
    c.showPage()
    c.save()
    from pypdf import PdfReader
    reader=PdfReader(str(OUT))
    assert len(reader.pages)==2
    assert all(abs(float(p.mediabox.width)-P_W)<.01 for p in reader.pages)
    sources=[R14/"result.json",E13/"run_manifest.json",BOARD,
        R14/"synthetic_mask_optimization.png",R13/"relative3d_uniform_step32_review.png",
        R13/"relative3d_topology_weighted_step32_review.png"]
    evidence={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (HERE/"layout_audit.json").write_text(json.dumps({"pages":2,"minimum_body_font_pt":8.55,
        "blocks":AUDIT,"sources_sha256":evidence,"pdf_sha256":hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "note":"All crops are layout-only, not retouched. Visual page inspection is required after every build."},ensure_ascii=False,indent=2),encoding="utf-8")
    (HERE/"report_ja.md").write_text("# FoodStateEdit 中間発表資料 (2026-09-07)\n\n2530030 GUO ZHENGPENG\n\n"+"\n\n".join(TEXT),encoding="utf-8")
    print(OUT)
    print("Verified: 2 A4 pages; numeric evidence gates; text blocks within bottom margin.")


if __name__ == "__main__":
    build()
