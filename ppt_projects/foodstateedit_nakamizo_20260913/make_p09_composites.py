#!/usr/bin/env python3
"""Build large-label 3-panel comparison composites for slide 10 (baselines).

Each composite: title bar (case + edit description) over three panels:
  [Input (real photo)] [ChordEdit | Qwen Image] [Ours (proposal)]
with big colored label bars so the viewer can tell at a glance which image
belongs to which method.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
IMG = ROOT / "images"

W = 1136
TITLE_H = 32
LABEL_H = 30
IMAGE_H = 100
PANEL_H = LABEL_H + IMAGE_H
TOTAL_H = TITLE_H + PANEL_H

# Panel widths: Input | Others(2 cells) | Ours
W_INPUT = 300
W_OTHERS = 536
W_OURS = 300

BLUE = "#0070C0"
CHORD = "#5D83A6"
QWEN = "#E31B23"
GREEN = "#2E8B57"
TEXT = "#222222"
MUTED = "#666666"

FONT_BOLD = str(Path("C:/Windows/Fonts/YuGothB.ttc"))
FONT_MED = str(Path("C:/Windows/Fonts/meiryo.ttc"))


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def label_bar(draw: ImageDraw.ImageDraw, x: int, w: int, y: int, text: str, color: str) -> None:
    draw.rectangle((x, y, x + w - 1, y + LABEL_H - 1), fill=color)
    f = font(FONT_BOLD, 20)
    tw = draw.textlength(text, font=f)
    if tw > w - 16:  # shrink if the label is too wide
        f = font(FONT_BOLD, max(13, int(20 * (w - 16) / tw)))
        tw = draw.textlength(text, font=f)
    draw.text((x + (w - tw) / 2, y + (LABEL_H - 20) / 2 + 1), text, fill="#FFFFFF", font=f)


def panel(canvas: Image.Image, x: int, w: int, y: int, image_path: str, label: str, color: str) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((x, y, x + w - 1, y + PANEL_H - 1), outline="#D8DDE3", width=2)
    label_bar(draw, x + 1, w - 2, y + 1, label, color)
    im = Image.open(IMG / image_path).convert("RGB")
    scale = IMAGE_H / im.height
    iw = int(im.width * scale)
    if iw > w - 8:
        scale = (w - 8) / im.width
        iw = w - 8
    ih = int(im.height * scale)
    im = im.resize((iw, ih), Image.LANCZOS)
    canvas.paste(im, (x + (w - iw) // 2, y + LABEL_H + (IMAGE_H - ih) // 2))


def composite(case_title: str, paths: list[str], out: Path) -> None:
    canvas = Image.new("RGB", (W, TOTAL_H), "#FFFFFF")
    draw = ImageDraw.Draw(canvas)
    f = font(FONT_BOLD, 20)
    tw = draw.textlength(case_title, font=f)
    if tw > W - 20:
        f = font(FONT_BOLD, max(14, int(20 * (W - 20) / tw)))
    draw.text((14, (TITLE_H - 20) / 2 + 1), case_title, fill=TEXT, font=f)
    y = TITLE_H
    panel(canvas, 0, W_INPUT, y, paths[0], "Input（実写真）", BLUE)
    panel(canvas, W_INPUT, W_OTHERS // 2, y, paths[1], "ChordEdit（既存）", CHORD)
    panel(canvas, W_INPUT + W_OTHERS // 2, W_OTHERS // 2, y, paths[2], "Qwen Image（既存）", QWEN)
    panel(canvas, W_INPUT + W_OTHERS, W_OURS, y, paths[3], "Ours（提案）", GREEN)
    canvas.save(out, optimize=True)
    print(out.name, canvas.size)


def main() -> None:
    composite(
        "Noodle｜編集：箸が麺を持ち上げる（実写真・seed 1）",
        ["p09_ramen_input.png", "p09_ramen_chordedit_seed1.png",
         "p09_ramen_qwen_seed1.png", "p09_ramen_ours_seed1.png"],
        IMG / "p09comp_noodle.png",
    )
    composite(
        "Soup｜編集：スプーンがスープをすくう（合成入力・seed 1）｜Outside SSIM 1.000/0.639/1.000｜Strict E2E ×/×/✓（ChordEdit/Qwen/Ours）",
        ["p10_success_soup_input.png", "p09_soup_chordedit_day34_seed1.png",
         "p09_soup_qwen_day34_seed1.png", "p10_success_soup_output.png"],
        IMG / "p09comp_soup.png",
    )
    composite(
        "Cake｜編集：フォークがケーキ片を持ち上げる（合成入力・seed 1）",
        ["p09_cake_input.png", "p09_cake_chordedit_seed1.png",
         "p09_cake_qwen_seed1.png", "p09_cake_ours_seed1.png"],
        IMG / "p09comp_cake.png",
    )


if __name__ == "__main__":
    main()
