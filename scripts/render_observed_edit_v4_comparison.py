"""Label the complete v4 comparison; never change individual result pixels."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--font', type=Path, required=True)
    a = p.parse_args()
    font = ImageFont.truetype(str(a.font), 21)
    small = ImageFont.truetype(str(a.font), 19)
    title = ImageFont.truetype(str(a.font), 28)
    canvas = Image.new('RGB', (1600, 754), '#f6f7f8')
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 12), '真实图片开发验证：全部结果与对照', font=title, fill='#132536')
    columns = [
        ('source', '原始照片'),
        ('v3_scaffold_only', '同状态输入，无约束'),
        ('v3_projected', 'v3：强约束'),
        ('v4_interior', 'v4：软内部约束'),
        ('v4_detail', 'v4：高通内部约束'),
    ]
    for j, (_, label) in enumerate(columns):
        draw.text((320*j+12, 62), label, font=font, fill='#132536')
    notes = [
        '披萨：新约束改善内部纹理保留；目标旁的多余碎屑仍在，背景亮度仍有变化。',
        '玉米：新结果中的明显白圈、旧位置亮斑不再可见；整体观感接近无约束基线。',
    ]
    for i, case in enumerate(['pizza', 'corn']):
        top = 100+i*288
        for j, (key, _) in enumerate(columns):
            path = a.bundle/'inputs'/case/'source.png' if key == 'source' else a.review/f'{case}_{key}.png'
            with Image.open(path) as im:
                canvas.paste(im.convert('RGB').resize((320, 240), Image.Resampling.LANCZOS), (320*j, top))
        draw.text((12, top+249), notes[i], font=small, fill='#132536')
    draw.text((12, 686), '所有编辑图均为原始模型生成，仅统一缩放展示；未做最终像素贴回或泊松融合。', font=small, fill='#394a59')
    draw.text((12, 715), '2 张已见开发图 × 1 个 seed；4 次新生成。不是封存测试，也没有真实托举终态真值。', font=small, fill='#394a59')
    output = a.review/'comparison_zh.png'
    if output.exists():
        raise FileExistsError(output)
    canvas.save(output)
    print(output)


if __name__ == '__main__':
    main()
