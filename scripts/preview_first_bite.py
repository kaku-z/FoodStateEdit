"""Make labeled development-only grids without replacing or scoring raw images."""
import argparse,json
from pathlib import Path
from PIL import Image,ImageDraw


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--supplement',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    b=root/('outputs/first_bite_20260928_supplement_bundle_v2' if a.supplement else 'outputs/first_bite_20260928_bundle_v1')
    c=json.loads((b/'config.json').read_text());case=next(x for x in c['cases'] if x['case_id']==a.case)
    if a.supplement and case['experiment_role']!='roi_repair':raise ValueError('Use the height comparison verifier for diagnostic levels')
    cells=[('SOURCE',b/case['files']['source.png']['path'])]
    for m in c['methods']:
        for s in [281,913]:
            runs=root/('outputs/first_bite_20260928_supplement_gp40' if a.supplement else 'outputs/first_bite_20260928_gp40')
            pattern=f'supplement_shard_*/{a.case}__{m}__{s}/raw.png' if a.supplement else f'shard_*/{a.case}__{m}__{s}/raw.png'
            matches=list(runs.glob(pattern))
            cells.append((f'{m} seed {s}',matches[0] if matches else None))
    canvas=Image.new('RGB',(1800,1680),'#edf1f5');d=ImageDraw.Draw(canvas)
    for i,(label,path) in enumerate(cells):
        x=i%3*600;y=i//3*560
        if path:
            im=Image.open(path).convert('RGB');im.thumbnail((600,530),Image.Resampling.LANCZOS)
            canvas.paste(im,(x+(600-im.width)//2,y+(530-im.height)//2))
        d.text((x+8,y+540),label+(' MISSING' if path is None else ''),fill='black')
    out=root/('outputs/first_bite_20260928_supplement_preview' if a.supplement else 'outputs/first_bite_20260928_preview');out.mkdir(exist_ok=True)
    path=out/(a.case+'_labeled.png');canvas.save(path);print(path)


if __name__=='__main__':main()
