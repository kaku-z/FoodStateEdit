"""Create an equal-scale four-way review panel and integrity summary for E5."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    records=json.loads((root/'files_manifest.json').read_text())
    bad=[]
    for rel,record in records.items():
        path=root/rel
        if not path.is_file() or path.stat().st_size!=record['size_bytes'] or sha(path)!=record['sha256']:
            bad.append(rel)
    return len(records),bad


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--e3',type=Path,required=True); p.add_argument('--e4',type=Path,required=True)
    p.add_argument('--e5',type=Path,required=True); p.add_argument('--e5b',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True); a=p.parse_args()
    n5,bad5=verify(a.e5); n5b,bad5b=verify(a.e5b)
    if bad5 or bad5b: raise ValueError({'e5':bad5,'e5b':bad5b})
    sources=[('E3 baseline',a.e3/'projected_final_hold.png'),
             ('E4 source-only repair',a.e4/'projected_final_hold.png'),
             ('E5 original reference',a.e5/'projected_final_hold.png'),
             ('E5b cavity reference',a.e5b/'projected_final_hold.png')]
    images=[Image.open(path).convert('RGB') for _,path in sources]; w,h=images[0].size
    crop=(275,195,490,430); cw,ch=crop[2]-crop[0],crop[3]-crop[1]
    scale=2; margin=30; title_h=36
    panel=Image.new('RGB',(2*w,2*(h+title_h)),'white'); draw=ImageDraw.Draw(panel)
    for i,((title,_),image) in enumerate(zip(sources,images)):
        x=(i%2)*w; y=(i//2)*(h+title_h)
        draw.text((x+8,y+9),title,fill='black'); panel.paste(image,(x,y+title_h))
    a.output_root.mkdir(parents=True,exist_ok=False)
    panel.save(a.output_root/'four_way_full.png')
    crops=Image.new('RGB',(4*cw*scale,ch*scale+title_h),'white'); draw=ImageDraw.Draw(crops)
    for i,((title,_),image) in enumerate(zip(sources,images)):
        c=image.crop(crop).resize((cw*scale,ch*scale),Image.Resampling.NEAREST)
        x=i*cw*scale; draw.text((x+5,9),title,fill='black'); crops.paste(c,(x,title_h))
    crops.save(a.output_root/'four_way_source_crop.png')
    summary={'schema_version':'foodstateedit.e5_review.v1','e5_records_verified':n5,
             'e5b_records_verified':n5b,'e5_bad':bad5,'e5b_bad':bad5b,
             'review_status':'partial_mechanism_support_visual_quality_failed',
             'findings':['E5 with original reference regenerates a block at the source location.',
                         'E5b with cavity reference exposes an empty notch and plate floor.',
                         'E5b cut walls remain overly planar/triangular and the projected payload partly occludes the notch.',
                         'Neither variant is a clear photographic success.'],
             'files':{}}
    for path in a.output_root.iterdir():
        if path.is_file(): summary['files'][path.name]={'sha256':sha(path),'size_bytes':path.stat().st_size}
    (a.output_root/'validation.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
