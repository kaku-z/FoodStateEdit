"""Full-resolution source/proxy/output pages for incremental assistant diagnosis."""
import argparse
import json
from pathlib import Path


def main():
    from PIL import Image,ImageDraw,ImageFont
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.snapshot.parent;notes=json.loads((root/'assistant_review_notes.json').read_text());done={(r['backend'],r['id']) for r in notes['rows']}
    inv=json.loads((a.snapshot/'formal/inventory.json').read_text());new=[r for r in inv['cells'] if r['status']=='generated' and (r['backend'],r['id']) not in done]
    a.output.mkdir(exist_ok=False);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20);pages=[]
    for cid in sorted(set(r['case_id'] for r in new)):
        rows=[r for r in new if r['case_id']==cid]
        for part in range((len(rows)+3)//4):
            batch=rows[part*4:(part+1)*4];sheet=Image.new('RGB',(1280,1530),'#eeeeee');draw=ImageDraw.Draw(sheet)
            for col,(name,path) in enumerate([('SOURCE',root/'input_bundle'/cid/'source.png'),('3D CONTROL (not an output)',root/'geometry_controls_v1/geometry'/cid/'rgb_control.png')]):
                draw.text((col*640+5,5),cid+' / '+name,font=font,fill='black')
                if path.exists():sheet.paste(Image.open(path).convert('RGB'),(col*640,30))
            for i,r in enumerate(batch):
                x,y=(i%2)*640,510+(i//2)*510
                draw.text((x+5,y+5),r['backend']+' / '+r['id'],font=font,fill='black')
                rel=r['raw_path'].split('/first_bite_complete_20260929/',1)[1]
                sheet.paste(Image.open(a.snapshot/rel).convert('RGB'),(x,y+30))
            path=a.output/f'{cid}_{part+1}.jpg';sheet.save(path,quality=97);pages.append({'page':str(path),'cells':[{'backend':r['backend'],'id':r['id'],'raw_sha256':r['raw_sha256']} for r in batch]})
    (a.output/'pages.json').write_text(json.dumps(pages,indent=2)+'\n');print(json.dumps({'new_outputs':len(new),'pages':[p['page'] for p in pages]}))


if __name__=='__main__':main()
