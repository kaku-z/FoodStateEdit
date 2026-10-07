"""Extract short-path review boards without exceeding Windows path limits."""
import argparse, json, zipfile
from io import BytesIO
from pathlib import Path
from PIL import Image,ImageDraw

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--gate',type=int,required=True);a=ap.parse_args();r=a.root;g=f'gate_v{a.gate}';out=r/f'r{a.gate}';out.mkdir(exist_ok=True)
    with zipfile.ZipFile(r/(g+'_complete.zip')) as z:
        for n in z.namelist():
            if n.startswith(g+'/') and n.count('/')==1 and n.endswith('_review.jpg'):(out/Path(n).name).write_bytes(z.read(n))
        if g+'/config.json' not in z.namelist():
            print('DERIVED_REVIEW_BOARDS_EXTRACTED',a.gate,len(list(out.glob('*_review.jpg'))),flush=True);return
        config=json.loads(z.read(g+'/config.json'));cases=sorted(set(j['case_id'] for j in config['jobs']))
        for cid in cases:
            js=[j for j in config['jobs'] if j['case_id']==cid];arms=sorted(set(j['method'] for j in js));board=Image.new('RGB',(1920,430*len(arms)),'white');d=ImageDraw.Draw(board)
            for row,arm in enumerate(arms):
                for col,s in enumerate([41,163,907]):
                    jid=next(j['id'] for j in js if j['method']==arm and j['seed']==s)
                    names=[n for n in z.namelist() if '/worker_' in n and n.endswith('/'+jid+'/raw.png')];assert len(names)==1
                    im=Image.open(BytesIO(z.read(names[0]))).convert('RGB');im.thumbnail((632,384));board.paste(im,(col*640+(632-im.width)//2,row*430+28));d.text((col*640+4,row*430+5),arm+' raw seed '+str(s),fill='black')
            board.save(out/(cid+'_raw_review.jpg'),quality=95)
    print('REVIEW_BOARDS_EXTRACTED',a.gate,len(cases),flush=True)

if __name__=='__main__':main()
