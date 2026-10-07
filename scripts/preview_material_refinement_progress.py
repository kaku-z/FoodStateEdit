"""Read completed G41 cells only; material previews do not mark a stage accepted."""
import json, zipfile, argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gate',choices=['gate_v41','gate_v42'],default='gate_v41');arg=ap.parse_args()
    gate=ROOT/arg.gate
    cfg=json.loads((gate/'config.json').read_text())
    completed={json.loads(p.read_text())['id']:p.parent for p in gate.glob('worker_*/*/result.json')}
    cases=json.loads((ROOT/'inputs/manifest.json').read_text())['cases']
    paths=[]
    for case in cases:
        cid=case['case_id']
        jobs=[j for j in cfg['jobs'] if j['case_id']==cid]
        if not all(j['id'] in completed for j in jobs): continue
        methods=['prior']+list(dict.fromkeys(j['method'] for j in jobs))
        board=Image.new('RGB',(1600,360*len(methods)),'white'); draw=ImageDraw.Draw(board)
        for row,method in enumerate(methods):
            for col,seed in enumerate([41,163,907]):
                ctx=Path(next(j['transform'] for j in jobs if j['seed']==seed)).parent
                tr=json.loads((ctx/'transform.json').read_text())
                x0,y0,x1,y1=tr['box']; base=np.asarray(Image.open(tr['parent']).convert('RGB'))
                mask=np.asarray(Image.open(ctx/'full_mask.png'))>0
                if method=='prior': final=base
                else:
                    p=completed[cid+'__'+method+'__'+str(seed)]/'raw.png'
                    raw=np.asarray(Image.open(p).convert('RGB').resize((192,192),Image.Resampling.LANCZOS),float)
                    target=base.astype(float).copy(); target[y0:y1,x0:x1]=raw
                    alpha=np.clip(distance_transform_edt(mask)/3,0,1)[...,None]
                    final=np.uint8(np.clip(np.rint(base*(1-alpha)+target*alpha),0,255))
                im=Image.fromarray(final).crop(tr['box']).resize((480,480),Image.Resampling.LANCZOS)
                # Top 310px preserves the entire crop width without stretching.
                im.thumbnail((520,310)); board.paste(im,(col*533+5,row*360+35))
                draw.text((col*533+5,row*360+5),method+' seed '+str(seed),fill='black')
        path=gate/(cid+'_progress_material.jpg'); board.save(path,quality=95); paths.append(path)
    with zipfile.ZipFile(ROOT/(arg.gate+'_progress_previews.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,p.name)
    print('completed_cases',len(paths),'completed_raw_cells',len(completed))
if __name__=='__main__':main()
