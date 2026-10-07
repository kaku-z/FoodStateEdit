"""Complete descriptive plots and a fixed-seed comparison, with no best-output selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


METHODS=['A_direct','C_prior_two_reference','E_full_single_reference','L0_no_cut_rgb','L1_free_hole','L2_context_locked']
LABELS=['A Direct','C Prior 2-ref','E State prompt','L0 No cut RGB','L1 Free hole','L2 Context lock']
DIMS=['lift','utensil_support','matching_source_notch','scene_identity','no_person','photographic_realism']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--views',type=Path);a=ap.parse_args();r=a.root
    views=a.views or r/'review/views'
    d=json.loads((r/'analysis/summary.json').read_text());out=r/'figures';out.mkdir(exist_ok=True)
    rows={x['method']:x for x in d['summary']}
    vals=np.array([[rows[m]['dimensions'][k]['P'] for k in DIMS] for m in METHODS])
    fig,ax=plt.subplots(figsize=(10.8,4.7),layout='constrained');im=ax.imshow(vals,vmin=0,vmax=12,cmap='Blues',aspect='auto')
    ax.set(xticks=range(6),xticklabels=['Visible\nlift','Fork\nsupport','Matching\nnotch','Scene\nidentity','No\nperson','Photo\nrealism'],yticks=range(6),yticklabels=LABELS)
    for i in range(6):
        for j in range(6):ax.text(j,i,f'{vals[i,j]}/12',ha='center',va='center',color='white' if vals[i,j]>7 else 'black')
    ax.set_title('Criterion passes on four real source photographs × three seeds\nOne unblinded assistant diagnostic; uncertain is not pass',fontsize=12)
    fig.savefig(out/'criteria_passes.png',dpi=180);fig.savefig(out/'criteria_passes.pdf');plt.close(fig)
    cases=sorted({x['case_id'] for x in d['photo_averages']});matrix=np.array([[next(x['successes'] for x in d['photo_averages'] if x['case_id']==c and x['method']==m) for m in METHODS] for c in cases])
    fig,ax=plt.subplots(figsize=(10.5,3.8),layout='constrained');ax.imshow(matrix,vmin=0,vmax=3,cmap='Greens',aspect='auto')
    ax.set(xticks=range(6),xticklabels=['A\nDirect','C\nPrior 2-ref','E\nState prompt','L0\nNo cut RGB','L1\nFree hole','L2\nContext lock'],yticks=range(4),yticklabels=[c.split('_')[-1] for c in cases],ylabel='Source photograph')
    for i in range(4):
        for j in range(6):ax.text(j,i,f'{matrix[i,j]}/3',ha='center',va='center',color='white' if matrix[i,j]>1 else 'black')
    ax.set_title('All six criteria passed, per source photograph\nInput-aware geometry development; descriptive validation, no population inference',fontsize=12)
    fig.savefig(out/'per_photo_success.png',dpi=180);fig.savefig(out/'per_photo_success.pdf');plt.close(fig)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    panelw,panelh=360,320;board=Image.new('RGB',(panelw*7,panelh*4+56),'#eeeeee');draw=ImageDraw.Draw(board)
    draw.text((12,12),'All four sources, fixed seed 41. Every method shown; no best-seed selection. L methods include declared layer composition.',font=font,fill='black')
    selected=[]
    for i,c in enumerate(cases):
        strip=Image.new('RGB',(panelw*7,panelh),'#eeeeee');sd=ImageDraw.Draw(strip)
        for j,m in enumerate(['SOURCE']+METHODS):
            if m=='SOURCE':p=views/('source_'+c+'.png');label=c+' / real source'
            else:
                cell=next(x for x in d['cells'] if x['case_id']==c and x['method']==m and x['seed']==41)
                p=views/(cell['id']+'.png');label=LABELS[j-1]+' | '+''.join(cell[k] for k in DIMS);selected.append(cell['id'])
            im=Image.open(p).convert('RGB');im.thumbnail((panelw,panelh-40));strip.paste(im,((j*panelw)+(panelw-im.width)//2,35+(panelh-40-im.height)//2));sd.text((j*panelw+6,9),label,font=small,fill='black')
        strip.save(out/(c+'_seed41.jpg'),quality=97);board.paste(strip,(0,56+i*panelh))
    board.save(out/'all_sources_seed41.jpg',quality=97)
    (out/'selection.json').write_text(json.dumps({'rule':'All four sources, all six methods, fixed seed 41; declared before all outputs were inspected. Not selected by outcome.','ids':selected},indent=2)+'\n')
    print('Saved complete criterion and per-photo plots, plus fixed-seed overview.')


if __name__=='__main__':main()
