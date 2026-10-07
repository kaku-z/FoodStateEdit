"""Disclosed source-preserving composition of separately synthesized local states."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--gate',default='gate_v1')
    a = ap.parse_args(); a.output.mkdir(exist_ok=False,parents=True)
    cfg = json.loads((a.bundle/(a.gate+'.json')).read_text())
    inv = json.loads((a.bundle/(a.gate+'_inventory.json')).read_text())
    assert inv['status']=='complete'
    bymethod = {}
    for r in inv['results']:
        p = a.bundle/r['raw_path'].split('/first_bite_coupled_20260929/',1)[1]
        assert sha(p)==r['raw_sha256']
        bymethod[r['seed'],r['method']] = p
    inputs = a.bundle/(a.gate+'_inputs')
    transforms = json.loads((inputs/'transforms.json').read_text())
    source = np.asarray(Image.open(inputs/'source.png').convert('RGB'))
    masks = {k:np.asarray(Image.open(inputs/(k+'_edit.png'))) > 0 for k in ['hole','target']}
    settings = [
        ('L1_independent_local', 'G_removal_crop', 'K_target_crop'),
        ('L2_frozen_context_local', 'H_removal_crop_locked', 'J_target_crop_locked'),
        ('L3_no_3d_removal_control', 'I_removal_source_crop_locked', 'J_target_crop_locked')]
    if a.gate=='gate_v2':
        settings=[('L1_free_hole_locked_target','G_removal_crop','J_target_crop_locked'),
                  ('L2_frozen_context_local','H_removal_crop_locked','J_target_crop_locked')]
    settings=[(str(seed)+'__'+name,seed,removal,target) for seed in sorted({r['seed'] for r in inv['results']}) for name,removal,target in settings]
    receipts = []
    for name, seed, removal, target in settings:
        final = source.astype(float)
        steps = []
        for kind, method in [('hole',removal),('target',target)]:
            x0,y0,x1,y1 = transforms[kind]['box_xyxy']
            im = Image.open(bymethod[seed,method]).convert('RGB').resize((x1-x0,y1-y0),Image.Resampling.LANCZOS)
            canvas = source.copy(); canvas[y0:y1,x0:x1] = np.asarray(im)
            alpha = np.clip(distance_transform_edt(masks[kind])/5.,0,1)
            final = final*(1-alpha[...,None])+canvas*alpha[...,None]
            steps.append({'region':kind,'raw_method':method,'raw_sha256':sha(bymethod[seed,method]),'crop':transforms[kind],
                          'alpha':'Inward 5-pixel distance feather within frozen edit mask; exactly zero outside.'})
        final = np.uint8(np.round(final))
        outside = ~(masks['hole']|masks['target'])
        assert np.array_equal(final[outside],source[outside])
        p = a.output/(name+'.png'); Image.fromarray(final).save(p)
        receipts.append({'method':name,'output_sha256':sha(p),'source_sha256':sha(inputs/'source.png'),
            'steps':steps,'outside_edit_exact_source_pixels':True,'outside_pixel_count':int(outside.sum()),
            'generated_raw':False,'scope':'Algorithm output includes explicit pixel-layer composition. Raw generator images are separately retained. No claim of physical 3-D truth.'})
    (a.output/'composition_audit.json').write_text(json.dumps(receipts,indent=2)+'\n')
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    panels = [('SOURCE',inputs/'source.png'),('GEOMETRY CONTROL',inputs/'proxy.png')]
    panels += [(str(s)+' '+m,p) for (s,m),p in sorted(bymethod.items())]
    panels += [(r['method']+' | COMPOSITED',a.output/(r['method']+'.png')) for r in receipts]
    pages=[]
    for i in range(0,len(panels),4):
        group=panels[i:i+4];board=Image.new('RGB',(1280,1040),'#eeeeee');d=ImageDraw.Draw(board)
        for k,(label,p) in enumerate(group):
            x,y=(k%2)*640,(k//2)*520
            d.text((x+8,y+4),label,font=font,fill='black')
            im=Image.open(p).convert('RGB'); im.thumbnail((640,480))
            board.paste(im,(x+(640-im.width)//2,y+36))
        fn=a.output/f'gate_review_{i//4+1}.jpg';board.save(fn,quality=95);pages.append(str(fn))
    print(json.dumps({'compositions':len(receipts),'raw_images':len(bymethod),'pages':pages}))


if __name__ == '__main__':
    main()
