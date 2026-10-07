"""Source-only prospective selection for the September 30 structural experiment."""
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
from PIL import Image,ImageDraw

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'outputs/first_bite_structure_20260930'
OLD=BASE/'outputs/first_bite_coupled_20260929'
ANNOTATIONS=[
 ('7473',[(134,189),(174,99),(186,87),(321,107),(322,135),(298,233),(281,242),(197,250),(142,227)],[(288,179),(205,220),(311,164)],'A low white tofu block with red and yellow peppers in a transparent glass bowl; a wooden spoon rests beside it.'),
 ('7441',[(104,111),(110,93),(120,71),(145,59),(185,72),(204,121),(196,150),(187,165),(124,166),(110,157)],[(190,112),(151,148),(200,138)],'A white tofu block with green garnish and tiny fish in a white flower shaped bowl.'),
 ('11160',[(72,217),(153,36),(174,38),(312,82),(305,125),(234,294),(217,305),(97,274),(80,254)],[(151,195),(151,271),(268,192)],'An elongated white tofu block with chopped green onion on a rectangular patterned plate.'),
 ('7498',[(17,302),(86,211),(98,207),(284,249),(287,267),(262,337),(244,359),(159,358),(66,337),(20,318)],[(224,312),(130,336),(273,304)],'A low rectangular tofu slab with ginger and bonito flakes in a round beige bowl, with a bottle in the background.')
]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=OUT/'prospective_inputs';out.mkdir(parents=True,exist_ok=False);cases=[]
    for i,(name,poly,anchors,desc) in enumerate(ANNOTATIONS,1):
        src=OLD/'new_source_candidates'/(name+'.jpg');im=Image.open(src).convert('RGB');w,h=im.size
        scale=min(640/w,480/h);rw,rh=round(w*scale),round(h*scale);left,top=(640-rw)//2,(480-rh)//2
        a=np.asarray(im.resize((rw,rh),Image.Resampling.LANCZOS));a=np.pad(a,((top,480-rh-top),(left,640-rw-left),(0,0)),mode='edge')
        cid=f'prospective_{i:02d}_{name}';d=out/cid;d.mkdir();shutil.copy2(src,d/'original.jpg');Image.fromarray(a).save(d/'source.png')
        mask=Image.new('L',(640,480));ImageDraw.Draw(mask).polygon([(x*rw/w+left,y*rh/h+top) for x,y in poly],fill=255);mask.save(d/'food_mask.png')
        row={'case_id':cid,'file_name':src.name,'source_sha256':sha(src),'source_size':[w,h],
             'preprocessing':{'resized':[rw,rh],'pad_left_top':[left,top],'method':'Lanczos resize and edge letterbox'},
             'mask_polygon_source_pixels':poly,'anchors_source_pixels':anchors,
             'anchors':[(round(x*rw/w+left),round(y*rh/h+top)) for x,y in anchors],
             'description':desc,'cluster_id':name,'annotation_type':'Approximate assistant source-only annotation; no measured geometric truth',
             'split':'Prospective: no edited outputs from these sources inspected in this iteration before selection.'}
        (d/'annotation.json').write_text(json.dumps(row,indent=2)+'\n');row['files']={p.name:sha(p) for p in d.iterdir()};cases.append(row)
    m={'created_unix':time.time(),'cases':cases,'status':'source_selection_frozen',
       'source_order':'Continue fixed 2026092902 candidate order after 7459; source-only eligibility review.',
       'excluded':{'7476':'top surface heavily occluded by avocado and topping','7466':'top heavily occluded by chopped topping','7438':'already divided into several pieces; incompatible intact first-bite starting state'},
       'scope':'Four curated block-tofu real photographs, including low slab and narrow-side challenges. Historical training exposure unknown.',
       'script_sha256':sha(Path(__file__))}
    (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    with zipfile.ZipFile(OUT/'prospective_inputs.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(out))
    print({'cases':len(cases),'manifest_sha256':sha(out/'manifest.json')})

if __name__=='__main__':main()
