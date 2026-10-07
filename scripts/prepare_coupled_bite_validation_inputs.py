"""Source-only annotations for four prospective repair-validation photographs."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/first_bite_coupled_20260929'
ANNOTATIONS = [
 ('7442',[(73,108),(157,49),(307,87),(305,135),(285,205),(266,219),(198,219),(111,203),(80,183),(69,152)],[(252,118),(155,171),(279,165)],'A thick white tofu block with ginger and green garnish on a black plate.'),
 ('7496',[(19,171),(117,72),(131,67),(259,130),(262,154),(232,238),(181,299),(172,301),(47,243),(20,215),(17,183)],[(168,242),(109,256),(228,222)],'A white tofu block topped with brown chopped sauce on a blue patterned plate.'),
 ('7443',[(125,100),(233,47),(332,91),(362,156),(350,219),(342,228),(156,261),(144,250),(128,167)],[(187,168),(137,188),(245,221)],'A tofu block with pale shredded vegetables on a dark round plate, with a narrow visible left side.'),
 ('7459',[(120,79),(198,53),(270,124),(260,163),(239,187),(176,198),(159,195),(124,131)],[(192,135),(140,133),(217,172)],'A white tofu block with red brown sauce and small toppings in a blue gray handled dish on a bamboo mat.')
]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out=OUT/'validation_inputs';out.mkdir(exist_ok=False)
    cases=[]
    for i,(name,polygon,anchors,description) in enumerate(ANNOTATIONS,1):
        src=OUT/'new_source_candidates'/(name+'.jpg');im=Image.open(src).convert('RGB');w,h=im.size
        scale=min(640/w,480/h);rw,rh=round(w*scale),round(h*scale);left,top=(640-rw)//2,(480-rh)//2
        a=np.asarray(im.resize((rw,rh),Image.Resampling.LANCZOS));a=np.pad(a,((top,480-rh-top),(left,640-rw-left),(0,0)),mode='edge')
        cid=f'new_{i:02d}_{name}';d=out/cid;d.mkdir();shutil.copy2(src,d/'original.jpg');Image.fromarray(a).save(d/'source.png')
        poly=[(x*rw/w+left,y*rh/h+top) for x,y in polygon]
        mask=Image.new('L',(640,480));ImageDraw.Draw(mask).polygon(poly,fill=255);mask.save(d/'food_mask.png')
        row={'case_id':cid,'file_name':src.name,'source_sha256':sha(src),'source_size':[w,h],
            'preprocessing':{'resized':[rw,rh],'pad_left_top':[left,top],'method':'Lanczos + edge letterbox, no generated pixels'},
            'mask_polygon_source_pixels':polygon,'anchors_source_pixels':anchors,
            'anchors':[(round(x*rw/w+left),round(y*rh/h+top)) for x,y in anchors],
            'description':description,'cluster_id':name,'stratum':'narrow_side_challenge' if name=='7443' else 'main',
            'annotation_type':'Assistant approximate source-only annotations, not geometric truth',
            'split':'Prospective sources excluded from old candidate 24 and development 7489/7490; no output seen when selected.'}
        (d/'annotation.json').write_text(json.dumps(row,indent=2)+'\n')
        row['files']={p.name:sha(p) for p in d.iterdir()};cases.append(row)
    m={'status':'SOURCE_SELECTION_FROZEN_BEFORE_REPAIRED_OUTPUTS','cases':cases,'sampling':'First four eligible sources in fixed candidate order; curated block-tofu test, not a population sample.',
       'screened_before_four_selected':['7501','7442','7496','7495','7443','7533','7521','7459'],
       'exclusions':{'7501':'top heavily occluded','7495':'top heavily occluded and rounded shape','7533':'multiple tofu dishes','7521':'heavy top and side occlusion'},
       'historical_pretraining_or_project_exposure':'Not guaranteed absent from model training or unrelated historical projects; held from this repair development and prior 24-source screening.',
       'preparer_sha256':sha(Path(__file__))}
    (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    with zipfile.ZipFile(OUT/'validation_input_bundle.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(out))
    print({'cases':len(cases),'manifest_sha256':sha(out/'manifest.json')})


if __name__=='__main__':main()
