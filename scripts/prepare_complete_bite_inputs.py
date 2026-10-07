"""Freeze source-only annotations for a narrow block-tofu experiment."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/first_bite_complete_20260929'

# Coordinates refer to unchanged source pixels, never generated outputs.
ANNOTATIONS=[
 ('7445','main',[(81,100),(188,59),(213,84),(243,134),(235,163),(130,222),(114,205),(90,151)],[(114,171),(101,166),(170,189)],'A small white tofu block topped with scallions and bonito flakes in a pale bowl.'),
 ('7532','main',[(43,100),(115,55),(172,54),(211,68),(214,112),(194,155),(133,218),(65,172),(49,142)],[(162,103),(87,162),(176,139)],'A thick white tofu block with two pieces of fish roe on a dark plate.'),
 ('10646','main',[(88,126),(179,45),(194,48),(390,164),(383,207),(358,263),(305,314),(283,311),(175,255),(108,190)],[(144,127),(177,237),(347,241)],'A white tofu block covered with brown soy sauce and central ginger and scallions on a patterned square plate.'),
 ('7469','main',[(168,71),(288,28),(325,37),(369,75),(374,143),(348,166),(219,196),(190,180),(171,133)],[(182,83),(207,149),(350,123)],'A tall white tofu block with minced sauce and scallions on a dark green plate. Existing resting chopsticks stay on the table.'),
 ('7492','main',[(64,49),(155,37),(199,68),(193,105),(174,147),(154,151),(76,127),(58,108)],[(81,72),(95,110),(179,119)],'A white tofu block with green herbs and brown soy sauce in a pale shallow bowl. Existing resting chopsticks stay below the bowl.'),
 ('11433','challenge',[(200,80),(306,89),(320,98),(319,161),(302,241),(280,273),(257,284),(185,264),(178,252),(187,174)],[(236,224),(225,273),(307,178)],'A white tofu block with ginger and chopped green herbs on a decorated square plate, viewed from high above.'),
 ('7526','challenge',[(89,113),(176,104),(349,120),(416,193),(421,219),(409,272),(401,306),(109,344),(84,338),(76,326),(79,187)],[(109,199),(86,249),(259,302)],'A softly rounded white tofu block with green onion and red brown sauce in a pale blue bowl.'),
 ('7537','challenge',[(104,79),(251,94),(246,153),(240,238),(231,253),(178,253),(85,239),(67,224),(77,168),(93,104)],[(154,151),(74,212),(159,238)],'A plain white tofu block on a green leaf in a blue and white patterned round bowl, viewed almost from overhead.'),
]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    target=OUT/'input_bundle';target.mkdir(exist_ok=False)
    cases=[]
    for number,(name,stratum,polygon,anchors,description) in enumerate(ANNOTATIONS,1):
        src=OUT/'candidates'/(name+'.jpg');im=Image.open(src).convert('RGB');w,h=im.size
        scale=min(640/w,480/h);rw,rh=round(w*scale),round(h*scale);left,top=(640-rw)//2,(480-rh)//2
        arr=np.asarray(im.resize((rw,rh),Image.Resampling.LANCZOS));arr=np.pad(arr,((top,480-rh-top),(left,640-rw-left),(0,0)),mode='edge')
        poly=[(x*rw/w+left,y*rh/h+top) for x,y in polygon];points=[(round(x*rw/w+left),round(y*rh/h+top)) for x,y in anchors]
        case_id=f'test_{number:02d}_{name}';folder=target/case_id;folder.mkdir()
        shutil.copyfile(src,folder/'original.jpg');Image.fromarray(arr).save(folder/'source.png')
        mask=Image.new('L',(640,480));ImageDraw.Draw(mask).polygon(poly,fill=255);mask.save(folder/'food_mask.png')
        ann={'case_id':case_id,'file_name':name+'.jpg','stratum':stratum,'description':description,
             'source_sha256':sha(src),'source_size':[w,h],'normalized_size':[640,480],
             'preprocessing':{'resized':[rw,rh],'pad_left_top':[left,top],'method':'Lanczos + edge letterbox, no generated pixels'},
             'mask_polygon_source_pixels':polygon,'anchors_source_pixels':anchors,'anchors':points,
             'cluster_id':'patterned_square_plate' if name in ['10646','11433'] else name,
             'annotation_type':'Assistant source-only approximate segmentation and semantic anchors, not ground truth',
             'split':'not used for selecting renderer or tuning parameters in this experiment'}
        (folder/'annotation.json').write_text(json.dumps(ann,indent=2)+'\n')
        ann['files']={p.name:sha(p) for p in folder.iterdir()};cases.append(ann)
    manifest={'status':'SOURCE_SELECTION_FROZEN_BEFORE_FORMAL_GENERATION','cases':cases,'selection_seed':20260929,
        'scope':'8 source-selected block-tofu cases, including 3 weak-view/rounded-shape challenges; 7 conservative photo clusters.',
        'sampling':'24 randomly ordered pure-cold-tofu metadata candidates screened from source photographs. This is a curated feasibility sample, not an unbiased food population sample.',
        'exclusions':{'7499':'top occluded by seaweed','7464':'already cut into multiple pieces','11099':'same plate and closely similar sauce composition as 10646',
            '7444':'base too occluded by flakes','7505':'rounded cylinder and heavy garnish','7503':'base too occluded by herbs','7463':'only one side visible',
            '7522':'visible text/watermark','7439':'possible multiple adjacent pieces under heavy garnish','7468':'already cut','7535':'multiple pieces and low resolution',
            '7517':'visible watermark','7446':'base too occluded by vegetables','7458':'multiple cut pieces','7457':'multiple round pieces','7518':'multiple cut pieces'},
        'model_pretraining_contamination':'Unknown; test split means held from this experiment development, not guaranteed absent from pretrained model data.',
        'dev_images':['7489.jpg','7490.jpg'],'preparer_sha256':sha(Path(__file__))}
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(OUT/'input_bundle.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in target.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(target))
    print(json.dumps({'cases':len(cases),'manifest_sha256':sha(target/'manifest.json'),'archive_sha256':sha(OUT/'input_bundle.zip')}))


if __name__=='__main__':main()
