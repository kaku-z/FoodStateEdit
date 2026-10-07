"""Bind real source photos and freeze the 64-cell first-bite development batch."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from foodstateedit.first_bite.geometry import render_control


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def preprocess(image):
    w,h=image.size
    factor=1152/max(w,h)
    W,H=max(32,round(w*factor/32)*32),max(32,round(h*factor/32)*32)
    scale=min(W/w,H/h);rw,rh=round(w*scale),round(h*scale)
    a=np.array(image.resize((rw,rh),Image.Resampling.LANCZOS))
    l,t=(W-rw)//2,(H-rh)//2
    a=np.pad(a,((t,H-rh-t),(l,W-rw-l),(0,0)),mode='edge')
    return Image.fromarray(a),{'source_size':[w,h],'resized_size':[rw,rh],'output_size':[W,H],'pad_ltrb':[l,t,W-rw-l,H-rh-t],'method':'Lanczos and at most small edge padding; no generative upscaling'}


def prompts(case,rule,audit,diagnostic=False):
    W,H=case['output_size'];target=np.array(audit['target_uv'])/[W,H]*100
    source=case['source_center'];hand=np.array(audit['hand_uv'])/[W,H]*100
    utensil={'cohesive':'one realistic four-tined stainless-steel eating fork','granular':'one realistic stainless-steel eating spoon',
             'strand':'exactly two separate wooden chopsticks held together by one hand','liquid':'one realistic stainless-steel soup spoon'}[case['family']]
    action='Show the instant before the first mouthful enters the mouth. The mouth and face are outside the frame.'
    clauses=[
        'Edit the source photograph into a natural, believable food photograph of exactly the same scene. '+action,
        'Food: '+case['food_description']+'. Take '+case['bite_description']+'. The lifted portion is '+rule['payload']+', suitable for a single mouthful, never the entire dish.',
        'Utensil: '+utensil+'. '+rule['contact']+'. A single natural human hand enters from the right image edge and firmly holds the utensil handle; show a believable grip and anatomically normal fingers.',
        ('Height: follow the support height specified by the spatial guide.' if diagnostic else 'Height: the working end of the utensil AND the first mouthful are clearly raised into free air above the food surface and container rim. Show an unmistakable air gap below the utensil. It must look lifted, not resting on the plate, merely relocated sideways, or stuck into the food.'),
        f'Layout in the same camera frame: take from around ({source[0]*100:.0f}% width, {source[1]*100:.0f}% height); place the utensil-food contact near ({target[0]:.0f}% width, {target[1]:.0f}% height); hand grip near the right edge at {hand[1]:.0f}% height. Keep the source change visible.',
        'Source change: '+rule['source_change']+'. The first mouthful and source change belong to one action, with no duplicated food or unrelated missing food.',
        'Appearance: realistic material, consistent lighting, plausible shadows and depth. Allow new cut surfaces and perspective changes in the lifted portion. Keep the rest of the food, plate or bowl, background objects, decorations and camera viewpoint unchanged. No graphic marks, text, diagrams, plastic-looking food, floating unsupported food, extra hands or extra working utensils.'
    ]
    if case['existing_utensil_box']:
        clauses.append('The source already contains a spoon: lift and reuse that same spoon with the new hand, leaving its old resting position empty. Do not leave a second spoon behind.')
    else:clauses.append('Add only the specified eating utensil and the one holding hand.')
    direct=' '.join(clauses)
    structured='\n'.join(f'{i+1}. {s}' for i,s in enumerate(clauses))
    control='\nThe FIRST image is the sole reference for photographic appearance and scene identity. The SECOND image is a spatial guide, not a second photograph. Green marks the source bite; gold marks the lifted food; gray/brown marks the utensil; the right endpoint marks hand entry. Use the guide for placement and support relationships only. Never render its colors, labels, lines, white canvas or schematic style into the photograph.'
    return {'A_direct':direct,'B_action_spec':structured,'C_planar_guide':structured+control,'D_geometry_guide':structured+control}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    design=json.loads((root/'configs/first_bite_experiment_v1_20260928.json').read_text())
    annotations=json.loads((root/'benchmark/first_bite_20260928/cases.bound.v1.json').read_text())
    old=json.loads((root/'outputs/observed_edit_v4_20260928_bundle/config.json').read_text())
    a.output.mkdir(parents=True,exist_ok=False);cases=[]
    overview=Image.new('RGB',(960,280*8),'white');od=ImageDraw.Draw(overview)
    for i,ann in enumerate(annotations):
        case=dict(ann);d=a.output/'inputs'/case['case_id'];d.mkdir(parents=True)
        source=root/case['source']['local_path'];assert sha(source)==case['source']['sha256']
        shutil.copyfile(source,d/'source_original.jpg')
        with Image.open(source) as im:photo,prep=preprocess(im.convert('RGB'))
        photo.save(d/'source.png');W,H=photo.size
        # Remap original normalized source annotations through the tiny letterbox.
        rw,rh=prep['resized_size'];l,t,_,_=prep['pad_ltrb']
        for key in ['container_center','target_ground_xy','hand_entry_xy','source_center']:
            x,y=case[key];case[key]=[(x*rw+l)/W,(y*rh+t)/H]
        for key in ['source_box','existing_utensil_box']:
            if case[key]:
                x0,y0,x1,y1=case[key];case[key]=[(x0*rw+l)/W,(y0*rh+t)/H,(x1*rw+l)/W,(y1*rh+t)/H]
        case['container_diameter']*=rw/W
        case['output_size']=[W,H];case['preprocessing']=prep
        audits={}
        for mode in ['planar','geometry']:
            img,audit=render_control(W,H,case,case['family'],mode,.16)
            img.save(d/(mode+'.png'));audits[mode]=audit
            (d/(mode+'.json')).write_text(json.dumps(audit,indent=2))
            assert 0<audit['target_uv'][0]<W and 0<audit['target_uv'][1]<H
            assert audit['projected_lift_pixels']>35
        assert np.allclose(audits['planar']['target_uv'],audits['geometry']['target_uv'])
        case['prompts']=prompts(case,design['families'][case['family']],audits['geometry'])
        mask=Image.new('L',(W,H),0);md=ImageDraw.Draw(mask)
        x0,y0,x1,y1=case['source_box'];md.rectangle([x0*W,y0*H,x1*W,y1*H],fill=255)
        txy=audits['geometry']['target_uv'];hxy=audits['geometry']['hand_uv']
        radius=case['container_diameter']*W*.16
        md.ellipse([txy[0]-radius,txy[1]-radius,txy[0]+radius,txy[1]+radius],fill=255)
        md.line([tuple(txy),tuple(hxy)],fill=255,width=max(8,int(W*.18)))
        if case['existing_utensil_box']:
            x0,y0,x1,y1=case['existing_utensil_box'];md.rectangle([x0*W,y0*H,x1*W,y1*H],fill=255)
        mask.save(d/'diagnostic_edit_region.png')
        case['files']={f.name:{'path':f.relative_to(a.output).as_posix(),'sha256':sha(f)} for f in d.iterdir()}
        cases.append(case)
        for j,name in enumerate(['source.png','planar.png','geometry.png']):
            im=Image.open(d/name).convert('RGB');im.thumbnail((320,250));overview.paste(im,(320*j,280*i))
            od.text((320*j+4,280*i+254),case['case_id']+' '+name,fill='black')
    overview.save(a.output/'input_review.png')
    cfg={'schema':'first_bite_execution.v1','status':'FROZEN_INPUTS_BEFORE_GENERATION','cases':cases,
         'methods':[m['id'] for m in design['methods']],'seeds':design['pilot']['seeds'],'expected_calls':64,
         'backend':old['backend'],'expected_pipeline_sha256':old['expected_pipeline_sha256'],
         'inference':{'num_inference_steps':40,'true_cfg_scale':4.,'guidance_scale':None,'negative_prompt':' '},
         'runtime':{'device_map':'balanced','local_gpu_pair':[0,1],'max_memory_per_gpu':'44GiB','cpu_offload':False},
         'resource_gate':dict(old['resource_gate'],min_available_system_memory_mib=32000),
         'no_pixel_compositor':True,'no_latent_projection':True,'new_model_downloads':False,
         'design_sha256':sha(root/'configs/first_bite_experiment_v1_20260928.json'),
         'annotations_sha256':sha(root/'benchmark/first_bite_20260928/cases.bound.v1.json'),
         'geometry_sha256':sha(root/'foodstateedit/first_bite/geometry.py'),
         'preparer_sha256':sha(Path(__file__)),
         'claim_limit':'8 real seen development photos; relative geometry is assumed; no paired eating target or independent ratings.'}
    (a.output/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    print(json.dumps({'cases':len(cases),'calls':64,'config_sha256':sha(a.output/'config.json')}))


if __name__=='__main__':main()
