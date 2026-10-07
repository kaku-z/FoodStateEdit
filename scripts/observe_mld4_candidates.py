"""Independent SAM3 crop observations of MLD4 candidates, isolated to GPU 3.

Preserves every returned mask/score above the declared raw threshold. Geometry
only localizes candidate selection; observed silhouettes are SAM components,
never copies or intersections of the guide silhouette. Absence/presence is a
fallible diagnostic, not realism certification. Existing images are read only.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi


VERSION = 'sam3_target_crop_v2'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def getmask(path):
    return np.asarray(Image.open(path).convert('L')) > 0


def bbox(mask, pad=0):
    y, x = np.where(mask)
    if not len(x):
        raise ValueError('Empty localization guide')
    return [max(0, int(x.min())-pad), max(0, int(y.min())-pad),
            min(mask.shape[1], int(x.max())+pad+1), min(mask.shape[0], int(y.max())+pad+1)]


def extract(array, box):
    return array[box[1]:box[3], box[0]:box[2]]


def save_mask(path, arr):
    Image.fromarray(arr.astype('uint8')*255).save(path)


def guide_folder(root, case_id, variant, request):
    named = {'geometry_only':'frozen_probe_guides', 'material_ref':'frozen_probe_guides',
             'owned_material':'guides_v2', 'owned_boundary':'guides_boundary','owned_anchors':'guides_anchored'}
    primary = root/named.get(variant,'guides')/case_id
    options = [primary] + [root/name/case_id for name in ('frozen_probe_guides','guides_v2','guides_boundary','guides_anchored','guides') if root/name/case_id != primary]
    expected = request.get('input_hashes', {})
    required = [name for name in ('source.png','coarse.png','inpaint_mask.png','edge.png') if name in expected]
    for folder in options:
        if folder.exists() and required and all((folder/name).exists() and digest(folder/name)==expected[name] for name in required):
            return folder
    raise ValueError(f'No guide phase matches generation request hashes: {case_id}/{variant}')


def resolve_input_protocol(root, candidate, case_id, variant):
    if (candidate/'request.json').exists():
        request=load(candidate/'request.json')
        guide=guide_folder(root,case_id,variant,request)
        return guide, {'kind':'full_frame_request','request_sha256':digest(candidate/'request.json'),
                       'verified_input_hashes':request['input_hashes']}
    # Factorized outputs deliberately have stage requests rather than a fake
    # full-frame request. Verify that complete, separate provenance chain.
    generation=load(candidate/'generation.json')
    if {r['stage'] for r in generation.get('stages',[])} != {'head','source'}:
        raise ValueError('Factorized candidate lacks both completed stage records')
    factor=None
    for folder in root.glob('guides_factorized*/'+case_id):
        if (folder/'factorization.json').exists() and digest(folder/'factorization.json')==generation.get('factorization_sha256'):
            factor=folder; break
    if factor is None: raise ValueError('No exact factorization manifest match')
    fm=load(factor/'factorization.json')
    guide=None
    for folder in root.glob('guides*/'+case_id):
        if (folder/'guide_manifest.json').exists() and digest(folder/'guide_manifest.json')==fm['parent_manifest_sha256']:
            guide=folder; break
    if guide is None: raise ValueError('Factorization parent guide manifest does not match')
    def require(path, expected, label):
        if not path.exists() or digest(path)!=expected: raise ValueError('Hash mismatch: '+label)
    require(factor/'full_source.png',generation['source_sha256'],'factorized original source')
    require(guide/'source.png',generation['source_sha256'],'parent/source equality')
    require(factor/'full_inpaint_mask.png',generation['frozen_edit_sha256'],'factorized editing envelope')
    require(guide/'inpaint_mask.png',generation['frozen_edit_sha256'],'parent/envelope equality')
    require(factor/'full_guide_channels.npz',fm['copied_geometry_channels_sha256'],'factorized full geometry')
    require(guide/'guide_channels.npz',fm['original_geometry_channels_sha256'],'parent full geometry')
    if fm['copied_geometry_channels_sha256']!=fm['original_geometry_channels_sha256']:
        raise ValueError('Factorized geometry differs from verified parent')
    stage_bindings={}
    base=candidate.parent/generation['base_variant']/'unprojected.png'
    for stage in ('head','source'):
        request_path=candidate/stage/'request.json'; request=load(request_path)
        stage_generation=load(candidate/stage/'generation.json')
        sm=load(factor/stage/'stage_manifest.json')
        if request['bbox_xyxy']!=sm['bbox_xyxy']: raise ValueError('Stage crop coordinates mismatch')
        if request['base_variant']!=generation['base_variant']: raise ValueError('Stage base variant mismatch')
        require(base,request['base_sha256'],stage+' unchanged base')
        require(candidate/stage/'conditioning_used.png',request['conditioning_sha256'],stage+' actual conditioning')
        for name,expected in request['inputs'].items(): require(factor/stage/name,expected,stage+'/'+name)
        require(candidate/stage/'generated.png',stage_generation['generated_sha256'],stage+' generated output')
        top_record=next(r for r in generation['stages'] if r['stage']==stage)
        if top_record['generated_sha256']!=stage_generation['generated_sha256']:
            raise ValueError('Full generation / stage generation disagreement')
        stage_bindings[stage]={'request_sha256':digest(request_path),'generation_sha256':digest(candidate/stage/'generation.json'),
                               'crop_xyxy':request['bbox_xyxy'],'verified_inputs':request['inputs']}
    return guide, {'kind':'factorized_head_source_v1','generation_sha256':digest(candidate/'generation.json'),
                   'factorization_folder':str(factor),'factorization_sha256':generation['factorization_sha256'],
                   'parent_manifest_sha256':fm['parent_manifest_sha256'],'stage_bindings':stage_bindings,
                   'full_source_sha256':generation['source_sha256'],'full_edit_sha256':generation['frozen_edit_sha256']}


def prompt_bank(manifest, white_only):
    noun = str(manifest.get('food_prompt', manifest.get('source_prompt','food'))).lower()
    if white_only:
        specific = ['egg white', 'piece of egg white']
        contrast = ['egg yolk']
    elif any(s in noun for s in ('noodle','udon','ramen','soba','pasta')):
        specific = ['noodles', 'udon noodles' if 'udon' in noun else 'noodle strands']
        contrast = ['seaweed', 'vegetables']
    elif 'chicken' in noun:
        specific = ['chicken', 'roasted chicken']
        contrast = []
    else:
        specific = [noun]
        contrast = []
    return {'generic_food':['food','food on a spoon'], 'specific_food':specific,
            'contrast_ingredient':contrast, 'utensil':['spoon','metal spoon']}


def observe_group(processor, image, group, prompts, box, hint, selection_box, output, confidence):
    """Select real SAM components by broad location, never by guide pixel shape."""
    full_shape = hint.shape
    crop = image.crop(tuple(box)); crop.save(output/f'{group}_input_crop.png')
    state = processor.set_image(crop)
    selected = np.zeros(full_shape, bool); detections = []
    hint_crop = extract(hint, box)
    selection = np.zeros(full_shape, bool)
    selection[selection_box[1]:selection_box[3],selection_box[0]:selection_box[2]] = True
    selection_crop = extract(selection, box)
    for pidx, prompt in enumerate(prompts):
        processor.reset_all_prompts(state)
        result = processor.set_text_prompt(state=state, prompt=prompt)
        masks = result['masks'].detach().bool().cpu().numpy().reshape(-1,crop.height,crop.width)
        scores = result['scores'].detach().float().cpu().numpy().reshape(-1)
        boxes = result['boxes'].detach().float().cpu().numpy().reshape(-1,4)
        np.savez_compressed(output/f'{group}_{pidx:02d}_raw.npz', masks=masks,scores=scores,boxes_crop_xyxy=boxes)
        for index,(m,score) in enumerate(zip(masks,scores)):
            labels, n = ndi.label(m, structure=np.ones((3,3)))
            record = {'group':group,'prompt':prompt,'prompt_index':pidx,'raw_instance_index':index,
                      'score':float(score),'raw_mask_pixels':int(m.sum()),'raw_box_crop_xyxy':boxes[index].tolist(),'components':[]}
            for component in range(1,n+1):
                cm=labels==component; area=int(cm.sum()); overlap=int((cm&hint_crop).sum())
                yy,xx=np.where(cm); cy=float(yy.mean()+box[1]); cx=float(xx.mean()+box[0])
                in_selection=float((cm&selection_crop).sum()/max(1,area))
                centroid_selected=(selection_box[0]<=cx<selection_box[2] and selection_box[1]<=cy<selection_box[3])
                reasons=[]
                if score<confidence: reasons.append('below_selected_score_threshold')
                if area<8: reasons.append('fewer_than_8_pixels')
                if overlap<4: reasons.append('does_not_intersect_localization_hint')
                if in_selection<.5: reasons.append('mostly_outside_target_selection_rectangle')
                if not centroid_selected: reasons.append('centroid_outside_target_selection_rectangle')
                accepted=not reasons
                if accepted:
                    full=selected[box[1]:box[3],box[0]:box[2]]; full |= cm
                record['components'].append({'label':component,'pixels':area,'hint_intersection_pixels':overlap,
                    'inside_selection_rectangle_fraction':in_selection,'centroid_full_xy':[cx,cy],
                    'accepted':accepted,'rejected_because':reasons})
            detections.append(record)
    save_mask(output/f'{group}_selected_mask.png',selected)
    return selected, {'prompts':prompts,'crop_xyxy':box,'crop_offset_xy':box[:2],
                      'crop_native_size':[crop.width,crop.height],'selection_rectangle_xyxy':selection_box,
                      'selected_pixels':int(selected.sum()),'raw_detections':detections,
                      'max_raw_score':max((x['score'] for x in detections),default=None),
                      'max_selected_score':max((x['score'] for x in detections if any(c['accepted'] for c in x['components'])),default=None)}


def overlay(image, food, utensil, path):
    arr=np.asarray(image).astype(float)
    arr[utensil]=arr[utensil]*.6+np.array([20,150,255])*.4
    arr[food]=arr[food]*.55+np.array([20,230,80])*.45
    result=Image.fromarray(np.clip(arr,0,255).astype('uint8'))
    ImageDraw.Draw(result).text((5,5),'SAM: specific food green / introduced utensil blue',fill='red')
    result.save(path)


def observe_candidate(root, candidate, processor, args):
    case_id=candidate.parent.parent.name; variant=candidate.name
    input_path=candidate/args.image
    guide,input_protocol=resolve_input_protocol(root,candidate,case_id,variant)
    out=candidate/args.output_name
    if (out/'observation_metadata.json').exists():
        old=load(out/'observation_metadata.json')
        if old.get('input_final_sha256')==digest(input_path) and old.get('runner_sha256')==digest(__file__):
            print(json.dumps({'case_id':case_id,'variant':variant,'status':'existing_matching_observation'}),flush=True)
            return old
        raise ValueError(f'Existing observation differs; choose another --output-name: {out}')
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Partial observation exists; choose another --output-name: {out}')
    out.mkdir(parents=True,exist_ok=True)
    (out/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    image=Image.open(input_path).convert('RGB'); source=Image.open(guide/'source.png').convert('RGB')
    food_hint=getmask(guide/'target_food_mask.png'); utensil_hint=getmask(guide/'spoon_mask.png')
    if food_hint.shape!=(image.height,image.width): raise ValueError('Candidate/native guide size mismatch')
    fb=bbox(food_hint); maxdim=max(fb[2]-fb[0],fb[3]-fb[1]); context=max(18,round(.4*maxdim))
    food_crop=bbox(food_hint,context); food_selection=bbox(food_hint,max(4,round(.13*maxdim)))
    utensil_crop=bbox(utensil_hint,12); utensil_selection=bbox(utensil_hint,6)
    prompts=prompt_bank(load(guide/'guide_manifest.json'),case_id in args.white_only_case)
    records={}; selected={}
    for group in ('generic_food','specific_food','contrast_ingredient','utensil'):
        crop,selection,hint=(utensil_crop,utensil_selection,utensil_hint) if group=='utensil' else (food_crop,food_selection,food_hint)
        selected[group],records[group]=observe_group(processor,image,group,prompts[group],crop,hint,selection,out,args.selected_threshold)
    # A same-location source-image control exposes accidental selection of
    # unmoved plated food or background near the intended lifted bite.
    for group in ('generic_food','specific_food'):
        key='source_control_'+group
        selected[key],records[key]=observe_group(processor,source,key,prompts[group],food_crop,food_hint,food_selection,out,args.selected_threshold)
    save_mask(out/'observed_final_food_mask.png',selected['specific_food'])
    save_mask(out/'observed_final_utensil_mask.png',selected['utensil'])
    overlay(image,selected['specific_food'],selected['utensil'],out/'selected_overlay.png')
    for key in ('generic_food','specific_food','contrast_ingredient'):
        overlay(image,selected[key],np.zeros_like(food_hint),out/(key+'_overlay.png'))
    specific=bool(selected['specific_food'].sum()>=8); generic=bool(selected['generic_food'].sum()>=8); utensil=bool(selected['utensil'].sum()>=8)
    diagnostics={'specific_target_detected':specific,'generic_food_detected':generic,'introduced_utensil_detected':utensil,
        'no_food_detected_on_visible_utensil':utensil and not specific and not generic,
        'specific_target_missing_generic_ambiguous':not specific and generic,
        'contrast_ingredient_detected':bool(selected['contrast_ingredient'].sum()>=8),
        'source_control_specific_pixels':int(selected['source_control_specific_food'].sum()),
        'source_control_generic_pixels':int(selected['source_control_generic_food'].sum()),
        'specific_target_guide_iou':float((selected['specific_food']&food_hint).sum()/max(1,(selected['specific_food']|food_hint).sum())),
        'scope':'Operational absence/presence flags at declared thresholds. False positives/negatives require actual image review. No realism or physical correctness certification.'}
    metadata={'version':VERSION,'case_id':case_id,'variant':variant,'input_image':str(input_path),
        'input_final_sha256':digest(input_path),'guide_folder':str(guide),'runner_sha256':digest(__file__),
        'verified_input_protocol':input_protocol,
        'guide_sha256':{p.name:digest(p) for p in [guide/'source.png',guide/'coarse.png',guide/'target_food_mask.png',guide/'spoon_mask.png',guide/'inpaint_mask.png']},
        'method':'Independent SAM3 image/text segmentation of candidate crops; connected-component localization by broad rectangles and guide intersection, without guide-mask clipping or copying.',
        'raw_return_threshold':args.raw_threshold,'selected_score_threshold':args.selected_threshold,
        'selection_protocol':{'minimum_component_pixels':8,'minimum_hint_intersection_pixels':4,'minimum_component_in_selection_rectangle_fraction':.5,
            'component_centroid_must_lie_in_selection_rectangle':True,'observed_mask_is_unclipped_SAM_component':True,
            'food_context_padding_px':context,'food_selection_padding_px':max(4,round(.13*maxdim)),
            'unmoved_food_exclusion':'Reject components centered outside lifted-target rectangle or mostly outside it; same-location source control diagnoses leakage. This spatial heuristic can still fail.'},
        'white_only_target_explicitly_declared':case_id in args.white_only_case,
        'sam_checkpoint_sha256':args.checkpoint_sha256,'sam_model_builder_sha256':args.model_builder_sha256,
        'runtime':{'physical_gpu':3,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'python':sys.executable},
        'records':records,'diagnostics':diagnostics,'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'visual_review_status':'not_yet_reviewed','realism_certified':False}
    write(out/'observation_metadata.json',metadata)
    print(json.dumps({'case_id':case_id,'variant':variant,'diagnostics':diagnostics}),flush=True)
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--sam-root',type=Path,default=Path('/host/space0/guo-z/Evol-SAM3'))
    p.add_argument('--variants',nargs='+',default=['geometry_only','material_ref'])
    p.add_argument('--indices',type=int,nargs='+')
    p.add_argument('--image',default='projected.png')
    p.add_argument('--output-name',default='observation_sam3_v2')
    p.add_argument('--white-only-case',nargs='*',default=['real_13_7118'])
    p.add_argument('--raw-threshold',type=float,default=.10)
    p.add_argument('--selected-threshold',type=float,default=.30)
    args=p.parse_args(); root=args.root.resolve()
    jobs=[]
    for case in sorted((root/'real').glob('real_*')):
        if args.indices and int(case.name.split('_')[1]) not in args.indices: continue
        for variant in args.variants:
            folder=case/'candidates'/variant
            full_request=(folder/'request.json').exists()
            factorized_complete=(folder/'generation.json').exists() and all((folder/s/'request.json').exists() for s in ('head','source'))
            if (folder/args.image).exists() and (full_request or factorized_complete): jobs.append(folder)
    if not jobs: raise SystemExit('No completed candidate inputs found')
    import subprocess
    preflight=subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    free=int(preflight.split(',')[0]); print(json.dumps({'physical_gpu':3,'preflight':preflight,'jobs':len(jobs)}),flush=True)
    if free<6000: raise SystemExit('GPU 3 has less than 6000 MiB free; not starting')
    sys.path.insert(0,str(args.sam_root))
    import torch
    torch.set_num_threads(2)
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    checkpoint=args.sam_root/'sam3/sam3.pt'; args.checkpoint_sha256=digest(checkpoint)
    args.model_builder_sha256=digest(args.sam_root/'sam3/model_builder.py')
    model=build_sam3_image_model(bpe_path=str(args.sam_root/'assets/bpe_simple_vocab_16e6.txt.gz'),
        checkpoint_path=str(checkpoint),load_from_HF=False,device='cuda',eval_mode=True,
        enable_segmentation=True,enable_inst_interactivity=False,compile=False)
    processor=Sam3Processor(model,device='cuda',confidence_threshold=args.raw_threshold)
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for candidate in jobs:
                observe_candidate(root,candidate,processor,args)
    finally:
        del processor; del model; torch.cuda.empty_cache()


if __name__=='__main__':
    main()
