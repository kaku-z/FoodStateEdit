"""Source-only ingredient semantics probe; no image generation or target use.

Retains raw VLM replies and every candidate for auditing. This diagnostic does
not overwrite previous observations and must not be treated as ground truth.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def parse_json(reply):
    text = re.sub(r'<think>.*?</think>', '', reply, flags=re.S).strip()
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find('{'), text.rfind('}')
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return {'parse_failed': True, 'raw_reply': reply}


class SourceVLM:
    def __init__(self, model_root):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Glm4vForConditionalGeneration
        torch.set_num_threads(4)
        self.torch = torch
        processor_root = model_root/'processor' if (model_root/'processor').is_dir() else model_root
        weights_root = model_root/'text_encoder' if (model_root/'text_encoder').is_dir() else model_root
        config = json.loads((weights_root/'config.json').read_text())
        model_class = Glm4vForConditionalGeneration if config['model_type'] == 'glm4v' else Qwen3VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(str(processor_root), local_files_only=True)
        self.model = model_class.from_pretrained(
            str(weights_root), local_files_only=True,
            torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa').eval()

    def query(self, image, prompt, max_new_tokens=1024):
        images = image if isinstance(image, (tuple,list)) else [image]
        content = [{'type': 'image', 'image': im} for im in images] + [{'type': 'text', 'text': prompt}]
        messages = [{'role': 'user', 'content': content}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors='pt', enable_thinking=False).to(self.model.device)
        inputs.pop('token_type_ids', None)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        reply = self.processor.batch_decode(output[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True)[0]
        return {'raw_reply': reply, 'parsed': parse_json(reply), 'model_root': str(self.model.config._name_or_path)}


INVENTORY_PROMPT = '''Inspect only this source photograph. Identify the food and distinguish actual
wheat/rice/buckwheat noodle strands from lookalike toppings, especially egg ribbons,
bonito/fish flakes, ginger strips, shredded vegetables, meat, and translucent toppings.
Do not identify something as noodles merely because it is thin, curved, or yellow.
Give uncertainty when the resolution does not permit ingredient identification.
Return JSON with dish_description, visible_ingredients (list of objects with name,
appearance, location, confidence), noodle_appearance, noodle_visibility
(clear/partial/insufficient), and noodle_regions. noodle_regions is up to three boxes
around visibly exposed true noodles, excluding toppings as much as possible. Each
region has box_xyxy normalized to 0..1000, confidence, evidence, and contamination.
Also provide lookalike_regions with name, box_xyxy, and evidence. These proposals will
be independently checked and may be rejected; do not infer hidden noodles.'''


def draw_regions(image, data):
    canvas = image.copy().resize((image.width * 3, image.height * 3))
    draw = ImageDraw.Draw(canvas)
    for i, region in enumerate(data.get('noodle_regions', [])):
        box = region.get('box_xyxy', [])
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(x) / 1000 for x in box]
        coords = [x1 * canvas.width, y1 * canvas.height, x2 * canvas.width, y2 * canvas.height]
        draw.rectangle(coords, outline=(0, 255, 0), width=3)
        draw.text((coords[0], coords[1]), f'N{i}', fill=(0, 255, 0))
    return canvas


def make_candidate_panels(image, food, out):
    """Geometry only proposals; semantics is supplied independently by the VLM."""
    ys, xs = np.nonzero(food)
    if not len(xs):
        return []
    radius = max(12, .14 * min(image.size))
    candidates = []
    for y in np.linspace(np.percentile(ys, 8), np.percentile(ys, 92), 4):
        for x in np.linspace(np.percentile(xs, 8), np.percentile(xs, 92), 4):
            yy, xx = np.indices(food.shape)
            disk = (xx - x) ** 2 + (yy - y) ** 2 < radius ** 2
            fraction = float(food[disk].mean()) if disk.any() else 0.
            if fraction < .65:
                continue
            i = len(candidates)
            box = [int(max(0, x-radius)), int(max(0, y-radius)), int(min(image.width, x+radius)), int(min(image.height, y+radius))]
            candidates.append({'id': i, 'center_xy': [float(x), float(y)], 'radius': radius,
                'crop_box_xyxy': box, 'food_fraction': fraction})
    # Every crop includes pixels only from the original source; borders are labels.
    panel_size = 224
    columns = 4
    rows = (len(candidates) + columns - 1) // columns
    panel = Image.new('RGB', (panel_size * columns, (panel_size + 24) * rows), 'white')
    draw = ImageDraw.Draw(panel)
    for record in candidates:
        i = record['id']; x = i % columns * panel_size; y = i // columns * (panel_size + 24)
        crop = image.crop(record['crop_box_xyxy']).resize((panel_size, panel_size))
        panel.paste(crop, (x, y+24)); draw.text((x+5,y+5), f'Region {i}', fill='black')
    if candidates:
        panel.save(out / 'candidate_panels.png')
    write(out / 'candidates.json', candidates)
    return candidates


def segment_sources(args):
    """Keep competing ingredient hypotheses, including zero-detection prompts."""
    sys.path.insert(0, '/host/space0/guo-z/Evol-SAM3')
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model(
        bpe_path='/host/space0/guo-z/Evol-SAM3/assets/bpe_simple_vocab_16e6.txt.gz',
        checkpoint_path='/host/space0/guo-z/Evol-SAM3/sam3/sam3.pt',
        load_from_HF=False, device='cuda', eval_mode=True)
    processor = Sam3Processor(model, confidence_threshold=.20)
    positive_prompts = ['noodles', 'brown noodles', 'visible noodle strands', 'cooked noodles']
    negative_prompts = ['bonito flakes', 'fish flakes', 'cabbage', 'pink pickled ginger',
        'ginger strips', 'yellow egg ribbons', 'shredded omelette', 'thin strips of cooked egg',
        'egg strips', 'cucumber strips', 'tomato', 'shredded vegetables', 'meat', 'translucent topping']
    prompts = positive_prompts + negative_prompts
    for folder in sorted(args.root.glob('real_*')):
        if int(folder.name.split('_')[1]) not in args.indices:
            continue
        out = args.output / folder.name; out.mkdir(parents=True, exist_ok=True)
        im = Image.open(folder/'source.png').convert('RGB')
        rgb = np.asarray(im); food = np.asarray(Image.open(folder/'food_mask.png')) > 0
        state = processor.set_image(im)
        records, mask_list, prompt_records = [], [], []
        panels = Image.new('RGB', (320*3, 270*((len(prompts)+2)//3)), '#e9edf4')
        draw = ImageDraw.Draw(panels)
        pos = np.zeros(food.shape, np.float32); neg = pos.copy()
        for j, prompt in enumerate(prompts):
            processor.reset_all_prompts(state)
            pred = processor.set_text_prompt(state=state, prompt=prompt)
            masks = pred['masks'].detach().cpu().numpy().astype(bool).reshape(-1, *food.shape)
            scores = pred['scores'].detach().cpu().numpy().reshape(-1)
            union = np.zeros(food.shape, bool)
            for mask, score in zip(masks, scores):
                i = len(mask_list)
                records.append({'mask_id':i, 'prompt':prompt, 'positive':prompt in positive_prompts,
                    'score':float(score), 'pixels':int(mask.sum()), 'food_overlap':int((mask&food).sum())})
                mask_list.append(mask)
                if score >= .35:
                    union |= mask & food
                    dest = pos if prompt in positive_prompts else neg
                    dest[mask & food] = np.maximum(dest[mask & food], float(score))
            prompt_records.append({'prompt':prompt,'detections':len(scores), 'scores':scores.tolist(),
                'confident_union_pixels':int(union.sum())})
            preview = (rgb * np.where(union[...,None], 1., .15)).astype(np.uint8)
            tile = Image.fromarray(preview); tile.thumbnail((310,238))
            x = j%3*320; y = j//3*270
            panels.paste(tile, (x,y+30));draw.text((x+5,y+4),prompt,fill='black')
            draw.text((x+5,y+16), str([round(float(v),2) for v in scores]),fill='black')
        panels.save(out/'expanded_ingredient_hypotheses.jpg')
        np.savez_compressed(out/'expanded_ingredient_candidates.npz', masks=np.asarray(mask_list,bool),
            positive_max_score=pos, negative_max_score=neg)
        write(out/'expanded_ingredients.json', {'prompts':prompt_records,'detections':records,
            'source_only':True,'target_image_used':False,'identity_status':'competing_hypotheses_not_ground_truth'})
        # An ambiguous pixel never becomes a confident noodle just from a ridge.
        strict = (pos >= .5) & (neg < .40) & food
        Image.fromarray(strict.astype(np.uint8)*255).save(out/'conservative_noodle_hypothesis.png')
        print(folder.name, json.dumps(prompt_records), flush=True)


def summarize_sources(args):
    """A coarse source-material patch selector with fail-closed noodle identity.

    The patch keeps every observed ingredient pixel inside its disk. It is not a
    strand mask and does not invent an ingredient boundary that the source cannot
    support. A negative identity decision never erases the source food.
    """
    for folder in sorted(args.root.glob('real_*')):
        if int(folder.name.split('_')[1]) not in args.indices:
            continue
        out = args.output/folder.name
        replies = json.loads((out/'individual_semantics.json').read_text())['regions']
        expected = json.loads((out/'candidates.json').read_text())
        complete = len(replies) == len(expected) and {r['id'] for r in replies} == {r['id'] for r in expected}
        inventory = json.loads((out/'source_inventory.json').read_text())['parsed']
        eligible = []
        for record in replies:
            result = record.get('parsed', {})
            reasons = []
            if not complete: reasons.append('incomplete_candidate_audit')
            if result.get('decision') != 'noodle_dominant': reasons.append('not_noodle_dominant')
            if result.get('noodle_identity_confidence', 0) < .80: reasons.append('identity_confidence_below_0.80')
            if result.get('noodle_area_fraction', 0) < .60: reasons.append('noodle_area_below_0.60')
            if inventory.get('noodle_visibility') not in ['clear','partial']: reasons.append('whole_source_identity_insufficient')
            record['selection_rejection_reasons'] = reasons
            if not reasons:
                eligible.append(record)
        best = max(eligible, key=lambda r:(r['parsed']['noodle_area_fraction'],
            r['parsed']['noodle_identity_confidence'],r['food_fraction'])) if eligible else None
        selected = dict(source_only=True, target_image_used=False, manual_case_rule_used=False,
            source_sha256=hashlib.sha256((folder/'source.png').read_bytes()).hexdigest(),
            candidate_audit_complete=complete,
            candidate_generation='4x4 spatial grid over source food quantiles; fixed radius=0.14*min(image_size); food_fraction>=0.65',
            rule='Whole-source visible noodle agreement, crop independently noodle_dominant, confidence >=0.80, noodle area >=0.60; rank by area then confidence then food support.',
            candidates=replies, selected_candidate_id=best['id'] if best else None,
            selected_mask='selected_coherent_source_patch.png' if best else None,
            identity_status='source_model_consensus_hypothesis' if best else 'no_verified_noodle_region',
            recommended_representation='coherent_observed_material_patch' if best else 'mixed_visible_material_patch_or_abstain',
            segment_identity_not_proven=True,
            limitations='VLM confidences are uncalibrated. Source patch includes visible toppings; semantic agreement is coarse and does not certify individual curves.')
        if best:
            food = np.asarray(Image.open(folder/'food_mask.png')) > 0
            yy,xx=np.indices(food.shape);x,y=best['center_xy'];radius=best['radius']
            mask=((xx-x)**2+(yy-y)**2 < radius**2)&food
            selected.update(center_xy=best['center_xy'],radius_pixels=radius,source_pixels=int(mask.sum()))
            Image.fromarray(mask.astype(np.uint8)*255).save(out/'selected_coherent_source_patch.png')
            im=Image.open(folder/'source.png').convert('RGB'); rgb=np.asarray(im)
            Image.fromarray((rgb*np.where(mask[...,None],1.,.18)).astype(np.uint8)).resize((im.width*3,im.height*3)).save(out/'selected_source_preview.png')
        write(out/'semantic_selection.json',selected)
        print(folder.name, selected['identity_status'], selected['selected_candidate_id'], flush=True)


MATERIAL_PROMPT = '''This image shows ONLY the selected owned food material from a
source photograph; neutral gray replaces every unselected pixel. Analyze only the
visible food material inside the silhouette. Gray background is not an ingredient.
The silhouette boundary is an ARTIFICIAL SELECTION MASK, not the natural outer shape
of the food: do not infer a wedge, slice, cut edge, or whole food shape from this mask.
Describe appearance before assigning a food name. Do not complete the whole dish.
Report ingredient parts supported by these selected pixels. Also state any contrasting
ingredient parts that are NOT visible and must not be added. Absence is local to this
selection only, not a claim about the original dish. Do not invent contextual parts.
Be specific about color, gloss, texture, shape and repeated structures. If semantic
identity is uncertain, preserve a color/texture description instead of guessing.
Return JSON with observed_material, visible_colors, visible_textures, supported_parts,
contrasting_parts_not_observed, confidence (0..1), material_description (one concise
sentence using ONLY selected color/texture/interior-structure evidence, with NO food
class noun or inferred whole-object shape), and preservation_instruction (one
concise sentence about maintaining this selected appearance without adding absent
parts). Do not refer to a spoon, target image, edited result, or output composition.'''


def describe_materials(args, vlm):
    for folder in sorted(args.root.glob('real_*')):
        if int(folder.name.split('_')[1]) not in args.indices:
            continue
        out=args.output/folder.name;out.mkdir(parents=True,exist_ok=True)
        rgb=np.asarray(Image.open(folder/'source_reference.png').convert('RGB'))
        mask=np.asarray(Image.open(folder/'source_reference_mask.png').convert('L')) > 127
        if not mask.any():
            write(out/'material_description.json', {'status':'empty_source_selection','target_image_used':False})
            continue
        isolated=rgb.copy();isolated[~mask]=[127,127,127]
        Image.fromarray(isolated).save(out/'masked_source_reference.png')
        Image.fromarray(mask.astype(np.uint8)*255).save(out/'source_reference_mask.png')
        original=Image.fromarray(rgb)
        # Uniform resampling is only for inspection; saved source pixels remain unchanged.
        scale=max(1,336/max(original.size))
        size=tuple(round(v*scale) for v in original.size)
        answer=vlm.query(Image.fromarray(isolated).resize(size), MATERIAL_PROMPT, 500)
        parsed=answer.get('parsed',{})
        colors=parsed.get('visible_colors',[])
        textures=parsed.get('visible_textures',[])
        if isinstance(colors,str): colors=[colors]
        if isinstance(textures,str): textures=[textures]
        appearance='Visible source material colors: '+', '.join(str(x) for x in colors)+'. Visible source surface textures: '+', '.join(str(x) for x in textures)+'. Preserve these selected-pixel appearances. Do not introduce any contrasting color region or ingredient component absent from the masked source reference. The selection boundary is artificial and does not define a natural food shape.'
        write(out/'material_description.json', dict(**answer, prompt=MATERIAL_PROMPT,
            appearance_only_conditioning=appearance,
            conditioning_rule='Compile only visible_colors and visible_textures; exclude observed_material, supported_parts, and material_description fields.',
            target_image_used=False, whole_dish_label_supplied=False,
            unmasked_context_supplied=False,
            manual_case_rule_used=False, source=str(folder/'source_reference.png'),
            source_sha256=hashlib.sha256((folder/'source_reference.png').read_bytes()).hexdigest(),
            mask_sha256=hashlib.sha256((folder/'source_reference_mask.png').read_bytes()).hexdigest(),
            selected_pixels=int(mask.sum()), masked_source_pixels_exact=True,
            semantic_status='independent_source_model_hypothesis'))
        print(folder.name,json.dumps(answer['parsed'],ensure_ascii=False),flush=True)


def index_materials(args):
    records=[]
    for folder in sorted(args.root.glob('real_*')):
        if int(folder.name.split('_')[1]) not in args.indices:
            continue
        out=args.output/folder.name
        descriptor_path=out/'material_description.json'
        if not descriptor_path.exists():
            records.append({'case':folder.name,'status':'missing_descriptor'})
            continue
        data=json.loads(descriptor_path.read_text())
        rgb=np.asarray(Image.open(folder/'source_reference.png').convert('RGB'))
        mask=np.asarray(Image.open(folder/'source_reference_mask.png').convert('L')) > 127
        masked=np.asarray(Image.open(out/'masked_source_reference.png').convert('RGB'))
        source_match=data.get('source_sha256') == hashlib.sha256((folder/'source_reference.png').read_bytes()).hexdigest()
        mask_match=data.get('mask_sha256') == hashlib.sha256((folder/'source_reference_mask.png').read_bytes()).hexdigest()
        owned_exact=bool(np.array_equal(masked[mask],rgb[mask]))
        outside_gray=bool(np.all(masked[~mask]==127))
        parsed=data.get('parsed',{})
        valid=source_match and mask_match and owned_exact and outside_gray and not parsed.get('parse_failed',False)
        records.append({'case':folder.name,'status':'verified_pixel_ownership' if valid else 'verification_failed',
            'source_hash_match':source_match,'mask_hash_match':mask_match,
            'owned_pixels_unchanged':owned_exact,'unowned_pixels_neutral_gray':outside_gray,
            'observed_material_hypothesis':parsed.get('observed_material'),
            'visible_colors':parsed.get('visible_colors'), 'visible_textures':parsed.get('visible_textures'),
            'appearance_only_conditioning':data.get('appearance_only_conditioning'),
            'target_image_used':data.get('target_image_used'),
            'unmasked_context_supplied':data.get('unmasked_context_supplied'),
            'whole_dish_label_supplied':data.get('whole_dish_label_supplied')})
    write(args.output/'material_manifest.json',{'cases':records,'case_count':len(records),
        'all_pixel_ownership_verified':all(r['status']=='verified_pixel_ownership' for r in records),
        'semantic_truth_not_verified':True,'confidence_calibrated':False,
        'source_only':True,'model_hypotheses_preserved':True})
    print(json.dumps({'case_count':len(records),'verified':sum(r['status']=='verified_pixel_ownership' for r in records)}),flush=True)


def srgb_to_lab(rgb):
    """D65 CIELAB from standard sRGB, float64, no learned color naming."""
    srgb=np.asarray(rgb,dtype=np.float64)/255.
    linear=np.where(srgb<=.04045,srgb/12.92,((srgb+.055)/1.055)**2.4)
    matrix=np.array([[.4124564,.3575761,.1804375],
                     [.2126729,.7151522,.0721750],
                     [.0193339,.1191920,.9503041]])
    xyz=linear@matrix.T/np.array([.95047,1.,1.08883])
    delta=6/29
    f=np.where(xyz>delta**3,np.cbrt(xyz),xyz/(3*delta**2)+4/29)
    return np.stack([116*f[:,1]-16,500*(f[:,0]-f[:,1]),200*(f[:,1]-f[:,2])],axis=-1)


def pixel_statistics(pixels):
    rgb=np.asarray(pixels,dtype=np.uint8).reshape(-1,3)
    if not len(rgb): return {'pixel_count':0}
    lab=srgb_to_lab(rgb)
    chroma=np.linalg.norm(lab[:,1:],axis=1)
    saturation=(rgb.max(1).astype(float)-rgb.min(1))/np.maximum(1,rgb.max(1))
    hue=np.mod(np.degrees(np.arctan2(lab[:,2],lab[:,1])),360.)
    qs=[0,5,25,50,75,95,100]
    bins=rgb.astype(np.int32)//16
    keys=bins[:,0]*256+bins[:,1]*16+bins[:,2]
    values,counts=np.unique(keys,return_counts=True)
    order=np.argsort(-counts,kind='stable')[:8]
    palette=[]
    for i in order:
        representative=np.rint(np.median(rgb[keys==values[i]],axis=0)).astype(int)
        palette.append({'rgb_median':representative.tolist(),
            'hex':'#'+''.join(f'{v:02x}' for v in representative),
            'fraction':float(counts[i]/len(rgb)), 'pixels':int(counts[i])})
    return {'pixel_count':int(len(rgb)), 'percentiles':qs,
        'rgb_mean':rgb.mean(0).tolist(), 'rgb_percentiles':np.percentile(rgb,qs,axis=0).tolist(),
        'lab_mean':lab.mean(0).tolist(), 'lab_percentiles':np.percentile(lab,qs,axis=0).tolist(),
        'lab_chroma_percentiles':np.percentile(chroma,qs).tolist(),
        'hsv_saturation_percentiles':np.percentile(saturation,qs).tolist(),
        'high_lab_chroma_fraction':float((chroma>40).mean()),
        'bright_yellow_color_fraction':float(((lab[:,0]>50)&(chroma>40)&(hue>=65)&(hue<=115)).mean()),
        'quantized_rgb_palette':palette}


def measure_materials(args):
    records=[]
    for folder in sorted(args.root.glob('real_*')):
        if int(folder.name.split('_')[1]) not in args.indices:
            continue
        out=args.output/folder.name;out.mkdir(parents=True,exist_ok=True)
        rgb=np.asarray(Image.open(folder/'source_reference.png').convert('RGB'))
        mask=np.asarray(Image.open(folder/'source_reference_mask.png').convert('L')) > 127
        isolated=rgb.copy();isolated[~mask]=[127,127,127]
        Image.fromarray(isolated).save(out/'masked_source_reference.png')
        Image.fromarray(mask.astype(np.uint8)*255).save(out/'source_reference_mask.png')
        owned=pixel_statistics(rgb[mask]);context=pixel_statistics(rgb[~mask])
        measurements={'case':folder.name, 'source_rgb_dimensions':[rgb.shape[1],rgb.shape[0]],
            'source_sha256':hashlib.sha256((folder/'source_reference.png').read_bytes()).hexdigest(),
            'mask_sha256':hashlib.sha256((folder/'source_reference_mask.png').read_bytes()).hexdigest(),
            'source_only':True,'target_image_used':False,'vlm_used':False,
            'manual_case_rule_used':False,'owned_pixels':owned,'unowned_context_pixels':context,
            'ownership_rule':'source_reference_mask >127; retained RGB copied exactly; all unowned pixels replaced with RGB(127,127,127)',
            'color_space':'sRGB8 converted to D65 CIELAB using standard sRGB inverse transfer function and D65 XYZ matrix',
            'bright_yellow_color_rule':'CIELAB L*>50, C*ab>40, hue_ab in [65,115] degrees; a color test only, not ingredient identification',
            'generic_preservation_conditioning':'Preserve the visible colors, texture and ingredient parts inside the masked source reference. Do not add any component absent from those owned source pixels. The mask silhouette is an artificial selection boundary and does not define a natural whole-food shape.',
            'learned_food_identity':'unknown_not_inferred'}
        write(out/'measured_material.json',measurements)
        records.append(measurements)
    write(args.output/'measured_material_manifest.json',{'case_count':len(records),'cases':records,
        'source_only':True,'vlm_used':False,'target_image_used':False,
        'production_candidate':'exact masked owned-pixel reference plus deterministic source color constraints; no free-form VLM food names or textures'})
    print(json.dumps({'case_count':len(records),'measured_owned_pixels':sum(r['owned_pixels']['pixel_count'] for r in records)}),flush=True)


CONTEXT_PROMPT_VERSION='context_owned_v2_surroundings'
CONTEXT_MATERIAL_PROMPTS = {
    'A': '''Both images come from the same original food photograph. Image 1 shows
the full meal with a magenta outline immediately outside the selected pixels and
a cyan rectangle locating the crop. Image 2 shows only those selected pixels on
neutral gray. The colored lines and gray are annotations, not food.
Identify the visible component INSIDE the magenta outline, using the surrounding
meal only to disambiguate that component. Do not assign the whole dish or any
nearby ingredient outside the outline to this selection. Describe only visible
colors and textures supported by the selected pixels; say uncertain when needed.
Return JSON: component_label (short noun phrase), visible_colors (list),
visible_textures (list), confidence (0..1), alternatives (list), context_support
(short evidence), nearby_parts_outside_selection (list), uncertain_details (list),
visible_surroundings (list of visible container/table/nearby food context), and
source_context_caption (one short descriptive sentence about the selected
component in its visible source setting). Separate selected material from its
surroundings. Do not infer unseen interiors or a future edited target. Use
descriptive phrases, not editing instructions, prohibitions, or a recipe.''',
    'B': '''Consider the exact selected region of this source photo. The first
image retains the whole meal for context: magenta marks the selection boundary,
cyan marks the crop box. The second image repeats the selected material alone;
everything outside its mask is neutral gray. Ignore annotation colors.
What local food component is actually visible within the selection? Distinguish
it from visually different adjacent components seen elsewhere in the meal.
Give a short descriptive noun phrase and visible color/texture evidence for that
region alone. Context may clarify identity but cannot add an unselected part.
If ambiguous, lower confidence and list alternatives instead of guessing.
Return JSON with component_label, visible_colors, visible_textures, confidence
(0..1), alternatives, context_support, nearby_parts_outside_selection,
uncertain_details, visible_surroundings (visible container/table/nearby food), and
source_context_caption (one short sentence placing the selected component in
the visible source setting). Keep selected attributes separate from surrounding
objects. Describe no unseen interior or imagined edited target. Keep fields
brief and descriptive, without instructions, prohibitions, or recipes.'''
}


def prepare_context_materials(args):
    from scipy.ndimage import binary_dilation
    records=[]
    for folder in sorted(args.root.glob('real_*')):
        if int(folder.name.split('_')[1]) not in args.indices:continue
        out=args.output/folder.name;out.mkdir(parents=True,exist_ok=True)
        manifest=json.loads((folder/'guide_manifest.json').read_text())
        source=Image.open(folder/'source.png').convert('RGB');pixels=np.asarray(source)
        ref=Image.open(folder/'source_reference.png').convert('RGB')
        mask=np.asarray(Image.open(folder/'source_reference_mask.png').convert('L'))>127
        box=manifest['source_reference_bbox'];x0,y0,x1,y1=box
        crop=pixels[y0:y1,x0:x1]
        if crop.shape!=np.asarray(ref).shape or not np.array_equal(crop[mask],np.asarray(ref)[mask]):
            raise ValueError('Ownership reference does not exactly match source pixels: '+folder.name)
        owned=np.zeros(pixels.shape[:2],bool);owned[y0:y1,x0:x1]=mask
        annotated=pixels.copy();annotated[binary_dilation(owned,iterations=2)&~owned]=(255,0,255)
        panel=Image.fromarray(annotated);draw=ImageDraw.Draw(panel)
        draw.rectangle([max(0,x0-3),max(0,y0-3),min(source.width-1,x1+2),min(source.height-1,y1+2)],outline=(0,200,255),width=1)
        panel=panel.resize((source.width*2,source.height*2),Image.Resampling.NEAREST)
        masked=np.full_like(crop,127);masked[mask]=crop[mask]
        selected=Image.fromarray(masked);scale=336/max(selected.size)
        selected=selected.resize((round(selected.width*scale),round(selected.height*scale)),Image.Resampling.NEAREST)
        source.save(out/'source_original.png');panel.save(out/'source_ownership_overlay.png')
        Image.fromarray(owned.astype(np.uint8)*255).save(out/'source_ownership_mask.png')
        Image.fromarray(masked).save(out/'selected_owned_pixels_native.png');selected.save(out/'selected_owned_reference.png')
        inputs={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ['source.png','source_reference.png','source_reference_mask.png','guide_manifest.json']}
        artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.png'))}
        row=dict(case_id=folder.name,source_folder=str(folder),source_reference_bbox=box,
            owned_pixel_count=int(mask.sum()),selection_pixels_exact=True,source_input_sha256=inputs,
            input_artifacts_sha256=artifacts,model_inputs=['source_ownership_overlay.png','selected_owned_reference.png'],
            no_food_metadata_in_prompt=True,no_generated_image_input=True,
            unmasked_context_reference_used_as_selected_exemplar=False,
            semantic_status='provisional_source_model_hypothesis_requires_review')
        write(out/'input_provenance.json',row);records.append(row)
    write(args.output/'input_manifest.json',dict(cases=records,prompt_version=CONTEXT_PROMPT_VERSION,prompt_templates=CONTEXT_MATERIAL_PROMPTS,
        all_inputs_source_only=True,automatic_generator_injection=False))
    return records


def context_materials(args,vlm):
    records=prepare_context_materials(args);answers=[]
    for row in records:
        out=args.output/row['case_id'];images=[Image.open(out/n).convert('RGB') for n in row['model_inputs']]
        for template,prompt in CONTEXT_MATERIAL_PROMPTS.items():
            start=time.time();reply=vlm.query(images,prompt,600)
            item=dict(case_id=row['case_id'],template_id=template,prompt_version=CONTEXT_PROMPT_VERSION,prompt=prompt,**reply,
                seconds=time.time()-start,source_only=True,automatic_generator_injection=False,
                semantic_status='provisional_source_model_hypothesis_requires_review')
            write(out/('reply_'+template+'.json'),item);answers.append(item)
            write(args.output/'responses.json',dict(responses=answers,expected_queries=len(records)*2,complete=len(answers)==len(records)*2))
            print(row['case_id'],template,json.dumps(reply['parsed'],ensure_ascii=False),flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--indices', type=int, nargs='+', default=[10, 11])
    p.add_argument('--model-root', type=Path, default=Path('/mnt/tmp/guo-z_first_bite_structure_20260930_models/qwen-image-2.1'))
    p.add_argument('--mode', choices=['propose', 'panels', 'segment', 'individual', 'summarize', 'material', 'material_index', 'measure','context_index','context_material'], default='propose')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode=='context_index':
        prepare_context_materials(args)
        return
    if args.mode == 'segment':
        segment_sources(args)
        return
    if args.mode == 'summarize':
        summarize_sources(args)
        return
    if args.mode == 'material_index':
        index_materials(args)
        return
    if args.mode == 'measure':
        measure_materials(args)
        return
    vlm = SourceVLM(args.model_root)
    if args.mode=='context_material':
        context_materials(args,vlm)
        return
    if args.mode == 'material':
        describe_materials(args, vlm)
        return
    for folder in sorted(args.root.glob('real_*')):
        if int(folder.name.split('_')[1]) not in args.indices:
            continue
        out = args.output / folder.name; out.mkdir(parents=True, exist_ok=True)
        image = Image.open(folder / 'source.png').convert('RGB')
        food = np.asarray(Image.open(folder / 'food_mask.png')) > 0
        start = time.time()
        if args.mode == 'propose':
            answer = vlm.query(image, INVENTORY_PROMPT, 1600)
            write(out / 'source_inventory.json', dict(**answer, source=str(folder / 'source.png'),
                target_image_used=False, prompt=INVENTORY_PROMPT, seconds=time.time()-start,
                semantic_status='unvalidated_source_model_hypothesis'))
            draw_regions(image, answer['parsed']).save(out / 'source_proposals.png')
        elif args.mode == 'individual':
            candidates = make_candidate_panels(image, food, out)
            prompt = '''Inspect only this crop of a food source photograph. Describe the dominant
visible ingredients. Are actual cooked noodle strands the dominant visible material,
or is this mostly toppings such as bonito flakes, egg ribbons, ginger, meat, vegetables,
or uncertain material? Do not use thin/curved/yellow appearance alone as noodle evidence.
Return JSON with dominant_material, visible_ingredients, noodle_identity_confidence
(0..1), noodle_area_fraction (0..1), evidence, and decision (noodle_dominant,
mixed_ingredients, topping_dominant, or uncertain). A noodle_dominant decision requires
more than half the visible food area to be identifiable actual noodles.'''
            replies=[]
            for record in candidates:
                crop = image.crop(record['crop_box_xyxy']).resize((336,336))
                answer = vlm.query(crop, prompt, 300)
                replies.append(dict(**record, **answer))
                write(out/'individual_semantics.json', {'regions':replies,'prompt':prompt,
                    'target_image_used':False, 'seconds':time.time()-start,
                    'semantic_status':'unvalidated_source_model_hypotheses'})
                print(folder.name, record['id'], json.dumps(answer['parsed'], ensure_ascii=False), flush=True)
            continue
        else:
            candidates = make_candidate_panels(image, food, out)
            prompt = '''These are numbered unmodified crops from one source food photo.
Classify every region's dominant VISIBLE ingredients. Specifically distinguish actual
noodles from yellow egg ribbons, bonito/fish flakes, pink ginger, vegetables and meat.
Thin or curved shapes alone are not evidence of noodle identity. Return JSON with
regions, a list of objects with id, dominant_ingredient, noodle_fraction_estimate
(0..1), noodle_identity_confidence (0..1), contamination, and suitable_noodle_bite
(true only if visibly actual noodles dominate with high confidence). Include
best_region_id or null if none can confidently supply a small coherent noodle bite.
Retain uncertain/failing regions in the list. Do not guess hidden ingredient content.'''
            answer = vlm.query(Image.open(out / 'candidate_panels.png'), prompt, 2200)
            write(out / 'panel_semantics.json', dict(**answer, prompt=prompt,
                candidate_count=len(candidates), target_image_used=False, seconds=time.time()-start))
        print(folder.name, json.dumps(answer['parsed'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
