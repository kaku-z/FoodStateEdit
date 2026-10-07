"""Observe visible scene support independently of selected ingredient identity."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation


SCHEMA = '''Return JSON with support_candidates, a list of objects having category
(plate, pan, bowl, tray, board, table, broth, sauce, or unknown), visibly_exposed
(boolean), observed_colors (list of short color words), observed_materials
(list of short material words), confidence (0..1), and visible_support_evidence
(one short sentence locating the actually visible support). Also return
uncertainty (short sentence). Mention no ingredient names or unseen interiors.
Annotations are not physical surfaces. Do not infer the appearance hidden
behind food. Physical material can be unknown when appearance is insufficient.'''
PROMPTS = {
    'A': '''Image 1 is an original photograph. Image 2 is the same photograph with
a magenta outline marking an automatic food-region hypothesis. The mask may
include several components or errors. Identify the visible physical support
holding the outlined food: its serving container and any visibly exposed
supporting liquid. Describe the support only, not the food. Use the whole
original image to locate the main container and its exposed surface. If broth
is visible within a bowl, describe both the bowl and broth separately. Report
only directly visible color/material evidence; be uncertain if not visible.
Do not describe an edited or imagined image. '''+SCHEMA,
    'B': '''Study the full original scene in image 1; image 2 adds an automatic
food-boundary outline only to locate the relevant region. What observable
container or supporting surface holds that region? Distinguish the serving
surface from the food resting on it. Account separately for a visible liquid
surface and the vessel containing it. Do not fill missing evidence with food
identity or guesses about covered areas. List compatible support candidates
using the exposed portions of the original scene, their visible color and
apparent material, and a short location as evidence. '''+SCHEMA}
CATEGORIES = {'plate','pan','bowl','tray','board','table','broth','sauce'}
COLORS = {'white','cream','beige','tan','brown','black','gray','red','pink','orange','yellow','green','blue','purple','silver'}
MATERIALS = {'ceramic','porcelain','metal','iron','steel','plastic','wood','paper','stone','glass','liquid'}
MODIFIERS = {'light','dark','pale','off','and','with','a','an','the','colored','colour','color','cast','stainless'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def prepare(args):
    root=args.root.resolve();out=root/'semantic_probe/support_context_v2'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);scripts=out/'snapshot';scripts.mkdir()
    for name in ['prepare_mld4_support_semantics.py','diagnose_mld4_semantics.py']:
        shutil.copy2(Path(__file__).parent/name,scripts/name)
    routing=read(args.routing);cases=[]
    for row in routing['cases']:
        case=row['case_id'];geometry=Path(row['resolved_geometry_case']);guide=Path(row['production_guides']['boundary'])
        source=Image.open(geometry/'source.png').convert('RGB');pixels=np.array(source)
        if not np.array_equal(pixels,np.array(Image.open(guide/'source.png').convert('RGB'))):
            raise ValueError('Source mismatch: '+case)
        food=np.array(Image.open(geometry/'source_food_mask.png').convert('L'))>127
        overlay=pixels.copy();overlay[binary_dilation(food,iterations=2)&~food]=(255,0,255)
        folder=out/case;folder.mkdir();source.save(folder/'original.png');Image.fromarray(overlay).save(folder/'outlined.png')
        shutil.copy2(geometry/'source_food_mask.png',folder/'source_food_mask.png')
        record=dict(case_id=case,source_path=str(geometry/'source.png'),source_sha256=sha(geometry/'source.png'),
            mask_path=str(geometry/'source_food_mask.png'),mask_sha256=sha(geometry/'source_food_mask.png'),
            mask_semantics_verified=False,source_only=True,generated_target_used=False,
            model_inputs={str(folder/n):sha(folder/n) for n in ['original.png','outlined.png']})
        write(folder/'provenance.json',record);cases.append(record)
    manifest=dict(cases=cases,templates=PROMPTS,source_only=True,expected_queries=2*len(cases),
        model_root=str(args.model_root),model_config_sha256=sha(args.model_root/'config.json'),
        snapshot_sha256={str(p):sha(p) for p in scripts.iterdir()},routing_sha256=sha(args.routing))
    write(out/'manifest.json',manifest);print('Prepared',len(cases),'source-only support cases',flush=True)


def query(args):
    out=args.root.resolve()/'semantic_probe/support_context_v2';manifest=read(out/'manifest.json')
    for p,digest in manifest['snapshot_sha256'].items():
        if sha(p)!=digest:raise ValueError('Source snapshot changed')
    for row in manifest['cases']:
        for p,digest in row['model_inputs'].items():
            if sha(p)!=digest:raise ValueError('Source image changed')
    from diagnose_mld4_semantics import SourceVLM
    vlm=SourceVLM(Path(manifest['model_root']));completed=[]
    for row in manifest['cases']:
        folder=out/row['case_id'];images=[Image.open(folder/n).convert('RGB') for n in ['original.png','outlined.png']]
        for template,prompt in manifest['templates'].items():
            destination=folder/('reply_'+template+'.json')
            if destination.exists():raise FileExistsError(destination)
            started=time.time();reply=vlm.query(images,prompt,700)
            record=dict(case_id=row['case_id'],template_id=template,prompt=prompt,**reply,
                seconds=time.time()-started,source_only=True,generated_image_used=False,
                provenance_sha256=sha(folder/'provenance.json'),support_semantics_ground_truth=False)
            write(destination,record);completed.append(str(destination))
            write(out/'worker.json',dict(status='running',completed=len(completed),expected=manifest['expected_queries']))
            print(row['case_id'],template,json.dumps(reply['parsed'],ensure_ascii=False),flush=True)
    write(out/'worker.json',dict(status='complete',completed=len(completed),expected=manifest['expected_queries']))


def words(value,allowed):
    values=[value] if isinstance(value,str) else value if isinstance(value,list) else []
    result=set()
    for item in values:
        if not isinstance(item,str):continue
        tokens=set(re.findall('[a-z]+',item.lower().replace('grey','gray').replace('wooden','wood').replace('metallic','metal')))
        if not tokens-(COLORS|MATERIALS|MODIFIERS):result|=tokens&allowed
    return result


def candidates(parsed):
    result={}
    if not isinstance(parsed,dict):return result
    for item in parsed.get('support_candidates',[]) or []:
        if not isinstance(item,dict):continue
        category=str(item.get('category','')).strip().lower()
        try:confidence=float(item.get('confidence',0))
        except (ValueError,TypeError):continue
        if category in CATEGORIES and item.get('visibly_exposed') is True and confidence>=.5:
            if category not in result:result[category]=item
    return result


def compile_pair(a,b):
    aa,bb=candidates(a),candidates(b);common=set(aa)&set(bb)
    # A visible supporting liquid is more immediate than its surrounding vessel.
    chosen=next((c for c in ['broth','sauce','pan','bowl','plate','tray','board','table'] if c in common),None)
    if chosen is None:return dict(caption='A natural photograph of an empty underlying surface, with a continuous exposed surface and the original scene lighting.',category=None,paired_agreement=False,colors=[],materials=[])
    colors=sorted(words(aa[chosen].get('observed_colors'),COLORS)&words(bb[chosen].get('observed_colors'),COLORS))[:3]
    shared_materials=words(aa[chosen].get('observed_materials'),MATERIALS)&words(bb[chosen].get('observed_materials'),MATERIALS)
    materials=sorted(shared_materials)
    if len(materials)>1:
        if shared_materials <= {'metal','iron','steel'}:materials=[next(v for v in ['steel','iron','metal'] if v in shared_materials)]
        elif shared_materials <= {'ceramic','porcelain'}:materials=['porcelain' if 'porcelain' in shared_materials else 'ceramic']
        else:materials=[]
    material=' '.join(materials)
    # Keep only observed attributes of the selected support, never free-form evidence.
    description=' '.join(v for v in [', '.join(colors),material,chosen] if v)
    if chosen in {'broth','sauce'}:
        container='a bowl' if 'bowl' in common else 'the existing container'
        liquid=' '.join(v for v in [', '.join(colors),chosen] if v)
        caption='A natural photograph of the continuous surface of '+liquid+' in '+container+', with the original container and lighting.'
    else:
        caption='A natural photograph of an empty '+description+', with a continuous exposed surface and the original scene lighting.'
    return dict(caption=caption,category=chosen,paired_agreement=True,compatible_categories=sorted(common),colors=colors,materials=materials,
        incompatible_shared_materials_omitted=sorted(shared_materials) if shared_materials and not materials else [])


def compile_all(args):
    out=args.root.resolve()/'semantic_probe/support_context_v2';manifest=read(out/'manifest.json');captions=read(args.captions)
    if args.output.exists():raise FileExistsError(args.output)
    records=[]
    for row in manifest['cases']:
        case=row['case_id'];folder=out/case;paths=[folder/('reply_'+t+'.json') for t in ['A','B']];replies=[read(p) for p in paths]
        if any(r.get('generated_image_used') or not r.get('source_only') or r.get('provenance_sha256')!=sha(folder/'provenance.json') for r in replies):
            raise ValueError('Source support provenance mismatch')
        compiled=compile_pair(*[r['parsed'] for r in replies])
        c=captions[case];c['source_surface_previous']=c['source_surface'];c['source_surface']=compiled['caption'];c['source_surface_v2']=compiled['caption']
        c['support_context_v2']=dict(**compiled,inputs={str(p):sha(p) for p in paths},
            source_only=True,ground_truth_verified=False,unseen_support_verified=False,
            empty_support_is_declared_counterfactual_target=True,
            evidence_injected=False,ingredient_names_injected=False)
        records.append(dict(case_id=case,**compiled))
    captions['_metadata']['support_surface_revision']=dict(version='whole_scene_visible_support_paired_v2',
        source_captions_sha256=sha(args.captions),support_manifest_sha256=sha(out/'manifest.json'),
        compiler_sha256=sha(__file__),source_only=True,hidden_surface_ground_truth=False)
    write(args.output,captions);write(out/'compiled_support.json',dict(cases=records,caption_sha256=sha(args.output)))
    print(json.dumps(dict(path=str(args.output),sha256=sha(args.output),resolved=sum(r['paired_agreement'] for r in records),cases=len(records))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('--root',type=Path,required=True);p.add_argument('--routing',type=Path,required=True)
    p.add_argument('--model-root',type=Path,default=Path('/host/space0/guo-z/FluxSAM-Seg/models/GLM-4.6V-Flash'));p.set_defaults(function=prepare)
    p=sub.add_parser('query');p.add_argument('--root',type=Path,required=True);p.set_defaults(function=query)
    p=sub.add_parser('compile');p.add_argument('--root',type=Path,required=True);p.add_argument('--captions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.set_defaults(function=compile_all)
    args=parser.parse_args();args.function(args)
