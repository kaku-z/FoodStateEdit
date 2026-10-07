"""Compile source-only pairs after checking exact selected-source ownership.

Consistency is not semantic verification. Context ingredients and free-form
captions are never copied into generation conditions.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def strings(value):
    if isinstance(value,str):return [value.strip()] if value.strip() else []
    if isinstance(value,(list,tuple)):return [s for item in value for s in strings(item)]
    return []


def clean(value):
    return ' '.join(re.sub(r'[^a-z0-9 ]+',' ',' and '.join(strings(value)).lower()).split())


FAMILIES=[
    ('egg white',r'\begg whites?\b|\balbumen\b'),('egg yolk',r'\byolks?\b'),
    ('noodle',r'\bnoodles?\b|\budon\b|\bramen\b|\bspaghetti\b|\bpasta\b'),
    ('pork',r'\bpork\b|\bbacon\b'),('chicken',r'\bchicken\b'),('beef',r'\bbeef\b'),
    ('fish',r'\bfish\b|\bsalmon\b|\btuna\b|\bsaury\b|\bmackerel\b'),
    ('meat',r'\bmeat\b'),('rice',r'\brice\b'),('cabbage',r'\bcabbage\b'),
    ('eggplant',r'\beggplant\b|\baubergine\b'),('takoyaki',r'\btakoyaki\b|\boctopus balls?\b'),
    ('pancake',r'\bpancake\b|\bokonomiyaki\b'),('carrot',r'\bcarrots?\b'),
    ('mushroom',r'\bmushrooms?\b'),('egg',r'\beggs?\b|\bomelette\b|\bomelet\b'),
    ('nut',r'\balmonds?\b|\bnuts?\b'),('herb',r'\bherbs?\b|\bcilantro\b|\bparsley\b'),
    ('scallion',r'\bscallions?\b|\bgreen onions?\b')]
ANNOTATION=re.compile(r'\b(?:magenta|cyan|selection|selected|outline|rectangle|highlight|bbox|image|annotation)\b',re.I)


def family(value):
    value=clean(value)
    if ANNOTATION.search(value):return None
    found={name for name,pattern in FAMILIES if re.search(pattern,value)}
    if found&{'egg white','egg yolk'}:found.discard('egg')
    if found&{'chicken','beef','pork','fish'}:found.discard('meat')
    return next(iter(found)) if len(found)==1 else None


def agree(a,b):
    x,y=clean(a),clean(b);fx,fy=family(a),family(b)
    if not fx or not fy:return 'food','incompatible or missing component labels'
    if fx==fy:
        if fx in {'egg white','egg yolk'}:return fx,'matching specific egg component'
        if re.search(r'\bskin\b',x) and re.search(r'\bskin\b',y):
            return fx+' skin','shared component family and skin part'
        if fx=='fish' and (re.search(r'\bfin\b|\btail\b',x) or re.search(r'\bfin\b|\btail\b',y)):
            return 'fish surface','shared fish family; fin/tail specificity discarded'
        if fx=='egg' and re.search(r'omelette|omelet',x) and re.search(r'omelette|omelet',y):
            return 'omelette','matching egg preparation'
        return fx,'shared coarse food family'
    if {fx,fy}<={'meat','pork','chicken','beef'} and 'meat' in [fx,fy]:
        return 'cooked meat','generic meat agreement; species unresolved'
    return 'food','incompatible or missing component labels'


TEXTURES={'smooth','glossy','rough','bubbly','frothy','grainy','porous','wrinkled','crinkled',
    'flaky','crispy','blistered','ridged','fibrous','translucent','matte','shiny','speckled'}
COLORS={'white','cream','beige','tan','brown','black','gray','grey','red','pink','orange','yellow','green','blue','purple'}
ATTRIBUTE_MODIFIERS={'and','slightly','somewhat','very','fine','light','dark','pale','golden','off',
    'lightly','surface','texture','appearance','with','visible','small','tiny','spots','specks',
    'patches','a','an','the','colored','colour','color'}


def safe_words(value,vocabulary):
    result=set()
    for phrase in strings(value):
        for fragment in re.split(r'[,;]',phrase):
            words=set(re.findall(r'[a-z]+',fragment.lower()))
            if ANNOTATION.search(fragment) or words-(TEXTURES|COLORS|ATTRIBUTE_MODIFIERS):continue
            result |= words&vocabulary
    return result


def attributes(a,b,key):
    vocabulary=COLORS if key=='visible_colors' else TEXTURES
    return ', '.join(sorted(safe_words(a.get(key),vocabulary)&safe_words(b.get(key),vocabulary))[:4])


def support(a,b):
    x=' '.join(strings(a.get('visible_surroundings'))).lower()
    y=' '.join(strings(b.get('visible_surroundings'))).lower()
    for pattern,caption in [(r'\bpan\b|\bgriddle\b','in the existing pan'),
        (r'\bplate\b','on the existing plate'),(r'\btray\b','on the existing tray'),
        (r'\bbowl\b','in the existing bowl')]:
        if re.search(pattern,x) and re.search(pattern,y):return caption
    return 'in its original setting'


def source_surface(a,b):
    x=' '.join(strings(a.get('visible_surroundings'))).lower()
    y=' '.join(strings(b.get('visible_surroundings'))).lower()
    for pattern,name in [(r'\bbroth\b','broth surface'),(r'\bpan\b|\bgriddle\b','pan surface'),
        (r'\bplate\b','plate surface'),(r'\btray\b','tray surface'),(r'\bbowl\b','bowl surface')]:
        if re.search(pattern,x) and re.search(pattern,y):
            return 'A close-up photograph of the continuous original '+name+' with natural texture and lighting.'
    return 'A close-up photograph of the continuous original underlying surface with natural texture and lighting.'


def measured_color(measurement):
    L,a,b=[float(x) for x in measurement['owned_pixels']['lab_mean']]
    if not all(math.isfinite(x) for x in (L,a,b)):raise ValueError('Nonfinite measured source color')
    C=math.hypot(a,b);h=math.degrees(math.atan2(b,a))%360
    if C<10:
        color='off-white' if L>=80 else 'light gray' if L>=65 else 'gray' if L>=40 else 'dark gray' if L>=22 else 'near-black'
    elif h<25 or h>=350:color='reddish brown' if L<58 else 'muted pink' if C<30 else 'red'
    elif h<70:color='brown' if L<60 else 'tan' if C<35 else 'orange-brown'
    elif h<115:color='brown' if L<55 else 'cream' if L>=76 and C<35 else 'beige' if C<35 else 'golden yellow'
    elif h<195:color='olive green' if L<65 else 'pale green'
    elif h<260:color='blue-green'
    elif h<320:color='blue-purple'
    else:color='muted pink' if C<30 else 'pink'
    if 10<=C<25 and not color.startswith(('muted','pale','near')):color='muted '+color
    return color


def confidence(value):
    try:result=float(value)
    except (ValueError,TypeError):return None
    return result if math.isfinite(result) and 0<=result<=1 else None


def validate_inputs(case_id,paths,provenance_path,measurement_path):
    provenance=read(provenance_path);measurement=read(measurement_path)
    if provenance.get('case_id')!=case_id or measurement.get('case')!=case_id:raise ValueError('Case provenance mismatch')
    if not provenance.get('selection_pixels_exact') or not provenance.get('no_generated_image_input'):
        raise ValueError('Exact source-only ownership provenance required')
    source=Path(provenance['source_folder']);hashes=provenance['source_input_sha256']
    for name,digest in hashes.items():
        if sha(source/name)!=digest:raise ValueError('Changed source input: '+str(source/name))
    if measurement.get('source_sha256')!=hashes['source_reference.png'] or measurement.get('mask_sha256')!=hashes['source_reference_mask.png']:
        raise ValueError('Measured colors belong to different source ownership')
    if not measurement.get('source_only') or measurement.get('vlm_used') or measurement.get('target_image_used'):
        raise ValueError('Measured colors must come directly from source pixels')
    if measurement['owned_pixels']['pixel_count']!=provenance['owned_pixel_count']:
        raise ValueError('Owned source pixel count mismatch')
    for path in paths:
        reply=read(path)
        if reply.get('case_id')!=case_id or not reply.get('source_only'):raise ValueError('Reply is not bound to source case')
        if Path(path).parent.resolve()!=Path(provenance_path).parent.resolve():raise ValueError('Reply/provenance folder mismatch')
    if {read(path).get('template_id') for path in paths}!={'A','B'}:raise ValueError('Both neutral templates are required')
    for name,digest in provenance.get('input_artifacts_sha256',{}).items():
        if sha(Path(provenance_path).parent/name)!=digest:raise ValueError('Changed model source input artifact')
    return provenance,measurement


def compile_case(folder=None,*,case_id=None,paths=None,provenance_path=None,measurement_path=None,strict=False):
    if paths is None:
        folder=Path(folder);paths=[folder/'reply_A.json',folder/'reply_B.json'];case_id=case_id or folder.name
    paths=[Path(p) for p in paths];replies=[read(p) for p in paths]
    if len(replies)!=2 or any(not isinstance(r.get('parsed'),dict) for r in replies):raise ValueError('Expected two parsed reply objects')
    provenance=None;measurement=None
    if provenance_path and measurement_path:
        provenance,measurement=validate_inputs(case_id,paths,provenance_path,measurement_path)
    elif strict:raise ValueError('Manifest compilation requires exact ownership and measured-color provenance')
    a,b=[r['parsed'] for r in replies];label,reason=agree(a.get('component_label'),b.get('component_label'))
    scores=[confidence(r.get('confidence')) for r in (a,b)]
    consistent=reason!='incompatible or missing component labels'
    if any(v is not None and v<.5 for v in scores):consistent=False;label='food';reason='low self-reported confidence'
    color=measured_color(measurement) if measurement else attributes(a,b,'visible_colors')
    texture=attributes(a,b,'visible_textures') if consistent else ''
    appearance=('The visible material has '+color+' coloring and the same tonal variation as the source reference. ') if color else 'The material retains its observed source colors. '
    if texture:appearance+='Its visible surface is '+texture+'. '
    head=(f'A close-up natural food photograph of a small bite of {label} supported inside a shallow stainless steel spoon, '
        'clearly lifted above the original meal. '+appearance+'The bite matches the source material in the reference and the arrangement in the structural guide. '
        'Soft natural food shading, realistic silver reflections, and the original camera view and surrounding scene.')
    source=(f'A close-up natural food photograph of {label} {support(a,b)}. '+appearance+
        'The visible surrounding surface and remaining material have the appearance and lighting of the original photograph.')
    return dict(head=head,source=source,source_surface=source_surface(a,b),component_label=label,component_consistency=consistent,
        consistency_reason=reason,raw_labels=[a.get('component_label'),b.get('component_label')],
        self_reported_confidences=scores,component_family=family(label),requires_topology_uncertainty=not consistent,
        inputs={str(p):sha(p) for p in paths},source_only=True,oracle=False,ground_truth_verified=False,
        nearby_ingredients_injected=False,freeform_context_caption_injected=False,
        unseen_interior_appearance_claimed=False,source_cavity_realism_verified=False,
        ownership_provenance_verified=provenance is not None,
        source_ownership=None if provenance is None else dict(source_folder=provenance['source_folder'],
            source_reference_bbox=provenance['source_reference_bbox'],owned_pixel_count=provenance['owned_pixel_count'],
            source_input_sha256=provenance['source_input_sha256'],provenance_file=str(provenance_path),provenance_sha256=sha(provenance_path)),
        measured_colors=None if measurement is None else dict(path=str(measurement_path),sha256=sha(measurement_path),
            color_phrase=color,rgb_mean=measurement['owned_pixels']['rgb_mean'],lab_mean=measurement['owned_pixels']['lab_mean'],
            lab_chroma_percentiles=measurement['owned_pixels']['lab_chroma_percentiles']),
        fallback_appearance='measured source colors; no VLM texture or component identity' if not consistent and measurement else None)


def main():
    p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--input',type=Path);group.add_argument('--manifest',type=Path)
    p.add_argument('--measurements',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve prior caption outputs; choose a new output path')
    cases={}
    if args.manifest:
        manifest=read(args.manifest)
        for row in manifest['cases']:
            case=row['case_id']
            if case in cases:raise ValueError('Duplicate manifest case: '+case)
            measured=row.get('measured_material_path') or (args.measurements/case/'measured_material.json' if args.measurements else None)
            cases[case]=compile_case(case_id=case,paths=row['reply_paths'],
                provenance_path=row['source_input_provenance'],measurement_path=measured,strict=True)
            if row.get('resolved_guide_folder'):
                if not row.get('ownership_equivalence',{}).get('exact_source_RGB_and_full_ownership_mask'):
                    raise ValueError('Resolved source ownership equivalence missing')
                for name,digest in row['resolved_guide_sha256'].items():
                    if sha(Path(row['resolved_guide_folder'])/name)!=digest:raise ValueError('Resolved source guide changed')
                cases[case]['resolved_source_guide']=dict(folder=row['resolved_guide_folder'],sha256=row['resolved_guide_sha256'],
                    ownership_equivalence=row['ownership_equivalence'])
        if len(cases)!=manifest.get('expected_cases',len(cases)):raise ValueError('Incomplete source case manifest')
    else:
        for folder in sorted(args.input.glob('real_*')):
            if not all((folder/('reply_'+t+'.json')).exists() for t in ('A','B')):continue
            measured=args.measurements/folder.name/'measured_material.json' if args.measurements else None
            cases[folder.name]=compile_case(folder,provenance_path=folder/'input_provenance.json' if measured else None,measurement_path=measured)
    cases['_metadata']=dict(method='source-only paired component agreement with exact ownership and measured colors',
        oracle=False,automatic=True,unselected_ingredient_names_used=False,model_consistency_is_not_ground_truth=True,
        source_manifest=None if args.manifest is None else str(args.manifest),
        source_manifest_sha256=None if args.manifest is None else sha(args.manifest),compiler_sha256=sha(__file__))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(cases=len(cases)-1,uncertain=[k for k,v in cases.items() if k!='_metadata' and not v['component_consistency']])))


if __name__=='__main__':main()

