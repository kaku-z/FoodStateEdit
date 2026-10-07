"""Freeze additional source-only validation inputs in an existing hash order.

No models, generated images, subjective quality ranking, or output selectors are
used. Existing selections and their historical source records remain unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image, ImageOps

PREFIX = 'mld-real-probe-20261004:'
FORMAT = 'mld4.frozen_additional_source_holdout.v1'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def rank_hash(name):
    return hashlib.sha256((PREFIX+name).encode('utf-8')).hexdigest()


def decode(path):
    with Image.open(path) as im: im.verify()
    with Image.open(path) as im: return ImageOps.exif_transpose(im).convert('RGB').copy()


def dhash(image):
    pixels=np.asarray(image.convert('L').resize((9,8),Image.Resampling.LANCZOS))
    result=0
    for bit in (pixels[:,1:]>pixels[:,:-1]).reshape(-1): result=(result<<1)|int(bit)
    return result


def optional_prompt(metadata):
    caption=metadata.get('text')
    match=re.fullmatch(r'A high quality food photography of (.+) on a plate, UECFood style\.',caption or '')
    labels=[part.strip() for part in match.group(1).split(',')] if match else []
    # The first provided label is used without image-driven reinterpretation.
    first=labels[0] if labels else None
    food_prompt=first if first and first.lower() not in {'other','others'} else 'food'
    return dict(food_prompt=food_prompt, metadata_caption=caption,
        metadata_food_labels=labels, food_prompt_origin='first_caption_label' if food_prompt!='food' else 'generic_food_fallback',
        caption_used_to_construct_optional_prompt=food_prompt!='food',
        caption_used_for_selection=False, caption_supplied_to_model_at_freeze=False,
        rule='Parse the existing exact caption template; use its first comma-separated label unless other/others, else food. No VLM label or manual case label.')


def references(snapshots):
    prior=read(snapshots/'prior_manifest.json')['cases']
    history=read(snapshots/'history_sources_excluded.json')
    historical=read(snapshots/'historical_dhash_reference.json')
    names=set(history['source_file_names'])|{r['source_file_name'] for r in prior}
    hashes=set(history['known_source_sha256'])|{r['source_sha256'] for r in historical['references']}|{r['source_sha256'] for r in prior}
    near=[(r['file_name'],int(r['dhash64'],16),'historical') for r in historical['references']]
    near += [(r['source_file_name'],int(r['dhash64'],16),'prior_development') for r in prior]
    return names,hashes,near


def select(source, snapshots, count, threshold):
    names,hashes,near=references(snapshots)
    candidates=read(snapshots/'candidate_order.json')['candidates']
    selected=[];decisions=[]
    for row in candidates:
        name=row['file_name'];decision=dict(candidate_rank=row['rank'],file_name=name,rank_sha256=row['rank_sha256'])
        if name in names:
            decision['status']='rejected_prior_or_historical_filename';decisions.append(decision);continue
        path=source/name
        try:
            image=decode(path);digest=sha256(path)
        except (OSError,ValueError) as exc:
            decision.update(status='rejected_invalid_decode',error=str(exc));decisions.append(decision);continue
        decision['source_sha256']=digest
        if digest in hashes:
            decision['status']='rejected_prior_historical_or_selected_exact_duplicate';decisions.append(decision);continue
        dh=dhash(image)
        distance,nearest,scope=min((((dh^rhash).bit_count(),rname,scope) for rname,rhash,scope in near),default=(65,None,None))
        decision.update(dhash64=f'{dh:016x}',nearest_reference_file=nearest,
            nearest_reference_scope=scope,nearest_reference_hamming_distance=distance)
        if distance<=threshold:
            decision['status']='rejected_prior_historical_or_selected_near_duplicate';decisions.append(decision);continue
        case_id=f'holdout_{len(selected):02d}_{Path(name).stem}'
        decision.update(status='selected',case_id=case_id)
        decisions.append(decision);selected.append(dict(decision))
        names.add(name);hashes.add(digest);near.append((name,dh,'new_holdout'))
        if len(selected)==count:break
    return selected,decisions


def prepare(args):
    out=args.output
    if out.exists() and any(out.iterdir()):raise FileExistsError('Never overwrite a frozen holdout; output must be empty')
    prior_recipe=read(args.prior/'selection_recipe.json');prior_manifest=read(args.prior/'manifest.json')
    metadata_path=args.source_directory/'metadata.jsonl'
    history_path=Path(prior_recipe['history_path'])
    required={'metadata':(metadata_path,prior_recipe['metadata_sha256']),
        'candidate_order':(args.prior/'candidate_order.json',prior_recipe['candidate_order_sha256']),
        'history':(history_path,prior_recipe['history_sha256']),
        'prior_recipe':(args.prior/'selection_recipe.json',prior_manifest['selection_recipe_sha256'])}
    for name,(path,digest) in required.items():
        if sha256(path)!=digest:raise ValueError(f'Prior frozen {name} hash changed')
    candidates=read(args.prior/'candidate_order.json')['candidates']
    metadata=[json.loads(x) for x in metadata_path.read_text(encoding='utf-8').splitlines() if x.strip()]
    by_name={r['file_name']:r for r in metadata}
    if len(by_name)!=len(metadata):raise ValueError('Ambiguous duplicate metadata filenames')
    if set(by_name)!={r['file_name'] for r in candidates}:raise ValueError('Prior candidate pool differs from metadata')
    for i,row in enumerate(candidates):
        name=row['file_name']
        if Path(name).name!=name or name in ('','.', '..'):raise ValueError('Unsafe source basename')
        if row['rank']!=i or row['rank_sha256']!=rank_hash(name):raise ValueError('Prior hash order inconsistent')
    if candidates!=sorted(candidates,key=lambda r:(r['rank_sha256'],r['file_name'])):raise ValueError('Unsorted prior candidates')
    threshold=prior_recipe['dhash']['max_distance_rejected']
    out.mkdir(parents=True,exist_ok=True);snapshots=out/'input_snapshots';snapshots.mkdir()
    snapshot_sources={'candidate_order.json':args.prior/'candidate_order.json',
        'historical_dhash_reference.json':args.prior/'historical_dhash_reference.json',
        'prior_selection_recipe.json':args.prior/'selection_recipe.json',
        'prior_selection_decisions.json':args.prior/'selection_decisions.json',
        'prior_manifest.json':args.prior/'manifest.json','history_sources_excluded.json':history_path,
        'metadata.jsonl':metadata_path}
    for name,path in snapshot_sources.items():shutil.copyfile(path,snapshots/name)
    names,hashes,near=references(snapshots)
    recipe=dict(format_version=FORMAT,status='rules_frozen_before_any_new_candidate_image_decode',
        created_utc=utc(),count=args.count,source_directory=str(args.source_directory),
        source_code_sha256=sha256(__file__),prior_case_count=len(prior_manifest['cases']),
        rank_rule=prior_recipe['rank_rule'],rank_prefix=PREFIX,
        max_dhash_distance_rejected=threshold,dhash=prior_recipe['dhash'],
        excluded_filename_count=len(names),excluded_exact_sha256_count=len(hashes),near_reference_count=len(near),
        input_snapshots={name:{'original_path':str(path),'sha256':sha256(snapshots/name)} for name,path in snapshot_sources.items()},
        normalization=dict(exif_transpose=True,color_mode='RGB',maximum_edge=args.max_edge,upscale=False,padding=False,interpolation='Pillow LANCZOS'),
        selection_rules=['Reuse original ascending filename SHA256 order','Exclude recorded historical and prior development filenames',
            'Require valid original image decode','Exclude historical, prior development and earlier holdout exact SHA256 matches',
            f'Exclude dHash64 Hamming distance <= {threshold} to any available historical, prior development or earlier holdout source'],
        quality_filter='Decode validity only; no visual cherry-picking',caption_selection=False,
        optional_prompt_policy='Metadata first label may be explicitly supplied by downstream caller; not inferred from VLM and not used for selection',
        generated_outputs_read=False,models_loaded=False,gpu_used=False,
        scope='Additional source holdout independent of recorded development/historical selections under these filters, frozen before any generated outputs for these selected sources.',
        limitations='dHash detects some near duplicates only; recorded history may not cover all prior human viewing or foundation-model training. No paired after-images, measured geometry, or success labels.')
    write(out/'selection_recipe.json',recipe)
    chosen,decisions=select(args.source_directory,snapshots,args.count,threshold)
    write(out/'selection_decisions.json',{'decisions':decisions})
    if len(chosen)!=args.count:raise RuntimeError(f'Only {len(chosen)} eligible sources; expected {args.count}')
    cases=[];pipeline=[]
    for selected in chosen:
        name=selected['file_name'];case=out/selected['case_id'];case.mkdir()
        path=args.source_directory/name;original=case/('original'+path.suffix.lower())
        shutil.copyfile(path,original)
        if sha256(original)!=selected['source_sha256']:raise RuntimeError('Original copy changed source bytes')
        image=decode(path);original_size=list(image.size)
        image.thumbnail((args.max_edge,args.max_edge),Image.Resampling.LANCZOS)
        image.save(case/'source.png')
        prompt=optional_prompt(by_name[name]);write(case/'food_prompt.json',prompt)
        record=dict(selected,selected_index=len(cases),source_file_name=name,original_source_path=str(path),
            original_path=str(original.relative_to(out)),original_sha256=sha256(original),
            image=str((case/'source.png').relative_to(out)),source_png_sha256=sha256(case/'source.png'),
            original_size_wh=original_size,normalized_size_wh=list(image.size),normalization=recipe['normalization'],
            food_prompt=prompt['food_prompt'],food_prompt_origin=prompt['food_prompt_origin'],
            metadata_caption=prompt['metadata_caption'],metadata_food_labels=prompt['metadata_food_labels'],
            caption_used_for_selection=False,caption_used_to_construct_optional_prompt=prompt['caption_used_to_construct_optional_prompt'],
            supervision='Source image only; no edited target, hidden material, geometry label or success label')
        write(case/'source_record.json',record);cases.append(record)
        pipeline.append(dict(case_id=record['case_id'],image=record['image'],
            image_absolute=str(case/'source.png'),food_prompt=record['food_prompt'],
            metadata_prompt_used=record['caption_used_to_construct_optional_prompt'],
            source_record=f"{record['case_id']}/source_record.json"))
    write(out/'raw_pipeline_inputs.json',{'format_version':FORMAT,'base_directory':str(out),
        'inputs':pipeline,'generated_outputs_present':False,'caption_supplied_to_model_at_freeze':False})
    files=[{'path':str(p.relative_to(out)),'sha256':sha256(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()]
    manifest=dict(format_version=FORMAT,status='frozen_before_holdout_generation',freeze_completed_utc=utc(),
        selected_count=len(cases),cases=cases,selection_recipe_sha256=sha256(out/'selection_recipe.json'),
        selection_decisions_sha256=sha256(out/'selection_decisions.json'),
        generated_outputs_read=False,models_loaded=False,gpu_used=False,files=files,
        scope=recipe['scope'],limitations=recipe['limitations'])
    write(out/'manifest.json',manifest)
    write(out/'freeze_lock.json',dict(format_version=FORMAT,frozen_utc=utc(),
        manifest_sha256=sha256(out/'manifest.json'),selection_recipe_sha256=manifest['selection_recipe_sha256'],
        selected_source_filenames=[r['source_file_name'] for r in cases],selected_source_sha256=[r['source_sha256'] for r in cases],
        generation_must_use_all_frozen_cases=True,no_reselection_by_output_quality=True))
    return manifest


def verify(args):
    out=args.output;manifest=read(out/'manifest.json');lock=read(out/'freeze_lock.json');recipe=read(out/'selection_recipe.json')
    errors=[]
    if sha256(out/'manifest.json')!=lock['manifest_sha256']:errors.append('manifest_lock_mismatch')
    for row in manifest['files']:
        if sha256(out/row['path'])!=row['sha256']:errors.append('artifact_hash:'+row['path'])
    chosen,decisions=select(args.source_directory,out/'input_snapshots',recipe['count'],recipe['max_dhash_distance_rejected'])
    if decisions!=read(out/'selection_decisions.json')['decisions']:errors.append('deterministic_selection_replay_mismatch')
    if [r['file_name'] for r in chosen]!=lock['selected_source_filenames']:errors.append('selected_identity_replay_mismatch')
    for row in manifest['cases']:
        im=decode(out/row['original_path']);im.thumbnail((recipe['normalization']['maximum_edge'],)*2,Image.Resampling.LANCZOS)
        if not np.array_equal(np.asarray(im),np.asarray(Image.open(out/row['image']).convert('RGB'))):errors.append('normalization_pixel_mismatch:'+row['case_id'])
    result=dict(status='passed' if not errors else 'failed',verified_utc=utc(),errors=errors,
        selected_count=manifest['selected_count'],deterministic_selection_replayed=True,original_and_normalized_hashes_verified=True,
        normalized_pixels_recomputed=True,models_loaded=False,gpu_used=False,generated_outputs_read=False)
    write(out/'verification.json',result)
    if errors:raise RuntimeError(json.dumps(result))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory',type=Path,required=True)
    parser.add_argument('--prior',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--count',type=int,default=8)
    parser.add_argument('--max-edge',type=int,default=640)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.count<1 or args.max_edge<8:raise ValueError('Invalid count or maximum edge')
    if not args.verify and args.prior is None:parser.error('--prior required when freezing')
    result=verify(args) if args.verify else prepare(args)
    print(json.dumps({'status':result['status'],'selected_count':result['selected_count'],
        'selected_sources':[r['source_file_name'] for r in result.get('cases',[])]}),flush=True)


if __name__=='__main__':main()
