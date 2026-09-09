"""Verify retrieved scientific outputs and compose labelled, unretouched comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for extra in ['cache/day19_build_deps', 'cache/day18_validation_deps']:
    sys.path.insert(0, str(ROOT / extra))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def remote_verify(root, listing):
    records = {}
    for line in listing.read_text(encoding='utf-8').splitlines():
        expected, rel = line.split(None, 1)
        rel = rel.strip().removeprefix('./')
        path = root / rel
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Unsafe listing path')
        actual = digest(path)
        if actual != expected:
            raise ValueError('Transfer mismatch: ' + str(path))
        records[rel] = actual
    local = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if set(records) != local:
        raise ValueError('Remote/local file set mismatch')
    return records

def decode(path):
    cap = cv2.VideoCapture(str(path)); frames = []
    while True:
        ok, frame = cap.read()
        if not ok: break
        frames.append(frame)
    cap.release()
    if len(frames) != 21: raise ValueError('Wrong decoded frame count')
    return np.stack(frames)

def verify_condition(root, record, data):
    for rec in record['files'].values():
        p = root / rec['path']
        if p.stat().st_size != rec['size_bytes'] or digest(p) != rec['sha256']:
            raise ValueError('Condition manifest mismatch')
    case = record['case_id']; arm = record['arm']; condition = record['condition']
    ref = cv2.imread(str(data / case / 'reference.png'))
    alpha = cv2.imread(str(data / case / 'edit_alpha.png'), cv2.IMREAD_GRAYSCALE)
    raw = decode(root / condition / 'raw.mp4')
    lossy = decode(root / condition / 'projected.mp4')
    weight = alpha.astype(np.float32)[None, ..., None] / 255
    preencode = np.rint(raw.astype(np.float32) * weight + ref[None] * (1 - weight)).clip(0,255).astype(np.uint8)
    final = cv2.imread(str(root / condition / 'projected_final_hold.png'))
    if not np.array_equal(preencode[-1], final): raise ValueError('Final PNG not reproducible')
    outside = alpha == 0
    inside = alpha > 0
    diff_raw = np.abs(raw.astype(np.int16) - ref[None].astype(np.int16))
    diff_proj = np.abs(preencode.astype(np.int16) - ref[None].astype(np.int16))
    diff_lossy = np.abs(lossy.astype(np.int16) - ref[None].astype(np.int16))
    if int(diff_proj[:,outside].max()) != 0: raise ValueError('Projection protected-pixel failure')
    return dict(condition=condition, frames=21, size=[ref.shape[1], ref.shape[0]],
                seed=record['seed'], steps=record['steps'],
                source_dataset_case=case, wall_time_seconds=record['wall_time_seconds'],
                support_fraction=float(inside.mean()),
                native_outside_mae=float(diff_raw[:,outside].mean()),
                preencode_outside_max=int(diff_proj[:,outside].max()),
                encoded_outside_mae=float(diff_lossy[:,outside].mean()),
                encoded_outside_max=int(diff_lossy[:,outside].max()),
                preencode_inside_mae=float(diff_proj[:,inside].mean()),
                final_sha256=digest(root / condition / 'projected_final_hold.png'),
                metrics_note='Pixel changes are diagnostics, not semantic scores. Exact outside preservation is imposed by projection.',
                root=str(root))

def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/arial.ttf', size)

def compose(data, selected, out):
    w,h=688,512; margin=24; gap=18; label=42
    headers=['Input image', 'Planar control + VACE', 'Relative 3D control + VACE']
    sheet=Image.new('RGB',(margin*2+3*w+2*gap,130+3*(h+label+gap)+70),'white')
    draw=ImageDraw.Draw(sheet)
    draw.text((margin,18),'FoodStateEdit  Non-noodle paired pilot',font=font(34),fill='black')
    draw.text((margin,63),'Same seed 1 | 20 steps | Final frame 20 / 21 | No LoRA | No retouching',font=font(22),fill='black')
    for col,title in enumerate(headers):draw.text((margin+col*(w+gap),100),title,font=font(24),fill='black')
    for row,case in enumerate(['soup','rice','cake']):
        y=140+row*(h+label+gap)
        source_note='real / previously used' if case != 'cake' else 'synthetic input / pre-cut bite'
        draw.text((margin,y),case.upper()+'   '+source_note,font=font(23),fill='black')
        images=[Image.open(data/case/'reference.png').convert('RGB')]
        for arm in ['planar','relative3d']:
            root=selected.get(case+'__'+arm)
            if root is None:raise ValueError('Cannot publish incomplete comparison')
            images.append(Image.open(root/(case+'__'+arm)/'projected_final_hold.png').convert('RGB'))
        for col,im in enumerate(images):sheet.paste(im,(margin+col*(w+gap),y+label))
        detail=Image.new('RGB',(margin*2+4*w+3*gap,h+145),'white');dd=ImageDraw.Draw(detail)
        dd.text((margin,12),case.upper()+' | input / procedural 3D control / planar result / relative 3D result',font=font(27),fill='black')
        im4=[images[0],Image.open(data/case/'relative3d_final.png').convert('RGB'),images[1],images[2]]
        for col,im in enumerate(im4):detail.paste(im,(margin+col*(w+gap),65))
        dd.text((margin,595),'Control is not a generated result. Frame 20 fixed for both arms. Single-image pilot; no effectiveness claim.',font=font(24),fill='black')
        detail.save(out/(case+'_comparison.png'))
    draw.text((margin,sheet.height-50),'Three single-image pilots; mixed sources. These images do not establish 3D superiority or photorealism.',font=font(22),fill='black')
    sheet.save(out/'non_noodle_comparison.png')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',action='append',type=Path,required=True)
    p.add_argument('--remote-hashes',action='append',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--allow-incomplete',action='store_true')
    args=p.parse_args()
    if len(args.run_root)!=len(args.remote_hashes):raise ValueError('One listing per run')
    args.output_root.mkdir(parents=True,exist_ok=False)
    data=ROOT/'artifacts/day19_multimaterial_dataset_v3';selected={};records=[];runs=[]
    for root,listing in zip(args.run_root,args.remote_hashes):
        hashes=remote_verify(root,listing)
        manifest=json.loads((root/'run_manifest.json').read_text())
        if manifest['pipeline_load_count']!=1:raise ValueError('Pipeline count')
        runs.append(dict(root=str(root),status=manifest['status'],pipeline_load_count=1,
                         verified_file_count=len(hashes),all_file_hashes=hashes))
        for record in manifest['completed_conditions']:
            if record['case_id']=='noodle':continue
            key=record['condition']
            if key in selected:raise ValueError('Duplicate selected condition')
            selected[key]=root
            records.append(verify_condition(root,record,data))
    expected={c+'__'+a for c in ['soup','rice','cake'] for a in ['planar','relative3d']}
    missing=sorted(expected-set(selected))
    if missing and not args.allow_incomplete:raise ValueError('Missing: '+repr(missing))
    if not missing:compose(data,selected,args.output_root)
    report=dict(schema_version='foodstateedit.presentation_verification.v1',runs=runs,conditions=records,
                missing_conditions=missing,complete_non_noodle_pairs=not missing,
                visual_review_required=True,no_generalization_claim=True,
                chosen_frame_policy='Final hold frame 20, fixed before reviewing missing cake relative3d.',
                figure_files={p.name:digest(p) for p in args.output_root.glob('*.png')})
    (args.output_root/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(conditions=len(records),missing=missing,verified_files=sum(x['verified_file_count'] for x in runs))))
if __name__=='__main__':main()
