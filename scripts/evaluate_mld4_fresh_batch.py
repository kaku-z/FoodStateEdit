"""Read-only experimental audit and portable review export for fresh RGB runs.

Visual outcomes must be recorded after inspection, separately from these exact
implementation invariants. Can export completed cases while other jobs run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import distance_transform_edt

from evaluate_mld4_production import evaluate as audit_generation


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rgb(p):
    return np.asarray(Image.open(p).convert('RGB'))


def write(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def audit_geometry(case):
    route=read(case/'routing_resolved_manifest.json')['cases'][0]
    folder=Path(route['resolved_geometry_case']); state=read(folder/'state.json')
    raw=read(Path(state['source_observation'])/'geometry_raw.json')
    with np.load(folder/'transport.npz') as z:
        rest=z['rest_xyz']; moved=z['target_xyz']; faces=z['faces']
        edge=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
        _,counts=np.unique(edge,axis=0,return_counts=True)
        def volume(vertices):
            tri=(vertices.astype(float)-vertices.mean(0))[faces]
            return abs(float(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6))
        vr,vt=volume(rest),volume(moved)
        checks=dict(source_ID_partition=np.array_equal(np.sort(np.r_[z['carried_ids'],z['remaining_ids']]),np.sort(z['source_id'])),
            carried_target_ID_equal=np.array_equal(z['carried_ids'],z['target_ids']),
            material_UV_equal=np.array_equal(z['source_uv'],z['target_source_uv']),
            material_RGB_equal=np.array_equal(z['source_material_rgb'],z['target_material_rgb']),
            finite_geometry=bool(np.isfinite(rest).all() and np.isfinite(moved).all()),
            closed_edge_incidence=bool(np.all(counts==2)))
    return dict(mode=state['mode'],checks=checks,all_checks=all(checks.values()),
        computed_proxy_volume_before=vr,computed_proxy_volume_after=vt,
        computed_proxy_volume_relative_change=abs(vt/vr-1) if vr else None,
        prior_diagnostics=raw.get('hidden_prior_diagnostics'),
        prior_applies_to_final_closure=state['mode']!='strand',
        frame_connection=state['spoon'].get('frame_connection'),
        automatic_source_label=route['component_label'],automatic_label_is_truth=False,
        scope='Actual saved proxy vertices/IDs/UV/colors, not measured physical mass or generated-image identity.')


def evaluate(root):
    protocol=read(root/'protocol.json')
    out=root/'evaluation'; out.mkdir(exist_ok=True)
    outer_code_bad=[n for n,h in protocol.get('code_hashes',{}).items() if sha(root/'code'/n)!=h]
    rows=[]
    for item in protocol['cases']:
        cid=item['case_id']; case=root/'cases'/cid
        state=read(case/'status.json') if (case/'status.json').exists() else {'status':'pending'}
        row=dict(case_id=cid,index=item['index'],status=state['status'],stage=state.get('stage'),
                 food_prompt=item['food_prompt'],group='regression_previously_observed')
        if state['status']!='complete':
            row['error']=state.get('error'); rows.append(row); continue
        production=case/'production'
        audit_generation(argparse.Namespace(production=production,
            protocol_sha256=state['production_protocol_sha256'],require_complete=False))
        pa=read(production/'evaluation/audit.json')['cases'][0]
        aa=production/'evaluation/assets/real_00_input'
        bound=case/'material_bound/real/real_00_input'
        original=ImageOps.exif_transpose(Image.open(item['image'])).convert('RGB')
        original.thumbnail((640,640),Image.Resampling.LANCZOS)
        source=rgb(aa/'source.png'); a=rgb(aa/'final.png'); final=rgb(bound/'final.png')
        edit=rgb(aa/'edit_mask.png')[...,0]>127
        contract=np.load(bound/'material_contract.npz')
        core=contract['core']; known=contract['known']
        br=read(bound/'result.json'); ip=read(case/'input_protocol.json')
        guide=Path(read(production/'protocol.json')['cases'][0]['source_guides']['boundary'])
        texture=rgb(guide/'material_texture.png')
        food=rgb(guide/'target_food_mask.png')[...,0]>127
        with np.load(guide/'guide_channels.npz') as channels:
            expected_known=food & (channels['material_confidence']>=.75)
        expected_alpha=np.clip((distance_transform_edt(expected_known)-.5)/1.5,0,1)
        tex=texture.astype(np.float32)/255
        tex=np.where(tex<=.04045,tex/12.92,((tex+.055)/1.055)**2.4)
        relit=np.maximum(tex*contract['gain'][...,None],0)
        expected=np.where(relit<=.0031308,12.92*relit,1.055*relit**(1/2.4)-.055)
        expected=np.rint(255*expected.clip(0,1)).clip(0,255).astype(np.uint8)
        core_error=int(np.abs(final.astype(int)-expected.astype(int))[core].max()) if core.any() else None
        code_bad=[n for n,h in ip['code_sha256'].items() if sha(case/'snapshot'/n)!=h]
        checks=dict(source_hash=sha(item['image'])==item['source_sha256']==ip['input_sha256'],
            source_pixels_unchanged=np.array_equal(source,np.asarray(original)),
            final_hash=sha(bound/'final.png')==state['final_sha256']==br['final_sha256'],
            A_hash=sha(aa/'final.png')==br['base_sha256'],
            exterior_unchanged=not np.any(final[~edit]!=source[~edit]),
            B_outside_known_unchanged=not np.any(final[~known]!=a[~known]),
            source_snapshot_unchanged=not code_bad,
            production_provenance=pa['all_provenance_checks'],
            material_core_contract=core_error==0,
            known_map=np.array_equal(known,expected_known),
            alpha_map=np.array_equal(contract['alpha'],expected_alpha),
            core_map=np.array_equal(core,expected_alpha==1),
            gain_finite_bounded=bool(np.isfinite(contract['gain']).all() and
                contract['gain'].min()>=.72 and contract['gain'].max()<=1.4),
            material_texture_hash=sha(guide/'material_texture.png')==br['texture_sha256'],
            upstream_extended_handle=ip['handle_extension_integrated'],
            fresh_perception=state['fresh_perception'])
        geometry=audit_geometry(case)
        checks['geometry_contracts']=geometry['all_checks']
        if 'code_hashes' in protocol:
            checks['same_frozen_generation_code']=ip['code_sha256']==protocol['code_hashes'] and not outer_code_bad
        target=out/'assets'/cid; target.mkdir(parents=True,exist_ok=True)
        copies={'source.png':aa/'source.png','base.png':aa/'base.png','A.png':aa/'final.png',
            'final.png':bound/'final.png','selected_reference.png':aa/'selected_reference.png',
            'edit_mask.png':aa/'edit_mask.png','source_removed_mask.png':aa/'source_removed_mask.png',
            'core_mask.png':bound/'bound_core_mask.png','status.json':case/'status.json',
            'input_protocol.json':case/'input_protocol.json','material_contract.json':bound/'result.json',
            'geometry_state.json':Path(read(case/'routing_resolved_manifest.json')['cases'][0]['resolved_geometry_case'])/'state.json'}
        for name,path in copies.items(): shutil.copy2(path,target/name)
        for label,arr in [('source',source),('A',a),('final',final)]:
            for kind,box in [('head',pa['head_bbox']),('removal',pa['source_bbox'])]:
                Image.fromarray(arr).crop(box).save(target/f'{label}_{kind}.png')
        sheet=Image.new('RGB',(1290,1080),(238,241,246));draw=ImageDraw.Draw(sheet)
        draw.text((12,8),cid+' | fresh RGB + coarse prompt | observed-data regression',fill='black')
        panels=[('Original photograph','source.png'),('A: generative','A.png'),('B: final material bound','final.png'),
                ('Original removal region','source_removal.png'),('A removal region','A_removal.png'),('B removal region','final_removal.png'),
                ('Selected source texture','selected_reference.png'),('A carried bite','A_head.png'),('B carried bite','final_head.png')]
        for j,(title,name) in enumerate(panels):
            x=j%3*430;y=j//3*352+24;draw.text((x+12,y+4),title,fill='black')
            im=ImageOps.contain(Image.open(target/name).convert('RGB'),(406,314),Image.Resampling.LANCZOS)
            sheet.paste(im,(x+(430-im.width)//2,y+24+(314-im.height)//2))
        sheet.save(target/'comparison.jpg',quality=95)
        row.update(checks=checks,all_checks=all(checks.values()),
            pixels=dict(changed_outside_edit=int(np.any(final!=source,2)[~edit].sum()),
                bound_core=int(core.sum()),known=int(known.sum()),core_contract_max_rgb_error=core_error,
                changed_B_from_A=int(np.any(final!=a,2).sum())),
            material_contract=br,geometry_diagnostics=geometry,route=state['route'],initial_mode=state['initial_mode'],
            head_bbox=pa['head_bbox'],source_bbox=pa['source_bbox'],
            artifacts={name:sha(path) for name,path in copies.items()},visual_review='pending')
        rows.append(row)
    result=dict(evaluated_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(root/'protocol.json'),
        complete=sum(r['status']=='complete' for r in rows),failed=sum(r['status']=='failed' for r in rows),
        pending=sum(r['status'] not in ['complete','failed'] for r in rows),cases=rows,
        frozen_outer_code_mismatches=outer_code_bad,
        interpretation='All 24 inputs were previously observed. Exact pixel/material contracts do not prove photographic realism, physical mass conservation or generalization.',
        evaluator_sha256=sha(Path(__file__)))
    write(out/'audit.json',result)
    for name in ['protocol.json','launch.json']:
        if (root/name).exists():shutil.copy2(root/name,out/name)
    print(json.dumps({k:result[k] for k in ['complete','failed','pending']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    evaluate(p.parse_args().root)
