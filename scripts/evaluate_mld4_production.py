"""Independent provenance/pixel audit and review assets for frozen MLD4 production.

Visual judgments are supplied separately after observing each actual output.
No geometric mask overlap is interpreted as generated semantic correctness.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def rgb(p):return np.asarray(Image.open(p).convert('RGB'))
def mask(p):return np.asarray(Image.open(p).convert('L'))>127


def evaluate(args):
    r=args.production.resolve();out=r/'evaluation';out.mkdir(exist_ok=True)
    protocol=read(r/'protocol.json');digest=sha(r/'protocol.json')
    assert digest==args.protocol_sha256==((r/'protocol.sha256').read_text().strip())
    rows=[];pending=[]
    # Independent read of every declared frozen input and executable snapshot.
    frozen_checks={scope:{p:sha(p)==h for p,h in protocol[scope].items()}
                   for scope in ['snapshot_hashes','input_hashes']}
    bad=[p for checks in frozen_checks.values() for p,passed in checks.items() if not passed]
    write(out/'frozen_input_audit.json',dict(protocol_sha256=digest,
        checked_counts={k:len(v) for k,v in frozen_checks.items()},mismatches=bad,
        evaluator_sha256=sha(Path(__file__))))
    assert not bad,bad
    for case in protocol['cases']:
        name=case['case_id'];p=r/'real'/name
        if not (p/'reform_state.json').exists():pending.append(name);continue
        state=read(p/'reform_state.json');base=p/'candidates/owned_anchors';final=p/'candidates/production'
        head=r/'guides/depth'/name/'head';depth=r/'guides/depth'/name;anchors=r/'guides/anchored'/name
        br=read(base/'request.json');bg=read(base/'generation.json');hr=read(final/'head/request.json');hg=read(final/'head/generation.json')
        fr=read(final/'generation.json')
        checks={'state_protocol':state['protocol_sha256']==digest,'source':sha(p/'source.png')==case['source_sha256'],
            'edit':sha(p/'edit_mask.png')==case['frozen_edit_mask_sha256'],
            'final_is_uniform_unprojected':sha(p/'final.png')==sha(final/'unprojected.png'),
            'head_base':hr['base_sha256']==sha(base/'unprojected.png'),
            'head_source_caption':hr['captions_sha256']==sha(r/'snapshot/captions.json'),
            'head_runner':hr['runner_sha256']==sha(r/'snapshot/scripts/run_mld4_factorized.py'),
            'head_generated':hg['generated_sha256']==sha(final/'head/generated.png'),
            'head_conditioning':hr['conditioning_sha256']==sha(final/'head/conditioning_used.png'),
            'head_keep_mask':hr['conditioning_mask_used_sha256']==sha(final/'head/conditioning_mask_used.png'),
            'head_acceptance':hr['acceptance_mask_sha256']==sha(final/'head/acceptance_mask.png'),
            'head_depth':hr['depth_control_sha256']==sha(head/'depth_control.png'),
            'no_lora':hr['binding_adapter'] is None and hr['binding_enabled'] is False,
            'head_only_stage':len(fr['stages'])==1 and fr['stages'][0]['stage']=='head',
            'head_cfg':hr['true_cfg_scale']==1.,'head_steps':hr['steps']==40,'head_seed':hr['seed']==41,
            'base_seed':br['seed']==41,'base_steps':br['steps']==40,'base_cfg':br['true_cfg_scale']==3.,
            'base_runner':br['runner_sha256']==sha(r/'snapshot/scripts/run_mld4_reconstruction.py'),
            'base_generated':bg['generated_sha256']==sha(base/'generated.png'),
            'base_conditioning':br['conditioning_image_sha256']==sha(base/'conditioning_used.png'),
            'base_keep_mask':br['conditioning_mask_sha256']==sha(base/'conditioning_mask_used.png'),
            'base_reference':br['reference_sha256']==sha(base/'reference_used.png')}
        for n,h in state['artifact_sha256'].items():checks['state_artifact.'+n]=sha(p/n)==h
        for n,h in hr['inputs'].items():checks['head_guide.'+n]=sha(head/n)==h
        for n,h in br['input_hashes'].items():checks['base_guide.'+n]=sha(anchors/n)==h
        # Request schemas differ between the full-frame and factorized runner; copy them for review.
        source=rgb(p/'source.png');b=rgb(base/'unprojected.png');f=rgb(p/'final.png')
        edit=mask(p/'edit_mask.png');hm=mask(depth/'head_stage_full_mask.png');removed=mask(p/'source_removed_mask.png')
        assert source.shape==b.shape==f.shape
        pixels=dict(base_changed_outside_edit=int(np.any(b!=source,2)[~edit].sum()),
            final_changed_outside_edit=int(np.any(f!=source,2)[~edit].sum()),
            refinement_changed_outside_head=int(np.any(f!=b,2)[~hm].sum()),
            refinement_changed_head_pixels=int(np.any(f!=b,2)[hm].sum()),
            final_changed_in_source_removal=int(np.any(f!=source,2)[removed].sum()),
            source_removal_pixels=int(removed.sum()))
        dest=out/'assets'/name;dest.mkdir(parents=True,exist_ok=True)
        assets={'source.png':p/'source.png','base.png':base/'unprojected.png','final.png':p/'final.png',
            'source_removed_mask.png':p/'source_removed_mask.png','edit_mask.png':p/'edit_mask.png',
            'selected_reference.png':head/'source_reference.png','head_raw.png':final/'head/generated.png',
            'base_request.json':base/'request.json','head_request.json':final/'head/request.json',
            'state.json':p/'reform_state.json'}
        for n,src in assets.items():shutil.copy2(src,dest/n)
        hb=hr['bbox_xyxy'];sb=read(depth/'source/stage_manifest.json')['bbox_xyxy']
        for label,arr in [('source',source),('base',b),('final',f)]:
            Image.fromarray(arr).crop(hb).save(dest/(label+'_head.png'))
            Image.fromarray(arr).crop(sb).save(dest/(label+'_removal.png'))
        # Three aligned comparison rows: full image, source hole, carried bite.
        sheet=Image.new('RGB',(1290,1080),(236,240,245));draw=ImageDraw.Draw(sheet)
        group='development (16)' if case['index']<16 else 'frozen new source (8)'
        draw.text((12,8),name+' | '+group+' | automatic label: '+case['semantic_target'],fill=(18,25,35))
        panels=[('Original photograph','source.png'),('Full-frame base','base.png'),('Uniform final','final.png'),
                ('Original source location','source_removal.png'),('Base source location','base_removal.png'),('Final source location','final_removal.png'),
                ('Observed selected material','selected_reference.png'),('Base carried bite','base_head.png'),('Final carried bite','final_head.png')]
        for j,(label,n) in enumerate(panels):
            x=(j%3)*430;y=(j//3)*352+24
            draw.text((x+12,y+4),label,fill=(18,25,35))
            im=Image.open(dest/n).convert('RGB');im=ImageOps.contain(im,(406,314),Image.Resampling.LANCZOS)
            sheet.paste(im,(x+(430-im.width)//2,y+24+(314-im.height)//2))
        sheet.save(dest/'comparison.jpg',quality=96)
        row=dict(case_id=name,index=case['index'],group='development' if case['index']<16 else 'frozen_new_source',
            automatic_semantic_label=case['semantic_target'],label_is_ground_truth=False,route=case['route'],
            checks=checks,all_provenance_checks=all(checks.values()),pixel_diagnostics=pixels,
            head_bbox=hb,source_bbox=sb,artifacts={n:sha(dest/n) for n in assets},comparison_sha256=sha(dest/'comparison.jpg'),
            visual_review_status='pending_direct_inspection')
        rows.append(row)
    if args.require_complete:assert len(rows)==24 and not pending,{'completed':len(rows),'pending':pending}
    write(out/'audit.json',dict(status='complete_pixels_and_provenance' if not pending else 'partial_pixels_and_provenance',
        protocol_sha256=digest,case_count=len(rows),pending=pending,cases=rows,
        frozen_inputs_ok=not bad,source_groups={'development':16,'frozen_new_source':8},
        criteria=['source_material_identity','visible_lift','food_spoon_support','source_removal','texture_and_seams','no_added_humans','handle_frame_connection'],
        interpretation='Exterior equality is an editing invariant, not semantic or visual success. No paired real after-images, measured mass or 3D ground truth exist. Automatic labels may be wrong.',
        lora='Not used in this frozen production. Separate200step pilot had small reconstruction gains without clear gain in three real head edits.',
        evaluated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    print(json.dumps({'completed':len(rows),'pending':len(pending),'frozen_input_mismatches':len(bad),
        'failed_provenance_cases':[x['case_id'] for x in rows if not x['all_provenance_checks']]}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--production',type=Path,required=True);p.add_argument('--protocol-sha256',required=True)
    p.add_argument('--require-complete',action='store_true');evaluate(p.parse_args())
