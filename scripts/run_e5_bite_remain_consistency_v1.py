"""Run one frozen E5 shared-volume completion and preserve protected pixels."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw

from bite_remain_consistency import composite
from run_high_lift_vace_pilot import resource_preflight, run_text, sha256, validate_file, write_json


def make_sheet(frames):
    w, h = frames[0].size
    result = Image.new('RGB', (w * 3, (h + 30) * 2), 'white'); draw = ImageDraw.Draw(result)
    for k, i in enumerate([0, 8, 9, 12, 15, 20]):
        x, y = (k % 3) * w, (k // 3) * (h + 30)
        result.paste(frames[i], (x, y + 30)); draw.text((x + 4, y + 5), f'E5 result / f{i}', fill='black')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controls', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--base-config', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--preflight-report', type=Path, required=True)
    a = p.parse_args(); manifest_path = a.controls / 'manifest.json'
    if sha256(manifest_path) != a.manifest_sha256: raise ValueError('controls manifest mismatch')
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema_version'] != 'foodstateedit.bite_remain_controls.v1': raise ValueError('unexpected controls schema')
    cfg = manifest['config']
    if sha256(a.base_config) != cfg['base_config_sha256']: raise ValueError('base config mismatch')
    if sha256(Path(__file__).with_name('bite_remain_consistency.py')) != manifest['helper_sha256']:
        raise ValueError('helper changed after control freeze')
    for rel, record in manifest['files'].items(): validate_file(a.controls / rel, record)
    base_config = json.loads(a.base_config.read_text())
    if a.output_root.exists() or a.preflight_report.exists(): raise FileExistsError('new paths required')
    gate = resource_preflight(base_config, a.output_root, a.preflight_report, a.gpu)
    a.output_root.mkdir(parents=True, exist_ok=False)
    started=time.monotonic(); done=threading.Event()
    state=dict(schema_version='foodstateedit.bite_remain_run.v1', status='initializing',
        started_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), physical_gpu=a.gpu,
        pipeline_load_count=0, controls_manifest_sha256=a.manifest_sha256,
        runner_sha256=sha256(Path(__file__)), base_config_sha256=sha256(a.base_config),
        seed=cfg['seed'], num_frames=cfg['num_frames'], num_inference_steps=cfg['num_inference_steps'],
        vace_scale=cfg['vace_scale'], lora_enabled=False, enable_ttm=False, claim_limit=cfg['claim_limit'])
    write_json(a.output_root/'run_manifest.json', state)

    def abort(signum, frame): raise RuntimeError('own worker stopped at resource or deadline boundary')
    def check_resource():
        rows=run_text(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'])
        for row in rows.splitlines():
            fields=[f.strip() for f in row.split(',')]
            if fields[0] == gate['gpu']['uuid'] and int(fields[1]) != os.getpid():
                raise RuntimeError('foreign GPU process appeared: '+row)
    def watch():
        while not done.wait(10):
            try: check_resource()
            except Exception as exc:
                (a.output_root/'resource_stop.txt').write_text(repr(exc)); os.kill(os.getpid(), signal.SIGTERM); return
    signal.signal(signal.SIGALRM, abort); signal.signal(signal.SIGTERM, abort); signal.alarm(cfg['hard_timeout_seconds'])
    try:
        runtime=base_config['runtime']; validate_file(Path(runtime['model_hash_audit']['path']), runtime['model_hash_audit'])
        for rel,digest in runtime['geoedit_files'].items():
            if sha256(Path(runtime['geoedit_root'])/rel) != digest: raise ValueError('frozen runtime changed: '+rel)
        for rel,record in runtime['model_files'].items(): validate_file(Path(runtime['model_root'])/rel, record, hash_file=False)
        with np.load(a.controls/'controls.npz', allow_pickle=False) as z:
            base,condition,mask,alpha,payload=[z[k] for k in
                ['baseline','condition','generation_mask','projection_alpha','planned_payload']]
        if base.shape != condition.shape or base.shape[0] != cfg['num_frames']: raise ValueError('control shape mismatch')
        if mask.shape != base.shape[:3] or alpha.shape != mask.shape or payload.shape != mask.shape: raise ValueError('mask shape mismatch')
        if not np.isin(mask,[0,255]).all() or np.any(alpha[mask==0]): raise ValueError('invalid mask contract')
        if np.any(condition[mask==0] != base[mask==0]): raise ValueError('inactive condition changed')
        if np.any(alpha[payload>0]) or mask[:cfg['first_lift_frame']].any(): raise ValueError('protected payload/prelift violation')
        h,w=base.shape[1:3]
        reference_name={'original':'reference.png','cavity_anchor':'cavity_anchor.png'}[cfg.get('reference_mode','original')]
        reference=Image.open(a.controls/reference_name).convert('RGB')
        state['reference_mode']=cfg.get('reference_mode','original')
        write_json(a.output_root/'run_manifest.json',state)
        os.environ.update(runtime['offline_environment'])
        os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), HF_HOME=str(a.output_root/'cache'),
            TRANSFORMERS_CACHE=str(a.output_root/'cache'), DIFFSYNTH_MODEL_BASE_PATH=str(Path(runtime['model_root']).parent.parent))
        sys.path.insert(0,runtime['geoedit_root']); check_resource(); threading.Thread(target=watch,daemon=True).start()
        from geoedit import inference
        with (a.output_root/'inference.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            state['status']='loading_pipeline'; write_json(a.output_root/'run_manifest.json',state)
            pipe=inference.load_pipeline(inference.resolve_vram_limit(None))
            state.update(pipeline_load_count=1,status='running'); write_json(a.output_root/'run_manifest.json',state)
            video=pipe(prompt=cfg['prompt'],negative_prompt=cfg['negative_prompt'],height=h,width=w,
                num_frames=cfg['num_frames'],num_inference_steps=cfg['num_inference_steps'],
                vace_video=[Image.fromarray(f) for f in condition],
                vace_video_mask=[Image.fromarray(f).convert('RGB') for f in mask],
                vace_reference_image=[reference],vace_scale=cfg['vace_scale'],enable_ttm=False,
                seed=cfg['seed'],tiled=True)
            if len(video) != cfg['num_frames']: raise ValueError('output frame count mismatch')
            raw_dir=a.output_root/'raw_frames'; final_dir=a.output_root/'projected_frames'; raw_dir.mkdir(); final_dir.mkdir()
            projected=[]; outside_error=payload_error=0
            for i,image in enumerate(video):
                raw=np.asarray(image.convert('RGB')); image.convert('RGB').save(raw_dir/f'{i:02d}.png')
                final=composite(base[i],raw,alpha[i]); diff=np.abs(final.astype(np.int16)-base[i].astype(np.int16))
                outside_error=max(outside_error,int(diff[alpha[i]==0].max(initial=0)))
                payload_error=max(payload_error,int(diff[payload[i]>0].max(initial=0)))
                rendered=Image.fromarray(final); rendered.save(final_dir/f'{i:02d}.png'); projected.append(rendered)
            if outside_error or payload_error: raise AssertionError('protected pixels changed')
            make_sheet(projected).save(a.output_root/'projected_review.png')
            projected[-1].save(a.output_root/'projected_final_hold.png')
            inference.save_video(projected,str(a.output_root/'projected.mp4'),fps=cfg['fps'],quality=5)
        check_resource(); state.update(status='complete_requires_visual_review',projected_frames=len(projected),
            outside_repair_max_difference=outside_error,planned_payload_max_difference=payload_error)
    except BaseException as exc:
        state.update(status='technical_failure_preserved',error=repr(exc)); (a.output_root/'failure.txt').write_text(traceback.format_exc())
    finally:
        done.set(); signal.alarm(0); state.update(finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-started)
        write_json(a.output_root/'run_manifest.json',state)
        records={f.relative_to(a.output_root).as_posix():dict(sha256=sha256(f),size_bytes=f.stat().st_size)
                 for f in a.output_root.rglob('*') if f.is_file()}
        write_json(a.output_root/'files_manifest.json',records)
    return 0 if state['status']=='complete_requires_visual_review' else 3


if __name__=='__main__': raise SystemExit(main())
