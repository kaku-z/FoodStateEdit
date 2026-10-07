"""Local appearance synthesis with explicit frozen-context diffusion and raw receipts.

The generator supplies appearance inside an editable region. Any final layer
composition is a separate, disclosed algorithm stage, never called a raw output.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import traceback


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(16*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write(p, obj):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2)+'\n')
    tmp.replace(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--gpus', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    a = ap.parse_args()
    c = json.loads(a.config.read_text())
    jobs = [j for i, j in enumerate(c['jobs']) if i % a.shards == a.shard]
    a.output.mkdir(exist_ok=False, parents=True)
    mp = a.output/'manifest.json'
    state = dict(status='preflight', pid=os.getpid(), gpus=a.gpus, started_unix=time.time(),
                 script_sha256=sha(__file__), config_sha256=sha(a.config),
                 expected_jobs=[j['id'] for j in jobs], completed=[])
    (a.output/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    write(mp, state)
    try:
        from run_qwen_image_edit_direct_baseline import gpu_snapshot, foreign_process_check
        snaps = [gpu_snapshot(int(g)) for g in a.gpus.split(',')]
        assert len(snaps) == 2
        assert all(not s['compute_processes'] and s['memory_free_mib'] > 47500 for s in snaps)
        assert snaps[0]['host_mem_available_mib'] > 50000
        for job in jobs:
            for f in job['files'].values():
                assert sha(f['path']) == f['sha256']
        for name, info in c['backend']['weight_files'].items():
            assert (Path(c['backend']['model_root'])/name).stat().st_size == info['size_bytes']
        os.environ.update(CUDA_VISIBLE_DEVICES=a.gpus, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', TORCH_COMPILE_DISABLE='1',
                          PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        import numpy as np
        from PIL import Image
        import torch
        import torch.nn.functional as F
        from diffusers import QwenImageEditPlusPipeline
        torch.set_num_threads(4)
        assert sha(inspect.getfile(QwenImageEditPlusPipeline)) == c['expected_pipeline_sha256']
        state.update(status='loading_model', updated_unix=time.time())
        write(mp, state)
        pipe = QwenImageEditPlusPipeline.from_pretrained(c['backend']['model_root'], torch_dtype=torch.bfloat16,
            local_files_only=True, device_map='balanced', max_memory={0:'44GiB', 1:'44GiB'})
        device = pipe._execution_device
        channels = pipe.transformer.config.in_channels//4
        for j in jobs:
            for s in snaps:
                foreign_process_check(s['uuid'])
            start = time.time()
            out = a.output/j['id']
            out.mkdir(exist_ok=False)
            images = [Image.open(j['files'][k]['path']).convert('RGB') for k in j['input_order']]
            W, H = images[0].size
            assert W % 32 == H % 32 == 0
            gen = torch.Generator(device=device).manual_seed(j['seed'])
            with torch.inference_mode():
                noise, _ = pipe.prepare_latents(None, 1, channels, H, W, torch.bfloat16, device, gen)
            target = keep = structural = None
            steps_audit = []
            if j.get('lock_context'):
                with torch.inference_mode():
                    x = pipe.image_processor.preprocess(images[0], height=H, width=W).unsqueeze(2)
                    grid = pipe._encode_vae_image(x.to(pipe.vae.device, torch.bfloat16), gen).to(device)
                    lh, lw = grid.shape[-2:]
                    target = pipe._pack_latents(grid, 1, channels, lh, lw)
                    mask = np.asarray(Image.open(j['files']['edit_mask']['path']).convert('L')).copy()/255.
                    edit = F.interpolate(torch.tensor(mask, device=device)[None,None].float(), size=(lh,lw), mode='area')
                    # Keep only cells entirely outside the declared editable region.
                    edit = F.max_pool2d(edit, 3, stride=1, padding=1)
                    keep_grid = (edit < .001).to(torch.bfloat16)[:,:,None].expand(1,channels,1,lh,lw).contiguous()
                    keep = pipe._pack_latents(keep_grid, 1, channels, lh, lw)
            if j.get('structure'):
                assert target is not None
                weight = np.asarray(Image.open(j['files']['structure_weight']['path']).convert('L')).copy()/255.
                weight = F.interpolate(torch.tensor(weight, device=device)[None,None].float(), size=(lh,lw), mode='area')
                weight = F.max_pool2d(weight,3,stride=1,padding=1)
                weight = weight.to(torch.bfloat16)[:,:,None].expand(1,channels,1,lh,lw).contiguous()
                structural = pipe._pack_latents(weight,1,channels,lh,lw)*(1-keep)
            state.update(status='generating', current=j['id'], step=-1, updated_unix=time.time())
            write(mp, state)
            def callback(pipeline, step, timestep, kwargs):
                before = kwargs['latents']
                after = before
                if keep is not None:
                    sigma = pipeline.scheduler.sigmas[step+1].to(device, before.dtype)
                    context = (1-sigma)*target+sigma*noise
                    after = keep*context+(1-keep)*before
                    editable = keep == 0
                    anchor_strength = 0.
                    if structural is not None:
                        cfg = j['structure'];fraction=(step+1)/c['inference']['steps']
                        anchor_strength=max(cfg['tail_weight'], min(1., (cfg['end_fraction']-fraction)/max(.001,cfg['end_fraction']-cfg['begin_taper_fraction'])))
                        delta=context-after
                        if cfg.get('lowpass'):
                            # Low spatial frequencies in VAE latents, not a depth guarantee.
                            d=delta.reshape(1,lh//2,lw//2,channels,2,2).permute(0,3,1,4,2,5).reshape(1,channels,lh,lw)
                            d=F.avg_pool2d(d.float(),cfg['kernel'],stride=1,padding=cfg['kernel']//2).to(before.dtype)[:,:,None]
                            delta=pipe._pack_latents(d,1,channels,lh,lw)
                        after=after+structural*anchor_strength*delta
                        assert torch.equal(after[keep==1], context[keep==1])
                    else:
                        assert torch.equal(after[editable], before[editable])
                    steps_audit.append({'step':step, 'sigma':float(sigma),
                        'editable_latent_update_max':float((after-before)[editable].abs().max()), 'structural_strength':anchor_strength, 'fixed_fraction':float(keep.float().mean())})
                state.update(step=int(step), updated_unix=time.time())
                write(mp, state)
                return {'latents':after}
            request = dict(j, noise_sha256=hashlib.sha256(noise.float().cpu().numpy().tobytes()).hexdigest())
            write(out/'request.json', request)
            for g in range(2):
                torch.cuda.reset_peak_memory_stats(g)
            inf = c['inference']
            with torch.inference_mode():
                result = pipe(image=images if len(images)>1 else images[0], prompt=j['prompt'],
                    negative_prompt=inf['negative_prompt'], true_cfg_scale=inf['true_cfg_scale'], guidance_scale=None,
                    num_inference_steps=inf['steps'], width=W, height=H, generator=gen, latents=noise.clone(),
                    callback_on_step_end=callback).images[0]
            result.save(out/'raw.png')
            write(out/'context_audit.json', {'enabled':keep is not None, 'steps':steps_audit,
                'scope':'Outside context plus typed spatial/temporal anchoring inside the edit when configured. Structural constraints are hypotheses, not output guarantees.',
                'raw_output_no_pixel_compositor':True})
            row = {'id':j['id'], 'method':j['method'], 'case_id':j['case_id'], 'seed':j['seed'],
                'seconds':time.time()-start, 'peak_allocated_gib':[torch.cuda.max_memory_allocated(g)/1024**3 for g in range(2)],
                'files':{p.name:sha(p) for p in out.iterdir()}}
            write(out/'result.json', row)
            state['completed'].append(row)
            write(mp, state)
            del target, keep, noise, structural
        state.update(status='complete_unreviewed', finished_unix=time.time())
        write(mp, state)
    except BaseException as exc:
        state.update(status='technical_failure', error=repr(exc), updated_unix=time.time())
        write(mp, state)
        (a.output/'FAILED.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
