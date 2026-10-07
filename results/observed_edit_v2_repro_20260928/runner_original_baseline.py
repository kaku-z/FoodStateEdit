"""Run frozen real-photo v2 development generation with an existing offline Qwen."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import traceback
import numpy as np
from PIL import Image
from run_qwen_image_edit_direct_baseline import gpu_snapshot, gate_snapshot, foreign_process_check, sha256_file


def write(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--physical-gpu', type=int, required=True)
    args = p.parse_args()
    config_path = args.bundle/'config.json'
    c = json.loads(config_path.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.time()
    manifest = {'experiment_id': c['experiment_id'], 'host': socket.gethostname(),
                'status': 'preflight', 'config_sha256': sha256_file(config_path),
                'runner_sha256': sha256_file(Path(__file__)), 'completed': [],
                'claim_limit': c['claim_limit'], 'started_unix': start}
    path = args.output/'run_manifest.json'
    write(path, manifest)
    try:
        for case in c['cases']:
            for asset in case['files'].values():
                if sha256_file(args.bundle/asset['path']) != asset['sha256']:
                    raise ValueError('Input checksum mismatch: '+asset['path'])
        model_root = Path(c['backend']['model_root'])
        checks = {}
        for relative, expected in c['backend']['weight_files'].items():
            weight = model_root/relative
            checks[relative] = weight.stat().st_size == expected['size_bytes'] and sha256_file(weight) == expected['sha256']
            manifest['weight_files_checked'] = len(checks)
            write(path, manifest)
        snapshot = gpu_snapshot(args.physical_gpu)
        resource = gate_snapshot(snapshot, c['resource_gate'])
        write(args.output/'preflight.json', {'weights': checks, 'gpu': snapshot, 'resource': resource})
        if not all(checks.values()) or not all(resource.values()):
            raise RuntimeError('Preflight failed; no generation attempted')
        os.environ.update(CUDA_VISIBLE_DEVICES=str(args.physical_gpu), HF_HUB_OFFLINE='1',
                          TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                          TORCH_COMPILE_DISABLE='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        import torch
        from diffusers import QwenImageEditPlusPipeline
        manifest.update(status='loading_model', torch_version=torch.__version__)
        write(path, manifest)
        pipe = QwenImageEditPlusPipeline.from_pretrained(str(model_root), torch_dtype=torch.bfloat16, local_files_only=True)
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=False)
        inference = c['inference']
        for case in c['cases']:
            def asset(name):
                return args.bundle/case['files'][name]['path']
            source = Image.open(asset('source.png')).convert('RGB')
            source_arr = np.array(source)
            state = np.load(asset('state.npz'), allow_pickle=False)
            for seed in c['seeds']:
                for condition in c['conditions']:
                    foreign_process_check(snapshot['uuid'])
                    name = f"{case['case_id']}_{seed}_{condition}"
                    manifest.update(status='generating', current=name, updated_unix=time.time())
                    write(path, manifest)
                    if condition == 'original':
                        model_images = source
                    else:
                        guide = Image.open(asset('control.png' if condition == 'direct' else 'scaffold.png')).convert('RGB')
                        model_images = [source, guide]
                    t0 = time.time()
                    result = pipe(image=model_images, prompt=case['prompts'][condition],
                                  negative_prompt=inference['negative_prompt'], num_inference_steps=inference['num_inference_steps'],
                                  true_cfg_scale=inference['true_cfg_scale'], guidance_scale=inference['guidance_scale'],
                                  generator=torch.Generator(device='cuda').manual_seed(seed)).images[0]
                    folder = args.output/name
                    folder.mkdir()
                    result.save(folder/'raw.png')
                    resized = np.array(result.resize(source.size, Image.Resampling.LANCZOS))
                    local = source_arr.copy()
                    local[state['editable']] = resized[state['editable']]
                    Image.fromarray(local).save(folder/'localized.png')
                    if condition == 'grounded':
                        anchored = local.copy()
                        anchored[state['payload_core']] = state['transported_rgb'][state['payload_core']]
                        Image.fromarray(anchored).save(folder/'observed_locked.png')
                    record = {'name': name, 'duration_seconds': time.time()-t0,
                              'raw_size': list(result.size), 'files': {f.name: sha256_file(f) for f in folder.iterdir()}}
                    manifest['completed'].append(record)
                    write(path, manifest)
        manifest.update(status='complete_unreviewed', duration_seconds=time.time()-start)
        write(path, manifest)
        (args.output/'COMPLETE').write_text('complete_unreviewed\n')
    except BaseException as exc:
        manifest.update(status='technical_failure', error=repr(exc), duration_seconds=time.time()-start)
        write(path, manifest)
        (args.output/'FAILED').write_text(traceback.format_exc())
        raise
    print(json.dumps({'status': manifest['status'], 'completed': len(manifest['completed'])}), flush=True)


if __name__ == '__main__':
    main()
