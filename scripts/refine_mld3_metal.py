"""Optional photographed metal appearance on a frozen MLD3 utensil silhouette."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import subprocess

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import binary_dilation


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def selected_folders(root, indices, shard, shards):
    folders = sorted(x for x in (root / 'real').glob('real_*') if x.is_dir())
    selected = [x for x in folders if indices is None or int(x.name.split('_')[1]) in indices]
    return selected[shard::shards]


def generate(root, steps=24, indices=None, shard=0, shards=1):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                      VIDEOX_ATTENTION_TYPE='SDPA', OMP_NUM_THREADS='4')
    sys.path.insert(0, '/mnt/tmp/guo-z_first_bite_structure_20260930_runtime/vendor')
    import torch
    from safetensors.torch import load_file
    from diffusers import FlowMatchEulerDiscreteScheduler
    from videox_fun.models import (AutoencoderKLQwenImage21, Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor, QwenImage21ControlTransformer2DModel)
    from videox_fun.pipeline.pipeline_qwenimage21_reference_control import QwenImage21ControlPipeline
    torch.set_num_threads(4)
    models = Path('/mnt/tmp/guo-z_first_bite_structure_20260930_models')
    model = models / 'qwen-image-2.1'; dtype = torch.bfloat16
    transformer = QwenImage21ControlTransformer2DModel.from_pretrained(str(model), subfolder='transformer',
        low_cpu_mem_usage=True, torch_dtype=dtype,
        transformer_additional_kwargs={'control_layers': list(range(0, 32, 2)), 'control_in_dim': 129})
    adapter = load_file(str(models / 'controlnet-union/Qwen-Image-2.1-Fun-Controlnet-Union.safetensors'))
    transformer.load_state_dict(adapter, strict=False); del adapter
    vae = AutoencoderKLQwenImage21.from_pretrained(str(model), subfolder='vae', torch_dtype=dtype, local_files_only=True)
    processor = Qwen3VLProcessor.from_pretrained(str(model), subfolder='processor', local_files_only=True)
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(str(model), subfolder='text_encoder',
        torch_dtype=dtype, local_files_only=True, attn_implementation='sdpa')
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(model), subfolder='scheduler', local_files_only=True)
    pipe = QwenImage21ControlPipeline(vae=vae, text_encoder=encoder, processor=processor,
                                     transformer=transformer, scheduler=scheduler)
    pipe.enable_model_cpu_offload(gpu_id=0)
    selected = selected_folders(root, indices, shard, shards)
    rows = []
    for folder in selected:
        source = Image.open(folder / 'source.png').convert('RGB')
        guide = Image.open(folder / 'deformed.png').convert('RGB')
        spoon = np.asarray(Image.open(folder / 'spoon_mask.png')) > 0
        food = np.asarray(Image.open(folder / 'moved_food_mask.png')) > 0
        removed = np.asarray(Image.open(folder / 'source_removed_mask.png')) > 0
        editable = binary_dilation(spoon, iterations=3) & ~food & ~removed
        prompt = ('Retouch only the existing shallow elliptical stainless steel spoon into realistic photographed silver metal. '
            'Keep its exact outline, thin shallow bowl, neck, curved tapering handle, pose, scale and contact with the food. '
            'Use natural silver reflections matching the original source illumination. Preserve the food and every noodle strand, '
            'their texture, shape, source removal, dish and background. No hand, person, new ingredients or deep bowl.')
        negative = 'hand, fingers, person, face, arm, mouth, deep cup, black cup, plastic, cartoon, duplicate food, changed food, new ingredients'
        edges = np.asarray(guide.filter(ImageFilter.FIND_EDGES).convert('L')) > 45
        control = Image.fromarray(np.repeat((edges.astype(np.uint8) * 255)[..., None], 3, axis=-1))
        directory = folder / 'metal_candidate'; directory.mkdir(exist_ok=True)
        mask = Image.fromarray(editable.astype(np.uint8) * 255)
        control.save(directory / 'control.png'); mask.save(directory / 'edit_mask.png')
        request = dict(case_id=folder.name, seed=41, steps=steps, prompt=prompt,
            negative_prompt=negative, locked_food=True, model='Qwen-Image-2.1 + Fun ControlNet-Union',
            conditioning='Masked deformed guide plus edge control; no extra pre-lift source reference image.',
            reference_image_count=0,
            guide_sha256=hashlib.sha256((folder / 'deformed.png').read_bytes()).hexdigest())
        write(directory / 'request.json', request); started = time.time()
        with torch.inference_mode():
            raw = pipe(prompt=prompt, negative_prompt=negative, image=guide, mask_image=mask.convert('RGB'),
                reference_images=None, reference_resolution=512, control_image=control,
                control_context_scale=1., true_cfg_scale=3., height=max(256, round(source.height / 32) * 32),
                width=max(256, round(source.width / 32) * 32), num_inference_steps=steps,
                generator=torch.Generator('cuda').manual_seed(41), use_kv_cache=False).images[0]
        raw.save(directory / 'generated.png')
        candidate = np.asarray(Image.alpha_composite(guide.convert('RGBA'),
            raw.convert('RGBA').resize(source.size, Image.Resampling.LANCZOS)).convert('RGB'))
        Image.fromarray(candidate).save(directory / 'candidate.png')
        rows.append(dict(case_id=folder.name, seconds=time.time() - started))
        print('metal_candidate', folder.name, rows[-1]['seconds'], flush=True)
        write(root / ('metal_worker_' + str(shard) + '.json'), dict(status='generating', cases=rows))
    write(root / ('metal_worker_' + str(shard) + '.json'), dict(status='generated', cases=rows, seed=41, steps=steps))


def project(root, steps=24, indices=None, shard=0, shards=1):
    import torch
    torch.set_num_threads(4)
    selected = selected_folders(root, indices, shard, shards)
    rows = []
    sys.path.insert(0, '/host/space0/guo-z/Evol-SAM3')
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model(bpe_path='/host/space0/guo-z/Evol-SAM3/assets/bpe_simple_vocab_16e6.txt.gz',
        checkpoint_path='/host/space0/guo-z/Evol-SAM3/sam3/sam3.pt', load_from_HF=False, device='cuda', eval_mode=True)
    processor = Sam3Processor(model, confidence_threshold=.18)
    for folder in selected:
        guide = np.asarray(Image.open(folder / 'deformed.png').convert('RGB'))
        candidate_image = Image.open(folder / 'metal_candidate' / 'candidate.png').convert('RGB')
        spoon = np.asarray(Image.open(folder / 'spoon_mask.png')) > 0
        image_state = processor.set_image(candidate_image)
        segments = []
        for prompt in ['spoon', 'metal spoon']:
            processor.reset_all_prompts(image_state)
            result = processor.set_text_prompt(state=image_state, prompt=prompt)
            masks = result['masks'].detach().cpu().numpy().astype(bool).reshape(-1, *spoon.shape)
            scores = result['scores'].detach().cpu().numpy().reshape(-1)
            segments.extend((float((m & spoon).sum()) / max(1, int(spoon.sum())) * float(s), m)
                            for s, m in zip(scores, masks))
        selected_mask = max(segments, key=lambda x: x[0])[1] if segments else np.zeros(spoon.shape, bool)
        accepted = selected_mask & spoon
        final = guide.copy(); final[accepted] = np.asarray(candidate_image)[accepted]
        Image.fromarray(final).save(folder / 'final.png')
        Image.fromarray(accepted.astype(np.uint8) * 255).save(folder / 'metal_candidate' / 'accepted_mask.png')
        state = json.loads((folder / 'state.json').read_text(encoding='utf-8'))
        state['final_appearance'] = dict(seed=41, steps=steps, generated_food_pixels=0,
            reference_image_count=0, conditioning='Masked deformed guide plus edge control, no extra source reference.',
            accepted_metal_pixels=int(accepted.sum()), visible_spoon_pixels=int(spoon.sum()),
            accepted_metal_fraction=float(accepted.sum()) / max(1, int(spoon.sum())),
            geometry_changed=False, material_changed=False,
            projection='Candidate SAM3 metal intersect frozen visible spoon, at original coordinates. PBR retained elsewhere.')
        write(folder / 'state.json', state)
        rows.append(dict(case_id=folder.name, **state['final_appearance']))
        print('metal_projected', folder.name, int(accepted.sum()), flush=True)
    write(root / ('metal_worker_' + str(shard) + '.json'), dict(status='complete', cases=rows, seed=41, steps=steps))


def refine(root, steps=24, indices=None, shard=0, shards=1):
    arguments = [str(Path(__file__).resolve()), '--root', str(root), '--steps', str(steps),
                 '--shard', str(shard), '--shards', str(shards)]
    if indices is not None:
        arguments += ['--indices'] + [str(x) for x in indices]
    subprocess.run(['/mnt/tmp/guo-z_first_bite_structure_20260930_runtime/venv/bin/python',
                    *arguments, '--generate-only'], check=True)
    subprocess.run(['/host/space0/guo-z/tf-ufi/food3d_pilot_20260928/venv/bin/python',
                    *arguments, '--project-only'], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True); p.add_argument('--steps', type=int, default=24)
    p.add_argument('--indices', type=int, nargs='+'); p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--generate-only', action='store_true'); mode.add_argument('--project-only', action='store_true')
    args = p.parse_args()
    action = generate if args.generate_only else project if args.project_only else refine
    action(args.root, args.steps, args.indices, args.shard, args.shards)


if __name__ == '__main__':
    main()
