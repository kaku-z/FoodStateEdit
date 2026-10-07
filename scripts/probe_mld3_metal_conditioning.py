"""Fixed-state comparison of native Qwen reference conditioning for metal only.

Geometry, guide, food/removal masks, prompt, control, 24 steps and seed41 stay
identical across source_ref / guide_ref / no_ref.  No food candidate pixels are
accepted.  All candidates remain saved, with request hashes and model audits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import binary_dilation


PROMPT = ('Retouch only the existing shallow elliptical stainless steel spoon into realistic photographed silver metal. '
    'Keep its exact outline, thin shallow bowl, neck, curved tapering handle, pose, scale and contact with the food. '
    'Use natural silver reflections matching the original source illumination. Preserve the food and every noodle strand, '
    'their texture, shape, source removal, dish and background. No hand, person, new ingredients or deep bowl.')
NEGATIVE = 'hand, fingers, person, face, arm, mouth, deep cup, black cup, plastic, cartoon, duplicate food, changed food, new ingredients'
VENDOR = Path('/mnt/tmp/guo-z_first_bite_structure_20260930_runtime/vendor')
MODELS = Path('/mnt/tmp/guo-z_first_bite_structure_20260930_models')


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_pipeline():
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                      VIDEOX_ATTENTION_TYPE='SDPA', OMP_NUM_THREADS='4')
    sys.path.insert(0, str(VENDOR))
    import torch
    from safetensors.torch import load_file
    from diffusers import FlowMatchEulerDiscreteScheduler
    from videox_fun.models import (AutoencoderKLQwenImage21, Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor, QwenImage21ControlTransformer2DModel)
    from videox_fun.pipeline.pipeline_qwenimage21_reference_control import QwenImage21ControlPipeline
    torch.set_num_threads(4)
    model = MODELS / 'qwen-image-2.1'; dtype = torch.bfloat16
    transformer = QwenImage21ControlTransformer2DModel.from_pretrained(str(model), subfolder='transformer',
        low_cpu_mem_usage=True, torch_dtype=dtype,
        transformer_additional_kwargs={'control_layers': list(range(0, 32, 2)), 'control_in_dim': 129})
    adapter = load_file(str(MODELS / 'controlnet-union/Qwen-Image-2.1-Fun-Controlnet-Union.safetensors'))
    result = transformer.load_state_dict(adapter, strict=False)
    adapter_audit = dict(adapter_tensors=len(adapter), missing_base_keys=len(result.missing_keys),
                         unexpected_adapter_keys=result.unexpected_keys)
    del adapter
    vae = AutoencoderKLQwenImage21.from_pretrained(str(model), subfolder='vae', torch_dtype=dtype, local_files_only=True)
    processor = Qwen3VLProcessor.from_pretrained(str(model), subfolder='processor', local_files_only=True)
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(str(model), subfolder='text_encoder',
        torch_dtype=dtype, local_files_only=True, attn_implementation='sdpa')
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(model), subfolder='scheduler', local_files_only=True)
    pipe = QwenImage21ControlPipeline(vae=vae, text_encoder=encoder, processor=processor,
                                     transformer=transformer, scheduler=scheduler)
    pipe.enable_model_cpu_offload(gpu_id=0)
    return pipe, adapter_audit


def generate(args):
    import torch
    pipe, adapter_audit = load_pipeline()
    write(args.output / 'pipeline_audit.json', dict(adapter=adapter_audit,
        vendor_file=str(VENDOR / 'videox_fun/pipeline/pipeline_qwenimage21_reference_control.py'),
        vendor_sha256=sha(VENDOR / 'videox_fun/pipeline/pipeline_qwenimage21_reference_control.py'),
        guide_path='Only masked VAE control input; never shown to text encoder',
        reference_path='Native vision/text reference prefix plus image latent prefix',
        mask_semantics='White regenerates; keep mask latent is1-white',
        combined_interface='Vendor explicitly documents reference+target control combination nottrained jointly',
        stochasticity='VAE argmax; seed41 creates the same initial diffusion noise for source/guide/no-reference recipes'))
    rows = []
    for folder in args.cases:
        source = Image.open(folder / 'source.png').convert('RGB')
        guide = Image.open(folder / 'deformed.png').convert('RGB')
        spoon = np.asarray(Image.open(folder / 'spoon_mask.png')) > 0
        food = np.asarray(Image.open(folder / 'moved_food_mask.png')) > 0
        removed = np.asarray(Image.open(folder / 'source_removed_mask.png')) > 0
        editable = binary_dilation(spoon, iterations=3) & ~food & ~removed
        mask = Image.fromarray(editable.astype(np.uint8) * 255)
        edges = np.asarray(guide.filter(ImageFilter.FIND_EDGES).convert('L')) > 45
        control = Image.fromarray(np.repeat((edges.astype(np.uint8) * 255)[..., None], 3, axis=-1))
        case = args.output / folder.name; case.mkdir(exist_ok=True)
        source.save(case / 'source.png'); guide.save(case / 'guide.png')
        mask.save(case / 'edit_mask.png'); control.save(case / 'control.png')
        Image.fromarray(spoon.astype(np.uint8) * 255).save(case / 'spoon_mask.png')
        Image.fromarray(food.astype(np.uint8) * 255).save(case / 'food_mask.png')
        for variant in ['source_ref', 'guide_ref', 'no_ref']:
            out = case / variant; out.mkdir(exist_ok=True)
            refs = [source] if variant == 'source_ref' else [guide] if variant == 'guide_ref' else None
            request = dict(case_id=folder.name, input=str(folder), protocol=variant, seed=41, steps=args.steps,
                prompt=PROMPT, negative_prompt=NEGATIVE, control_context_scale=1., true_cfg_scale=3.,
                input_hashes={name: sha(case / name) for name in ['source.png', 'guide.png', 'edit_mask.png', 'control.png', 'spoon_mask.png', 'food_mask.png']},
                reference=['source.png'] if variant == 'source_ref' else ['guide.png'] if variant == 'guide_ref' else [],
                source_food_geometry_frozen=True, outside_spoon_candidate_pixels_discarded=True,
                canvas=[max(256, round(source.width / 32) * 32), max(256, round(source.height / 32) * 32)])
            request_text = json.dumps(request, sort_keys=True)
            request['request_sha256'] = hashlib.sha256(request_text.encode()).hexdigest()
            write(out / 'request.json', request)
            started = time.perf_counter()
            with torch.inference_mode():
                raw = pipe(prompt=PROMPT, negative_prompt=NEGATIVE, image=guide, mask_image=mask.convert('RGB'),
                    reference_images=refs, reference_resolution=512, control_image=control,
                    control_context_scale=1., true_cfg_scale=3., height=request['canvas'][1], width=request['canvas'][0],
                    num_inference_steps=args.steps, generator=torch.Generator('cuda').manual_seed(41), use_kv_cache=False).images[0]
            raw.save(out / 'generated.png')
            Image.alpha_composite(guide.convert('RGBA'),
                raw.convert('RGBA').resize(source.size, Image.Resampling.LANCZOS)).convert('RGB').save(out / 'candidate.png')
            row = dict(case_id=folder.name, protocol=variant, seconds=time.perf_counter() - started,
                       conditioning_audit=pipe.last_conditioning_audit, generated_sha256=sha(out / 'generated.png'))
            write(out / 'generation.json', row); rows.append(row)
            print(json.dumps(row), flush=True)
            write(args.output / 'generation_manifest.json', dict(status='running', cases=rows))
    write(args.output / 'generation_manifest.json', dict(status='complete', cases=rows, seed=41, steps=args.steps))


def project(args):
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0, '/host/space0/guo-z/Evol-SAM3')
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model(bpe_path='/host/space0/guo-z/Evol-SAM3/assets/bpe_simple_vocab_16e6.txt.gz',
        checkpoint_path='/host/space0/guo-z/Evol-SAM3/sam3/sam3.pt', load_from_HF=False, device='cuda', eval_mode=True)
    processor = Sam3Processor(model, confidence_threshold=.18)
    rows = []
    for case in sorted(p for p in args.output.glob('real_*') if p.is_dir()):
        guide = np.asarray(Image.open(case / 'guide.png').convert('RGB'))
        spoon = np.asarray(Image.open(case / 'spoon_mask.png')) > 0
        food = np.asarray(Image.open(case / 'food_mask.png')) > 0
        for variant in ['source_ref', 'guide_ref', 'no_ref']:
            out = case / variant
            im = Image.open(out / 'candidate.png').convert('RGB')
            image_state = processor.set_image(im)
            candidates = []
            for prompt in ['spoon', 'metal spoon']:
                processor.reset_all_prompts(image_state)
                found = processor.set_text_prompt(state=image_state, prompt=prompt)
                masks = found['masks'].detach().cpu().numpy().astype(bool).reshape(-1, *spoon.shape)
                scores = found['scores'].detach().cpu().numpy().reshape(-1)
                candidates.extend((float((m & spoon).sum()) / max(1, int(spoon.sum())) * float(s), float(s), m)
                                  for s, m in zip(scores, masks))
            selected = max(candidates, key=lambda x: x[0]) if candidates else (0., 0., np.zeros(spoon.shape, bool))
            accepted = selected[2] & spoon
            candidate = np.asarray(im)
            final = guide.copy(); final[accepted] = candidate[accepted]
            Image.fromarray(final).save(out / 'final.png')
            Image.fromarray(accepted.astype(np.uint8) * 255).save(out / 'accepted_mask.png')
            row = dict(case_id=case.name, protocol=variant, spoon_pixels=int(spoon.sum()),
                accepted_pixels=int(accepted.sum()), accepted_fraction=float(accepted.sum()) / max(1, int(spoon.sum())),
                selected_score=selected[1], food_rgb_max_change=int(np.abs(final.astype(int)-guide.astype(int))[food].max()),
                nonspoon_rgb_max_change=int(np.abs(final.astype(int)-guide.astype(int))[~spoon].max()),
                generated_food_pixels=0, guide_geometry_changed=False,
                raw_candidate_spoon_region_mae=float(np.abs(candidate.astype(float)-guide.astype(float))[spoon].mean()))
            write(out / 'projection.json', row); rows.append(row); print(json.dumps(row), flush=True)
    write(args.output / 'projection_manifest.json', dict(status='complete', cases=rows,
        protocol_selection='No percase outputranking or retries; all fixed protocols retained for root visualreview'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['generate', 'project'], required=True)
    p.add_argument('--cases', type=Path, nargs='+', default=[]); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=24); args = p.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    (generate if args.stage == 'generate' else project)(args)


if __name__ == '__main__':
    main()
