"""Photographic first-bite experiment: observation anchoring, one field, analytic lift."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageOps

R1 = Path('/host/space0/guo-z/tf-ufi/material_lineage_training_20261004')
SAM_ROOT = Path('/host/space0/guo-z/Evol-SAM3')
MOGE_ROOT = Path('/host/space0/guo-z/tf-ufi/food3d_repair_20260928')

PROMPTS = ['beef', 'noodles', 'chicken', 'pork', 'Japanese pancake', 'sashimi',
           'takoyaki', 'fish shaped pancake', 'noodles', 'omelet', 'noodles',
           'noodles', 'grilled fish', 'fried egg', 'noodles', 'beef']


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def source(case, max_size=640):
    im = ImageOps.exif_transpose(Image.open(R1/'real_probes'/case['copied_original_path'])).convert('RGB')
    im.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return im


def perceive(args, cases):
    import torch
    torch.set_num_threads(4)
    if args.mode == 'segment':
        sys.path.insert(0, str(SAM_ROOT))
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        model = build_sam3_image_model(bpe_path=str(SAM_ROOT/'assets/bpe_simple_vocab_16e6.txt.gz'),
            checkpoint_path=str(SAM_ROOT/'sam3/sam3.pt'), load_from_HF=False, device='cuda', eval_mode=True)
        processor = Sam3Processor(model, confidence_threshold=.18)
    else:
        sys.path[:0] = [str(MOGE_ROOT/'moge'), str(MOGE_ROOT/'utils3d_moge')]
        from moge.model.v2 import MoGeModel
        ckpt = torch.load(MOGE_ROOT/'models/model.pt', map_location='cpu', weights_only=True)
        model = MoGeModel(**ckpt['model_config'])
        model.load_state_dict(ckpt['model'])
        model = model.eval().cuda()
        del ckpt
    rows = []
    for i, case in enumerate(cases):
        if i % args.shards != args.shard or (args.indices is not None and i not in args.indices):
            continue
        folder = args.output/case['case_id']; folder.mkdir(parents=True, exist_ok=True)
        im = source(case); im.save(folder/'source.png')
        started = time.time()
        with torch.inference_mode():
            if args.mode == 'segment':
                state = processor.set_image(im)
                candidates = []; prompts = [PROMPTS[i], 'food']
                if i in [0,15]:prompts=['beef slices','meat','beef','food']
                for prompt in prompts:
                    processor.reset_all_prompts(state)
                    result = processor.set_text_prompt(state=state, prompt=prompt)
                    masks = result['masks'].detach().cpu().numpy().astype(bool).reshape(-1, im.height, im.width)
                    scores = result['scores'].detach().cpu().numpy().reshape(-1)
                    candidates.extend([(float(s), prompt, m) for s, m in zip(scores, masks)])
                # Caption names select a semantic object; all candidates remain available.
                specific = [(s, p, m) for s,p,m in candidates if p != 'food' and .006 < m.mean() < .88]
                available = specific or [(s,p,m) for s,p,m in candidates if .006 < m.mean() < .88]
                selected = max(available, key=lambda v: v[0]) if available else (0., 'none', np.zeros((im.height, im.width),bool))
                masks = np.stack([v[2] for v in candidates]) if candidates else np.zeros((0,im.height,im.width),bool)
                np.savez_compressed(folder/'segmentation.npz', masks=masks, scores=[v[0] for v in candidates],
                    selected_mask=selected[2], prompts=np.asarray([v[1] for v in candidates]))
                Image.fromarray(selected[2].astype(np.uint8)*255).save(folder/'food_mask.png')
                row = dict(case_id=case['case_id'], selected_prompt=selected[1], confidence=selected[0],
                    candidate_count=len(candidates), mask_fraction=float(selected[2].mean()), seconds=time.time()-started)
                write(folder/'segmentation.json', row)
            else:
                tensor = torch.from_numpy(np.asarray(im).copy()).cuda().float().permute(2,0,1)/255
                result = model.infer(tensor, resolution_level=7, use_fp16=True)
                maps = {k: v.detach().cpu().numpy() for k,v in result.items()}
                np.savez_compressed(folder/'depth.npz', **maps)
                row = dict(case_id=case['case_id'], seconds=time.time()-started,
                    model='MoGe2', metric_truth=False, intrinsics=maps['intrinsics'].tolist())
                write(folder/'depth.json', row)
        rows.append(row)
        print(args.mode, json.dumps(row), flush=True)
        write(args.output/(args.mode+'_manifest.json'), dict(status='processing', cases=rows))
    write(args.output/(args.mode+'_manifest.json'), dict(status='complete', cases=rows))


def render(args, cases):
    import torch
    import traceback
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    helper='mld2_real_geometry'
    if args.geometry_variant=='local_scale':helper='mld2_real_geometry_v2'
    from importlib import import_module
    build_case=import_module(helper).build_case
    model = None
    evidence=dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path(__file__).with_name(helper+'.py').read_bytes()).hexdigest(),
        geometry_variant=args.geometry_variant,
        source_dataset_manifest_sha256=hashlib.sha256((R1/'real_probes/manifest.json').read_bytes()).hexdigest())
    if args.checkpoint:
        from foodstateedit.material_lineage.mld2 import MLD2Model, MLD2Config
        ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        model_type=MLD2Model
        if ckpt.get('model_variant') == 'depth_surface':
            from foodstateedit.material_lineage.mld2_depth import DepthSurfaceMLD2, DepthSurfaceDetachedMLD2
            model_type=DepthSurfaceDetachedMLD2 if ckpt.get('observation_maps_detached') else DepthSurfaceMLD2
        model=model_type(MLD2Config(**ckpt['model_config'])).cuda().eval()
        model.load_state_dict(ckpt['state_dict'])
        if 'coordinate_lowpass_max_frequency' in ckpt:
            from mld2_lowpass import attach_lowpass
            attach_lowpass(model,ckpt['coordinate_lowpass_max_frequency'])
            evidence['coordinate_lowpass_max_frequency']=ckpt['coordinate_lowpass_max_frequency']
        evidence.update(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            optimizer_steps=int(ckpt['optimizer_steps']),model_variant=ckpt.get('model_variant','mld2'))
    rows=[]
    for i,c in enumerate(cases):
        if i % args.shards != args.shard:continue
        if args.indices is not None and i not in args.indices:continue
        folder=args.output/c['case_id']
        try:
            with torch.inference_mode():row=build_case(folder,model,torch,baseline=args.baseline)
            row['case_id']=c['case_id'];row['status']='rendered'
        except Exception as e:
            (folder/'RENDER_FAILED.txt').write_text(traceback.format_exc())
            row=dict(case_id=c['case_id'],status='failed',error=repr(e))
        rows.append(row);print(json.dumps(row),flush=True)
        write(args.output/('render_baseline_manifest.json' if args.baseline else 'render_manifest.json'),
            dict(status='processing',checkpoint=str(args.checkpoint),evidence=evidence,cases=rows))
    write(args.output/('render_baseline_manifest.json' if args.baseline else 'render_manifest.json'),
        dict(status='complete',checkpoint=str(args.checkpoint),evidence=evidence,cases=rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['segment', 'depth', 'render'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--indices', type=int, nargs='+')
    p.add_argument('--baseline', action='store_true')
    p.add_argument('--geometry-variant',choices=['original','local_scale'],default='original')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = json.loads((R1/'real_probes/manifest.json').read_text())['cases']
    if args.mode in ['segment', 'depth']:
        perceive(args, cases)
    else:
        render(args, cases)


if __name__ == '__main__':
    main()
