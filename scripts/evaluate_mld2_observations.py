"""Evaluate predicted image observations on the held-out validation split."""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
import train_mld2 as base
from foodstateedit.material_lineage.mld2 import MLD2Config, MLD2Model
from foodstateedit.material_lineage.mld2_depth import DepthSurfaceMLD2, DepthSurfaceDetachedMLD2
from mld2_lowpass import attach_lowpass


@torch.inference_mode()
def main(args):
    torch.set_num_threads(4)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cls = MLD2Model
    if ck.get('model_variant') == 'depth_surface':
        cls = DepthSurfaceDetachedMLD2 if ck.get('observation_maps_detached') else DepthSurfaceMLD2
    model = cls(MLD2Config(**ck['model_config'])).cuda().eval()
    model.load_state_dict(ck['state_dict'])
    if 'coordinate_lowpass_max_frequency' in ck:
        attach_lowpass(model, ck['coordinate_lowpass_max_frequency'])
    data = base.Data(args.dataset_root)
    scenes = data.groups[1][:args.limit] if args.limit else data.groups[1]
    rows = []
    for start in range(0, len(scenes), 16):
        ids = scenes[start:start+16]
        image = np.asarray(data.a['source_rgb'][ids], dtype=np.float32).transpose(0, 3, 1, 2).copy()/255
        if ck.get('no_image'):
            image[:] = .5
        cache = model.encode_image(torch.from_numpy(image).cuda())
        mask = cache['mask_logits'][:, 0].sigmoid().cpu().numpy() > .5
        depth = cache['depth'][:, 0].cpu().numpy()
        truth = np.asarray(data.a['source_mask'][ids])
        true_depth = np.asarray(data.a['source_depth'][ids], dtype=np.float32)
        for j, index in enumerate(ids):
            occupied = truth[j]
            rows.append({'scene_index': int(index), 'shape_family': data.records[int(index)]['shape_family'],
                'mask_iou': float((mask[j]&occupied).sum()/max((mask[j]|occupied).sum(), 1)),
                'depth_mae': float(np.abs(depth[j]-true_depth[j])[occupied].mean()),
                'depth_bias': float((depth[j]-true_depth[j])[occupied].mean()),
                'predicted_foreground_fraction': float(mask[j].mean())})
    keys = ['mask_iou', 'depth_mae', 'depth_bias', 'predicted_foreground_fraction']
    result = {'checkpoint': args.checkpoint, 'optimizer_steps': ck['optimizer_steps'],
        'model_class': cls.__name__, 'split': 'validation', 'scene_count': len(rows),
        'metrics': {key: float(np.mean([row[key] for row in rows])) for key in keys},
        'per_family': {family: {key: float(np.mean([row[key] for row in rows if row['shape_family']==family]))
                                for key in keys} for family in sorted({row['shape_family'] for row in rows})},
        'per_scene': rows}
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key not in ['per_scene', 'per_family']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--limit', type=int, default=0)
    main(p.parse_args())
