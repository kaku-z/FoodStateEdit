"""Full validation, including per-scene mass and independent uniform queries."""
from pathlib import Path
import argparse
import torch
import train_mld2 as base
from train_mld2_continuous import ContinuousData, uniform_metrics
from foodstateedit.material_lineage.mld2 import MLD2Config, MLD2Model
from foodstateedit.material_lineage.mld2_depth import DepthSurfaceMLD2, DepthSurfaceDetachedMLD2
from mld2_lowpass import attach_lowpass
from export_mld2_metrics import export


def main(args):
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda')
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model_class = DepthSurfaceMLD2 if ck.get('model_variant') == 'depth_surface' else MLD2Model
    if ck.get('observation_maps_detached'):
        model_class = DepthSurfaceDetachedMLD2
    model = model_class(MLD2Config(**ck['model_config'])).to(device)
    model.load_state_dict(ck['state_dict'])
    if 'coordinate_lowpass_max_frequency' in ck:
        attach_lowpass(model, ck['coordinate_lowpass_max_frequency'])
    data = ContinuousData(args.dataset_root, args.supplement_root, 41)
    data.groups[2] = data.groups[1]
    result = base.evaluate(model, data, root, device, ck['seed'], ck.get('no_image', False), residual=False)
    result['split'] = 'validation'
    result['validation_scene_count'] = result.pop('test_scene_count')
    result['uniform_validation'] = uniform_metrics(model, data, device, 1, ck.get('no_image', False),
                                                  output=root/'uniform_evaluation.json')
    base.json_write(root/'evaluation.json', result)
    export(root)
    print('Completed full validation: 1024 scenes, 4096 grid queries and 1024 independent uniform queries each.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--supplement-root', required=True)
    p.add_argument('--output-root', required=True)
    main(p.parse_args())
