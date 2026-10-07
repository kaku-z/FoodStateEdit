"""Export scalar paired-scene results without copying large predicted fields."""
from pathlib import Path
import argparse
import json
import numpy as np


def export(root):
    root = Path(root)
    evaluation = json.loads((root/'evaluation.json').read_text())
    with np.load(root/'evaluation_predictions.npz') as predictions:
        keys = [key for key in predictions.files if key.endswith('_l1_normalized') or
                key in ['source_iou', 'sdf_mae', 'occupied_linear_rgb_mse']]
        values = {key: predictions[key] for key in keys}
        scenes, families = predictions['scene_indices'], predictions['shape_families']
        rows = [{'scene_index': int(scene), 'shape_family': str(families[index]),
                 **{key: float(array[index]) for key, array in values.items()}}
                for index, scene in enumerate(scenes)]
    development = set(range(7168, 7176))
    blind = [row for row in rows if row['scene_index'] not in development]
    result = {'scope': evaluation['normalized_mass_metric'],
              'synthetic_development_scene_indices': sorted(development),
              'all_test_scene_count': len(rows), 'blind_remainder_scene_count': len(blind),
              'blind_remainder_macro': {key: float(np.mean([row[key] for row in blind])) for key in keys},
              'blind_remainder_per_family': {family: {'scene_count': sum(row['shape_family'] == family for row in blind),
                                                      **{key: float(np.mean([row[key] for row in blind if row['shape_family'] == family])) for key in keys}}
                                              for family in sorted({row['shape_family'] for row in blind})},
              'scene_indices': scenes.tolist(), 'shape_families': families.tolist(),
              **{key: value.tolist() for key, value in values.items()}}
    (root/'per_scene_metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'run': str(root), 'scenes': len(rows), 'scalar_metrics_per_scene': len(keys)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+')
    for root in p.parse_args().runs:
        export(root)
