"""Verify pulled Day18 bytes and produce local-only diagnostic comparisons."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def decode(path, width, height):
    process = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path),
                              '-vf', f'scale={width}:{height}:flags=lanczos',
                              '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                             check=True, capture_output=True)
    frames = np.frombuffer(process.stdout, np.uint8).reshape(-1, height, width, 3)
    if len(frames) != 21:
        raise ValueError(f'Expected 21 frames: {path}, found {len(frames)}')
    return frames


def composite(raw, reference, alpha):
    weight = alpha.astype(np.float32)[..., None] / 255.0
    return np.clip(np.rint(raw.astype(np.float32) * weight +
                          reference.astype(np.float32) * (1.0 - weight)), 0, 255).astype(np.uint8)


def sheet(images, labels, path, columns=3):
    width, height = images[0].size
    rows = (len(images) + columns - 1) // columns
    result = Image.new('RGB', (width * columns, (height + 32) * rows), 'white')
    draw = ImageDraw.Draw(result)
    for i, (im, label) in enumerate(zip(images, labels)):
        x, y = (i % columns) * width, (i // columns) * (height + 32)
        draw.text((x + 8, y + 8), label, fill='black')
        result.paste(im, (x, y + 32))
    result.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    config_path = repo / 'configs/day18_high_lift_swept_support_gp40_v1.json'
    config = read_json(config_path)
    run = read_json(args.run_root / 'run_manifest.json')
    preflight = read_json(args.preflight)
    assert run['status'] == 'complete_requires_action_and_photo_review'
    assert run['config_sha256'] == digest(config_path)
    assert run['preflight_sha256'] == digest(args.preflight)
    assert run['runner_sha256'] == digest(repo / config['implementation']['runner'])
    assert run['pipeline_load_count'] == 1 and run['new_vace_inference_count'] == 1
    assert run['decoded_frames'] == 21 and not run['lora_enabled']
    assert run['outside_support_max_pixel_difference'] == 0
    assert preflight['passed'] and all(preflight['checks'].values())
    verified = []
    for name, rec in run['artifacts'].items():
        path = args.run_root / Path(rec['path']).name
        assert path.stat().st_size == rec['size_bytes'], path
        assert digest(path) == rec['sha256'], path
        verified.append(dict(name=name, path=path.name, size_bytes=path.stat().st_size,
                             sha256=digest(path)))
    assert len(verified) == 6
    data = repo / 'artifacts/day18_high_lift_swept_support_v1'
    for rec in [config['dataset']['manifest'], *config['dataset']['files'].values()]:
        assert digest(data / rec['path']) == rec['sha256']
    reference = np.asarray(Image.open(data / config['dataset']['files']['reference']['path']).convert('RGB'))
    alpha = np.asarray(Image.open(data / config['dataset']['files']['edit_alpha']['path']).convert('L'))
    h, w = alpha.shape
    raw = decode(args.run_root / 'raw.mp4', w, h)
    encoded = decode(args.run_root / 'projected.mp4', w, h)
    projected = composite(raw, reference, alpha)
    outside = alpha == 0
    inside = ~outside
    raw_diff = np.abs(raw.astype(np.int16) - reference.astype(np.int16))
    projection_diff = np.abs(projected.astype(np.int16) - reference.astype(np.int16))
    encoded_diff = np.abs(encoded.astype(np.int16) - reference.astype(np.int16))
    assert int(projection_diff[:, outside].max(initial=0)) == 0
    old_root = repo / 'artifacts/day17_high_lift_vace_gp38_20260907_v1'
    old_run = read_json(old_root / 'run_manifest.json')
    assert old_run['config_sha256'] == digest(repo / 'configs/day17_high_lift_vace_pilot_v1.json')
    assert len(old_run['artifacts']) == 6
    for rec in old_run['artifacts'].values():
        old_file = old_root / Path(rec['path']).name
        assert old_file.stat().st_size == rec['size_bytes']
        assert digest(old_file) == rec['sha256']
    old_raw_path = old_root / 'raw.mp4'
    assert digest(old_raw_path) == '7199ad64468d7af47ce72cccb58922f93fe2b3360c55e630d6b5ec76a927cc99'
    old_raw = decode(old_raw_path, w, h)
    repaired_old = composite(old_raw, reference, alpha)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    indices = config['inference']['review_frame_indices']
    sheet([Image.fromarray(raw[i]) for i in indices], [f'Native VACE / frame {i}' for i in indices],
          output / 'native_raw_review.png')
    sheet([Image.open(old_root / 'projected_final_hold.png').convert('RGB'),
           Image.fromarray(repaired_old[-1]), Image.open(args.run_root / 'projected_final_hold.png').convert('RGB')],
          ['Day17: old support / same seed 1', 'Day17: same raw / final alpha repaired',
           'Day18: new control + VACE mask + final alpha'], output / 'final_hold_comparison.png')
    sheet([Image.open(old_root / 'projected_final_hold.png').convert('RGB').crop((400, 40, 688, 290)),
           Image.fromarray(repaired_old[-1]).crop((400, 40, 688, 290)),
           Image.open(args.run_root / 'projected_final_hold.png').convert('RGB').crop((400, 40, 688, 290))],
          ['Day17 old projection', 'Same raw, new alpha', 'Day18 regenerated'], output / 'contact_crops.png')
    report = dict(schema_version='foodstateedit.high_lift_local_verification.v1',
                  config_sha256=digest(config_path), run_manifest_sha256=digest(args.run_root / 'run_manifest.json'),
                  preflight_sha256=digest(args.preflight), verified_artifacts=verified,
                  verified_artifact_count=len(verified), mismatches=0,
                  preserved_day17_artifacts_reverified=len(old_run['artifacts']),
                  pipeline_load_count=run['pipeline_load_count'], raw_frames=len(raw), projected_frames=len(encoded),
                  support_fraction=float(inside.mean()),
                  native_inside_support_mae_to_source=float(raw_diff[:, inside].mean()),
                  native_outside_support_mae_to_source=float(raw_diff[:, outside].mean()),
                  recomposited_inside_support_mae_to_source=float(projection_diff[:, inside].mean()),
                  recomposited_outside_support_max_difference=int(projection_diff[:, outside].max(initial=0)),
                  encoded_outside_support_mae_to_source=float(encoded_diff[:, outside].mean()),
                  encoded_outside_support_max_difference=int(encoded_diff[:, outside].max(initial=0)),
                  metric_limit='pixel-change diagnostics only, not semantic quality or physical validity',
                  visualization_limit='agent technical review required; no automatic visual success',
                  derived_from_old_raw_sha256=digest(old_raw_path))
    with (output / 'verification.json').open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
