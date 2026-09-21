"""Create a private runtime copy and add optional soft projection to its loop.

Only the exact frozen E6 replacement site may be patched. Original runtime and
model weights remain untouched. Run before GPU loading on the remote machine.
"""
import argparse
import json
from pathlib import Path
import shutil

from run_high_lift_vace_pilot import sha256, write_json


OLD = '''                        inputs_shared["latents"] = (
                            inputs_shared["latents"] * (1.0 - replace_mask)
                            + noisy_ref * replace_mask
                        )'''
NEW = '''                        policy = getattr(self, "cavity_projection_policy", None)
                        if policy is None:
                            inputs_shared["latents"] = (
                                inputs_shared["latents"] * (1.0 - replace_mask)
                                + noisy_ref * replace_mask
                            )
                        else:
                            from cavity_state_projection import soft_cavity_projection
                            inputs_shared["latents"] = soft_cavity_projection(
                                inputs_shared["latents"], noisy_ref, replace_mask,
                                global_step, layer_endpoints["hole"],
                                max_strength=policy["max_strength"],
                                feather_kernel=policy["feather_kernel"],
                                trace=self.cavity_projection_trace,
                            )'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    a = p.parse_args(); bundle = a.bundle.resolve()
    base_path = bundle/'day25_same_condition_ablation_v2_retry.json'
    base = json.loads(base_path.read_text())
    source = Path(base['runtime']['geoedit_root'])
    target = bundle/'runtime'
    rel = 'diffsynth/pipelines/wan_video.py'
    for name, digest in base['runtime']['geoedit_files'].items():
        if sha256(source/name) != digest:
            raise ValueError('frozen runtime changed: '+name)
    code = (source/rel).read_text()
    if code.count(OLD) != 1:
        raise ValueError('expected exactly one projection replacement site')
    if target.exists():
        raise FileExistsError(target)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '.git'))
    (target/rel).write_text(code.replace(OLD, NEW))
    base['runtime']['geoedit_root'] = str(target)
    base['runtime']['geoedit_files'][rel] = sha256(target/rel)
    derived_path = bundle/'e7_runtime_config.json'
    if derived_path.exists():
        raise FileExistsError(derived_path)
    write_json(derived_path, base)
    for variant in ('hard', 'soft'):
        config = json.loads((bundle/f'e7_{variant}.json').read_text())
        config['base_config_sha256'] = sha256(derived_path)
        config['base_config'] = str(derived_path)
        config['controls'] = str(bundle/'controls')
        output = bundle/f'e7_{variant}_remote.json'
        if output.exists():
            raise FileExistsError(output)
        write_json(output, config)
    write_json(bundle/'runtime_derivation.json', {
        'original_config_sha256':sha256(base_path),
        'derived_config_sha256':sha256(derived_path),
        'original_pipeline_sha256':sha256(source/rel),
        'patched_pipeline_sha256':sha256(target/rel),
        'patch_script_sha256':sha256(Path(__file__)),
        'default_behavior':'hard projection unchanged; soft policy requires explicit pipe attribute',
    })
    print(json.dumps({'runtime':str(target),'config_sha256':sha256(derived_path)}))


if __name__ == '__main__':
    main()
