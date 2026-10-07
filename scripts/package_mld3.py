"""Package the implemented algorithm, frozen prior, tests and run instructions."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    paths = list((repo/'foodstateedit').glob('*.py')) + list((repo/'foodstateedit/material_lineage').glob('*.py'))
    script_names = ['run_mld3.py', 'run_mld3_gp40.sh', 'refine_mld3_metal.py', 'prepare_mld3_ingredients.py',
        'mld2_real_geometry.py', 'mld2_real_geometry_v2.py', 'mld2_real_spoon_v3.py',
        'evaluate_mld3_integrated.py', 'build_mld3_gallery.py', 'build_mld3_framework.py',
        'benchmark_mld3_deformation.py', 'probe_mld3_strands.py', 'probe_mld3_metal_conditioning.py', 'package_mld3.py']
    paths += [repo/'scripts'/name for name in script_names]
    paths += [repo/'tests'/name for name in ['test_material_deformation.py', 'test_material_strands.py', 'test_material_topology.py']]
    contents = {str(path.relative_to(repo)).replace('\\', '/'): path.read_bytes() for path in paths}
    for name in ['README.md', 'algorithm.md', 'analysis_summary.md', 'runtime_versions.json', 'tests.json', 'experiment_plan.json',
                 'artifact_verification.json', 'executed_artifact_hashes.json', 'visual_review.json',
                 'evaluation/integrated_summary.json', 'evaluation/independent_final_visual_review.json',
                 'framework.png', 'framework.svg', 'weights/mld2_source65k.pt']:
        contents[name] = (output/name).read_bytes()
    manifest = {name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) for name, data in sorted(contents.items())}
    with zipfile.ZipFile(output/'algorithm_code_and_weights.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
        archive.writestr('package_manifest.json', json.dumps(manifest, indent=2)+'\n')
    # Read all archive entries back and compare exact bytes to the source manifest.
    with zipfile.ZipFile(output/'algorithm_code_and_weights.zip') as archive:
        for name, entry in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == entry['sha256'], name
    record = dict(status='verified', files=len(manifest), sha256=hashlib.sha256((output/'algorithm_code_and_weights.zip').read_bytes()).hexdigest(),
        bytes=(output/'algorithm_code_and_weights.zip').stat().st_size, files_manifest=manifest,
        third_party_large_models_included=False, runtime='Existing gp40 runtimes and model paths documented in README.')
    (output/'package_verification.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: record[key] for key in ['status', 'files', 'sha256', 'bytes']}))


if __name__ == '__main__':
    main()
