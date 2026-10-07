"""Create the owner's private research archive after all 100 cells are verified."""
import datetime
import hashlib
import json
import zipfile
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    completion = json.loads((root/'results/FIRST_BITE_COMPLETION_20260928.json').read_text(encoding='utf-8'))
    assert completion['model_calls'] == completion['raw_outputs_verified'] == 100
    assert completion['generation_failures'] == 0 and completion['experiment_gpu_workers_remaining'] == 0
    paths = set()
    for pattern in ['scripts/*first_bite*.py', 'tests/test_first_bite*.py', 'configs/first_bite*',
                    'results/FIRST_BITE_*.json', 'docs/FIRST_BITE_*.md']:
        paths.update(root.glob(pattern))
    for name in ['foodstateedit/__init__.py', 'scripts/run_qwen_image_edit_direct_baseline.py',
                 'benchmark/DATASET_PROVENANCE.md', 'docs/OBSERVED_EDIT_CURRENT_STATUS.md']:
        paths.add(root/name)
    for name in ['foodstateedit/first_bite', 'benchmark/first_bite_20260928',
                 'outputs/first_bite_20260928_bundle_v1', 'outputs/first_bite_20260928_height_bundle_v1',
                 'outputs/first_bite_20260928_supplement_bundle_v2', 'outputs/first_bite_20260928_annotation_audit',
                 'outputs/first_bite_20260928_gp40', 'outputs/first_bite_20260928_supplement_gp40',
                 'outputs/first_bite_20260928_review', 'outputs/first_bite_20260928_supplement_review',
                 'outputs/first_bite_20260928_execution_logs']:
        directory = root/name
        assert directory.is_dir(), name
        paths.update(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for record in ['FIRST_BITE_EXECUTION_FREEZE_20260928.json', 'FIRST_BITE_SUPPLEMENT_FREEZE_20260928.json']:
        frozen = json.loads((root/'results'/record).read_text(encoding='utf-8'))
        for name, digest in frozen['files'].items():
            assert sha(root/name) == digest, name
            paths.add(root/name)
    paths = sorted(paths)
    assert all(p.is_file() for p in paths)
    manifest = {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'scope':'Private owner research archive; includes UECFOOD sources and private blind-review keys. Do not publish or give labeled galleries/keys to blind raters.',
                'files':{p.relative_to(root).as_posix():sha(p) for p in paths}}
    archive = root/'outputs/first_bite_20260928_evaluation_v1.zip'
    assert not archive.exists(), 'Archive is immutable; do not overwrite.'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in paths:
            z.write(p, p.relative_to(root).as_posix())
        z.writestr('ARCHIVE_MANIFEST.json', json.dumps(manifest,indent=2,ensure_ascii=False))
        z.writestr('READ_ME.txt', 'Private research archive. Start with docs/FIRST_BITE_RESULTS_20260928.md.\nOpen outputs/first_bite_20260928_review/index.html and outputs/first_bite_20260928_supplement_review/index.html locally.\nAll 100 outputs are model-generated edits of 8 real source photos; no paired real after-action ground truth or independent human ballots.\nModels are not bundled; use the frozen model/pipeline hashes and runtime records.\nDo not publicly redistribute UECFOOD images. Keep PRIVATE_blind_keys.json and labeled diagnostic galleries away from blind raters.\n')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, digest in manifest['files'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
    receipt = {'path':str(archive), 'sha256':sha(archive), 'bytes':archive.stat().st_size,
               'files_hashed':len(paths), 'zip_integrity_verified':True,
               'server_copy':'not yet confirmed'}
    (root/'results/FIRST_BITE_ARCHIVE_RECEIPT_20260928.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
