"""Portable evidence package; excludes redundant live snapshots and model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
    audit=json.loads((r/'analysis/integrity_audit.json').read_text());assert audit['status']=='verified'
    assert (r/'REPORT.md').exists() and (r/'review/all_results.html').exists()
    code=r/'code';code.mkdir(exist_ok=True)
    for p in Path('scripts').glob('*coupled_bite*.py'):shutil.copy2(p,code/p.name)
    shutil.copy2(Path('docs/COUPLED_FIRST_BITE_REPAIR_20260929.md'),r/'METHOD_AND_DEVELOPMENT.md')
    files=set()
    for folder in ['server_results','review','analysis','figures','code','development_evidence','new_source_candidates',
                   'gate_v1_inputs','gate_v1_results','gate_v1_review','gate_v2_results','gate_v2_review']:
        files.update(p for p in (r/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for folder in r.glob('review_batch_*'):
        files.update(p for p in folder.iterdir() if p.suffix=='.jpg' or p.name=='pages.json')
    for name in ['REPORT.md','METHOD_AND_DEVELOPMENT.md','REPRODUCE.md','assistant_review_notes.json','REVIEW_DEFINITION.json','TECHNICAL_INCIDENTS.json',
                 'gate_v1.json','gate_v2.json','gate_v1_receipt.json','gate_v2_receipt.json',
                 'validation_receipt.json','postcompute_evidence_receipt.json','development_evidence_receipt.json',
                 'new_source_sheet.jpg','selected_sources_native.jpg','validation_geometry_review.jpg']:
        p=r/name
        if p.exists():files.add(p)
    manifest={'status':'complete','scope':'All 72 formal endpoints, 84 raw calls, 2 replay calls, 76 observer records, development evidence and code; weights and dense NPZ maps retained only on gp40.',
              'files':{str(p.relative_to(r)).replace('\\','/'):{'size_bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)}}
    mp=r/'DELIVERY_MANIFEST.json';mp.write_text(json.dumps(manifest,indent=2)+'\n');files.add(mp)
    dest=r.parent/'first_bite_coupled_20260929_delivery.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in sorted(files):z.write(p,p.relative_to(r))
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for n,meta in manifest['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==meta['sha256']
    receipt={'archive':str(dest),'size_bytes':dest.stat().st_size,'sha256':sha(dest),'file_count':len(files),'contents_verified':True}
    (r/'DELIVERY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
