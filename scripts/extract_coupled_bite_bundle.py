"""Hash-verified ZIP extraction, preserving case-distinct Linux receipts on Windows."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def main():
    ap=argparse.ArgumentParser();ap.add_argument('archive',type=Path);ap.add_argument('destination',type=Path)
    ap.add_argument('--sha256',required=True);a=ap.parse_args()
    digest=hashlib.sha256(a.archive.read_bytes()).hexdigest();assert digest==a.sha256
    a.destination.mkdir(parents=True,exist_ok=True);mapped=[];seen=set()
    with zipfile.ZipFile(a.archive) as z:
        for entry in z.infolist():
            p=PurePosixPath(entry.filename);assert not p.is_absolute() and '..' not in p.parts and '\\' not in entry.filename
            if entry.is_dir():continue
            parts=list(p.parts)
            if parts[-1]=='FROZEN.json':parts[-1]='freeze_decision.json'
            rel=Path(*parts);assert str(rel).lower() not in seen;seen.add(str(rel).lower())
            dest=a.destination/rel;dest.parent.mkdir(exist_ok=True,parents=True);data=z.read(entry)
            # Refreshing a running snapshot is not supported; final archives get new directories.
            if dest.exists():assert dest.read_bytes()==data,rel
            else:dest.write_bytes(data)
            if str(rel).replace('\\','/')!=entry.filename:mapped.append({'archive_path':entry.filename,'local_path':str(rel)})
    receipt={'archive':str(a.archive),'sha256':digest,'file_count':len(seen),'path_mappings':mapped}
    (a.destination/(a.archive.stem+'_extraction.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
