"""Combine independently verified visible-material and utensil-shaft channels."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from run_mld4_reconstruction import sha, write


def rgb(path):
    return np.asarray(Image.open(path).convert('RGB'))


def run(args):
    protocol=json.loads((args.production/'protocol.json').read_text(encoding='utf-8'))
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for case in protocol['cases']:
        cid=case['case_id']
        a=args.production/'real'/cid
        b=args.material/'real'/cid
        c=args.handle/'real'/cid
        guide=args.geometry/cid
        original=rgb(a/'source.png')
        base=rgb(a/'final.png')
        material=rgb(b/'final.png')
        shaft=rgb(c/'final.png')
        edit=np.asarray(Image.open(guide/'acceptance_mask.png'))>127
        old_edit=np.asarray(Image.open(a/'edit_mask.png'))>127
        with np.load(b/'material_contract.npz') as z:
            known=z['known'];core=z['core']
        assert not (edit & known).any(), 'Material and shaft ownership must be disjoint'
        assert np.array_equal(material[~known],base[~known])
        assert np.array_equal(shaft[~edit],base[~edit])
        final=material.copy();final[edit]=shaft[edit]
        allowed=old_edit|edit
        source_region=np.asarray(Image.open(guide/'source_inpaint_mask.png'))>127
        assert np.array_equal(final[source_region],base[source_region])
        assert np.array_equal(final[~allowed],original[~allowed])
        out=args.output/'real'/cid;out.mkdir(parents=True)
        Image.fromarray(final).save(out/'final.png')
        Image.fromarray(allowed.astype(np.uint8)*255).save(out/'edit_mask.png')
        for name in ['source.png','guide.png','moved_food_mask.png','source_removed_mask.png']:
            shutil.copy2(a/name,out/name)
        shutil.copy2(b/'bound_core_mask.png',out/'bound_core_mask.png')
        record=dict(case_id=cid,index=case['index'],group='development' if case['index']<16 else 'frozen_new_sources',
            final_sha256=sha(out/'final.png'),source_sha256=sha(out/'source.png'),
            inputs=dict(generative=sha(a/'final.png'),material=sha(b/'final.png'),shaft=sha(c/'final.png')),
            source_protocol_sha256=sha(args.production/'protocol.json'),
            changed_outside_final_edit_pixels=0,material_shaft_overlap_pixels=0,
            bound_core_pixels=int(core.sum()),known_pixels=int(known.sum()),
            source_region_unchanged_from_generative_base=True,
            uniform_method=True,per_case_selection=False,
            method='Full-frame reconstruction + local depth-conditioned bite + explicit observed-material radiance + camera-connected shaft',
            physical_ground_truth=False,photorealism_verified=False,
            limitation='Source exposure and unobserved material remain generative hypotheses; implementation contracts are not complete visual success.')
        write(out/'state.json',record);rows.append(record)
    write(args.output/'manifest.json',dict(status='complete',cases=rows,
        runner_sha256=sha(Path(__file__)),source_protocol_sha256=sha(args.production/'protocol.json'),
        material_manifest_sha256=sha(args.material/'manifest.json'),
        handle_protocol_sha256=sha(args.handle/'protocol.json'),
        selection='One uniform composition for all24; no output-specific parameter or candidate selection',
        scope='This retrospective integration combines two separately frozen branches; original A remains available for evaluation.'))
    print(json.dumps(dict(cases=len(rows),bound_core_pixels=sum(x['bound_core_pixels'] for x in rows))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--production',type=Path,required=True)
    p.add_argument('--material',type=Path,required=True)
    p.add_argument('--handle',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
