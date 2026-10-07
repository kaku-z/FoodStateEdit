"""New metal candidates on corrected v3: direct intersection, no shape warping."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--candidate-root',type=Path,required=True)
    p.add_argument('--indices',type=int,nargs='+');args=p.parse_args()
    sys.path.insert(0,'/host/space0/guo-z/Evol-SAM3')
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    torch.set_num_threads(4)
    base=Path('/host/space0/guo-z/Evol-SAM3')
    model=build_sam3_image_model(bpe_path=str(base/'assets/bpe_simple_vocab_16e6.txt.gz'),checkpoint_path=str(base/'sam3/sam3.pt'),load_from_HF=False,device='cuda',eval_mode=True)
    processor=Sam3Processor(model,confidence_threshold=.18)
    rows=[]
    for index,folder in enumerate(sorted(f for f in args.root.glob('real_*') if f.is_dir())):
        if args.indices is not None and index not in args.indices:continue
        raw=np.asarray(Image.open(folder/'final_v3.png').convert('RGB'));source=np.asarray(Image.open(folder/'source.png').convert('RGB'))
        d=args.candidate_root/folder.name
        candidate=Image.open(d/'candidate_composite.png').convert('RGB')
        old_spoon=np.asarray(Image.open(folder/'spoon_mask.png'))>0
        guide=binary_dilation(old_spoon,iterations=3)
        candidates=[]
        with torch.inference_mode():
            state=processor.set_image(candidate)
            for prompt in ['metal spoon','stainless steel spoon','spoon']:
                processor.reset_all_prompts(state)
                result=processor.set_text_prompt(state=state,prompt=prompt)
                masks=result['masks'].detach().cpu().numpy().astype(bool).reshape(-1,*raw.shape[:2])
                scores=result['scores'].detach().cpu().numpy().reshape(-1)
                for mask,score in zip(masks,scores):
                    overlap=float((mask&guide).sum()/max(1,guide.sum()))
                    candidates.append((mask,float(score),prompt,overlap))
        masks=np.stack([c[0] for c in candidates]) if candidates else np.zeros((0,*raw.shape[:2]),bool)
        np.savez_compressed(d/'candidate_spoon_segmentation.npz',masks=masks,scores=[c[1] for c in candidates],prompts=[c[2] for c in candidates],guide_overlap=[c[3] for c in candidates])
        usable=[c for c in candidates if c[1]>=.25 and c[3]>=.02 and .0001<c[0].mean()<.25]
        selected=max(usable,key=lambda c:c[1]*c[3]) if usable else None
        destination=np.asarray(Image.open(folder/'spoon_mask.png'))>0
        final=raw.copy();mapping={};candidate_mask=np.zeros(raw.shape[:2],bool)
        if selected is not None:
            candidate_mask=selected[0]
            overlap=candidate_mask&destination
            final[overlap]=np.asarray(candidate)[overlap]
            mapping=dict(method='Direct original-coordinate candidate_metal intersection with fixed v3 visible spoon; uncovered pixels keep PBR',used_pixels=int(overlap.sum()))
        Image.fromarray(candidate_mask.astype(np.uint8)*255).save(d/'candidate_spoon_mask.png')
        Image.fromarray(final).save(d/'final_v4.png')
        food=np.asarray(Image.open(folder/'food_mask.png'))>0
        core_folder=Path(json.loads((folder/'final_v3.json').read_text())['food_state_source'])
        cavity=(np.asarray(Image.open(core_folder/'source_bite_mask.png'))>0)&~food&~destination
        edit=np.asarray(Image.open(folder/'frozen_edit_mask.png'))>0
        row=dict(case_id=folder.name,index=index,round='Exploratory v4; all first outcomes previously inspected',
            method='Fixed v3 food/cavity/context/utensil shape; new Qwen metal-only candidate, SAM3 intersection without PCA or nearest warping',
            candidate_found=selected is not None,candidate_score=selected[1] if selected else None,
            candidate_prompt=selected[2] if selected else None,candidate_guide_overlap=selected[3] if selected else 0.,
            candidate_count=len(candidates),candidate_model='SAM3',candidate_source='New Qwen2.1 seed41/24steps metal-only generation on corrected v3 guide',
            generated_food_pixels_used=0,spoon_footprint_pixels=int(destination.sum()),spoon_opaque_coverage=1.,
            Qwen_metal_used_pixel_count=mapping.get('used_pixels',0),Qwen_metal_coverage_fraction=mapping.get('used_pixels',0)/max(1,int(destination.sum())),
            carried_food_vs_core_mae=float(np.abs(final.astype(float)-raw.astype(float))[food].mean()),
            source_cavity_vs_core_mae=float(np.abs(final.astype(float)-raw.astype(float))[cavity].mean()),
            non_spoon_vs_v3_mae=float(np.abs(final.astype(float)-raw.astype(float))[~destination].mean()),
            outside_context_vs_source_mae=float(np.abs(final.astype(float)-source.astype(float))[~edit].mean()),
            mapping=mapping,fallback='Core metal retained when no candidate' if selected is None else None,
            geometric_accuracy_truth_available=False)
        (d/'final_v4.json').write_text(json.dumps(row,indent=2));rows.append(row);print(json.dumps(row),flush=True)
        (args.candidate_root/'v4_projection_manifest.json').write_text(json.dumps(dict(status='processing',cases=rows),indent=2))
    (args.candidate_root/'v4_projection_manifest.json').write_text(json.dumps(dict(status='complete',cases=rows),indent=2))


if __name__=='__main__':main()
