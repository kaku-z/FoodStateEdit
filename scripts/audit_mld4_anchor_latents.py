"""CPU-only reproduction of the executed sparse anchor mask preprocessing.

Loads the installed VaeImageProcessor, not Qwen, VAE or any model weights.
Reports actual nearest-sampled latent keep cells and anchor/target overlap.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']=''

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import diffusers.image_processor as image_processor_module
from diffusers.image_processor import VaeImageProcessor


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--vendor',type=Path,default=Path('/mnt/tmp/guo-z_first_bite_structure_20260930_runtime/vendor/videox_fun/pipeline/pipeline_qwenimage21_reference_control.py'))
    args=p.parse_args(); root=args.root.resolve(); torch.set_num_threads(2)
    processor=VaeImageProcessor(vae_scale_factor=16,do_normalize=False)
    for candidate in sorted((root/'real').glob('*/candidates/owned_anchors')):
        actual=candidate/'conditioning_mask_used.png'; request_path=candidate/'request.json'
        if not actual.exists() or not request_path.exists(): continue
        request=json.loads(request_path.read_text()); cid=candidate.parent.parent.name
        guide=root/'guides_anchored'/cid
        target_path=guide/'target_food_mask.png'; edit_path=guide/'inpaint_mask.png'
        if sha(actual)!=request.get('conditioning_mask_sha256'): raise ValueError(f'Conditioning mask hash mismatch: {candidate}')
        if sha(edit_path)!=request['input_hashes']['inpaint_mask.png']: raise ValueError(f'Edit guide mismatch: {candidate}')
        width,height=map(int,request['resolution']); lh=2*(height//32); lw=2*(width//32)
        def preprocess(path):
            return (processor.preprocess(Image.open(path),height=height,width=width)>=.5).float()[:,:1]
        condition=preprocess(actual); edit=preprocess(edit_path); target=preprocess(target_path)
        keep=F.interpolate(1-condition,size=(lh,lw),mode='nearest')[0,0].numpy().astype(bool)
        edit_latent=F.interpolate(edit,size=(lh,lw),mode='nearest')[0,0].numpy().astype(bool)
        target_centers=F.interpolate(target,size=(lh,lw),mode='nearest')[0,0].numpy().astype(bool)
        anchor_canvas=((edit>.5)&(condition<.5)).float()
        anchor_centers=F.interpolate(anchor_canvas,size=(lh,lw),mode='nearest')[0,0].numpy().astype(bool)
        target_coverage=F.adaptive_avg_pool2d(target,(lh,lw))[0,0].numpy()
        anchor_coverage=F.adaptive_avg_pool2d(anchor_canvas,(lh,lw))[0,0].numpy()
        target_any=target_coverage>0; anchor_any=anchor_coverage>0
        native_edit=np.asarray(Image.open(edit_path).convert('L'))>127
        native_cond=np.asarray(Image.open(actual).convert('L'))>127
        native_target=np.asarray(Image.open(target_path).convert('L'))>127
        native_anchor=native_edit&~native_cond
        # Every introduced keep bit should have an actual retained anchor at
        # its sampled canvas position. This checks the mask path, not food truth.
        added_keep=keep&edit_latent
        ys,xs=np.where(anchor_any|target_any)
        cells=[{'latent_xy':[int(x),int(y)],'canvas_sample_xy':[int(x*width/lw),int(y*height/lh)],
                'kept':bool(keep[y,x]),'introduced_keep_anchor':bool(added_keep[y,x]),
                'target_at_nearest_sample':bool(target_centers[y,x]),
                'target_cell_occupancy':float(target_coverage[y,x]),'anchor_cell_occupancy':float(anchor_coverage[y,x])}
               for y,x in zip(ys,xs)]
        report={'status':'executed_mask_path_reproduced_without_model_loading','case_id':cid,
            'candidate_folder':str(candidate),'request_sha256':sha(request_path),'conditioning_mask_sha256':sha(actual),
            'guide_hashes':{p.name:sha(p) for p in [target_path,edit_path]},
            'vendor_sha256':sha(args.vendor),'vendor_matches_generation_request':sha(args.vendor)==request.get('vendor_sha256'),
            'processor_source_sha256':sha(image_processor_module.__file__),
            'processor_config':dict(processor.config),'torch_version':torch.__version__,
            'cpu_only':True,'cuda_initialized':torch.cuda.is_initialized(),
            'native_image_size':[native_cond.shape[1],native_cond.shape[0]],'request_canvas_wh':[width,height],
            'latent_grid_wh':[lw,lh],'vae_spatial_factor':16,
            'exact_operations':['VaeImageProcessor(vae_scale_factor=16, do_normalize=False).preprocess(actual mask, request height/width)',
                               'mask_condition = (mask_condition >= 0.5)[:, :1]',
                               'mask_latent = torch.nn.functional.interpolate(1 - mask_condition, size=(2*(height//32), 2*(width//32)), mode=nearest)'],
            'counts':{'native_anchor_pixels':int(native_anchor.sum()),'native_anchor_pixels_inside_target':int((native_anchor&native_target).sum()),
                'request_recorded_native_anchor_pixels':request.get('kept_material_anchor_pixels'),
                'canvas_anchor_pixels_after_preprocess_difference':int(anchor_canvas.sum()),
                'all_kept_latent_cells_including_unchanged_background':int(keep.sum()),
                'all_introduced_keep_latent_cells':int(added_keep.sum()),
                'latent_cells_with_any_target_overlap':int(target_any.sum()),
                'latent_cells_with_target_at_nearest_sample':int(target_centers.sum()),
                'kept_latent_cells_with_any_target_overlap':int((keep&target_any).sum()),
                'kept_latent_cells_with_target_at_nearest_sample':int((keep&target_centers).sum()),
                'latent_cells_with_any_anchor_overlap':int(anchor_any.sum()),
                'latent_cells_with_anchor_at_nearest_sample':int(anchor_centers.sum()),
                'introduced_keep_cells_with_any_target_overlap':int((added_keep&target_any).sum()),
                'introduced_keep_cells_with_target_at_nearest_sample':int((added_keep&target_centers).sum()),
                'introduced_keep_without_anchor_sample':int((added_keep&~anchor_centers).sum())},
            'target_and_anchor_cell_records':cells,
            'interpretation':'Any-overlap counts include partly covered cells; nearest-sample counts reproduce the actual 1-channel keep mask. Surviving bits are sparse control evidence, not a guarantee that decoder output or food semantics are frozen. The RGB image is also VAE-encoded, whose receptive field is not audited here.'}
        out=candidate/'anchor_latent_audit'; out.mkdir(exist_ok=True)
        (out/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
        (out/'audit.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
        np.savez_compressed(out/'latent_masks.npz',keep=keep,added_keep=added_keep,target_centers=target_centers,
                            target_occupancy=target_coverage,anchor_centers=anchor_centers,anchor_occupancy=anchor_coverage)
        vis=np.zeros((lh,lw,3),np.uint8); vis[target_any]=[90,90,90]; vis[target_centers]=[160,160,160]
        vis[anchor_any]=[230,150,30]; vis[added_keep]=[20,230,80]
        Image.fromarray(vis).resize((lw*10,lh*10),Image.Resampling.NEAREST).save(out/'latent_anchor_grid.png')
        print(json.dumps({'case_id':cid,'counts':report['counts'],'cpu_only':True}),flush=True)


if __name__=='__main__': main()
