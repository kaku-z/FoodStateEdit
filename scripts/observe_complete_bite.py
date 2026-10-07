"""Text-only SAM3 diagnostics. Masks are fallible observations, never ground truth."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(16*1024**2),b''):h.update(x)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    c=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4')
    from run_qwen_image_edit_direct_baseline import gpu_snapshot
    snapshot=gpu_snapshot(a.gpu);assert not snapshot['compute_processes'] and snapshot['memory_free_mib']>20000
    sam=c['sam3'];code=Path(sam['code_root'])
    for name in ['checkpoint','bpe']:assert sha(code/sam[name]['path'])==sam[name]['sha256']
    assert sha(code/'sam3/model_builder.py')==sam['model_builder_sha256']
    assert sha(code/'sam3/model/sam3_image_processor.py')==sam['image_processor_sha256']
    sys.path.insert(0,str(code))
    import torch
    import numpy as np
    from PIL import Image
    from scipy.ndimage import label,distance_transform_edt
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    torch.set_num_threads(4)
    model=build_sam3_image_model(bpe_path=str(code/sam['bpe']['path']),device='cuda',eval_mode=True,
        checkpoint_path=str(code/sam['checkpoint']['path']),load_from_HF=False,enable_segmentation=True,enable_inst_interactivity=False,compile=False)
    processor=Sam3Processor(model,device='cuda',confidence_threshold=.5)
    rows=[]
    for item in c['images']:
        start=time.time();path=Path(item['path']);assert sha(path)==item['sha256'];image=Image.open(path).convert('RGB')
        state=processor.set_image(image);row={'id':item['id'],'role':item['role'],'case_id':item['case_id'],'sha256':item['sha256'],'prompts':{}}
        unions={};arrays={}
        for prompt in ['fork','tofu','hand']:
            processor.reset_all_prompts(state)
            with torch.inference_mode():result=processor.set_text_prompt(prompt=prompt,state=state)
            scores=np.asarray(result['scores'].detach().cpu());raw=np.asarray(result['masks'].detach().cpu())
            good=[i for i,m in enumerate(raw) if np.asarray(m).squeeze().sum()>=50]
            masks=[np.asarray(raw[i]).squeeze().astype(bool) for i in good]
            union=np.logical_or.reduce(masks) if masks else np.zeros((image.height,image.width),bool)
            arrays[prompt]=np.stack(masks) if masks else np.zeros((0,image.height,image.width),bool);unions[prompt]=union
            row['prompts'][prompt]={'instance_count':len(masks),'scores':[float(scores[i]) for i in good],'union_pixels':int(union.sum())}
        if item.get('target_mask'):
            target=np.asarray(Image.open(item['target_mask']).convert('L'))>0
            components,count=label(unions['tofu']);candidates=[]
            for i in range(1,count+1):
                component=components==i;area=int(component.sum());overlap=int((component&target).sum())
                if area>=50 and overlap/max(1,area)>.15:candidates.append((overlap,component))
            if candidates:
                _,payload=max(candidates,key=lambda x:x[0]);yy,xx=np.where(payload);ty,tx=np.where(target)
                gap=float(distance_transform_edt(~unions['fork'])[payload].min()) if unions['fork'].any() else None
                row['payload_diagnostic']={'pixels':int(payload.sum()),'centroid_xy':[float(xx.mean()),float(yy.mean())],
                    'target_centroid_error_px':float(np.hypot(xx.mean()-tx.mean(),yy.mean()-ty.mean())),
                    'fork_to_payload_minimum_image_distance_px':gap,
                    'target_mask_iou':float((payload&target).sum()/max(1,(payload|target).sum()))}
            else:row['payload_diagnostic']=None
        np.savez_compressed(a.output/(item['id']+'.npz'),**arrays)
        row['seconds']=time.time()-start;rows.append(row)
        (a.output/'observations.json').write_text(json.dumps({'status':'running','images':rows,'automatic_success_assignment':False},indent=2)+'\n')
        print('OBSERVED',item['id'],flush=True)
    negatives=[r for r in rows if r['role']=='unedited_source']
    report={'status':'complete','images':rows,'threshold':.5,'minimum_mask_pixels':50,
        'negative_controls':{'count':len(negatives),'fork_detections_on_unedited_source':sum(r['prompts']['fork']['instance_count']>0 for r in negatives),
                             'hand_detections_on_unedited_source':sum(r['prompts']['hand']['instance_count']>0 for r in negatives)},
        'automatic_success_assignment':False,
        'scope':'Text-only observer, no expected boxes used as segmentation prompts. Target region is used only to match disconnected food components after inference.',
        'limits':'2-D proximity does not prove 3-D support, lift, source removal, realism or absence of undetected hands.',
        'script_sha256':sha(__file__),'config_sha256':sha(a.config)}
    (a.output/'observations.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
