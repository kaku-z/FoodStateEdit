"""Freeze disjoint real-photo patch reconstruction and cache the exact joint layout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps

PROMPT = ('A natural camera photograph. Reconstruct the missing observed surface using the material, '
          'color and fine texture in the reference image. Match the surrounding photograph and lighting.')
MODELS = Path('/mnt/tmp/guo-z_first_bite_structure_20260930_models')
VENDOR = Path('/mnt/tmp/guo-z_first_bite_structure_20260930_runtime/vendor')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def dhash(im):
    a = np.asarray(im.convert('L').resize((9, 8), Image.Resampling.LANCZOS))
    return int(''.join('1' if b else '0' for b in (a[:, 1:] > a[:, :-1]).ravel()), 2)


def prepare(args):
    import cv2
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'split_manifest.json').exists(), 'Frozen split already exists'
    hold = args.experiment / 'holdout_sources'
    historical = json.loads((hold / 'input_snapshots/historical_dhash_reference.json').read_text())
    history = json.loads((hold / 'input_snapshots/history_sources_excluded.json').read_text())
    prior = json.loads((hold / 'input_snapshots/prior_manifest.json').read_text())
    new = json.loads((hold / 'manifest.json').read_text())
    blocked = historical['references'] + prior['cases'] + new['cases']
    names = set(history['source_file_names'])
    hashes = set(history['known_source_sha256'])
    near = []
    normalized = set()
    for row in blocked:
        name = row.get('file_name', row.get('source_file_name'))
        names.add(name)
        h = row.get('original_sha256', row.get('source_sha256'))
        if h:
            hashes.add(h)
        p = args.corpus / name
        if p.exists():
            im = ImageOps.exif_transpose(Image.open(p)).convert('RGB')
            normalized.add(hashlib.sha256(np.asarray(im).tobytes()).hexdigest())
            near.append((name, dhash(im)))
        elif row.get('dhash64'):
            near.append((name, int(row['dhash64'], 16)))
    recipe = dict(status='frozen_before_source_selection_and_training', seed=args.seed,
        objective='Reconstruct a masked observed image patch from a geometrically reoriented reference. Targets are original real photographs, not food-removal ground truth.',
        train_count=args.train_count, val_count=args.val_count, target_size=args.size, reference_size=256,
        selection='Ascending sha256(material-binding-v1:+filename); valid images; reject filename, byte SHA256, normalized RGB SHA256, dHash<=8 versus all history/dev16/holdout8 and every earlier accepted source. Every fifth accepted source is validation.',
        masks='One deterministic central elliptical observed patch per image, whole patch hidden in conditioning. This is a visual material patch, not a claimed semantic food segmentation.',
        target='Central square crop resampled to target size, no generated pixels.',
        reference='Bounding crop of masked observed patch, outside ellipse gray, deterministic rotation by a multiple of 90deg; resized to256square. Exact target-to-reference pixel mapping saved.',
        structural_control='Canny edges outside missing patch; zero inside missing patch. No RGB/color of the hidden target leaks through structure control.',
        train_val_overlap='No source overlap; all accepted sources separated by dHash>8. This filter cannot rule out every near duplicate or foundation-model pretraining overlap.',
        prompt=PROMPT, code_sha256=sha(Path(__file__)),
        blocked_filename_count=len(names), blocked_byte_sha256_count=len(hashes), blocked_dhash_count=len(near),
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    write(out / 'frozen_recipe.json', recipe)
    paths = sorted(args.corpus.glob('*.jpg'), key=lambda p:hashlib.sha256(('material-binding-v1:'+p.name).encode()).hexdigest())
    rows, decisions = [], []
    used_rgb, used_sha, used_near = set(normalized), set(hashes), list(near)
    counts = {'train':0, 'val':0}
    for p in paths:
        d = {'file_name':p.name}
        if p.name in names:
            d['status']='blocked_filename'; decisions.append(d); continue
        h = sha(p); d['sha256']=h
        if h in used_sha:
            d['status']='exact_duplicate'; decisions.append(d); continue
        try:
            im = ImageOps.exif_transpose(Image.open(p)).convert('RGB')
        except OSError as e:
            d.update(status='invalid_decode',error=str(e)); decisions.append(d); continue
        rgb = hashlib.sha256(np.asarray(im).tobytes()).hexdigest()
        dh = dhash(im)
        nearest = min(((dh ^ v).bit_count(), n) for n,v in used_near)
        d.update(normalized_rgb_sha256=rgb,dhash64=f'{dh:016x}',nearest_distance=nearest[0],nearest_file=nearest[1])
        if rgb in used_rgb or nearest[0] <=8:
            d['status']='normalized_or_near_duplicate'; decisions.append(d); continue
        split='val' if len(rows)%5==4 else 'train'
        if counts[split] >= (args.val_count if split=='val' else args.train_count):
            split='train' if split=='val' else 'val'
        idx=len(rows); case=f'{idx:04d}_{p.stem}'; dest=out/'data'/case;dest.mkdir(parents=True)
        rng=np.random.default_rng(args.seed+idx)
        side=min(im.size); x=(im.width-side)//2;y=(im.height-side)//2
        target=im.crop((x,y,x+side,y+side)).resize((args.size,args.size),Image.Resampling.LANCZOS)
        a=np.asarray(target); s=args.size
        cx,cy=np.array([.5,.5])*s+rng.integers(-s//12,s//12+1,size=2)
        rx,ry=rng.integers(int(.17*s),int(.24*s)+1,size=2)
        yy,xx=np.mgrid[:s,:s];mask=((xx-cx)/rx)**2+((yy-cy)/ry)**2<=1
        bx0,by0,bx1,by1=int(cx-rx),int(cy-ry),int(cx+rx+1),int(cy+ry+1)
        owned=a.copy();owned[~mask]=128
        ref=Image.fromarray(owned).crop((bx0,by0,bx1,by1))
        # Pad to a square before the recorded 90-degree transform, preserving aspect.
        bside=max(ref.size);px=(bside-ref.width)//2;py=(bside-ref.height)//2
        square=Image.new('RGB',(bside,bside),(128,128,128));square.paste(ref,(px,py))
        rotation=idx%4
        ref=Image.fromarray(np.rot90(np.asarray(square),rotation).copy()).resize((256,256),Image.Resampling.LANCZOS)
        edge=cv2.Canny(a,80,160);edge[mask]=0
        target.save(dest/'target.png');ref.save(dest/'reference.png')
        Image.fromarray(mask.astype(np.uint8)*255).save(dest/'mask.png')
        Image.fromarray(edge).convert('RGB').save(dest/'control.png')
        # Forward mapping from every target pixel to reference pixel center before interpolation.
        u=xx-bx0+px;v=yy-by0+py
        for _ in range(rotation):u,v=v,bside-1-u
        uv=np.stack([(u+.5)*256/bside-.5,(v+.5)*256/bside-.5],-1).astype(np.float32)
        np.savez_compressed(dest/'correspondence.npz',target_to_reference_uv=uv,valid=mask)
        row=dict(case_id=case,split=split,original_path=str(p),original_sha256=h,original_size=list(im.size),
            normalized_rgb_sha256=rgb,dhash64=d['dhash64'],nearest_distance=nearest[0],nearest_file=nearest[1],
            crop_xyxy=[x,y,x+side,y+side],target_size=s,reference_size=256,reference_bbox=[bx0,by0,bx1,by1],
            reference_square_padding=[px,py],rotation_ccw_quarters=rotation,
            material_scope='observed image patch; no semantic food segmentation claimed',
            files={n:sha(dest/n) for n in ['target.png','reference.png','mask.png','control.png','correspondence.npz']})
        write(dest/'sample.json',row); rows.append(row);counts[split]+=1
        used_rgb.add(rgb);used_sha.add(h);used_near.append((p.name,dh))
        d['status']='selected';d['split']=split;decisions.append(d)
        if counts=={'train':args.train_count,'val':args.val_count}:break
    assert counts=={'train':args.train_count,'val':args.val_count},counts
    write(out/'selection_decisions.json',decisions)
    write(out/'split_manifest.json',dict(status='frozen',recipe_sha256=sha(out/'frozen_recipe.json'),
        cases=rows,counts=counts,decisions_sha256=sha(out/'selection_decisions.json'),
        frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    print(json.dumps({'status':'split_frozen','counts':counts,'manifest_sha256':sha(out/'split_manifest.json')}),flush=True)


def cache(args):
    import sys
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
    sys.path.insert(0,str(VENDOR))
    import torch
    import torch.nn.functional as F
    from scipy.ndimage import distance_transform_edt
    from diffusers import FlowMatchEulerDiscreteScheduler
    from videox_fun.models import AutoencoderKLQwenImage21,Qwen3VLForConditionalGeneration,Qwen3VLProcessor
    from videox_fun.pipeline.pipeline_qwenimage21_reference_control import QwenImage21ControlPipeline
    torch.set_num_threads(4);dtype=torch.bfloat16;device='cuda'
    model=MODELS/'qwen-image-2.1'
    vae=AutoencoderKLQwenImage21.from_pretrained(str(model),subfolder='vae',torch_dtype=dtype,local_files_only=True).to(device).eval()
    encoder=Qwen3VLForConditionalGeneration.from_pretrained(str(model),subfolder='text_encoder',torch_dtype=dtype,local_files_only=True,attn_implementation='sdpa').to(device).eval()
    processor=Qwen3VLProcessor.from_pretrained(str(model),subfolder='processor',local_files_only=True)
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(str(model),subfolder='scheduler',local_files_only=True)
    pipe=QwenImage21ControlPipeline(vae=vae,text_encoder=encoder,processor=processor,transformer=None,scheduler=scheduler)
    manifest=json.loads((args.output/'split_manifest.json').read_text());out=args.output/'cache';out.mkdir(exist_ok=True)
    def encode(im):
        t=pipe.image_processor.preprocess(im.convert('RGBA')).unsqueeze(2).to(device=device,dtype=dtype)
        z=pipe._encode_vae_image(t,None)
        return pipe._pack_latents(z,1,64,z.shape[-2],z.shape[-1])
    rows=[];started=time.perf_counter()
    for row in manifest['cases']:
        dest=out/(row['case_id']+'.pt')
        if dest.exists():
            rows.append({'case_id':row['case_id'],'cache_sha256':sha(dest),'reused':True});continue
        p=args.output/'data'/row['case_id'];t0=time.perf_counter()
        target=Image.open(p/'target.png').convert('RGB');ref=Image.open(p/'reference.png').convert('RGBA')
        mask=np.asarray(Image.open(p/'mask.png'))>0;size=target.width;lat=size//16
        with torch.inference_mode():
            emb,em,im=pipe.encode_prompt(prompt=PROMPT,image=[ref],device=device,num_images_per_prompt=1)
            z0=encode(target);zr=encode(ref);zc=encode(Image.open(p/'control.png'))
            t=pipe.image_processor.preprocess(target).to(device=device,dtype=dtype)
            m=torch.tensor(mask,device=device)[None,None]
            t=t*(~m);t=torch.cat([t,torch.ones_like(t[:,:1])],1).unsqueeze(2)
            zi=pipe._encode_vae_image(t,None);zi=pipe._pack_latents(zi,1,64,lat,lat)
            keep=F.interpolate((~m).to(dtype),size=(lat,lat),mode='nearest').flatten(2).transpose(1,2)
            control=torch.cat([zc,keep,zi],-1)
            control=torch.cat([torch.zeros(1,zr.shape[1],129,device=device,dtype=dtype),control],1)
            fullmask=torch.cat([im,torch.ones(1,z0.shape[1]//4,device=device,dtype=torch.bool)],1)
            region=F.interpolate(m.float(),size=(lat,lat),mode='area').flatten(2).transpose(1,2)
            core=torch.tensor(distance_transform_edt(mask)>16,device=device)[None,None].float()
            core=F.interpolate(core,size=(lat,lat),mode='area').flatten(2).transpose(1,2)>=.999
            tensors=dict(z0=z0,zref=zr,control_context=control,prompt_embeds=emb,encoder_mask=em,img_mask=fullmask,region=region,core=core)
            tensors={k:v.cpu() if v is not None else None for k,v in tensors.items()}
        tensors.update(case_id=row['case_id'],img_shapes=[[(1,16,16),(1,lat,lat)]],sample_sha256=sha(p/'sample.json'))
        torch.save(tensors,dest)
        r={'case_id':row['case_id'],'cache_sha256':sha(dest),'seconds':time.perf_counter()-t0,'layout':{'target_tokens':z0.shape[1],'reference_tokens':zr.shape[1],'prompt_tokens':emb.shape[1]}}
        rows.append(r);print(json.dumps(r),flush=True)
        write(out/'progress.json',dict(status='running',cases=rows,seconds=time.perf_counter()-started))
    write(out/'manifest.json',dict(status='complete',split_sha256=sha(args.output/'split_manifest.json'),
        code_sha256=sha(Path(__file__)),vendor_sha256=sha(VENDOR/'videox_fun/pipeline/pipeline_qwenimage21_reference_control.py'),
        seconds=time.perf_counter()-started,cases=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','cache']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--experiment',type=Path);p.add_argument('--corpus',type=Path,default=Path('/host/space0/guo-z/projects/multimodal_food/uecfood_dataset'))
    p.add_argument('--train-count',type=int,default=128);p.add_argument('--val-count',type=int,default=32);p.add_argument('--size',type=int,default=512);p.add_argument('--seed',type=int,default=20261005)
    a=p.parse_args();prepare(a) if a.mode=='prepare' else cache(a)
