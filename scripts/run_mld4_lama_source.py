"""Official Big-LaMa generator: exact-bite versus whole-food source completion."""
import argparse
import ast
import collections
import hashlib
import json
import numbers
import os
from pathlib import Path
import shutil
import sys
import time
import types
import typing

import numpy as np
from PIL import Image


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def rgb(path):
    return np.asarray(Image.open(path).convert('RGB')).copy()


def mask(path):
    return np.asarray(Image.open(path).convert('L')) > 127


class OpaqueLegacyTrainingMetadata:
    """Inert containers for unused Lightning/OmegaConf checkpoint metadata."""
    pass


def load_model(vendor):
    os.environ.update(CUDA_VISIBLE_DEVICES='5', OMP_NUM_THREADS='4')
    import torch
    import yaml
    torch.set_num_threads(4)
    torch.manual_seed(41)
    sys.path[:0] = [str(vendor / 'python_deps'), str(vendor / 'lama')]
    # Reuse the exact official helper without importing the training-only Lightning package.
    utils_path = vendor / 'lama/saicinpainting/utils.py'
    utils_tree = ast.parse(utils_path.read_text())
    helper = next(n for n in utils_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_shape')
    utilities = types.ModuleType('saicinpainting.utils')
    utilities.__dict__.update(torch=torch, numbers=numbers)
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(utils_path), 'exec'), utilities.__dict__)
    sys.modules['saicinpainting.utils'] = utilities
    from saicinpainting.training.modules.ffc import FFCResNetGenerator
    cfg = yaml.safe_load((vendor / 'big-lama/config.yaml').read_text())
    def resolve(value):
        if isinstance(value, dict):
            return {k: resolve(v) for k,v in value.items()}
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            result = cfg
            for key in value[2:-1].split('.'):
                result = result[key]
            return resolve(result)
        return value
    kwargs = resolve(cfg['generator'])
    assert kwargs.pop('kind') == 'ffc_resnet'
    model = FFCResNetGenerator(**kwargs)
    checkpoint = vendor / 'big-lama/models/best.ckpt'
    names = torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint)
    opaque_names = [n for n in names if n.startswith('omegaconf.') or n == 'pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint']
    with torch.serialization.safe_globals([int,dict,list,collections.defaultdict,typing.Any] +
               [(OpaqueLegacyTrainingMetadata,n) for n in opaque_names]):
        loaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = {k[len('generator.'):]: v for k,v in loaded['state_dict'].items() if k.startswith('generator.')}
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).cuda()
    audit = dict(torch_version=torch.__version__, physical_gpu=5, generator_kwargs=kwargs,
                 strict_generator_load=True, generator_state_entries=len(state), weights_only=True,
                 unused_training_metadata_substituted_as_inert_data=opaque_names,
                 checkpoint_sha256=sha(checkpoint), configuration_sha256=sha(vendor/'big-lama/config.yaml'),
                 official_code_sha256={str(p.relative_to(vendor/'lama')):sha(p) for p in (vendor/'lama/saicinpainting/training/modules').glob('*.py')},
                 get_shape_helper_source_sha256=sha(utils_path),
                 inference_contract='Official generator and weights, RGB [0,1], zero masked RGB, append mask channel, symmetric pad to modulo 8, native resolution, float32, eval mode; no optimization or prompt.')
    return model, audit


def prepare(a):
    root = a.output / 'lama_source_probe'
    root.mkdir(exist_ok=True)
    frozen = root / ('frozen' if a.experiment=='native' else 'frozen_dilated')
    frozen.mkdir(exist_ok=False)
    rows=[]
    for index in [2,13,14]:
        source_folder, = (a.output/'support_layer_probe').glob(f'real_{index:02d}_*')
        case = source_folder.name
        guide = a.output/'guides_factorized_depth'/case
        dest = frozen/case
        dest.mkdir()
        for n in ['source.png','whole_food_mask.png','removed_mask.png']:
            shutil.copy2(source_folder/n,dest/n)
        for n in ['source_stage_full_mask.png','head_stage_full_mask.png','full_inpaint_mask.png']:
            shutil.copy2(guide/n,dest/n)
        shutil.copy2(a.output/'real'/case/'candidates/factorized_auto_depth/unprojected.png',dest/'base.png')
        source, removed, whole = rgb(dest/'source.png'), mask(dest/'removed_mask.png'), mask(dest/'whole_food_mask.png')
        assert np.array_equal(source,rgb(guide/'full_source.png'))
        assert not np.any(removed & ~whole)
        assert not np.any(removed & mask(dest/'head_stage_full_mask.png'))
        radius=0.
        if a.experiment=='dilated':
            from scipy.ndimage import distance_transform_edt
            radius=12*max(source.shape[:2])/640.
            dilated=distance_transform_edt(~whole)<=radius
            Image.fromarray(dilated.astype(np.uint8)*255).save(dest/'whole_food_dilated_mask.png')
            overlay=source.copy()
            overlay[whole]=np.rint(.55*source[whole]+.45*np.array([255,0,255])).astype(np.uint8)
            overlay[dilated&~whole]=np.rint(.45*source[dilated&~whole]+.55*np.array([0,200,255])).astype(np.uint8)
            Image.fromarray(overlay).save(dest/'dilation_overlay.png')
        for mode in (['exact','whole'] if a.experiment=='native' else ['whole_dilated']):
            model_mask='removed_mask.png' if mode=='exact' else ('whole_food_mask.png' if mode=='whole' else 'whole_food_dilated_mask.png')
            request=dict(case_id=case,mode=mode,variant='source_lama_'+mode+'_v1',
                         source_bbox_xyxy=json.loads((guide/'source/stage_manifest.json').read_text())['bbox_xyxy'],
                         model_mask=model_mask,
                         acceptance_mask='removed_mask.png', mask_pixels=int(mask(dest/model_mask).sum()),
                         accepted_pixels=int(removed.sum()), seam_pixels=0,
                         synthesis_mask_dilation_radius_native_pixels=radius,
                         synthesis_mask_rule='Euclidean distance <= 12*max(H,W)/640' if a.experiment=='dilated' else 'Exact original mask',
                         input_sha256={p.name:sha(p) for p in dest.iterdir() if p.suffix=='.png'},
                         parent_support_request_sha256=sha(source_folder/'request.json'))
            write(dest/(mode+'_request.json'),request)
            rows.append(dict(case_id=case,mode=mode,request_sha256=sha(dest/(mode+'_request.json'))))
    write(frozen/'config.json',dict(cases=rows,runner_sha256=sha(__file__),
          frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
          vendor_provenance_sha256=sha(a.output/'vendor/lama_provenance.json'),
          pipeline='Whole original photograph; official LaMa completes model mask. Reset prior source stage to original, accept exactly removed_mask, preserve head/handle from factorized_auto_depth.',
          limitations='Whole-food masks are existing automatic observations, not verified food/background boundaries. No prompt or per-image manual donor choice. Hidden surface is an unpaired hypothesis.'))
    print(dict(frozen=str(frozen),config_sha256=sha(frozen/'config.json')))


def run(a):
    import torch
    root=a.output/'lama_source_probe';frozen=root/('frozen' if a.experiment=='native' else 'frozen_dilated')
    config=json.loads((frozen/'config.json').read_text())
    assert config['runner_sha256']==sha(__file__)
    model,audit=load_model(a.output/'vendor')
    suffix='' if a.experiment=='native' else '_dilated'
    write(root/('model_audit'+suffix+'.json'),audit)
    completed=[]
    for job in config['cases']:
        folder=frozen/job['case_id'];reqpath=folder/(job['mode']+'_request.json')
        assert sha(reqpath)==job['request_sha256']
        req=json.loads(reqpath.read_text())
        for n,h in req['input_sha256'].items():
            assert sha(folder/n)==h
        out=a.output/'real'/job['case_id']/'candidates'/req['variant']
        out.mkdir(parents=True,exist_ok=False)
        shutil.copy2(reqpath,out/'request.json')
        source=rgb(folder/'source.png');native_h,native_w=source.shape[:2]
        selected=mask(folder/req['model_mask']);removed=mask(folder/'removed_mask.png')
        image_chw=source.transpose(2,0,1).astype(np.float32)/255.
        mask_chw=selected[None].astype(np.float32)
        padding=((0,0),(0,(-native_h)%8),(0,(-native_w)%8))
        image_t=torch.from_numpy(np.pad(image_chw,padding,mode='symmetric'))[None].cuda()
        mask_t=torch.from_numpy(np.pad(mask_chw,padding,mode='symmetric'))[None].cuda()
        started=time.perf_counter()
        with torch.inference_mode():
            predicted=model(torch.cat([image_t*(1-mask_t),mask_t],dim=1))
        torch.cuda.synchronize()
        raw=np.clip(predicted[0,:,:native_h,:native_w].permute(1,2,0).cpu().numpy()*255,0,255).astype(np.uint8)
        Image.fromarray(raw).save(out/'raw_generator.png')
        model_completion=source.copy();model_completion[selected]=raw[selected]
        Image.fromarray(model_completion).save(out/'model_completion.png')
        Image.fromarray(selected.astype(np.uint8)*255).save(out/'model_mask.png')
        Image.fromarray(removed.astype(np.uint8)*255).save(out/'acceptance_mask.png')
        base=rgb(folder/'base.png');full=base.copy();source_stage=mask(folder/'source_stage_full_mask.png')
        full[source_stage]=source[source_stage];full[removed]=raw[removed]
        head=mask(folder/'head_stage_full_mask.png');editable=mask(folder/'full_inpaint_mask.png')
        kept=source_stage&~removed
        assert np.array_equal(full[head],base[head])
        assert np.array_equal(full[kept],source[kept])
        assert np.array_equal(full[~editable],source[~editable])
        Image.fromarray(full).save(out/'candidate.png');Image.fromarray(full).save(out/'unprojected.png')
        row=dict(case_id=job['case_id'],variant=req['variant'],seconds=time.perf_counter()-started,
                 final_sha256=sha(out/'unprojected.png'),raw_sha256=sha(out/'raw_generator.png'),
                 source_sha256=sha(folder/'source.png'),model_mask_sha256=sha(out/'model_mask.png'),
                 model_mask_pixels=int(selected.sum()),accepted_generated_pixels=int(removed.sum()),
                 head_changed_pixels=0,retained_source_changed_pixels=0,exterior_changed_pixels=0,
                 generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        write(out/'generation.json',row);completed.append(row)
        write(root/('worker'+suffix+'.json'),dict(status='running',completed=completed))
        print(json.dumps(row),flush=True)
    write(root/('worker'+suffix+'.json'),dict(status='complete_unreviewed',completed=completed))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check-model','prepare','run'])
    p.add_argument('--experiment',choices=['native','dilated'],default='native')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.action=='check-model':
        m,audit=load_model(a.output/'vendor')
        (a.output/'lama_source_probe').mkdir(exist_ok=True)
        write(a.output/'lama_source_probe/model_preflight.json',audit);print(json.dumps(audit))
    elif a.action=='prepare':prepare(a)
    else:run(a)
