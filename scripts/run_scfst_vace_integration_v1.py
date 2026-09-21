"""Attach the zero-init SCFST adapter and execute the frozen E7 VACE runner.

This is a wiring/full-inference launch experiment, not effectiveness training.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import torch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8')
    os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--controls',type=Path,required=True)
    p.add_argument('--base-config',type=Path,required=True)
    p.add_argument('--scfst-controls',type=Path,required=True)
    p.add_argument('--scfst-config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--checkpoint-sha256')
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--preflight-report',type=Path,required=True)
    a=p.parse_args()
    if a.output_root.exists() or a.preflight_report.exists():
        raise FileExistsError('fresh output and preflight paths required')
    scfg=json.loads(a.scfst_config.read_text(encoding='utf-8'))
    checkpoint_sha256=None
    if a.checkpoint is not None:
        if not a.checkpoint.is_file():
            raise FileNotFoundError(a.checkpoint)
        checkpoint_sha256=sha256(a.checkpoint)
        if a.checkpoint_sha256 and checkpoint_sha256!=a.checkpoint_sha256:
            raise ValueError('SCFST checkpoint hash mismatch')
    sman_path=a.scfst_controls/'manifest.json'
    if sha256(sman_path)!=scfg['controls_manifest_sha256']:
        raise ValueError('SCFST controls manifest mismatch')
    sman=json.loads(sman_path.read_text(encoding='utf-8'))
    token_path=a.scfst_controls/'state_transfer_tokens.npz'
    if sha256(token_path)!=sman['files']['state_transfer_tokens.npz']['sha256']:
        raise ValueError('SCFST token hash mismatch')
    with np.load(token_path,allow_pickle=False) as z:
        conditions=z['condition'].astype(np.float32,copy=True)
        active=z['active'].astype(bool,copy=True)
    if list(conditions.shape)!=sman['shape'] or active.shape!=conditions.shape[:2]:
        raise ValueError('SCFST token schema mismatch')

    base=json.loads(a.base_config.read_text(encoding='utf-8'))
    runtime_root=Path(base['runtime']['geoedit_root'])
    script_dir=Path(__file__).resolve().parent
    package_root=script_dir if (script_dir/'foodstateedit').is_dir() else script_dir.parent
    if not (package_root/'foodstateedit/state_transfer/routing.py').is_file():
        raise FileNotFoundError('bundled state-transfer package is missing')
    sys.path.insert(0,str(package_root))
    sys.path.insert(0,str(runtime_root))
    from foodstateedit.state_transfer.routing import StateTransferAdapter
    from geoedit import inference
    import run_e7_cavity_repair_v1 as e7
    if a.checkpoint is not None:
        from safetensors.torch import load_file
        checkpoint_state=load_file(str(a.checkpoint),device='cpu')
    else:
        checkpoint_state=None

    original_load=inference.load_pipeline
    pipe_box={}; stop=threading.Event()

    def summaries():
        pipe=pipe_box.get('pipe')
        rows={}
        if pipe is not None:
            for name in ('dit','dit2'):
                model=getattr(pipe,name,None)
                if model is not None:
                    trace=getattr(model,'state_transfer_trace',[])
                    rows[name]={'calls':len(trace),'last':trace[-1] if trace else None}
        return rows

    def progress_writer():
        while not stop.wait(5):
            if a.output_root.exists():
                atomic_json(a.output_root/'state_transfer_progress.json',
                            {'status':'running','updated_unix':time.time(),'branches':summaries()})

    def configured_load(vram_limit):
        pipe=original_load(vram_limit)
        branch_info={}
        for name,label in (('dit','high_noise'),('dit2','low_noise')):
            model=getattr(pipe,name,None)
            if model is None:
                continue
            channels=int(scfg['reference_channels'])
            if channels <= 0:
                raise ValueError('reference_channels must be positive')
            adapter=StateTransferAdapter(int(model.dim),channels,int(scfg['hidden_dim']))
            loaded_checkpoint=None
            if name=='dit' and checkpoint_state is not None:
                adapter.load_state_dict(checkpoint_state,strict=True)
                loaded_checkpoint=str(a.checkpoint)
            adapter=adapter.to(dtype=next(model.parameters()).dtype)
            # These are the only trainable output paths; zero means exact identity.
            output_l1=float(adapter.route_out.weight.abs().sum().item()+
                            adapter.state_net[-1].weight.abs().sum().item()+
                            adapter.state_net[-1].bias.abs().sum().item())
            if checkpoint_state is None and output_l1!=0:
                raise AssertionError('adapter is not zero initialized')
            model.state_transfer_adapter=adapter
            model.state_transfer_conditions=torch.from_numpy(conditions)
            model.state_transfer_active=torch.from_numpy(active)
            model.state_transfer_block_ids=tuple(map(int,scfg['block_ids']))
            model.state_transfer_strength=float(scfg['strength'])
            model.state_transfer_reference_channels=channels
            model.state_transfer_trace=[]
            model.state_transfer_branch=label
            branch_info[name]={'label':label,'token_dim':int(model.dim),
                               'reference_channels':channels,
                               'source_memory':'vace_context[:,:16,0] or explicit reference_latents',
                               'output_l1':output_l1,
                               'loaded_checkpoint':loaded_checkpoint,
                               'checkpoint_sha256':checkpoint_sha256 if loaded_checkpoint else None}
        if not branch_info:
            raise RuntimeError('no Wan DiT branch configured')
        pipe_box['pipe']=pipe
        atomic_json(a.output_root/'scfst_wiring_manifest.json',{
            'schema_version':'foodstateedit.scfst_vace_wiring.v1',
            'status':'trained_checkpoint_inference' if checkpoint_state is not None else 'zero_init_full_inference_wiring',
            'runner_sha256':sha256(Path(__file__)),
            'adapter_sha256':sha256(package_root/'foodstateedit/state_transfer/routing.py'),
            'runtime_pipeline_sha256':sha256(runtime_root/'diffsynth/pipelines/wan_video.py'),
            'controls_manifest_sha256':sha256(sman_path),
            'scfst_config_sha256':sha256(a.scfst_config),
            'condition_shape':list(conditions.shape),'active_tokens':int(active.sum()),
            'block_ids':scfg['block_ids'],'strength':scfg['strength'],'branches':branch_info,
            'checkpoint_sha256':checkpoint_sha256,
            'claim_limit':('seen synthetic checkpoint inference; visual comparison required; no generalization claim'
                           if checkpoint_state is not None else
                           'execution/wiring only; zero initialized; no algorithmic gain claim'),
        })
        threading.Thread(target=progress_writer,daemon=True).start()
        return pipe

    inference.load_pipeline=configured_load
    sys.argv=[str(Path(e7.__file__)), '--config',str(a.config), '--controls',str(a.controls),
              '--base-config',str(a.base_config),'--gpu',str(a.gpu),
              '--output-root',str(a.output_root),'--preflight-report',str(a.preflight_report)]
    try:
        code=e7.main()
    finally:
        stop.set()
        if a.output_root.exists():
            all_traces={}
            pipe=pipe_box.get('pipe')
            if pipe is not None:
                for name in ('dit','dit2'):
                    model=getattr(pipe,name,None)
                    if model is not None:
                        all_traces[name]=getattr(model,'state_transfer_trace',[])
            atomic_json(a.output_root/'state_transfer_trace.json',all_traces)
            atomic_json(a.output_root/'state_transfer_progress.json',
                        {'status':'process_finished','updated_unix':time.time(),'branches':summaries()})
    return code


if __name__=='__main__':
    raise SystemExit(main())
