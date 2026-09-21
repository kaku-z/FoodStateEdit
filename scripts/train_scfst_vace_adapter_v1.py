"""Train only the SCFST residual adapter on a frozen Wan/VACE backbone."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import accelerate
import numpy as np
import torch
from safetensors.torch import save_file


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8'); os.replace(tmp,path)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed%(2**32)); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args(); cfg=json.loads(a.config.read_text(encoding='utf-8'))
    if a.output_root.exists(): raise FileExistsError(a.output_root)
    a.output_root.mkdir(parents=True)
    started=time.time(); seed_all(int(cfg['training']['seed']))
    bundle=Path(__file__).resolve().parent
    package_root=bundle if (bundle/'foodstateedit').is_dir() else bundle.parent
    runtime=(a.config.parent/Path(cfg['runtime'])).resolve(); dataset_root=(a.config.parent/Path(cfg['dataset']['root'])).resolve()
    controls_root=(a.config.parent/Path(cfg['controls']['root'])).resolve()
    sys.path.insert(0,str(package_root)); sys.path.insert(0,str(runtime))
    from foodstateedit.state_transfer.routing import StateTransferAdapter
    from diffsynth.core import OffloadTrainingManager, UnifiedDataset
    from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing
    from examples.wanvideo.model_training.train import WanTrainingModule

    sman_path=controls_root/'manifest.json'
    if sha256(sman_path)!=cfg['controls']['manifest_sha256']: raise ValueError('controls manifest mismatch')
    sman=json.loads(sman_path.read_text(encoding='utf-8'))
    tokens_path=controls_root/'state_transfer_tokens.npz'
    if sha256(tokens_path)!=sman['files']['state_transfer_tokens.npz']['sha256']:
        raise ValueError('control token hash mismatch')
    with np.load(tokens_path,allow_pickle=False) as z:
        conditions=z['condition'].astype(np.float32,copy=True)
        active=z['active'].astype(bool,copy=True)
    # Train only local state-transfer tokens; KEEP and reference-prefix tokens are protected.
    roles=np.argmax(conditions[...,:4],axis=-1)
    active &= roles!=0
    if list(conditions.shape)!=cfg['controls']['shape'] or int(active.sum())!=cfg['controls']['train_active_tokens']:
        raise ValueError('training control layout mismatch')

    trace_path=a.output_root/'training_trace.jsonl'
    run_manifest=a.output_root/'run_manifest.json'
    state={'schema_version':'foodstateedit.scfst_adapter_training.v1','status':'initializing',
           'started_unix':started,'pid':os.getpid(),'config_sha256':sha256(a.config),
           'claim_limit':cfg['claim_limit']}
    atomic_json(run_manifest,state)

    class Module(WanTrainingModule):
        def __init__(self):
            tr=cfg['training']; model_root=Path(cfg['model']['root'])
            super().__init__(
                model_paths=json.dumps([str(model_root/path) for path in cfg['model']['files']]),
                tokenizer_path=str(model_root/cfg['model']['tokenizer']),
                trainable_models=None,lora_base_model=None,
                use_gradient_checkpointing=True,use_gradient_checkpointing_offload=True,
                extra_inputs='vace_video,vace_reference_image',device='cpu',task='sft',
                min_timestep_boundary=float(tr['min_timestep_boundary']),
                max_timestep_boundary=float(tr['max_timestep_boundary']),
            )
            model=self.pipe.dit
            adapter=StateTransferAdapter(int(model.dim),int(cfg['adapter']['reference_channels']),
                                         int(cfg['adapter']['hidden_dim']))
            adapter=adapter.to(dtype=self.pipe.torch_dtype)
            model.state_transfer_adapter=adapter
            model.state_transfer_conditions=torch.from_numpy(conditions)
            model.state_transfer_active=torch.from_numpy(active)
            model.state_transfer_block_ids=tuple(map(int,cfg['adapter']['block_ids']))
            model.state_transfer_strength=float(cfg['adapter']['strength'])
            model.state_transfer_reference_channels=int(cfg['adapter']['reference_channels'])
            model.state_transfer_trace=None
            model.state_transfer_branch='high_noise_train'
            self.scfst_adapter=adapter
            for param in adapter.parameters(): param.requires_grad_(True)
            self.forward_step=0
            prefix=int(cfg['controls']['prefix_tokens']); gh,gw=cfg['controls']['grid_hw']
            tokens_per_frame=gh*gw
            if prefix % tokens_per_frame:
                raise ValueError(f'prefix token count {prefix} is not divisible by grid size {tokens_per_frame}')
            prefix_frames=prefix//tokens_per_frame
            data_active=torch.from_numpy(active[:,prefix:]).reshape(1,len(cfg['controls']['frame_indices']),gh,gw)
            # The VACE training latent contains the reference prefix followed by
            # the six action-time slices. SCFST activity is defined only for the
            # action slices, so keep the reference prefix unweighted. The mask is
            # stored on the DiT patch grid and expanded to latent resolution in
            # forward() after the model reveals its exact output shape.
            prefix_active=torch.zeros((1,prefix_frames,gh,gw),dtype=data_active.dtype)
            loss_active=torch.cat((prefix_active,data_active),dim=1)
            self.register_buffer('loss_active',loss_active[:,None].float(),persistent=False)

        def forward(self,data,inputs=None):
            if inputs is not None: raise ValueError('cached inputs unsupported')
            pipeline_inputs=self.get_pipeline_inputs(dict(data))
            pipeline_inputs=self.transfer_data_to_device(pipeline_inputs,self.pipe.device,self.pipe.torch_dtype)
            for unit in self.pipe.units:
                pipeline_inputs=self.pipe.unit_runner(unit,self.pipe,*pipeline_inputs)
            inputs_shared,inputs_posi,_=pipeline_inputs
            values={**inputs_shared,**inputs_posi}; pipe=self.pipe; step=self.forward_step
            low=int(values.get('min_timestep_boundary',0)*len(pipe.scheduler.timesteps))
            high=int(values.get('max_timestep_boundary',1)*len(pipe.scheduler.timesteps))
            gen=torch.Generator(device='cpu'); gen.manual_seed(int(cfg['training']['seed'])+2*step)
            idx=int(torch.randint(low,high,(1,),generator=gen).item())
            timestep=pipe.scheduler.timesteps[idx:idx+1].to(dtype=pipe.torch_dtype,device=pipe.device)
            latents=values['input_latents']; ngen=torch.Generator(device=latents.device)
            ngen.manual_seed(int(cfg['training']['seed'])+2*step+1)
            noise=torch.randn(latents.shape,dtype=latents.dtype,device=latents.device,generator=ngen)
            values['latents']=pipe.scheduler.add_noise(latents,noise,timestep)
            target=pipe.scheduler.training_target(latents,noise,timestep)
            first='first_frame_latents' in values
            if first: values['latents'][:,:,0:1]=values['first_frame_latents']
            models={name:getattr(pipe,name) for name in pipe.in_iteration_models}
            prediction=pipe.model_fn(**models,**values,timestep=timestep)
            weight=self.loss_active.to(device=prediction.device,dtype=torch.float32)
            if first:
                prediction=prediction[:,:,1:]; target=target[:,:,1:]; weight=weight[:,:,1:]
            if tuple(target.shape)!=tuple(prediction.shape):
                raise RuntimeError(f'target mismatch {tuple(target.shape)} vs {tuple(prediction.shape)}')
            if weight.shape[2]!=prediction.shape[2]:
                raise RuntimeError(f'loss mask temporal mismatch {tuple(weight.shape)} vs {tuple(prediction.shape)}')
            if tuple(weight.shape[-2:])!=tuple(prediction.shape[-2:]):
                weight=torch.nn.functional.interpolate(
                    weight,size=tuple(prediction.shape[-3:]),mode='nearest')
            if tuple(weight.shape[-3:])!=tuple(prediction.shape[-3:]):
                raise RuntimeError(f'loss mask mismatch {tuple(weight.shape)} vs {tuple(prediction.shape)}')
            weight=1+float(cfg['training']['support_weight'])*weight
            weight=weight/weight.mean().clamp_min(1e-12)
            loss=(weight*(prediction.float()-target.float()).square()).mean()
            loss=loss*pipe.scheduler.training_weight(timestep)
            self.last_trace={'optimizer_step':step+1,'timestep_index':idx,
                             'loss':float(loss.detach().cpu()),'active_tokens':int(active.sum())}
            self.forward_step+=1
            return loss

    accelerator=accelerate.Accelerator(gradient_accumulation_steps=1)
    dataset=UnifiedDataset(
        base_path=str(dataset_root),metadata_path=str(dataset_root/'metadata.csv'),
        repeat=int(cfg['training']['steps']),
        data_file_keys='video,vace_video,vace_reference_image'.split(','),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=str(dataset_root),max_pixels=int(cfg['training']['max_pixels']),
            height=None,width=None,height_division_factor=16,width_division_factor=16,
            num_frames=21,time_division_factor=4,time_division_remainder=1),
    )
    model=Module(); adapter=model.scfst_adapter
    params=[p for p in adapter.parameters() if p.requires_grad]
    trainable=sum(p.numel() for p in params)
    optimizer=torch.optim.AdamW(params,lr=float(cfg['training']['learning_rate']),
                                weight_decay=float(cfg['training']['weight_decay']))
    loader=torch.utils.data.DataLoader(dataset,shuffle=False,collate_fn=lambda batch:batch[0],num_workers=0)
    optimizer,loader=accelerator.prepare(optimizer,loader)
    model.pipe.device=accelerator.device
    offload=OffloadTrainingManager(model,accelerator.device,False,None)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    state.update(status='running',trainable_parameters=trainable,device=str(accelerator.device),steps=cfg['training']['steps'])
    atomic_json(run_manifest,state)
    save_steps=set(map(int,cfg['training']['save_steps']))
    with trace_path.open('w',encoding='utf-8') as trace:
        for data in loader:
            if model.forward_step>=int(cfg['training']['steps']): break
            optimizer.zero_grad(); loss=model(data); accelerator.backward(loss)
            grad=float(torch.nn.utils.clip_grad_norm_(params,float(cfg['training']['gradient_clip'])).detach().cpu())
            offload.after_backward(); optimizer.step()
            record={**model.last_trace,'gradient_norm':grad,'elapsed_seconds':time.time()-started}
            trace.write(json.dumps(record,sort_keys=True)+'\n'); trace.flush()
            step=record['optimizer_step']
            if step in save_steps:
                weights={k:v.detach().float().cpu().contiguous() for k,v in adapter.state_dict().items()}
                save_file(weights,str(a.output_root/f'scfst_adapter_step{step}.safetensors'))
            state.update(last_step=step,last_loss=record['loss'],last_gradient_norm=grad)
            atomic_json(run_manifest,state)
    records=[json.loads(line) for line in trace_path.read_text(encoding='utf-8').splitlines()]
    if len(records)!=int(cfg['training']['steps']): raise RuntimeError('optimizer step count mismatch')
    state.update(status='complete_requires_evaluation',finished_unix=time.time(),
                 initial_loss=records[0]['loss'],final_loss=records[-1]['loss'],
                 loss_ratio=records[-1]['loss']/max(records[0]['loss'],1e-12),
                 checkpoint_sha256={path.name:sha256(path) for path in sorted(a.output_root.glob('*.safetensors'))})
    atomic_json(run_manifest,state); print(json.dumps(state,indent=2))


if __name__=='__main__': main()
