"""Joint-reference/ControlNet LoRA on held-out real observed patch reconstruction.

This trains no counterfactual target, no source-cavity ground truth, and no base weights.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

VENDOR=Path('/mnt/tmp/guo-z_first_bite_structure_20260930_runtime/vendor')
MODELS=Path('/mnt/tmp/guo-z_first_bite_structure_20260930_models')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')


def main(args):
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
    sys.path.insert(0,str(VENDOR))
    import torch
    from peft import LoraConfig,get_peft_model_state_dict
    from safetensors.torch import load_file,save_file
    from diffusers import FlowMatchEulerDiscreteScheduler
    from videox_fun.models import QwenImage21ControlTransformer2DModel
    from videox_fun.pipeline.pipeline_qwenimage21_reference_control import calculate_shift
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    output=args.data/args.run_name;output.mkdir(exist_ok=True)
    assert not (output/'complete.json').exists(),'Completed run is immutable'
    cachemanifest=json.loads((args.data/'cache/manifest.json').read_text())
    split=json.loads((args.data/'split_manifest.json').read_text())
    assert cachemanifest['split_sha256']==sha(args.data/'split_manifest.json')
    expected={r['case_id']:r['cache_sha256'] for r in cachemanifest['cases']}
    batches={}
    for r in split['cases']:
        p=args.data/'cache'/(r['case_id']+'.pt');assert sha(p)==expected[r['case_id']]
        batches[r['case_id']]=torch.load(p,map_location='cpu',weights_only=False)
    train=[r['case_id'] for r in split['cases'] if r['split']=='train']
    val=[r['case_id'] for r in split['cases'] if r['split']=='val']
    modules=[f'control_blocks.{i}.attn.{n}' for i in range(16) for n in ['to_q','to_k','to_v','to_out.0']]
    config=dict(status='frozen_before_optimizer',objective='real observed patch reconstruction; not real first-bite paired training',
        train_count=len(train),validation_count=len(val),steps=args.steps,accumulation=args.accumulation,
        rank=args.rank,alpha=args.rank,modules=modules,lr=args.lr,seed=args.seed,
        split_sha256=sha(args.data/'split_manifest.json'),cache_manifest_sha256=sha(args.data/'cache/manifest.json'),
        code_sha256=sha(Path(__file__)),vendor_sha256=sha(VENDOR/'videox_fun/models/qwenimage21_transformer2d_control.py'),
        frozen='VLM,VAE,base transformer,original ControlNet weights',
        loss='Region-normalized velocity MSE(mask)+0.1*MSE(context)+0.1*L1(z0_hat,true_z0) inside eroded observed patch for sigma<=0.5',
        validation='Same target/control/noise/sigma; correct versus cyclically shuffled validation reference, including VLM embeddings. Never use train references as validation targets.',
        eval_sigma=.5,validation_seed=9137,base_model=str(MODELS/'qwen-image-2.1'),
        adapter=str(MODELS/'controlnet-union/Qwen-Image-2.1-Fun-Controlnet-Union.safetensors'),
        timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write(output/'frozen_config.json',config)
    model=QwenImage21ControlTransformer2DModel.from_pretrained(str(MODELS/'qwen-image-2.1'),subfolder='transformer',
        low_cpu_mem_usage=True,torch_dtype=torch.bfloat16,
        transformer_additional_kwargs={'control_layers':list(range(0,32,2)),'control_in_dim':129})
    state=load_file(str(MODELS/'controlnet-union/Qwen-Image-2.1-Fun-Controlnet-Union.safetensors'))
    loaded=model.load_state_dict(state,strict=False);assert not loaded.unexpected_keys
    del state
    model.requires_grad_(False)
    model.add_adapter(LoraConfig(r=args.rank,lora_alpha=args.rank,target_modules=modules,lora_dropout=0.,bias='none'))
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert all('lora_' in n and n.startswith('control_blocks.') for n in trainable)
    for p in trainable.values():p.data=p.data.float()
    nparams=sum(p.numel() for p in trainable.values())
    assert nparams==16*4*args.rank*8192,nparams
    model.to('cuda');model.enable_gradient_checkpointing()
    optimizer=torch.optim.AdamW(list(trainable.values()),lr=args.lr,betas=(.9,.99),weight_decay=.01)
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(str(MODELS/'qwen-image-2.1'),subfolder='scheduler',local_files_only=True)
    nt=batches[train[0]]['z0'].shape[1]
    mu=calculate_shift(nt,scheduler.config.get('base_image_seq_len',256),scheduler.config.get('max_image_seq_len',4096),scheduler.config.get('base_shift',.5),scheduler.config.get('max_shift',1.15))
    scheduler.set_timesteps(sigmas=np.linspace(1.,1/1000,1000),device='cuda',mu=mu)
    rng=np.random.default_rng(args.seed);gen=torch.Generator('cuda').manual_seed(args.seed)

    def batch(case,reference=None):
        b={k:v.to('cuda') if torch.is_tensor(v) else v for k,v in batches[case].items()}
        if reference:
            ref=batches[reference]
            for k in ['zref','prompt_embeds','encoder_mask','img_mask']:
                b[k]=ref[k].to('cuda') if ref[k] is not None else None
        return b

    def predict(b,zt,sigma):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return model(hidden_states=torch.cat([b['zref'],zt],1),
                encoder_hidden_states=b['prompt_embeds'],encoder_hidden_states_mask=b['encoder_mask'],
                timestep=torch.as_tensor(sigma,device='cuda',dtype=torch.bfloat16).reshape(1),
                img_shapes=b['img_shapes'],img_mask=b['img_mask'],
                control_context=b['control_context'],control_context_scale=1.,return_dict=False)[0][:,-zt.shape[1]:]

    def errors(pred,b,noise,sigma):
        z0=b['z0'].float();velocity=noise.float()-z0
        err=(pred.float()-velocity).square();reg=b['region'].float();context=1-reg
        em=(err*reg).sum()/(reg.sum()*err.shape[-1]);ec=(err*context).sum()/(context.sum()*err.shape[-1])
        zh=(1-sigma)*z0+sigma*noise.float()-sigma*pred.float()
        core=b['core'];l1=((zh-z0).abs()*core).sum()/(core.sum()*z0.shape[-1]).clamp_min(1)
        return em,ec,l1,zh

    @torch.no_grad()
    def evaluate(step,all_val):
        model.eval();rows=[]
        for j,case in enumerate(val if all_val else val[:8]):
            rg=torch.Generator('cuda').manual_seed(9137+j)
            b=batch(case);noise=torch.randn(b['z0'].shape,generator=rg,device='cuda',dtype=torch.bfloat16)
            zt=.5*b['z0']+.5*noise
            for refkind in ['correct','shuffled']:
                refcase=case if refkind=='correct' else val[(j+1)%len(val)]
                inp=b if refkind=='correct' else batch(case,refcase)
                pred=predict(inp,zt,.5);em,ec,l1,zh=errors(pred,b,noise,.5)
                rows.append(dict(case_id=case,reference=refkind,reference_case=refcase,
                    masked_velocity_mse=em.item(),context_velocity_mse=ec.item(),observed_core_latent_l1=l1.item()))
                if j<2:
                    torch.save(dict(z0_hat=zh.cpu(),z0=b['z0'].cpu()),output/f'eval_{step:04d}_{case}_{refkind}.pt')
        mean={kind:{key:float(np.mean([r[key] for r in rows if r['reference']==kind])) for key in ['masked_velocity_mse','context_velocity_mse','observed_core_latent_l1']} for kind in ['correct','shuffled']}
        result=dict(step=step,rows=rows,means=mean,binding_gap_shuffled_minus_correct=mean['shuffled']['masked_velocity_mse']-mean['correct']['masked_velocity_mse'])
        write(output/f'validation_{step:04d}.json',result);model.train();return result

    # A zero LoRA output must preserve the existing exact joint-reference/control baseline.
    model.eval();b=batch(train[0]);noise=torch.randn(b['z0'].shape,generator=gen,device='cuda',dtype=torch.bfloat16);zt=.5*b['z0']+.5*noise
    with torch.no_grad():
        enabled=predict(b,zt,.5);model.disable_adapters();disabled=predict(b,zt,.5);model.enable_adapters()
    delta=(enabled.float()-disabled.float()).abs().max().item();assert delta==0,delta
    write(output/'step0_equivalence.json',dict(max_absolute_output_difference=delta,trainable_parameters=nparams,
        trainable_names=list(trainable),gpu=torch.cuda.get_device_name(),total_memory_bytes=torch.cuda.get_device_properties(0).total_memory))
    del b,noise,zt,enabled,disabled
    baseline=evaluate(0,True)
    model.train();log=[];started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for step in range(1,args.steps+1):
        t0=time.perf_counter();optimizer.zero_grad(set_to_none=True);losses=[]
        for _ in range(args.accumulation):
            case=train[int(rng.integers(len(train)))];b=batch(case)
            index=int(rng.integers(1000));sigma=float(scheduler.sigmas[index].item())
            noise=torch.randn(b['z0'].shape,generator=gen,device='cuda',dtype=torch.bfloat16)
            zt=(1-sigma)*b['z0']+sigma*noise
            pred=predict(b,zt,sigma);em,ec,l1,_=errors(pred,b,noise,sigma)
            loss=em+.1*ec+(.1*l1 if sigma<=.5 else 0)
            assert torch.isfinite(loss),{'case':case,'step':step,'loss':loss.item()}
            (loss/args.accumulation).backward();losses.append(loss.item())
        nonzero=[n for n,p in trainable.items() if p.grad is not None and bool(torch.any(p.grad!=0))]
        grad=torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.)
        assert torch.isfinite(grad) and nonzero,(step,grad)
        optimizer.step();torch.cuda.synchronize()
        row=dict(step=step,loss=float(np.mean(losses)),gradient_norm=float(grad),nonzero_gradient_tensors=len(nonzero),
            seconds=time.perf_counter()-t0,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        log.append(row);print(json.dumps(row),flush=True)
        if step==1:
            write(output/'smoke.json',dict(status='finite_forward_backward_and_optimizer_pass',**row,
                step0_max_abs_difference=delta,nonzero_gradient_names=nonzero,base_trainable_count=0))
        if step in [1,50,100,150,args.steps]:
            state={k:v.detach().cpu().contiguous() for k,v in get_peft_model_state_dict(model).items()}
            save_file(state,str(output/f'adapter_step_{step:04d}.safetensors'))
            write(output/'training_progress.json',dict(status='running',steps=log,seconds=time.perf_counter()-started))
        if step in [50,100,150,args.steps]:
            evaluate(step,step==args.steps)
    final=json.loads((output/f'validation_{args.steps:04d}.json').read_text())
    write(output/'complete.json',dict(status='completed_self_supervised_material_binding_pilot',optimizer_updates=args.steps,
        trainable_parameters=nparams,seconds=time.perf_counter()-started,
        measured_median_seconds_per_update=float(np.median([r['seconds'] for r in log])),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        final_adapter_sha256=sha(output/f'adapter_step_{args.steps:04d}.safetensors'),baseline=baseline['means'],final=final['means'],
        baseline_binding_gap=baseline['binding_gap_shuffled_minus_correct'],final_binding_gap=final['binding_gap_shuffled_minus_correct'],
        limitations='Observed patch reconstruction only. No true removal target, physical accuracy or final first-bite realism proof.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--run-name',default='pilot_rank16')
    p.add_argument('--steps',type=int,default=200);p.add_argument('--rank',type=int,default=16);p.add_argument('--accumulation',type=int,default=1)
    p.add_argument('--lr',type=float,default=5e-5);p.add_argument('--seed',type=int,default=20261005)
    main(p.parse_args())
