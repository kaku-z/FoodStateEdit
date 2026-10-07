"""Train a shared image-conditioned MLD pretraining model, with genuine updates.

This is procedural-3D pretraining, not supervision by real before/after images.
All geometry/material/cut truths are targets only. The generated canonical
state is independent of action and reused by every action branch.
"""
from pathlib import Path
import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from foodstateedit.material_lineage.learned_mld import MLDConfig, MLDModel
from foodstateedit.material_lineage.core import material_noise

SCOPE = 'Shared, image-conditioned procedural-3D pretraining for cut and rigid lift only; no real 3D truth, calibrated real contact, photographic realism or complete general-purpose MLD claim.'
INPUT_KEYS = ['source_rgb', 'canonical_xyz', 'source_uv', 'actions']
MODULES = ['image_encoder', 'coordinate_projection', 'source_transformer', 'source_geometry_head',
           'source_material_head', 'action_encoder', 'state_encoder', 'transition_head', 'denoiser']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write_json(path, value):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(path)

def save_checkpoint(path, value):
    path=Path(path);tmp=path.with_suffix('.pt.tmp');torch.save(value,tmp);tmp.replace(path)

def parameter_sha(model, prefix=None):
    h=hashlib.sha256()
    for name,p in sorted(model.named_parameters()):
        if prefix and not name.startswith(prefix):continue
        h.update(name.encode());h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class Data:
    def __init__(self, root):
        self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text())
        self.a={k:np.load(self.root/(k+'.npy'),mmap_mode='r') for k in
            ['canonical_xyz','source_uv','source_rgb','joint','occupancy','actions',
             'carried_fraction','cell_occupancy','action_validmask','splits']}
        self.groups=[np.flatnonzero(self.a['splits']==s) for s in range(3)]
        assert all(len(g)>0 for g in self.groups)
        self.tokens=len(self.a['canonical_xyz'])
    def batch(self, scene, token, action, device, no_image=False):
        scene=np.asarray(scene);token=np.asarray(token)
        image=np.asarray(self.a['source_rgb'][scene]).copy().transpose(0,3,1,2).astype(np.float32)/255
        if no_image:image[:]=0
        xyz=np.broadcast_to(self.a['canonical_xyz'][token],(len(scene),len(token),3)).copy()
        uv=np.broadcast_to(self.a['source_uv'][token],(len(scene),len(token),2)).copy()
        joint=np.asarray(self.a['joint'][scene[:,None],token[None,:]],dtype=np.float32).copy()
        occ=np.asarray(self.a['occupancy'][scene[:,None],token[None,:]],dtype=np.float32)[...,None].copy()
        valid=np.asarray(self.a['action_validmask'][scene[:,None],token[None,:]],dtype=np.float32)[...,None].copy()
        mass=np.asarray(self.a['cell_occupancy'][scene[:,None],token[None,:]],dtype=np.float32)[...,None].copy()
        act=np.asarray(self.a['actions'][scene,np.asarray(action)],dtype=np.float32).copy()
        frac=np.asarray(self.a['carried_fraction'][scene[:,None],np.asarray(action)[:,None],token[None,:]],dtype=np.float32)[...,None].copy()
        arrays={'image':image,'xyz':xyz,'uv':uv,'joint':joint,'occ':occ,'valid':valid,'mass':mass,'action':act,'fraction':frac}
        b={k:torch.from_numpy(np.ascontiguousarray(v)).to(device) for k,v in arrays.items()}
        b['remaining_delta']=torch.zeros_like(b['xyz'])
        b['carried_delta']=rigid_delta(b['xyz'],b['action'])
        return b

def rigid_delta(xyz,action):
    aa=action[:,7:10];theta=torch.linalg.vector_norm(aa,dim=1,keepdim=True)
    axis=aa/theta.clamp_min(1e-8);p=xyz
    cross=torch.cross(axis[:,None,:].expand_as(p),p,dim=-1)
    dot=(p*axis[:,None,:]).sum(-1,keepdim=True)
    c=theta.cos()[:,None,:];s=theta.sin()[:,None,:]
    rotated=p*c+cross*s+axis[:,None,:]*dot*(1-c)
    return rotated+action[:,None,4:7]-p

def weighted_mean(value,weight):
    return (value*weight).sum()/(weight.expand_as(value).sum().clamp_min(1))

def transition_for_state(model,source,action,state=None):
    if model.config.state_dim:
        if state is None:state=torch.cat((source['sdf'],source['rgb']),dim=-1)
        return model.transition(source['features'],action,state)
    return model.transition(source['features'],action)

def supervised_losses(source,transition,b):
    # Occupancy weights are used in losses only, never fed to the model.
    occ=b['occ'];count=occ.sum();neg=occ.numel()-count
    pos_weight=(neg/count.clamp_min(1)).clamp(1,12)
    lo=F.binary_cross_entropy_with_logits(source['occupancy_logits'],occ,pos_weight=pos_weight)
    ls=F.smooth_l1_loss(source['sdf'],b['joint'][...,:1])
    lm=weighted_mean((source['rgb']-b['joint'][...,1:]).square(),occ)
    p=transition['carried_fraction'].clamp(1e-6,1-1e-6)
    target=b['fraction'];valid=b['valid']
    # Unweighted BCE remains calibrated for fractional cell targets. A positive
    # class multiplier would incorrectly inflate partial material fractions.
    lf=weighted_mean(-(target*p.log()+(1-target)*(1-p).log()),valid)
    null=(b['action'].abs().sum(-1)==0).float()[:,None,None]
    flow_weight=b['mass']*(target+null*(1-target))
    lc=weighted_mean(F.smooth_l1_loss(transition['carried_delta'],b['carried_delta'],reduction='none'),flow_weight)
    lr=weighted_mean(transition['remaining_delta'].square(),b['mass'])
    return {'occupancy':lo,'sdf':ls,'material':lm,'fraction':lf,'carried_flow':lc,'remaining_flow':lr}

def alpha_schedule(device):
    beta=torch.linspace(.0001,.02,1000,device=device)
    return torch.cumprod(1-beta,dim=0)

@torch.inference_mode()
def sample_joint(model,features,noise,alpha,steps=50):
    x=noise.clone()
    ts=np.rint(np.linspace(len(alpha)-1,0,steps)).astype(int)
    for i,t in enumerate(ts):
        at=alpha[int(t)]; nxt=alpha[int(ts[i+1])] if i+1<len(ts) else torch.ones_like(at)
        time_input=torch.full((len(features),),t/(len(alpha)-1),device=x.device)
        eps=model.denoise(features,x,time_input)
        x0=((x-(1-at).sqrt()*eps)/at.sqrt()).clamp(-1,1)
        eps=(x-at.sqrt()*x0)/(1-at).sqrt()
        x=nxt.sqrt()*x0+(1-nxt).sqrt()*eps
    return x

def field_metrics(sdf,rgb,logits,b):
    occupied=b['occ']>0.5;prediction=logits>0
    union=(prediction|occupied).sum();intersection=(prediction&occupied).sum()
    return {'occupancy_iou':float(intersection/union.clamp_min(1)),
            'sdf_mae':float((sdf-b['joint'][...,:1]).abs().mean()),
            'occupied_linear_rgb_mse':float(weighted_mean(((rgb-b['joint'][...,1:])/2).square(),b['occ']))}

@torch.inference_mode()
def light_validation(model,data,device,seed,no_image=False):
    model.eval();rng=np.random.default_rng(seed+701)
    token=np.sort(rng.choice(data.tokens,min(512,data.tokens),replace=False))
    scenes=data.groups[1][:16];actions=np.arange(len(scenes))%3
    b=data.batch(scenes,token,actions,device,no_image)
    source=model.encode_source(b['image'],b['xyz'],b['uv']);tr=transition_for_state(model,source,b['action'])
    values=field_metrics(source['sdf'],source['rgb'],source['occupancy_logits'],b)
    values.update(fraction_mae=float(weighted_mean((tr['carried_fraction']-b['fraction']).abs(),b['valid'])),
                  carried_endpoint_rmse=float(weighted_mean((tr['carried_delta']-b['carried_delta']).square().sum(-1,keepdim=True),b['mass']*b['fraction']).sqrt()),
                  max_allocation_budget_error=float((tr['allocation'].sum(-1)-1).abs().max()))
    model.train();return values

@torch.inference_mode()
def final_evaluation(model,initial_state,model_config,data,root,device,seed,no_image=False):
    model.eval();initial=MLDModel(MLDConfig(**model_config)).to(device)
    initial.load_state_dict(initial_state);initial.eval()
    rng=np.random.default_rng(seed+991)
    token=np.sort(rng.choice(data.tokens,min(512,data.tokens),replace=False))
    scenes=data.groups[2][:32];acts=np.arange(len(scenes))%3
    b=data.batch(scenes,token,acts,device,no_image)
    source=model.encode_source(b['image'],b['xyz'],b['uv'])
    si=initial.encode_source(b['image'],b['xyz'],b['uv'])
    shuffled=model.encode_source(b['image'].roll(1,0),b['xyz'],b['uv'])
    # Nonzero action3 is entirely excluded from optimizer updates.
    ood=data.batch(scenes,token,np.full(len(scenes),3),device,no_image)
    alpha=alpha_schedule(device)
    scene_records=[json.loads(line) for line in (data.root/'scenes.jsonl').read_text().splitlines()]
    ids=[f"{scene_records[int(s)]['scene_id']}:lattice:{int(q)}" for s in scenes for q in token]
    noise=torch.from_numpy(material_noise(ids,seed+433,channels=4,stream='joint-field').astype(np.float32)).reshape(b['joint'].shape).to(device)
    sampled=sample_joint(model,source['features'],noise,alpha)
    sample_shuffled=sample_joint(model,shuffled['features'],noise,alpha)
    sample_initial=sample_joint(initial,si['features'],noise,alpha)
    # Every action replay reads exactly the same generated canonical state.
    tr=transition_for_state(model,source,b['action'],sampled)
    tri=transition_for_state(initial,si,b['action'],sample_initial)
    tras=transition_for_state(model,source,b['action'].roll(1,0),sampled)
    zero=transition_for_state(model,source,torch.zeros_like(b['action']),sampled)
    trood=transition_for_state(model,source,ood['action'],sampled)
    trposterior=transition_for_state(model,source,b['action'])
    trstate=transition_for_state(model,source,b['action'],sampled.roll(1,0))
    # Store arrays so a separate audit can recompute metrics and dataset truth.
    asnp=lambda v:v.detach().float().cpu().numpy()
    arrays={'scene_indices':scenes,'token_indices':token,'action_indices':acts,
       'split_codes':np.asarray(data.a['splits'][scenes]),'target_joint':asnp(b['joint']),
       'target_occupancy':asnp(b['occ']),'target_valid':asnp(b['valid']),'target_mass':asnp(b['mass']),
       'trained_joint':asnp(sampled),'initial_joint':asnp(sample_initial),'source_shuffled_joint':asnp(sample_shuffled),
       'transition_state_joint':asnp(sampled),'initial_transition_state_joint':asnp(sample_initial),
       'posterior_sdf':asnp(source['sdf']),'posterior_rgb':asnp(source['rgb']),'posterior_occupancy_logits':asnp(source['occupancy_logits']),
       'target_carried_fraction':asnp(b['fraction']),'trained_fraction':asnp(tr['carried_fraction']),
       'initial_fraction':asnp(tri['carried_fraction']),'action_shuffled_fraction':asnp(tras['carried_fraction']),
       'zero_fraction':asnp(zero['carried_fraction']),'trained_allocation':asnp(tr['allocation']),
       'posterior_state_fraction':asnp(trposterior['carried_fraction']),'state_shuffled_fraction':asnp(trstate['carried_fraction']),
       'target_carried_delta':asnp(b['carried_delta']),'trained_carried_delta':asnp(tr['carried_delta']),
       'action_shuffled_carried_delta':asnp(tras['carried_delta']),'zero_carried_delta':asnp(zero['carried_delta']),
       'ood_target_fraction':asnp(ood['fraction']),'ood_trained_fraction':asnp(trood['carried_fraction']),
       'ood_target_carried_delta':asnp(ood['carried_delta']),'ood_trained_carried_delta':asnp(trood['carried_delta']),
       'material_id_noise':asnp(noise)}
    np.savez_compressed(root/'evaluation_predictions.npz',**arrays)
    result={'scope':SCOPE,'split':'test','case_indices':scenes.tolist(),'token_indices':token.tolist(),
      'allowed_conditioning_inputs':INPUT_KEYS,'heldout_action_index':3,'sampling_steps':50,'sampling_seed':seed+433,
      'posterior':field_metrics(source['sdf'],source['rgb'],source['occupancy_logits'],b),
      'initial_posterior':field_metrics(si['sdf'],si['rgb'],si['occupancy_logits'],b),
      'source_shuffled_posterior':field_metrics(shuffled['sdf'],shuffled['rgb'],shuffled['occupancy_logits'],b),
      'sampled_joint':field_metrics(sampled[...,:1],sampled[...,1:],-sampled[...,:1],b),
      'initial_sampled_joint':field_metrics(sample_initial[...,:1],sample_initial[...,1:],-sample_initial[...,:1],b),
      'source_shuffled_sampled_joint':field_metrics(sample_shuffled[...,:1],sample_shuffled[...,1:],-sample_shuffled[...,:1],b)}
    result['transition']={'fraction_mae':float(weighted_mean((tr['carried_fraction']-b['fraction']).abs(),b['valid'])),
      'initial_fraction_mae':float(weighted_mean((tri['carried_fraction']-b['fraction']).abs(),b['valid'])),
      'posterior_state_fraction_mae':float(weighted_mean((trposterior['carried_fraction']-b['fraction']).abs(),b['valid'])),
      'state_shuffled_fraction_mae':float(weighted_mean((trstate['carried_fraction']-b['fraction']).abs(),b['valid'])),
      'state_shuffle_output_change_mae':float(weighted_mean((trstate['carried_fraction']-tr['carried_fraction']).abs(),b['valid'])),
      'action_shuffled_fraction_mae':float(weighted_mean((tras['carried_fraction']-b['fraction']).abs(),b['valid'])),
      'carried_endpoint_rmse':float(weighted_mean((tr['carried_delta']-b['carried_delta']).square().sum(-1,keepdim=True),b['mass']*b['fraction']).sqrt()),
      'zero_fraction_mean':float(weighted_mean(zero['carried_fraction'],b['valid'])),
      'zero_carried_delta_rms':float(weighted_mean(zero['carried_delta'].square(),b['mass']).sqrt()),
      'ood_fraction_mae':float(weighted_mean((trood['carried_fraction']-ood['fraction']).abs(),b['valid'])),
      'ood_carried_endpoint_rmse':float(weighted_mean((trood['carried_delta']-ood['carried_delta']).square().sum(-1,keepdim=True),b['mass']*ood['fraction']).sqrt()),
      'max_allocation_budget_error':float((tr['allocation'].sum(-1)-1).abs().max())}
    result['representation_invariants']={'canonical_generation_action_independent':True,
      'transition_reads_same_sampled_source_state':bool(model.config.state_dim),
      'transition_training_state':'Predicted posterior field, never target field. Stochastic sampled-field transition calibration remains to be validated.',
      'material_id_noise_scope':'One canonical lattice noise sample, generated once, shared by branch/action replay. Not a learned physical law.',
      'conservation_scope':'Column-normalized allocation of a model/proxy reference budget; not real measured mass.'}
    result['predictions_sha256']=sha(root/'evaluation_predictions.npz')
    write_json(root/'evaluation.json',result)
    del initial
    return result

def main(args):
    root=args.output_root;root.mkdir(parents=True,exist_ok=True)
    if (root/'initial_checkpoint.pt').exists():raise FileExistsError('Use a new run directory; initial checkpoint must not be overwritten')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=Data(args.dataset_root)
    cfg=MLDConfig(state_dim=4);model=MLDModel(cfg).to(device)
    model_config=dataclasses.asdict(cfg)
    rng=np.random.default_rng(args.seed)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(scope=SCOPE,model_config=model_config,training_action_indices=[0,1,2],heldout_action_index=3,
                  transition_training_state='predicted_posterior_sdf_rgb',transition_evaluation_state='shared_sampled_joint',
                  dataset_manifest_sha256=sha(data.root/'manifest.json'),input_keys=INPUT_KEYS,
                  dataset_split_counts=[len(g) for g in data.groups],torch_version=str(torch.__version__),
                  trainer_sha256=sha(Path(__file__)),model_source_sha256=sha(ROOT_MODEL),started_unix=time.time())
    write_json(root/'config.json',config)
    initial={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
    initial_hash={m:parameter_sha(model,m) for m in MODULES}
    save_checkpoint(root/'initial_checkpoint.pt',{'model_config':model_config,'state_dict':initial,'optimizer_steps':0,'scope':SCOPE})
    gradient_observations={m:0 for m in MODULES};max_grad={m:0. for m in MODULES}
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    alpha=alpha_schedule(device);start=time.monotonic();completed=0
    val=light_validation(model,data,device,args.seed,args.no_image)
    write_json(root/'initial_validation.json',val)
    log=(root/'train_metrics.jsonl').open('a',buffering=1)
    receipt={'status':'running','scope':SCOPE,'actual_optimizer_steps':0,'planned_optimizer_steps':args.steps,
      'parameter_count':sum(p.numel() for p in model.parameters()),'per_module_gradient_nonzero_observations':gradient_observations,
      'initial_parameter_sha256':initial_hash,'input_keys':INPUT_KEYS,'config_sha256':sha(root/'config.json'),
      'dataset_manifest_sha256':config['dataset_manifest_sha256'],'pid':os.getpid(),'seed':args.seed,'no_image':args.no_image}
    write_json(root/'training_receipt.json',receipt)
    print(json.dumps({'event':'training_started',**receipt}),flush=True)
    try:
        for step in range(1,args.steps+1):
            model.train();scene=rng.choice(data.groups[0],args.batch_size,replace=True)
            tokens=np.sort(rng.choice(data.tokens,min(args.tokens,data.tokens),replace=False))
            actions=rng.integers(0,3,size=len(scene))
            b=data.batch(scene,tokens,actions,device,args.no_image)
            source=model.encode_source(b['image'],b['xyz'],b['uv'])
            transition=transition_for_state(model,source,b['action'])
            loss=supervised_losses(source,transition,b)
            active=step>args.warmup_steps
            if active:
                t=torch.randint(0,len(alpha),(len(scene),),device=device)
                noise=torch.randn_like(b['joint']);abar=alpha[t][:,None,None]
                xt=abar.sqrt()*b['joint']+(1-abar).sqrt()*noise
                epsilon=model.denoise(source['features'],xt,t.float()/(len(alpha)-1))
                weight=torch.cat([torch.ones_like(b['occ']),b['occ'].expand(-1,-1,3)],-1)
                loss['diffusion']=weighted_mean((epsilon-noise).square(),weight)
            else:loss['diffusion']=source['sdf'].sum()*0
            total=loss['occupancy']+2*loss['sdf']+2*loss['material']+loss['fraction']+4*loss['carried_flow']+loss['remaining_flow']+loss['diffusion']
            if not torch.isfinite(total):raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);total.backward()
            if step<=3 or step%100==0 or step==args.warmup_steps+1:
                for module in MODULES:
                    norm=sum(float(p.grad.detach().float().square().sum()) for n,p in model.named_parameters() if n.startswith(module) and p.grad is not None)**.5
                    if math.isfinite(norm) and norm>0:gradient_observations[module]+=1
                    max_grad[module]=max(max_grad[module],norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scale=min(1.,step/100.)*(.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
            for group in optimizer.param_groups:group['lr']=args.lr*scale
            optimizer.step();completed=step
            if step%50==0 or step==1:
                row={'step':step,'phase':'joint_diffusion' if active else 'posterior_transition_warmup',
                     'elapsed_seconds':time.monotonic()-start,'total_loss':float(total.detach()),
                     **{k:float(v.detach()) for k,v in loss.items()},'lr':optimizer.param_groups[0]['lr']}
                log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
            if step%args.checkpoint_every==0 or step==args.steps:
                values=light_validation(model,data,device,args.seed,args.no_image)
                log.write(json.dumps({'step':step,'validation':values})+'\n')
                checkpoint={'model_config':model_config,'state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                    'optimizer_steps':step,'seed':args.seed,'scope':SCOPE,'validation':values,
                    'dataset_manifest_sha256':config['dataset_manifest_sha256']}
                save_checkpoint(root/'latest_checkpoint.pt',checkpoint)
                receipt.update(actual_optimizer_steps=step,elapsed_seconds=time.monotonic()-start,
                    per_module_gradient_nonzero_observations=dict(gradient_observations),max_gradient_norm=dict(max_grad),
                    current_parameter_sha256={m:parameter_sha(model,m) for m in MODULES},validation=values,
                    latest_checkpoint_sha256=sha(root/'latest_checkpoint.pt'))
                write_json(root/'training_receipt.json',receipt)
                print(json.dumps({'event':'checkpoint_saved','step':step,'validation':values}),flush=True)
        save_checkpoint(root/'final_checkpoint.pt',{'model_config':model_config,'state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),'optimizer_steps':completed,'seed':args.seed,'scope':SCOPE,
            'dataset_manifest_sha256':config['dataset_manifest_sha256']})
        receipt.update(status='evaluating',actual_optimizer_steps=completed);write_json(root/'training_receipt.json',receipt)
        evaluation=final_evaluation(model,initial,model_config,data,root,device,args.seed,args.no_image)
        receipt.update(status='complete_pretraining',elapsed_seconds=time.monotonic()-start,
            final_checkpoint_sha256=sha(root/'final_checkpoint.pt'),evaluation_sha256=sha(root/'evaluation.json'))
        write_json(root/'training_receipt.json',receipt)
        print(json.dumps({'event':'complete_pretraining','step':completed,'sampled_joint':evaluation['sampled_joint']}),flush=True)
    except BaseException as e:
        receipt.update(status='failed',actual_optimizer_steps=completed,error=repr(e),elapsed_seconds=time.monotonic()-start)
        write_json(root/'training_receipt.json',receipt);raise
    finally:log.close()

ROOT_MODEL=Path(__file__).resolve().parents[1]/'foodstateedit/material_lineage/learned_mld.py'

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--seed',type=int,default=41);p.add_argument('--steps',type=int,default=30000)
    p.add_argument('--warmup-steps',type=int,default=1000);p.add_argument('--batch-size',type=int,default=24)
    p.add_argument('--tokens',type=int,default=512);p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--checkpoint-every',type=int,default=500);p.add_argument('--device',default='cuda')
    p.add_argument('--no-image',action='store_true')
    a=p.parse_args()
    if a.steps<1 or a.warmup_steps>=a.steps or a.batch_size<2 or a.tokens<8:raise ValueError('Invalid training settings')
    main(a)
