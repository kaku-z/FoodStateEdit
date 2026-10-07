"""Frozen v1 fields on the complete new test split, with shared MLD2 actions."""
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodstateedit.material_lineage.learned_mld import MLDConfig, MLDModel
from train_mld2 import make_actions

MODES=['id','normal_ood','height_ood','rotation_ood','combined_ood','null']
def actions(ids,mode):
    if mode!='combined_ood':return make_actions(ids,mode)
    a=make_actions(ids,'normal_ood');a[:,4:7]=make_actions(ids,'height_ood')[:,4:7]
    a[:,7:10]=make_actions(ids,'rotation_ood')[:,7:10]
    return a
def sample_grid(field,points):
    # Grid storage axes are x,y,z, while torch sampling addresses z,y,x.
    b,n,c=field.shape;g=round(n**(1/3))
    grid=field.transpose(1,2).reshape(b,c,g,g,g)
    shape=points.shape
    uv=points.reshape(b,-1,3).flip(-1)[:,None,None]
    value=F.grid_sample(grid,uv,align_corners=False,padding_mode='border')
    return value[:,:,0,0].transpose(1,2).reshape(*shape[:-1],c)
def write_json(path,x):Path(path).write_text(json.dumps(x,indent=2),encoding='utf-8')
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

@torch.inference_mode()
def run(args):
    device=torch.device(args.device);torch.set_num_threads(4)
    data=args.dataset;out=args.output;out.mkdir(parents=True,exist_ok=True)
    keys=['canonical_xyz','source_uv','subcell_offsets','source_rgb','joint','corner_sdf',
          'occupancy','cell_occupancy','splits','surface_xyz','surface_rgb','surface_valid']
    a={k:np.load(data/(k+'.npy'),mmap_mode='r') for k in keys}
    records=[json.loads(x) for x in (data/'scenes.jsonl').read_text().splitlines()]
    scenes=np.flatnonzero(a['splits']==2)
    xyz=torch.tensor(np.asarray(a['canonical_xyz']),device=device)
    uv=torch.tensor(np.asarray(a['source_uv']),device=device)
    offsets=torch.tensor(np.asarray(a['subcell_offsets']),device=device)
    results={}
    for name,no_image in [('shared_s41',False),('no_image_s41',True)]:
        started=time.time();p=args.reference/'runs'/name/'final_checkpoint.pt'
        ck=torch.load(p,map_location='cpu',weights_only=True)
        model=MLDModel(MLDConfig(**ck['model_config'])).to(device).eval()
        model.load_state_dict(ck['state_dict'])
        rows=[];field_stats=np.zeros(6,dtype=np.float64)
        for start in range(0,len(scenes),args.batch):
            scene=scenes[start:start+args.batch];b=len(scene)
            image=torch.tensor(np.asarray(a['source_rgb'][scene]).copy().transpose(0,3,1,2),device=device,dtype=torch.float32)/255
            if no_image:image.zero_()
            q=xyz[None].expand(b,-1,-1)
            s=model.encode_source(image,q,uv[None].expand(b,-1,-1))
            true=torch.tensor(np.asarray(a['joint'][scene]).copy(),device=device,dtype=torch.float32)
            occ=torch.tensor(np.asarray(a['occupancy'][scene]).copy(),device=device)
            sub=q[...,None,:]+offsets
            corner_sdf=sample_grid(s['sdf'],sub)[...,0]
            predicted_corner=(corner_sdf<=0).float()
            truth_corner=torch.tensor(np.asarray(a['corner_sdf'][scene]).copy(),device=device)<=0
            pm=predicted_corner.mean(-1);tm=truth_corner.float().mean(-1)
            gtvolume=tm.sum(-1).clamp_min(1)
            pred=s['occupancy_logits'][...,0]>0
            inter=(pred&occ).sum(-1).float();union=(pred|occ).sum(-1).float().clamp_min(1)
            sdferr=(s['sdf'][...,0]-true[...,0]).abs().mean(-1)
            rgb_err=(((s['rgb']-true[...,1:])/2).square()*occ[...,None]).sum((1,2))/(occ.sum(-1)*3).clamp_min(1)
            field_stats += np.array([float(inter.sum()),float(union.sum()),float((s['sdf'][...,0]-true[...,0]).abs().sum()),float((((s['rgb']-true[...,1:])/2).square()*occ[...,None]).sum()),b*len(xyz),float(occ.sum())])
            source_relative=(pm-tm).abs().sum(-1)/gtvolume
            sv=torch.tensor(np.asarray(a['surface_xyz'][scene]).copy(),device=device,dtype=torch.float32)
            st=torch.tensor(np.asarray(a['surface_rgb'][scene]).copy(),device=device,dtype=torch.float32)
            valid=torch.tensor(np.asarray(a['surface_valid'][scene]).copy(),device=device,dtype=torch.float32)
            visible_rgb=sample_grid(s['rgb'],sv)
            visible_err=((((visible_rgb+1)/2-st).square())*valid[...,None]).sum((1,2))/(valid.sum(-1)*3).clamp_min(1)
            ids=[records[int(i)]['scene_id'] for i in scene]
            cuts={}
            for mode in MODES:
                command=torch.from_numpy(actions(ids,mode)).to(device)
                cut=(sub*command[:,None,None,:3]).sum(-1)>command[:,None,None,3]
                cut &= (command[:,:4].abs().sum(-1)>0)[:,None,None]
                truthmass=(truth_corner.float()*cut).mean(-1)
                inferred=(predicted_corner*cut).mean(-1)
                allsolid=cut.float().mean(-1)
                cuts[mode]={
                    'mass_l1_per_source_volume':((inferred-truthmass).abs().sum(-1)/gtvolume).cpu().numpy(),
                    'mass_total_error_per_source_volume':((inferred.sum(-1)-truthmass.sum(-1)).abs()/gtvolume).cpu().numpy(),
                    'allsolid_mass_l1_per_source_volume':((allsolid-truthmass).abs().sum(-1)/gtvolume).cpu().numpy()}
            for j,index in enumerate(scene):
                row={'scene_index':int(index),'scene_id':ids[j],'family':records[int(index)]['shape_family'],
                    'occupancy_iou':float(inter[j]/union[j]),'sdf_mae':float(sdferr[j]),
                    'occupied_linear_rgb_mse':float(rgb_err[j]),'visible_linear_rgb_mse':float(visible_err[j]),
                    'source_mass_l1_per_source_volume':float(source_relative[j]),
                    'actions':{m:{k:float(v[j]) for k,v in c.items()} for m,c in cuts.items()}}
                rows.append(row)
            if start%128==0:print(json.dumps({'baseline':name,'evaluated':start+b,'total':len(scenes)}),flush=True)
        aggregate={k:float(np.mean([x[k] for x in rows])) for k in ['occupancy_iou','sdf_mae','occupied_linear_rgb_mse','visible_linear_rgb_mse','source_mass_l1_per_source_volume']}
        aggregate['micro_occupancy_iou']=float(field_stats[0]/field_stats[1])
        aggregate['actions']={m:{k:float(np.mean([x['actions'][m][k] for x in rows])) for k in rows[0]['actions'][m]} for m in MODES}
        by_family={f:{k:float(np.mean([x[k] for x in rows if x['family']==f])) for k in ['occupancy_iou','source_mass_l1_per_source_volume','visible_linear_rgb_mse']} for f in sorted({x['family'] for x in rows})}
        result={'model':name,'source':'frozen original v1 training checkpoint; not a same-data architecture ablation',
                'checkpoint':str(p),'checkpoint_sha256':sha(p),'optimizer_steps':ck['optimizer_steps'],
                'scenes':len(scenes),'queries':len(xyz),'corner_inference':'trilinear interpolation of one complete4096 v1SDFfield',
                'visible_inference':'sample complete cached field at target surface locations, no targetquery context condition',
                'aggregate':aggregate,'by_family':by_family,'elapsed_seconds':time.time()-started}
        write_json(out/(name+'.json'),result)
        write_json(out/(name+'_per_scene.json'),rows)
        results[name]=result
        del model,ck
        torch.cuda.empty_cache()
    write_json(out/'reference_summary.json',results)
    print(json.dumps({n:v['aggregate'] for n,v in results.items()},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--device',default='cuda')
    run(p.parse_args())
