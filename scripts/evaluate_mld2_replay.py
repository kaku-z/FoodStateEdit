"""Replay one cached material field through two cuts and two rigid motions.

Budget/material/pose preservation are construction checks, not learned physics.
Query order and partition tests establish practical replay of the pointwise field.
Independent residual draws measure what is lost when each action replaces H.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodstateedit.material_lineage.mld2 import (
    MLD2Config, MLD2Model, rigid_transform, spatial_noise, sample_residual)
from mld2_scene import shape_sdf, axis_angle_matrix
from train_mld2 import make_actions


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding="utf-8")


def tensor_state_sha(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(np.asarray(value.detach().cpu().contiguous()).tobytes())
    return h.hexdigest()


def module_records(model,initial,checkpoint):
    optimizer=checkpoint.get("optimizer_state_dict",{}).get("state",{})
    modules={}
    for index,(name,parameter) in enumerate(model.named_parameters()):
        module=name.split(".")[0]
        entry=modules.setdefault(module,{"parameters":0,"max_abs_change_from_reference":None,"optimizer_step_min":None,"optimizer_step_max":0})
        entry["parameters"]+=parameter.numel()
        if name in initial:
            delta=float((parameter.detach().cpu()-initial[name]).abs().max())
            entry["max_abs_change_from_reference"]=max(entry["max_abs_change_from_reference"] or 0.,delta)
        step=optimizer.get(index,{}).get("step",0)
        step=int(step.item()) if hasattr(step,"item") else int(step)
        entry["optimizer_step_min"]=step if entry["optimizer_step_min"] is None else min(entry["optimizer_step_min"],step)
        entry["optimizer_step_max"]=max(entry["optimizer_step_max"],step)
    state=model.state_dict()
    for module,entry in modules.items():
        entry["loaded_weights_sha256"]=tensor_state_sha({k:v for k,v in state.items() if k.split(".")[0]==module})
    return modules


def cut_plane_queries(normal,offset,side=26):
    n=np.asarray(normal,dtype=float); n/=np.linalg.norm(n)
    basis=np.asarray([1.,0.,0.]) if abs(n[0])<.85 else np.asarray([0.,1.,0.])
    u=np.cross(n,basis);u/=np.linalg.norm(u);v=np.cross(n,u)
    s=np.linspace(-1.35,1.35,side)
    a,b=np.meshgrid(s,s,indexing="ij")
    return (n*float(offset)+a.reshape(-1,1)*u+b.reshape(-1,1)*v).astype(np.float32)


def query_chunks(model,cache,xyz,chunk=512):
    parts=[model.query(cache,xyz[:,lo:lo+chunk]) for lo in range(0,xyz.shape[1],chunk)]
    return {key:torch.cat([p[key] for p in parts],1) for key in parts[0]}


def source_joint(source): return torch.cat((source["sdf"],source["rgb"]),-1)


def array_error(a,b):
    d=np.abs(np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64))
    return {"mae":float(d.mean()),"max_abs":float(d.max())}


@torch.inference_mode()
def run(args):
    output=Path(args.output_root);output.mkdir(parents=True,exist_ok=True)
    started=time.time();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cp=Path(args.checkpoint)
    payload=cp.read_bytes(); checkpoint_sha=hashlib.sha256(payload).hexdigest()
    checkpoint=torch.load(io.BytesIO(payload),map_location="cpu",weights_only=True)
    if args.expected_steps is not None:
        assert int(checkpoint["optimizer_steps"])==args.expected_steps
    model=MLD2Model(MLD2Config(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"],strict=True)
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in checkpoint["state_dict"].items())
    initial_path=cp.parent/"initial_checkpoint.pt"
    if not initial_path.exists():initial_path=cp.parent/"resume_checkpoint.pt"
    reference=torch.load(initial_path,map_location="cpu",weights_only=True) if initial_path.exists() else None
    initial=reference["state_dict"] if reference else {}
    learned=module_records(model,initial,checkpoint)
    loaded_sha=tensor_state_sha(model.state_dict())
    model=model.to(device).eval()
    root=Path(args.dataset_root)
    manifest=json.loads((root/"manifest.json").read_text())
    records=[json.loads(line) for line in (root/"scenes.jsonl").read_text().splitlines()]
    arrays={name:np.load(root/(name+".npy"),mmap_mode="r") for name in [
        "source_rgb","canonical_xyz","splits","surface_xyz","surface_valid"]}
    test_indices=np.flatnonzero(arrays["splits"]==2)
    diagnostic_indices=set(map(int,test_indices[:8]))
    indices=test_indices[:args.scenes]
    xyznp=np.asarray(arrays["canonical_xyz"],dtype=np.float32)
    xyz=torch.from_numpy(xyznp.copy())[None].to(device)
    offsets=model.subcell_offsets.detach().cpu().numpy()
    subnp=(xyznp[:,None]+offsets).reshape(-1,3)
    subxyz=torch.from_numpy(subnp.copy())[None].to(device)
    cellvolume=(2/model.config.grid_size)**3
    sample_seeds=[int(v) for v in args.sample_seeds.split(",")]
    if len(sample_seeds)<2: raise ValueError("Use at least two fixed seeds for the independent-resampling control")
    rows=[];predictions={"scene_indices":indices,"canonical_xyz":xyznp,"subcell_offsets":offsets}
    storage={name:[] for name in ["source_joint","source_sub_sdf","source_sub_rgb","branch_id",
        "actions","cut_xyz","cut_valid","cut_samples","surface_joint"]}
    permutations=np.arange(len(xyznp)-1,-1,-1)
    inverse_permutations=np.argsort(permutations)
    model_role="final" if cp.name=="final_checkpoint.pt" else "intermediate_pipeline_check"
    print(json.dumps({"status":"running","checkpoint_role":model_role,"optimizer_steps":checkpoint["optimizer_steps"],"scene_count":len(indices)}),flush=True)
    for iteration,index in enumerate(indices):
        rec=records[int(index)];sid=rec["scene_id"]
        rgb=np.asarray(arrays["source_rgb"][index],dtype=np.float32).transpose(2,0,1)/255
        if checkpoint.get("no_image",False):rgb[:]=.5
        cache=model.encode_image(torch.from_numpy(rgb.copy())[None].to(device))
        source=model.query(cache,xyz)
        chunked=query_chunks(model,cache,xyz)
        reversed_source=model.query(cache,xyz[:,permutations])
        direct=source_joint(source).cpu().numpy()[0]
        chunk=source_joint(chunked).cpu().numpy()[0]
        reverse=source_joint(reversed_source).cpu().numpy()[0,inverse_permutations]
        noise=spatial_noise(xyznp,[sid],sample_seeds[0])[0]
        chunk_noise=np.concatenate([spatial_noise(xyznp[lo:lo+512],[sid],sample_seeds[0])[0] for lo in range(0,len(xyznp),512)])
        reverse_noise=spatial_noise(xyznp[permutations],[sid],sample_seeds[0])[0,inverse_permutations]
        sample=sample_residual(model,source,torch.from_numpy(noise.copy())[None].to(device),args.sample_steps)
        chunk_samples=[]
        for lo in range(0,len(xyznp),512):
            sliced={key:value[:,lo:lo+512] for key,value in chunked.items()}
            chunk_samples.append(sample_residual(model,sliced,torch.from_numpy(chunk_noise[lo:lo+512].copy())[None].to(device),args.sample_steps))
        chunk_sample=torch.cat(chunk_samples,1)
        reverse_sample=sample_residual(model,reversed_source,
            torch.from_numpy(spatial_noise(xyznp[permutations],[sid],sample_seeds[0]).copy()).to(device),args.sample_steps)
        sample_np=sample.cpu().numpy()[0]
        samplechunk_np=chunk_sample.cpu().numpy()[0]
        samplereverse_np=reverse_sample.cpu().numpy()[0,inverse_permutations]
        # The material IDs are (scene identity, parent lattice index, subsite index).
        material=model.query(cache,subxyz)
        subsdf=material["sdf"].cpu().numpy()[0,:,0].reshape(len(xyznp),8)
        subrgb=material["rgb"].cpu().numpy()[0].reshape(len(xyznp),8,3)
        soft=1/(1+np.exp(np.clip(subsdf/.025,-60,60)))
        hard=subsdf<=0
        actions=np.stack([make_actions([sid],"id")[0],make_actions([sid+":second"],"combined_ood")[0]])
        cut1=(subnp@actions[0,:3]>actions[0,3]).reshape(len(xyznp),8)
        cut2=(subnp@actions[1,:3]>actions[1,3]).reshape(len(xyznp),8)&~cut1
        remainder=~(cut1|cut2)
        gates=np.stack([remainder,cut1,cut2])
        per_id_error=np.abs((gates*soft[None]).sum(0)-soft)
        volume_soft=(gates*soft[None]).sum(axis=(1,2))*cellvolume/8
        volume_hard=(gates*hard[None]).sum(axis=(1,2))*cellvolume/8
        source_soft=float(soft.sum()*cellvolume/8);source_hard=float(hard.sum()*cellvolume/8)
        inverse_errors=[]
        for command in actions:
            moved=rigid_transform(subxyz,torch.from_numpy(command.copy())[None].to(device)).cpu().numpy()[0]
            restored=(np.asarray(moved,dtype=np.float64)-command[4:7])@axis_angle_matrix(command[7:10])
            inverse_errors.append(float(np.abs(restored-subnp).max()))
        # Copy the same stored material values into all child lineage tables.
        child_materials=[subrgb.copy(),subrgb.copy(),subrgb.copy()]
        material_copy_error=max(float(np.abs(values-subrgb).max()) for values in child_materials)
        cutxyz=cut_plane_queries(actions[0,:3],actions[0,3])
        cutvalid=(np.abs(cutxyz)<=1).all(-1)&(shape_sdf(cutxyz,rec["target_generator_parameters"])<=0)
        cutsource=model.query(cache,torch.from_numpy(cutxyz.copy())[None].to(device))
        cutdraws=[]
        for seed in sample_seeds:
            z=spatial_noise(cutxyz,[sid],seed)
            cutdraws.append(sample_residual(model,cutsource,torch.from_numpy(z).to(device),args.sample_steps).cpu().numpy()[0])
        cutdraws=np.stack(cutdraws)
        shared_remaining=cutdraws[0].copy()
        shared_carried=cutdraws[0].copy()
        shared_cut_difference=np.abs(shared_remaining[cutvalid]-shared_carried[cutvalid])
        shared_cut_sdf=float(shared_cut_difference[:,0].mean()) if cutvalid.any() else 0.
        shared_cut_rgb=float(shared_cut_difference[:,1:].mean()/2) if cutvalid.any() else 0.
        sdf_diffs=[];rgb_diffs=[]
        if cutvalid.any():
            for first in range(len(sample_seeds)):
                for second in range(first+1,len(sample_seeds)):
                    difference=np.abs(cutdraws[first,cutvalid]-cutdraws[second,cutvalid])
                    sdf_diffs.append(float(difference[:,0].mean()))
                    rgb_diffs.append(float(difference[:,1:].mean()/2))
        surface=model.query(cache,torch.from_numpy(np.asarray(arrays["surface_xyz"][index],dtype=np.float32).copy())[None].to(device))
        row={"index":int(index),"scene_id":sid,"family":rec["shape_family"],
            "used_for_development_diagnostic":int(index) in diagnostic_indices,
            "source_hard_proxy_volume":source_hard,"source_soft_proxy_volume":source_soft,
            "branch_soft_proxy_volumes":volume_soft.tolist(),"branch_hard_proxy_volumes":volume_hard.tolist(),
            "per_id_soft_budget_max_abs":float(per_id_error.max()),
            "soft_volume_budget_abs":abs(float(volume_soft.sum())-source_soft),
            "hard_volume_budget_abs":abs(float(volume_hard.sum())-source_hard),
            "child_material_copy_max_abs":material_copy_error,
            "inverse_pose_max_abs":max(inverse_errors),
            "posterior_chunk_error":array_error(direct,chunk),"posterior_reverse_error":array_error(direct,reverse),
            "noise_chunk_error":array_error(noise,chunk_noise),"noise_reverse_error":array_error(noise,reverse_noise),
            "sample_chunk_error":array_error(sample_np,samplechunk_np),"sample_reverse_error":array_error(sample_np,samplereverse_np),
            "cut_valid_points":int(cutvalid.sum()),"shared_state_cut_sdf_difference":shared_cut_sdf,"shared_state_cut_linear_rgb_difference":shared_cut_rgb,
            "independent_cut_sdf_mae_pairs":sdf_diffs,"independent_cut_linear_rgb_mae_pairs":rgb_diffs}
        rows.append(row)
        storage["source_joint"].append(direct)
        storage["source_sub_sdf"].append(subsdf)
        storage["source_sub_rgb"].append(subrgb.astype(np.float16))
        storage["branch_id"].append(np.where(cut1,1,np.where(cut2,2,0)).astype(np.uint8))
        storage["actions"].append(actions);storage["cut_xyz"].append(cutxyz)
        storage["cut_valid"].append(cutvalid);storage["cut_samples"].append(cutdraws)
        storage["surface_joint"].append(source_joint(surface).cpu().numpy()[0])
        if iteration%8==7 or iteration==len(indices)-1:
            print(json.dumps({"evaluated":iteration+1,"total":len(indices),"elapsed_seconds":round(time.time()-started,1)}),flush=True)
    for name,values in storage.items():predictions[name]=np.stack(values)
    np.savez_compressed(output/"predictions.npz",**predictions)
    write_json(output/"per_scene.json",rows)
    def maximum(name):return max(row[name] for row in rows)
    def error_summary(name):return {"mean_mae":float(np.mean([row[name]["mae"] for row in rows])),"max_abs":max(row[name]["max_abs"] for row in rows)}
    valid=[row for row in rows if row["cut_valid_points"]]
    sdf_pairs=[v for row in valid for v in row["independent_cut_sdf_mae_pairs"]]
    rgb_pairs=[v for row in valid for v in row["independent_cut_linear_rgb_mae_pairs"]]
    summary={"status":"complete","checkpoint_role":model_role,"checkpoint_path":str(cp),
        "checkpoint_sha256":checkpoint_sha,"optimizer_steps":int(checkpoint["optimizer_steps"]),
        "model_config":asdict(model.config),"parameter_count":sum(p.numel() for p in model.parameters()),
        "loaded_state_sha256":loaded_sha,"module_records":learned,
        "weight_reference_checkpoint":str(initial_path) if reference else None,
        "weight_reference_optimizer_steps":int(reference["optimizer_steps"]) if reference else None,"dataset_root":str(root),
        "dataset_manifest_sha256":sha(root/"manifest.json"),"model_source_sha256":sha(Path(__file__).resolve().parents[1]/"foodstateedit/material_lineage/mld2.py"),
        "evaluator_sha256":sha(__file__),"scene_count":len(indices),"selection":f"first {len(indices)} test identities in frozen order, no output-based selection",
        "development_diagnostic_scene_indices":[row["index"] for row in rows if row["used_for_development_diagnostic"]],
        "qualification":"First8 test scenes were used for the dense alias diagnostic; all replay preservation metrics are construction/numerical checks, not blind learned-accuracy measures.",
        "family_counts":{family:sum(row["family"]==family for row in rows) for family in sorted({row["family"] for row in rows})},
        "field_policy":"one cached image context, query-independent continuous H; hard/soft unit-density proxy at8subsites; no GT geometry/material in neural input",
        "noise_policy":{"function":"32-mode Gaussian Fourier field keyed by scene ID+seed, reused at canonical positions","sample_seeds":sample_seeds,"DDIM_steps":args.sample_steps},
        "construction_checks":{"per_id_soft_budget_max_abs":maximum("per_id_soft_budget_max_abs"),
            "soft_volume_budget_abs_max":maximum("soft_volume_budget_abs"),"hard_volume_budget_abs_max":maximum("hard_volume_budget_abs"),
            "child_material_copy_max_abs":maximum("child_material_copy_max_abs"),"inverse_pose_max_abs":maximum("inverse_pose_max_abs"),
            "interpretation":"Budget partition, material copying and pose inverses are by construction; these do not establish learned physical accuracy."},
        "query_replay":{"posterior_512chunk":error_summary("posterior_chunk_error"),"posterior_reverse":error_summary("posterior_reverse_error"),
            "spatial_noise_512chunk":error_summary("noise_chunk_error"),"spatial_noise_reverse":error_summary("noise_reverse_error"),
            "sample_512chunk":error_summary("sample_chunk_error"),"sample_reverse":error_summary("sample_reverse_error")},
        "independent_resampling_control":{"valid_cut_scenes":len(valid),"total_scenes":len(rows),"valid_cut_points":sum(row["cut_valid_points"] for row in rows),
            "shared_state_cut_sdf_mae":maximum("shared_state_cut_sdf_difference"),"shared_state_cut_linear_rgb_mae":maximum("shared_state_cut_linear_rgb_difference"),
            "independent_cut_sdf_mae_mean":float(np.mean(sdf_pairs)) if sdf_pairs else None,
            "independent_cut_linear_rgb_mae_mean":float(np.mean(rgb_pairs)) if rgb_pairs else None,
            "independent_cut_sdf_mae_scene_pairs":sdf_pairs,"independent_cut_linear_rgb_mae_scene_pairs":rgb_pairs,
            "interpretation":"Shared H copies one material state into both cut branches. Independently sampled H may disagree on the same canonical cut plane. This isolates persistence, not realism or correct hidden texture."},
        "prediction_sha256":sha(output/"predictions.npz"),"elapsed_seconds":time.time()-started}
    write_json(output/"summary.json",summary)
    print(json.dumps({"status":"complete","optimizer_steps":summary["optimizer_steps"],"construction_checks":summary["construction_checks"],"query_replay":summary["query_replay"],"elapsed_seconds":summary["elapsed_seconds"]}),flush=True)
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root",required=True);p.add_argument("--checkpoint",required=True)
    p.add_argument("--output-root",required=True);p.add_argument("--scenes",type=int,default=64)
    p.add_argument("--sample-seeds",default="20261004,20261005,20261006")
    p.add_argument("--sample-steps",type=int,default=50)
    p.add_argument("--expected-steps",type=int)
    run(p.parse_args())
