"""Whole-object near-surface targets, including surfaces facing away from camera."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mld2_scene import rng_for,shape_sdf,material_rgb,camera_basis


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def gradient(x,p,eps=.001):
    return np.stack([(shape_sdf(x+np.eye(3)[i]*eps,p)-shape_sdf(x-np.eye(3)[i]*eps,p))/(2*eps) for i in range(3)],-1)


def worker(args):
    record,queries=args;p=record["target_generator_parameters"]
    r=rng_for(record["scene_id"],"all-surface-near-v1")
    x=r.uniform(-.98,.98,(queries*2,3))
    for _ in range(10):
        sdf=shape_sdf(x,p)
        grad=gradient(x,p)
        step=sdf[:,None]*grad/np.maximum((grad*grad).sum(-1,keepdims=True),1e-8)
        length=np.linalg.norm(step,axis=-1,keepdims=True)
        step*=np.minimum(1.,.35/np.maximum(length,1e-8))
        x=np.clip(x-step,-.98,.98)
    sd=shape_sdf(x,p)
    valid=np.flatnonzero(np.abs(sd)<.003)
    chosen=r.choice(valid,queries//2,replace=len(valid)<queries//2)
    anchor=x[chosen];normal=gradient(anchor,p)
    normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-8)
    distance=r.uniform(.02,.035,(queries//2,1))
    near=np.concatenate([anchor+distance*normal,anchor-distance*normal],0)
    near=np.clip(near,-1,1).astype(np.float32)
    permutation=r.permutation(queries);near=near[permutation]
    sdf=shape_sdf(near,p);rgb=material_rgb(near,p)
    joint=np.concatenate([np.clip(sdf[:,None],-.5,.5)/.5,rgb*2-1],-1).astype(np.float16)
    statistics={"scene_index":record["index"],"scene_id":record["scene_id"],"shape_family":p["family"],
        "projection_valid_fraction":float(len(valid)/len(x)),
        "anchor_sdf_abs_max":float(np.abs(shape_sdf(anchor,p)).max()),
        "positive_sdf_fraction":float((joint[:,0]>0).mean()),
        "near_sdf_actual_abs_mean":float(np.abs(sdf).mean()),
        "near_sdf_actual_abs_p90":float(np.quantile(np.abs(sdf),.9)),
        "away_facing_anchor_fraction":float((normal@camera_basis()[2]<0).mean())}
    return near,joint,anchor.astype(np.float32),normal.astype(np.float16),statistics


def main(args):
    started=time.time();data=Path(args.dataset_root);out=Path(args.output_root)
    out.mkdir(parents=True,exist_ok=True)
    if (out/"manifest.json").exists():raise FileExistsError("Choose a fresh supplement output")
    records=[json.loads(line) for line in (data/"scenes.jsonl").read_text().splitlines()];n=len(records)
    q=args.queries
    arrays={name:np.lib.format.open_memmap(out/(name+".npy"),mode="w+",shape=shape,dtype=dtype) for name,shape,dtype in [
        ("all_surface_near_xyz",(n,q,3),np.float32),("all_surface_near_joint",(n,q,4),np.float16),
        ("all_surface_anchor_xyz",(n,q//2,3),np.float32),("all_surface_anchor_normal",(n,q//2,3),np.float16)]}
    recipe={"format_version":"mld2-all-surface-near-v1","source_dataset_root":str(data),
        "source_manifest_sha256":sha(data/"manifest.json"),"source_scenes_sha256":sha(data/"scenes.jsonl"),
        "generator_sha256":sha(__file__),"scene_helper_sha256":sha(Path(__file__).with_name("mld2_scene.py")),
        "scene_count":n,"queries_per_scene":q,"RNG_stream":"scene_identity:all-surface-near-v1",
        "method":"Uniform3D seeds, 10 bounded Newton signed-distance/gradient steps; select |SDF|<.003 anchors; pair +normal/-normal offsets uniform.02–.035 and shuffle. Does not use source visibility.",
        "condition_policy":"GT loss supervision only; original source RGB unchanged; preserve original identity indices and splits.",
        "joint_channels":["clamped_sdf_divided_by_0.5","linear_red_times2_minus1","linear_green_times2_minus1","linear_blue_times2_minus1"],
        "scope":"Synthetic surfaces across full object; away-facing orientation confirms back surfaces, not measured real visibility or geometry."}
    (out/"recipe.json").write_text(json.dumps(recipe,indent=2));rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool,(out/"per_scene.jsonl").open("w") as f:
        for i,result in enumerate(pool.map(worker,((record,q) for record in records),chunksize=8)):
            xyz,joint,anchor,normal,stats=result
            for name,value in zip(arrays,[xyz,joint,anchor,normal]):arrays[name][i]=value
            stats["split"]=records[i]["split"];f.write(json.dumps(stats)+"\n");rows.append(stats)
            if i%256==0 or i==n-1:
                print(json.dumps({"generated":i+1,"total":n,"elapsed_seconds":round(time.time()-started,2)}),flush=True)
    for array in arrays.values():array.flush()
    families=sorted({row["shape_family"] for row in rows})
    family_stats={}
    for family in families:
        selected=[row for row in rows if row["shape_family"]==family]
        family_stats[family]={"scene_count":len(selected),**{key:float(np.mean([row[key] for row in selected])) for key in [
            "projection_valid_fraction","positive_sdf_fraction","near_sdf_actual_abs_mean","near_sdf_actual_abs_p90","away_facing_anchor_fraction"]}}
    manifest={**recipe,"status":"complete","elapsed_seconds":time.time()-started,"per_family":family_stats,
        "split_counts":{split:sum(row["split"]==split for row in rows) for split in ["train","validation","test"]},
        "files":[{"path":path.name,"sha256":sha(path),"bytes":path.stat().st_size} for path in sorted(out.glob("*.npy"))]}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps({"status":"complete","elapsed_seconds":manifest["elapsed_seconds"],"per_family":family_stats}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root",required=True);p.add_argument("--output-root",required=True)
    p.add_argument("--queries",type=int,default=512);p.add_argument("--workers",type=int,default=12)
    main(p.parse_args())
