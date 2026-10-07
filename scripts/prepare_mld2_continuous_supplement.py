"""Add uniform continuous supervision to existing frozen scene identities."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mld2_scene import rng_for,shape_sdf,material_rgb


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def worker(args):
    record,queries=args
    r=rng_for(record["scene_id"],"uniform-continuous-v1")
    xyz=r.uniform(-1,1,(queries,3)).astype(np.float32)
    p=record["target_generator_parameters"]
    sdf=shape_sdf(xyz,p);rgb=material_rgb(xyz,p)
    joint=np.concatenate([np.clip(sdf[:,None],-.5,.5)/.5,rgb*2-1],-1).astype(np.float16)
    return xyz,joint


def main(args):
    started=time.time();data=Path(args.dataset_root);out=Path(args.output_root)
    out.mkdir(parents=True,exist_ok=True)
    if (out/"manifest.json").exists():raise FileExistsError("Choose a fresh supplement output")
    records=[json.loads(line) for line in (data/"scenes.jsonl").read_text().splitlines()]
    n=len(records)
    xyz=np.lib.format.open_memmap(out/"uniform_xyz.npy",mode="w+",shape=(n,args.queries,3),dtype=np.float32)
    joint=np.lib.format.open_memmap(out/"uniform_joint.npy",mode="w+",shape=(n,args.queries,4),dtype=np.float16)
    recipe={"format_version":"mld2-continuous-uniform-v1","source_dataset_root":str(data),
        "source_manifest_sha256":sha(data/"manifest.json"),"source_scenes_sha256":sha(data/"scenes.jsonl"),
        "generator_sha256":sha(__file__),"scene_helper_sha256":sha(Path(__file__).with_name("mld2_scene.py")),
        "scene_count":n,"queries_per_scene":args.queries,"RNG_stream":"scene_identity:uniform-continuous-v1",
        "query_policy":"Independent uniform[-1,1]^3 query points; targets sampled from original frozen scene parameters",
        "condition_policy":"Loss supervision only; source RGB remains the only image condition. Use original splits, never train on test identities.",
        "joint_channels":["clamped_sdf_divided_by_0.5","linear_red_times2_minus1","linear_green_times2_minus1","linear_blue_times2_minus1"],
        "scope":"Synthetic continuous supervision to reduce regular-grid high-frequency aliasing, not real geometry truth"}
    (out/"recipe.json").write_text(json.dumps(recipe,indent=2))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,(x,j) in enumerate(pool.map(worker,((record,args.queries) for record in records),chunksize=16)):
            xyz[i]=x;joint[i]=j
            if i%1024==0 or i==n-1:
                print(json.dumps({"generated":i+1,"total":n,"elapsed_seconds":round(time.time()-started,2)}),flush=True)
    xyz.flush();joint.flush()
    assert np.isfinite(xyz).all() and np.isfinite(joint).all()
    manifest={**recipe,"status":"complete","elapsed_seconds":time.time()-started,
        "files":[{"path":path.name,"sha256":sha(path),"bytes":path.stat().st_size} for path in [out/"uniform_xyz.npy",out/"uniform_joint.npy"]]}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps({"status":"complete","elapsed_seconds":manifest["elapsed_seconds"]}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root",required=True);p.add_argument("--output-root",required=True)
    p.add_argument("--queries",type=int,default=1024);p.add_argument("--workers",type=int,default=12)
    main(p.parse_args())
