"""Generate the MLD2 multi-topology field dataset and visible-surface targets."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mld2_scene import (SEED,FAMILIES,CAMERA_SCALE,canonical_grid,camera_basis,
    known_camera_projection,subcell_offsets,generate_scene)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


def worker(args): return generate_scene(*args)


def prepare(output,scenes=8192,seed=SEED,grid_size=16,image_size=64,workers=12):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    if (output/"manifest.json").exists():
        raise FileExistsError("This dataset is already generated; use a new output directory.")
    started=time.time(); tokens=grid_size**3
    specs={"source_rgb":((scenes,image_size,image_size,3),np.uint8),
        "joint":((scenes,tokens,4),np.float16),"corner_sdf":((scenes,tokens,8),np.float16),
        "cell_occupancy":((scenes,tokens),np.float16),"occupancy":((scenes,tokens),np.bool_),
        "source_mask":((scenes,image_size,image_size),np.bool_),
        "source_depth":((scenes,image_size,image_size),np.float16),
        "surface_xyz":((scenes,512,3),np.float16),"surface_rgb":((scenes,512,3),np.float16),
        "surface_uv":((scenes,512,2),np.float16),"surface_valid":((scenes,512),np.bool_),
        "near_xyz":((scenes,512,3),np.float16),"near_joint":((scenes,512,4),np.float16),
        "near_corner_sdf":((scenes,512,8),np.float16)}
    arrays={name:np.lib.format.open_memmap(output/f"{name}.npy",mode="w+",shape=shape,dtype=dtype) for name,(shape,dtype) in specs.items()}
    split=np.full(scenes,2,dtype=np.uint8); split[:scenes*3//4]=0
    split[scenes*3//4:scenes*3//4+scenes//8]=1
    np.save(output/"splits.npy",split)
    xyz=canonical_grid(grid_size)
    np.save(output/"canonical_xyz.npy",xyz); np.save(output/"source_uv.npy",known_camera_projection(xyz))
    np.save(output/"subcell_offsets.npy",subcell_offsets(grid_size))
    recipe={"format_version":"mld2-multitopology-v1","seed":seed,"scenes":scenes,
        "inputs":["source_rgb","canonical_xyz","source_uv","independent_user_action"],
        "targets":list(specs),"shape_families":FAMILIES,
        "generator_sha256":sha256(__file__),"scene_helper_sha256":sha256(Path(__file__).with_name("mld2_scene.py")),
        "started_unix":started}
    write_json(output/"recipe.json",recipe)
    with ProcessPoolExecutor(max_workers=workers) as pool, (output/"scenes.jsonl").open("w",encoding="utf-8") as f:
        for i,(record,meta) in enumerate(pool.map(worker,((i,seed,grid_size,image_size) for i in range(scenes)),chunksize=4)):
            for name,array in arrays.items(): array[i]=record[name]
            meta["split"]=["train","validation","test"][int(split[i])]
            f.write(json.dumps(meta,separators=(",",":"))+"\n")
            if i%128==0 or i==scenes-1:
                print(json.dumps({"generated":i+1,"total":scenes,"elapsed_seconds":round(time.time()-started,1)}),flush=True)
    for array in arrays.values(): array.flush()
    files=[]
    for path in sorted(output.iterdir()):
        entry={"path":path.name,"sha256":sha256(path),"bytes":path.stat().st_size}
        if path.suffix==".npy":
            a=np.load(path,mmap_mode="r"); entry.update(shape=list(a.shape),dtype=str(a.dtype))
        files.append(entry)
    manifest={**recipe,"status":"complete","grid_size":grid_size,"image_size":image_size,
        "tokens":tokens,"scene_count":scenes,"split_codes":{"train":0,"validation":1,"test":2},
        "split_counts":{name:int(np.sum(split==k)) for k,name in enumerate(["train","validation","test"])},
        "camera":{"projection":"known orthographic","yaw_degrees":30.,"elevation_degrees":35.,
            "scale":CAMERA_SCALE,"basis_right_up_toward_camera":camera_basis().tolist()},
        "joint_channels":["clamped_sdf_divided_by_0.5","linear_red_times2_minus1","linear_green_times2_minus1","linear_blue_times2_minus1"],
        "corner_semantics":"8 subcell centers per lattice cell at subcell_offsets, normalized clamped SDF/.5; they are not cube vertices",
        "surface_semantics":"512 first-visible ray hits and unlit persistent linear RGB; supervision only, not input. Replacement when fewer than 512 foreground pixels.",
        "near_semantics":"512 visible-surface points perturbed by independent 3D Gaussian sigma .035 clipped[-1,1], with exact generator SDF/material targets; loss supervision only, query locations are never source conditions.",
        "depth_semantics":"first hit xyz dot toward_camera, miss=0; source_mask identifies misses",
        "mass_scope":"unit-density discrete proxy mass, sum8 occupied subcell indicators ×cellvolume/8; no physical grams or true food geometry",
        "action_policy":"Sample plane normals independently from uniform sphere and sample translation/rotation independently; online trainer. Keep held-out normal sector, height and angle tests factored.",
        "sdf_scope":"Exact SDF composition for box/ring/capsules; conservative signed-distance proxies for ellipsoids and rough fields; occupied sign is generator truth.",
        "truth_scope":"Programmatic shape/material supervision, no real paired image, no measured 3D truth. All views/actions for scene identity remain in one split.",
        "elapsed_seconds":time.time()-started,"files":files}
    write_json(output/"manifest.json",manifest)
    return manifest


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--scenes",type=int,default=8192)
    p.add_argument("--seed",type=int,default=SEED); p.add_argument("--grid-size",type=int,default=16)
    p.add_argument("--image-size",type=int,default=64); p.add_argument("--workers",type=int,default=12)
    a=p.parse_args(); m=prepare(a.output,a.scenes,a.seed,a.grid_size,a.image_size,a.workers)
    print(json.dumps({"status":m["status"],"scene_count":m["scene_count"],"elapsed_seconds":m["elapsed_seconds"]}),flush=True)
