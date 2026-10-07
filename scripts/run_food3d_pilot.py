"""Run the frozen Hunyuan3D shape pilot in its isolated server environment."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "hunyuan/hy3dshape"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import numpy as np
    from PIL import Image
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.models.autoencoders import SurfaceExtractors
    manifest = json.loads((root / "input_manifest.json").read_text())
    receipt = json.loads((root / "model_receipt.json").read_text())
    assert receipt["revision"] == manifest["revision"]
    for name, digest in manifest["files"].items():
        assert sha(root / "inputs" / name) == digest, name
    for entry in receipt["files"]:
        assert sha(root / "models/Hunyuan3D-2.1" / entry["file"]) == entry["sha256"]
    out = root / "results/shape_v1"
    out.mkdir(exist_ok=False)
    record = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "status": "loading", "gpu": torch.cuda.get_device_name(0),
              "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "script_sha256": sha(Path(__file__)),
              "input_manifest_sha256": sha(root / "input_manifest.json"),
              "versions": {k:importlib.metadata.version(k) for k in ["torch","torchvision","transformers","diffusers","trimesh","numpy"]},
              "jobs": []}
    def save():
        (out / "run_manifest.json").write_text(json.dumps(record,indent=2)+"\n")
    save()
    start = time.monotonic()
    try:
        folder = root / "models/Hunyuan3D-2.1/hunyuan3d-dit-v2-1"
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
            str(folder / "model.fp16.ckpt"), str(folder / "config.yaml"),
            device="cuda", dtype=torch.float16, use_safetensors=False)
        pipeline.vae.surface_extractor = SurfaceExtractors[manifest["surface_extractor"]]()
        image = Image.open(root / "inputs/food_rgba.png").convert("RGBA")
        record["load_seconds"] = time.monotonic() - start
        for seed in manifest["seeds"]:
            job = {"seed":seed,"status":"running"}
            record["jobs"].append(job)
            record["status"] = "running"
            save()
            torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(seed)
            np.random.seed(seed)
            t = time.monotonic()
            try:
                mesh = pipeline(image=image, generator=torch.Generator(device="cuda").manual_seed(seed),
                    num_inference_steps=manifest["steps"], guidance_scale=manifest["guidance_scale"],
                    octree_resolution=manifest["octree_resolution"], num_chunks=manifest["num_chunks"],
                    output_type="trimesh")[0]
                if mesh is None:
                    raise RuntimeError("Surface extractor returned no mesh")
                file = out / f"seed_{seed}_raw.ply"
                mesh.export(file)
                # Keep full-resolution raw geometry; do not silently repair or remove pieces.
                mesh.export(out / f"seed_{seed}_raw.glb")
                job.update(status="complete", seconds=time.monotonic()-t,
                           vertices=len(mesh.vertices), faces=len(mesh.faces),
                           is_watertight=bool(mesh.is_watertight),
                           is_winding_consistent=bool(mesh.is_winding_consistent),
                           signed_volume=float(mesh.volume), bounds=mesh.bounds.tolist(),
                           connected_components=len(mesh.split(only_watertight=False)),
                           peak_allocated_gb=torch.cuda.max_memory_allocated()/1024**3,
                           output_sha256=sha(file))
                print("JOB_COMPLETE",json.dumps(job),flush=True)
            except Exception:
                job.update(status="failed",error=traceback.format_exc())
                print(job["error"],flush=True)
            save()
        record["status"] = "complete" if all(j["status"]=="complete" for j in record["jobs"]) else "completed_with_failures"
    except Exception:
        record.update(status="failed",error=traceback.format_exc())
        print(record["error"],flush=True)
    record["elapsed_seconds"] = time.monotonic()-start
    save()
    print("FINAL",record["status"],flush=True)


if __name__ == "__main__":
    main()
