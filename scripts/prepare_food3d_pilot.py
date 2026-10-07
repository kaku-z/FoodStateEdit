"""Freeze a source-only single-image geometry pilot, separate from prior edits."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs/food3d_pilot_20260928"
    inputs = out / "inputs"
    inputs.mkdir(exist_ok=False)
    cases = json.loads((root / "benchmark/first_bite_20260928/cases.bound.v1.json").read_text(encoding="utf-8"))
    case = next(c for c in cases if c["case_id"] == "cohesive_01")
    source = root / case["source"]["local_path"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == case["source"]["sha256"]
    (inputs / "source.jpg").write_bytes(source.read_bytes())
    im = Image.open(source).convert("RGB")
    assert im.size == (640, 480)
    polygon = [(156,139),(355,39),(364,35),(390,48),(540,179),(555,193),
               (562,216),(551,252),(538,310),(530,354),(514,373),(371,456),
               (349,465),(336,453),(277,385),(227,319),(191,258),(172,212),(163,178)]
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).polygon(polygon, fill=255)
    mask.save(inputs / "food_mask.png")
    rgba = im.convert("RGBA")
    rgba.putalpha(mask)
    rgba.save(inputs / "food_rgba.png")
    review = im.copy()
    ImageDraw.Draw(review).line(polygon + [polygon[0]], fill=(0,255,255), width=3)
    review.save(inputs / "mask_review.png")
    manifest = {
        "scope": "One real development photo, two fixed seeds; geometry feasibility only, no paired 3D ground truth.",
        "source": case["source"], "case_id": case["case_id"],
        "mask": {"method": "assistant source-only manual silhouette polygon", "vertices_px": polygon,
                 "not_ground_truth": True, "excludes": "plate and soy-sauce pool"},
        "model": "tencent/Hunyuan3D-2.1", "revision": "0b94677654c57bb9a6b6845cd7b704ccf551d327",
        "code_commit": "82920d643c0dc2f7bfd7255f45f62d386edfe60c",
        "selection": "SAM 3D weight request returned 401; public Hunyuan3D 2.1 shape-only fallback. TRELLIS.2 not run.",
        "seeds": [281,913], "steps": 50, "guidance_scale": 5.0,
        "octree_resolution": 384, "num_chunks": 8000, "surface_extractor": "mc",
        "checks": ["raw mesh closure and components", "multi-view shape", "source silhouette registration",
                   "shared-source boolean cut and lift", "volume conservation in reconstructed mesh"],
        "limits": ["Hidden geometry is a model prediction.", "Silhouette fitting is an in-sample fit, not 3D accuracy.",
                   "Geometric lift is not a VACE output or evidence of photorealistic editing.",
                   "No texture model, MoGe, physical deformation, or utensil-contact solver in this pilot."],
        "files": {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs.iterdir())},
    }
    (out / "input_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"input": str(inputs), "source_sha256": manifest["source"]["sha256"]}))


if __name__ == "__main__":
    main()
