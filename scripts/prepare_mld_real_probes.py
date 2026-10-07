"""Freeze and copy unpaired real-image probes without inspecting model outputs.

The selected pictures have no real after-image, measured shape, or hidden-food
labels. They are domain probes, not proof that a synthetic-trained model edits
real food correctly. Captions are provenance only and never model conditions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageOps


RANK_PREFIX = "mld-real-probe-20261004:"
FORMAT_VERSION = "mld.unpaired_real_probes.v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, obj):
    path = Path(path)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def ranking(file_name):
    return hashlib.sha256((RANK_PREFIX + file_name).encode("utf-8")).hexdigest()


def decoded_rgb(path):
    """Validate the original file, then apply orientation only for model pixels."""
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB").copy()


def dhash64(image):
    value = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    bits = (value[:, 1:] > value[:, :-1]).reshape(-1)
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def hamming64(first, second):
    return (first ^ second).bit_count()


def letterbox(image, size=64):
    width, height = image.size
    scale = min(size / width, size / height)
    resized = (max(1, min(size, int(round(width * scale)))),
               max(1, min(size, int(round(height * scale)))))
    small = image.resize(resized, Image.Resampling.LANCZOS)
    left, top = (size - resized[0]) // 2, (size - resized[1]) // 2
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(small, (left, top))
    return np.asarray(canvas, dtype=np.uint8), {
        "source_size_wh": [width, height], "scale": scale,
        "resized_wh": list(resized), "pad_left_top": [left, top],
        "content_bbox_xyxy": [left, top, left + resized[0], top + resized[1]],
        "canvas_size_wh": [size, size], "padding_rgb": [255, 255, 255],
        "interpolation": "Pillow LANCZOS", "orientation": "EXIF transpose before letterbox",
    }


def prepare(source_directory, history_path, output, count=16, size=64,
            max_dhash_distance=8):
    source_directory = Path(source_directory)
    history_path = Path(history_path)
    output = Path(output)
    if count < 1 or size < 8 or max_dhash_distance < 0:
        raise ValueError("Invalid count, image size, or near-duplicate threshold")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output must be empty; never replace a frozen selection")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    names_excluded = set(history["source_file_names"])
    hashes_excluded = set(history["known_source_sha256"])
    metadata_path = source_directory / "metadata.jsonl"
    metadata = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    file_names = [row["file_name"] for row in metadata]
    if len(file_names) != len(set(file_names)):
        raise ValueError("Metadata has duplicate file names; candidate identity is ambiguous")
    for name in file_names:
        if Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("Metadata file names must be single safe basenames")
    candidates = sorted(metadata, key=lambda row: (ranking(row["file_name"]), row["file_name"]))
    output.mkdir(parents=True, exist_ok=True)
    candidate_order = [{"rank": i, "file_name": row["file_name"],
                        "rank_sha256": ranking(row["file_name"])}
                       for i, row in enumerate(candidates)]
    write_json(output / "candidate_order.json", {"candidates": candidate_order})
    recipe = {
        "format_version": FORMAT_VERSION, "status": "frozen_before_source_selection",
        "created_unix": time.time(), "count": count, "source_rgb_size": size,
        "metadata_path": str(metadata_path), "metadata_sha256": sha256(metadata_path),
        "metadata_count": len(metadata), "source_directory": str(source_directory),
        "candidate_order_sha256": sha256(output / "candidate_order.json"),
        "rank_rule": "sha256('mld-real-probe-20261004:' + file_name), ascending, filename tie-break",
        "history_path": str(history_path), "history_sha256": sha256(history_path),
        "history_excluded_file_names": len(names_excluded),
        "history_excluded_sha256": len(hashes_excluded),
        "source_code_sha256": sha256(__file__),
        "selection_rules": ["Filename not in historical exclusion list", "Valid original image decode",
                            "Original SHA256 not in historical exclusion list or selected identities",
                            "dHash64 Hamming distance greater than threshold to every available historical source and previously selected source"],
        "dhash": {"version": "grayscale9x8_lanczos_horizontal_right_greater_than_left_rowmajor_msbfirst",
                  "max_distance_rejected": max_dhash_distance,
                  "scope": "Detects some near duplicates; cannot prove source instances or all prior human inspection are independent"},
        "scope": "Unpaired domain probes selected before viewing model outputs; no real after-images, calibrated geometry, hidden material, or edit-success labels.",
        "model_inputs": ["source_rgb"], "captions_policy": "Saved only as provenance, never supplied to the model",
    }
    write_json(output / "selection_recipe.json", recipe)
    # The frozen recipe exists before any historical/candidate image is opened.
    historical_hashes = []
    historical_unreadable = []
    for name in sorted(names_excluded):
        path = source_directory / name
        try:
            image = decoded_rgb(path)
            digest = sha256(path)
            dh = dhash64(image)
            historical_hashes.append({"file_name": name, "source_sha256": digest,
                                      "dhash64": f"{dh:016x}"})
        except (OSError, ValueError) as exc:
            historical_unreadable.append({"file_name": name, "error": str(exc)})
    write_json(output / "historical_dhash_reference.json", {
        "references": historical_hashes, "unreadable": historical_unreadable,
        "history_sha256": recipe["history_sha256"],
    })
    near_reference = [(row["file_name"], int(row["dhash64"], 16)) for row in historical_hashes]
    selected = []
    seen_hashes = set()
    pixels = []
    decisions = []
    for rank, row in enumerate(candidates):
        name = row["file_name"]
        decision = {"candidate_rank": rank, "file_name": name, "rank_sha256": ranking(name)}
        if name in names_excluded:
            decision["status"] = "rejected_historical_filename"
            decisions.append(decision)
            continue
        path = source_directory / name
        try:
            image = decoded_rgb(path)
            digest = sha256(path)
        except (OSError, ValueError) as exc:
            decision.update(status="rejected_invalid_decode", error=str(exc))
            decisions.append(decision)
            continue
        decision["source_sha256"] = digest
        if digest in hashes_excluded or digest in seen_hashes:
            decision["status"] = "rejected_historical_or_selected_sha256"
            decisions.append(decision)
            continue
        dh = dhash64(image)
        nearest = min(((hamming64(dh, reference), reference_name)
                       for reference_name, reference in near_reference), default=(65, None))
        decision.update(dhash64=f"{dh:016x}", nearest_reference_file=nearest[1],
                        nearest_reference_hamming_distance=nearest[0])
        if nearest[0] <= max_dhash_distance:
            decision["status"] = "rejected_historical_or_selected_near_duplicate"
            decisions.append(decision)
            continue
        case_id = f"real_{len(selected):02d}_{Path(name).stem}"
        case = output / case_id
        case.mkdir()
        copied = case / ("original" + path.suffix.lower())
        shutil.copyfile(path, copied)
        copied_hash = sha256(copied)
        if copied_hash != digest:
            raise RuntimeError("Copy changed original source bytes")
        canvas, preprocessing = letterbox(image, size)
        np.save(case / "source_rgb.npy", canvas, allow_pickle=False)
        Image.fromarray(canvas).save(case / "source.png")
        selected_row = {
            "case_id": case_id, "selected_index": len(selected), "candidate_rank": rank,
            "rank_sha256": ranking(name), "source_file_name": name,
            "source_path": str(path), "source_sha256": digest,
            "copied_original_path": str(copied.relative_to(output)),
            "copied_original_sha256": copied_hash, "dhash64": f"{dh:016x}",
            "source_rgb_path": str((case / "source_rgb.npy").relative_to(output)),
            "source_rgb_sha256": sha256(case / "source_rgb.npy"),
            "source_preview_path": str((case / "source.png").relative_to(output)),
            "preprocessing": preprocessing,
            "metadata_caption_not_model_input": row.get("text"),
            "nearest_historical_or_previously_selected_distance": nearest[0],
            "supervision": "Unpaired real source only; no edited after-image or measured 3D labels",
        }
        write_json(case / "source_record.json", selected_row)
        selected.append(selected_row)
        pixels.append(canvas)
        seen_hashes.add(digest)
        near_reference.append((name, dh))
        decision.update(status="selected", case_id=case_id)
        decisions.append(decision)
        if len(selected) == count:
            break
    write_json(output / "selection_decisions.json", {"decisions": decisions})
    if len(selected) != count:
        raise RuntimeError(f"Only {len(selected)} images pass frozen selection; requested {count}")
    np.save(output / "source_rgb.npy", np.stack(pixels), allow_pickle=False)
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            files.append({"path": str(path.relative_to(output)), "sha256": sha256(path),
                          "bytes": path.stat().st_size})
    manifest = {
        "format_version": FORMAT_VERSION, "status": "complete_unpaired_domain_probes",
        "selected_count": len(selected), "cases": selected,
        "source_rgb_array": {"path": "source_rgb.npy", "shape": [count, size, size, 3],
                             "dtype": "uint8", "sha256": sha256(output / "source_rgb.npy")},
        "metadata_sha256": recipe["metadata_sha256"], "history_sha256": recipe["history_sha256"],
        "selection_recipe_sha256": sha256(output / "selection_recipe.json"),
        "candidate_order_sha256": recipe["candidate_order_sha256"],
        "historical_reference_count": len(historical_hashes),
        "historical_reference_unreadable": historical_unreadable,
        "near_duplicate_rule": "Reject dHash64 distance <= 8 to available history or earlier selections",
        "near_duplicate_scope": recipe["dhash"]["scope"],
        "conditions": ["source_rgb"], "caption_used_by_model": False,
        "source_preprocessing": "Orientation corrected, LANCZOS resize preserving aspect ratio, white letterbox to 64x64; original file copied byte for byte",
        "scope": recipe["scope"],
        "validation_claim": "No edit realism or geometry accuracy success can be concluded from these unpaired sources alone",
        "files": files,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--max-dhash-distance", type=int, default=8)
    args = parser.parse_args()
    result = prepare(args.source_directory, args.history, args.output, args.count,
                     args.size, args.max_dhash_distance)
    print(json.dumps({"status": result["status"], "selected_count": result["selected_count"],
                      "selected_file_names": [case["source_file_name"] for case in result["cases"]]}), flush=True)


if __name__ == "__main__":
    main()
