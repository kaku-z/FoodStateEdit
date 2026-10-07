"""Find auditable real incremental food-photo pairs from public Nutrition5k.

Pair selection uses metadata only; target images are not examined until after
the candidate list is frozen. Nutrition5k is CC BY 4.0 (see its official README).
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.error
import urllib.request

BASE = "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/"
META = ("dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv")


def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "FoodStateEdit-research"}), timeout=30).read()


def ingredients(row):
    # First six fields are dish attributes; each ingredient occupies seven.
    return tuple((row[i], row[i + 1], row[i + 2]) for i in range(6, len(row), 7))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--ingredient", action="append", default=[], help="Limit to exact added-ingredient names; repeat")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    metadata = {}
    for name in META:
        data = fetch(BASE + "metadata/" + name)
        (args.output / name).write_bytes(data)
        metadata[name] = hashlib.sha256(data).hexdigest()
        for row in csv.reader(io.StringIO(data.decode())):
            rows.append((name, row))
    groups = {}
    for cafe, row in rows:
        groups.setdefault((cafe, ingredients(row)), []).append(row)
    candidates = []
    for cafe, new in rows:
        contents = ingredients(new)
        if len(contents) != 2:
            continue
        for removed in contents:
            # Earlier scan must retain exactly one ingredient with exactly the
            # same recorded identity/weight; only one new ingredient appears.
            preserved = tuple(x for x in contents if x != removed)
            if len(preserved) != 1:
                continue
            for old in groups.get((cafe, preserved), []):
                dt = int(new[0][5:]) - int(old[0][5:])
                if 0 < dt <= 300:
                    candidates.append({"cafe": cafe, "source_after_addition": new[0],
                                       "target_before_addition": old[0],
                                       "added_ingredient": list(removed), "preserved": list(preserved[0]),
                                       "seconds_between_scans": dt,
                                       "source_mass_g": float(new[2]), "target_mass_g": float(old[2])})
    # Deterministic order, frozen before any target images are opened.
    candidates.sort(key=lambda r: (r["source_after_addition"], r["target_before_addition"]))
    if args.ingredient:
        candidates = [c for c in candidates if c["added_ingredient"][1] in args.ingredient]
    (args.output / "candidate_manifest.json").write_text(json.dumps({
        "selection": "one shared ingredient identity and weight; one added; <=300s; metadata only",
        "candidate_count": len(candidates), "metadata_sha256": metadata, "candidates": candidates}, indent=2))
    chosen = []
    used_source = set()
    for candidate in candidates:
        if len(chosen) >= args.count:
            break
        if candidate["source_after_addition"] in used_source:
            continue
        items = []
        for tag, dish in (("source", candidate["source_after_addition"]),
                          ("target", candidate["target_before_addition"])):
            for modality in ("rgb.png", "depth_raw.png"):
                remote = f"imagery/realsense_overhead/{dish}/{modality}"
                try:
                    data = fetch(BASE + remote)
                except urllib.error.HTTPError as exc:
                    if exc.code == 404:
                        items = []
                        break
                    raise
                out = args.output / f"pair_{len(chosen):03d}" / f"{tag}_{modality}"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(data)
                items.append({"tag": tag, "modality": modality, "relative_path": str(out.relative_to(args.output)),
                              "source_url": BASE + remote, "sha256": hashlib.sha256(data).hexdigest(),
                              "size": len(data)})
            if not items:
                break
        if len(items) == 4:
            chosen.append({**candidate, "files": items})
            used_source.add(candidate["source_after_addition"])
    (args.output / "selected_manifest.json").write_text(json.dumps({
        "dataset": "Nutrition5k", "source": "https://github.com/google-research-datasets/Nutrition5k",
        "license": "CC BY 4.0", "sampling": "first available candidates sorted by dish ID; no image quality screening",
        "selected": chosen}, indent=2))
    print(json.dumps({"candidate_count": len(candidates), "downloaded_pairs": len(chosen),
                      "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
