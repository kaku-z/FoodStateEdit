"""Source-only food removal test on audited Nutrition5k incremental scan pairs.

The four source rectangles were fixed by visual inspection of source RGB only.
Target RGB is loaded solely after all edited images are generated, for scoring.
This tests revealed-background synthesis, not food transport or spoon geometry.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


# (x, y, width, height): source-only rough annotations, compact and multi-piece.
CASES = {
    0: (267, 45, 98, 110),   # orange beside almonds
    3: (391, 40, 141, 147),  # apple beside bagel
    5: (334, 42, 186, 174),  # pizza portion beside corn
    7: (146, 115, 124, 125), # corn slice beside pizza
    9: (407, 182, 215, 285), # corn slices beside pizza, multi-piece stress case
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_mask(rgb, rect):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    labels = np.zeros(rgb.shape[:2], np.uint8)
    bg = np.zeros((1, 65), np.float64)
    fg = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, labels, rect, bg, fg, 8, cv2.GC_INIT_WITH_RECT)
    mask = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
    return mask


def mae(a, b, area):
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=2)
    return float(diff[area].mean()) if area.any() else None


def plate_surface(source, depth, mask, rect):
    """Fit local plate RGB/depth from source-only neutral, valid-depth pixels."""
    h, w = mask.shape
    x, y, rw, rh = rect
    yy, xx = np.mgrid[:h, :w]
    expanded = (xx >= x - 55) & (xx < x + rw + 55) & (yy >= y - 55) & (yy < y + rh + 55)
    rgb = source.astype(np.float64)
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    neutral = ((mx - mn) / np.maximum(mx, 1)) < 0.22
    sample = expanded & (mask == 0) & neutral & (mx > 85) & (depth > 0) & (depth < 10000)
    # A high sample count is expected; subsample deterministically for speed.
    sample[::1, ::1] &= ((xx[::1, ::1] + 3 * yy[::1, ::1]) % 3 == 0)
    u, v = (xx - x) / 100, (yy - y) / 100
    basis = np.stack((np.ones_like(u), u, v, u*u, u*v, v*v), axis=-1)
    X = basis[sample]
    Y = np.column_stack((rgb[sample], depth[sample].astype(np.float64)))
    if len(X) < 200:
        raise ValueError(f"Only {len(X)} plate samples for {rect}")
    keep = np.ones(len(X), bool)
    for _ in range(4):
        coef = np.linalg.lstsq(X[keep], Y[keep], rcond=None)[0]
        residual = np.abs(X @ coef - Y)
        # Reject nearby food edges and depth anomalies; independent of target.
        keep = (np.max(residual[:, :3], axis=1) < 30) & (residual[:, 3] < 100)
    fit = (basis.reshape(-1, 6) @ coef).reshape(h, w, 4)
    return np.clip(fit[:, :, :3], 0, 255).astype(np.uint8), np.clip(fit[:, :, 3], 0, 65535).astype(np.uint16), int(keep.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pairs = json.loads((args.pairs / "selected_manifest.json").read_text())["selected"]
    rows = []
    for index, rect in CASES.items():
        pair = pairs[index]
        folder = args.pairs / f"pair_{index:03d}"
        source_path, target_path = folder / "source_rgb.png", folder / "target_rgb.png"
        for item in pair["files"]:
            actual = folder / f'{item["tag"]}_{item["modality"]}'
            if sha256(actual) != item["sha256"]:
                raise ValueError(f"Dataset checksum mismatch: {actual}")
        source = cv2.cvtColor(cv2.imread(str(source_path)), cv2.COLOR_BGR2RGB)
        source_depth = cv2.imread(str(folder / "source_depth_raw.png"), cv2.IMREAD_UNCHANGED)
        if source_depth.dtype != np.uint16 or source_depth.shape != source.shape[:2]:
            raise ValueError(f"Invalid source RGB-D pair: {folder}")
        # No target read until predictions are written to disk.
        mask = get_mask(source, rect)
        source_bgr = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
        predicted = {}
        out = args.output / f"pair_{index:03d}"
        out.mkdir()
        cv2.imwrite(str(out / "source.png"), source_bgr)
        cv2.imwrite(str(out / "source_mask.png"), mask)
        for method, flag in (("telea", cv2.INPAINT_TELEA), ("ns", cv2.INPAINT_NS)):
            for radius in (3, 7, 15):
                key = f"{method}_{radius}"
                img = cv2.inpaint(source_bgr, mask, radius, flag)
                path = out / f"{key}.png"
                cv2.imwrite(str(path), img)
                predicted[key] = path
        plate_rgb, plate_depth, plate_samples = plate_surface(source, source_depth, mask, rect)
        plate_pred = source.copy()
        plate_pred[mask != 0] = plate_rgb[mask != 0]
        plate_path = out / "plate_fit.png"
        cv2.imwrite(str(plate_path), cv2.cvtColor(plate_pred, cv2.COLOR_RGB2BGR))
        predicted["plate_fit"] = plate_path
        predicted_depth = source_depth.copy()
        predicted_depth[mask != 0] = plate_depth[mask != 0]
        cv2.imwrite(str(out / "plate_fit_depth.png"), predicted_depth)
        # Fixed rectangular evaluation area includes whole object and inpaint rim.
        x, y, w, h = rect
        area = np.zeros(mask.shape, bool)
        area[max(0, y - 10):min(area.shape[0], y + h + 10),
             max(0, x - 10):min(area.shape[1], x + w + 10)] = True
        # Read target only for scoring, and preserve its original unregistered frame.
        target = cv2.cvtColor(cv2.imread(str(target_path)), cv2.COLOR_BGR2RGB)
        target_depth = cv2.imread(str(folder / "target_depth_raw.png"), cv2.IMREAD_UNCHANGED)
        if target_depth.dtype != np.uint16 or target_depth.shape != target.shape[:2] or target.shape != source.shape:
            raise ValueError(f"Invalid target RGB-D pair: {folder}")
        cv2.imwrite(str(out / "target.png"), cv2.cvtColor(target, cv2.COLOR_RGB2BGR))
        score = {"pair": index, "source_dish": pair["source_after_addition"],
                 "target_dish": pair["target_before_addition"],
                 "added_ingredient": pair["added_ingredient"][1],
                 "rect_xywh": list(rect), "source_sha256": sha256(source_path),
                 "target_sha256": sha256(target_path),
                 "mask_pixels": int((mask != 0).sum()), "region_pixels": int(area.sum()),
                 "plate_fit_samples": plate_samples,
                 "unchanged_mae": mae(source, target, area),
                 "outside_region_mae": mae(source, target, ~area)}
        valid_depth = (mask != 0) & (source_depth > 0) & (source_depth < 10000) & (target_depth > 0) & (target_depth < 10000)
        score["valid_depth_pixels"] = int(valid_depth.sum())
        score["unchanged_depth_cm_mae"] = (float(np.abs(source_depth.astype(np.float64) - target_depth.astype(np.float64))[valid_depth].mean()) / 100)
        score["plate_fit_depth_cm_mae"] = (float(np.abs(predicted_depth.astype(np.float64) - target_depth.astype(np.float64))[valid_depth].mean()) / 100)
        for name, path in predicted.items():
            pred = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            score[name + "_mae"] = mae(pred, target, area)
        rows.append(score)
    fields = list(rows[0].keys())
    with (args.output / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    cols = ["source.png", "target.png", "telea_15.png", "plate_fit.png"]
    overview = np.full((len(CASES) * 260, len(cols) * 320, 3), 255, np.uint8)
    for row_no, index in enumerate(CASES):
        for col_no, name in enumerate(cols):
            image = cv2.imread(str(args.output / f"pair_{index:03d}" / name))
            image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
            top, left = row_no * 260, col_no * 320
            overview[top:top + 240, left:left + 320] = image
            cv2.putText(overview, f"{index:03d} {name[:-4]}", (left + 6, top + 255),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(args.output / "overview.png"), overview)
    compact = {0, 3, 5, 7}
    def means(subset):
        return {key: float(np.mean([r[key] for r in subset])) for key in fields
                if key.endswith("_mae")}
    summary = {"dataset": "Nutrition5k", "cases": list(CASES),
               "scope": "reverse addition, single object removal/background restoration only",
               "evaluation": "RGB mean absolute error, 0-255, fixed source-annotated rectangle plus 10px; depth MAE in cm inside source-only mask; no registration",
               "target_used_in_generation": False,
               "results": rows,
               "mean_mae": means(rows),
               "compact_mean_mae": means([r for r in rows if r["pair"] in compact]),
               "multi_piece_mean_mae": means([r for r in rows if r["pair"] not in compact])}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["mean_mae"], indent=2))


if __name__ == "__main__":
    main()
