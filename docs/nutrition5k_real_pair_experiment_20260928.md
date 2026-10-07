# Real-photo RGB-D pilot: revealed-plate recovery

## Question and scope

Can a source-only food-removal method reconstruct the plate region revealed when food is removed? This is one measurable component of FoodStateEdit. The experiment does **not** test spoon insertion, lifted-food appearance, object identity at a destination, volume conservation, or a complete source-to-target food manipulation. No real paired spoon-lift images with calibrated geometry are currently available in this project.

## Data and curation

The public [Nutrition5k dataset](https://github.com/google-research-datasets/Nutrition5k) provides real overhead RGB-D images, incremental ingredient scans, and ingredient masses under CC BY 4.0. We compare an image **after addition** (source) to the earlier image **before addition** (target), thereby evaluating the reverse removal problem. The official README gives raw-depth scale as 10,000 units per meter.

The download script selects candidates from metadata where exactly one ingredient was added, the other ingredient retained the same identity and recorded mass, and the scans are at most five minutes apart. This metadata rule is insufficient by itself: the first unfiltered sample contained distinct plates. We downloaded 16 solid-food candidates and visually audited the image pairs. The final set has four compact objects (orange, apple, pizza portion, corn slice: indices 0, 3, 5, 7) and one multi-piece stress case (corn slices: index 9). Pair 4 was retained in the preliminary audit output but excluded from the primary comparison because the whole plate moved substantially between captures (outside-edit RGB MAE 31.06). The other candidate pairs were not forced into the compact-object test. This is a small, post-hoc curated pilot, not a held-out benchmark.

For each final pair, a rough rectangle was drawn from the **source image only**. GrabCut obtains the object mask. The methods use source RGB-D and the rectangle; the target is loaded after predicted images have been written, solely for evaluation. All input file SHA-256 values are checked against the download manifest.

## Methods and metrics

Comparators: unchanged source; OpenCV Telea and Navier-Stokes inpainting (radii 3, 7, 15); local plate-surface fit. The last method robustly fits a quadratic in image coordinates to neutral-colored, valid-depth source pixels near the object, replacing masked RGB and depth. It has no camera-calibration or learned hidden-surface model.

RGB error is mean absolute channel error (0–255) within the fixed source rectangle expanded by 10 pixels. Depth error is mean absolute error in centimeters inside the source-only object mask, at pixels where both scans have valid raw depth. Images are compared in their original coordinates without registration. Outside-region source/target RGB MAE measures scan-to-scan change and is not an algorithm error. Means are unweighted across pairs; no significance test is justified at this sample size.

| Pair | Added food | Source RGB MAE | Telea-15 | Plate fit | Source depth MAE (cm) | Plate depth MAE (cm) |
|---|---|---:|---:|---:|---:|---:|
| 0 | Orange | 38.65 | 25.30 | 25.25 | 3.37 | 0.64 |
| 3 | Apple | 55.12 | 28.24 | 30.70 | 4.85 | 0.12 |
| 5 | Pizza portion | 34.02 | 11.61 | 5.98 | 1.28 | 0.15 |
| 7 | Corn slice | 26.92 | 21.01 | 13.87 | 2.06 | 0.65 |
| 9 | Corn slices (multi-piece) | 21.78 | 19.87 | 6.57 | 1.70 | 0.36 |
| **Mean, 5** | | **35.30** | **21.21** | **16.47** | **2.65** | **0.38** |

The plate fit reduces mean RGB MAE by 53.3% relative to unchanged source and mean masked depth MAE by 85.6%. It is worse than Telea-15 on the apple, and all five outputs retain visible seams, flat patches, or object-edge remnants. These visual failures matter even where numeric error improves. The depth result is conditional on these source masks and on the simple plate-background cases; it does not prove 3D food mass conservation or manipulation success.

## Reproduce and inspect

This real-photo run first used local Python 3.13, NumPy, and OpenCV on Windows. It was then repeated on gp40 using Python 3.11 from the existing `ditseg` environment. Interactive password authentication to im00 established a temporary SSH tunnel; gp40 itself accepted the local key. Because the account's persistent home filesystem exceeded its quota, the server run used `/tmp/codex_foodstateedit_real_20260928_v1` and its results were copied back to `outputs/nutrition5k_real_removal_20260928_gp40/`. The server and local `scores.csv` are byte-for-byte identical, the overview PNGs have identical SHA-256 hashes, and the summary JSON objects compare equal (their raw bytes differ only in platform text formatting). To re-download the curated candidate pool (network needed):

```powershell
py -3.13 scripts/prepare_nutrition5k_pairs.py --output outputs/nutrition5k_solid_pairs_rebuild --count 18 --ingredient orange --ingredient sausage --ingredient apple --ingredient tofu --ingredient pizza --ingredient chicken --ingredient 'corn on the cob' --ingredient 'grilled chicken'
```

The exact downloaded files and SHA-256 values used in this run are in `outputs/nutrition5k_solid_pairs_20260928_v1/selected_manifest.json`. Re-run the fixed evaluation:

```powershell
py -3.13 scripts/run_nutrition5k_real_removal.py --pairs outputs/nutrition5k_solid_pairs_20260928_v1 --output outputs/nutrition5k_real_removal_rebuild
```

Primary artifacts are `outputs/nutrition5k_real_removal_20260928_validated/summary.json`, `scores.csv`, `overview.png`, and per-pair source, target, mask, RGB predictions, and depth prediction. The preliminary six-case audit, including excluded pair 4, is in `outputs/nutrition5k_real_removal_20260928_v4/`.

The full proposed FoodStateEdit experiment still requires photographs of the **same food actually moved onto a spoon**, synchronized before/after RGB-D, camera calibration, spoon pose, object mask, and a measured or otherwise verified mass/volume target. Sequential ingredient additions provide none of the lifted-food or spoon-contact target views, so they cannot certify the complete algorithm.
