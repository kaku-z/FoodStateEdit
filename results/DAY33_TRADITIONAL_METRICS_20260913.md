# Day 33 Traditional Image Metrics

This table compares each edited output with its input. Higher PSNR/SSIM and lower MAE mean stronger input fidelity, not a more correct utensil action.

| Method | n | Full PSNR ↑ | Full SSIM ↑ | Outside PSNR ↑ | Outside SSIM ↑ | Outside MAE ↓ | Outside exact ↑ | Inside change MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ChordEdit | 12 | 26.29 | 0.9438 | inf | 1.0000 | 0.000 | 1.0000 | 20.993 |
| Qwen-Image-Edit | 12 | 13.90 | 0.4506 | 14.66 | 0.4628 | 35.468 | 0.0017 | 51.121 |
| Native VACE | 12 | 34.60 | 0.9823 | inf | 1.0000 | 0.000 | 1.0000 | 8.330 |
| 2D planar proxy | 12 | 24.86 | 0.9473 | inf | 1.0000 | 0.000 | 1.0000 | 20.180 |
| fixed relative3D | 12 | 24.60 | 0.9452 | inf | 1.0000 | 0.000 | 1.0000 | 20.733 |
| FoodStateEdit material-policy candidate | 12 | 23.29 | 0.9384 | inf | 1.0000 | 0.000 | 1.0000 | 25.863 |

## Interpretation

- ChordEdit and all projected VACE variants preserve protected pixels exactly by construction; this is a preservation property, not proof that the requested action occurred.
- Qwen-Image-Edit globally regenerates/reframes the image, so input-fidelity scores include registration and resize effects even when the output looks photographic.
- Full-image metrics penalize the intended edit. They must be read together with Action Success, Photo Success, and Strict End-to-End Success.
- The four case images are selected development examples. The clustered bootstrap intervals in the JSON resample cases and are descriptive only.

## Reproducibility

- Per-output CSV: `artifacts/day33_traditional_metrics_v1/per_output_metrics.csv`
- Machine-readable result: `results/day33_traditional_metrics_v1.json`
- Script: `scripts/compute_day33_traditional_metrics.py`
