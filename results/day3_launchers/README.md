# Exact Day 3 launcher snapshots

These are byte-for-byte copies fetched from the non-overwritten remote launcher
directories after all runs completed. They preserve the exact launch code used
for the reported smoke evidence; `scripts/run_geoedit_anchor.py` may continue to
gain diagnostics for later sprint days.

| File | Used for | SHA-256 |
| --- | --- | --- |
| `run_geoedit_anchor_v1.py` | four `vanilla_geoedit` runs | `9373002caf9924e35a8f9c1eda2e343972f5ded74dc0e152ad6a538b98933ea6` |
| `run_geoedit_anchor_v2.py` | four `geoedit_unified_action_mask` runs | `e839949a8bb2d1d6df30d352f210ed03c2192c27274a837a7ed5d0e8596b623c` |
| `summarize_geoedit_batch_v1.py` | both batch summaries | `b28d13508a8b8ad4ac8b7bf2d710690b98c62feea67eff226f63bdfffaa4d4dd` |

Both launchers set `DIFFSYNTH_SKIP_DOWNLOAD=true`, validate the one-time model
hash audit and current model sizes, verify the four frozen GeoEdit override
hashes, refuse to overwrite run directories, retain failure logs, and apply
the same exact-protection 2D projection.
