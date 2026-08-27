# Day 3 four-anchor smoke review

Review date: 2026-08-27 (Asia/Tokyo)

## Outcome

All eight stochastic runs completed at frozen seed `1`: four vanilla GeoEdit
and four same-proxy unified-action-mask runs. There were no technical failures,
no seed replacements, and all exact-protection projections had maximum pixel
difference 0 outside `edit_alpha`.

Technical completion is not counted as editing success. A single internal,
non-blind diagnostic review gives vanilla GeoEdit 1/4 provisional action
successes and the unified-mask baseline 0/4. This review is for pilot debugging;
it is not the later blinded human evaluation.

| Anchor | Vanilla GeoEdit | Unified action mask | Main diagnostic |
| --- | --- | --- | --- |
| Soup + spoon | action pass; photo fail | action fail; photo fail | unified window creates a ghost bowl and surface smear |
| Fried rice + spatula | action fail; photo fail | action fail; photo fail | source reduction is absent or turns into a broad blur |
| Udon + chopsticks | action fail; photo fail | action fail; photo fail | transparent sticks and broken/missing grip topology |
| Pasta + fork | action fail; photo fail | action fail; photo fail | fork collapses into a spoon-like bowl with no four tines |

The important result is a reproducible failure mode, not a paper-ready visual:
one strong schedule for rigid geometry, deformable payload, contact, and source
repair is insufficient. This justifies testing the proposed layer-specific
projection schedule next.

## Runtime finding

The single-case CLI repeatedly reads and swaps the two 34.7 GB Wan denoisers.
A cold high-noise load and the step-9/10 high-to-low-noise transition dominated
the final pair's runtime. Formal batches therefore require a per-GPU resident
pipeline worker before scaling beyond the pilot anchors.

Machine-readable evidence:

- `results/day3_vanilla_geoedit_smoke_v1/`
- `results/day3_unified_geoedit_smoke_v1/`
- `results/day3_no_edit_v1/`

Generated images and videos remain only in ignored local artifacts and hashed
remote output directories because the source dataset is restricted.
