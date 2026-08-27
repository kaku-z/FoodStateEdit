# Day 4 checklist: four-layer staged integration

Date opened: 2026-08-27

## Contract

- [x] Preserve byte-exact Day 3 GeoEdit override snapshots.
- [x] Add explicit rigid/contact/material/hole masks and endpoints.
- [x] Define half-open active windows and union composition.
- [x] Keep legacy `mask_old/non_hole` behavior available for prior baselines.
- [x] Freeze one family-independent candidate schedule before inference.
- [x] Pass the ten remote mask/argument/schedule unit tests.

## Runtime

- [x] Verify the per-GPU worker loads one pipeline for multiple cases.
- [x] Run all four anchors at frozen seed `1` without case/seed replacement.
- [x] Apply the same exact-protection 2D projection.
- [x] Retain the quota-aborted preflight and completed-run timing evidence.

## Review gate

- [x] Review utensil identity, contact topology, source repair, and photo quality.
- [x] Compare against Day 3 vanilla/unified results without changing v1 outputs.
- [x] Decide whether the candidate warrants a formal seed expansion.

Day 4 Gate status: `CLOSED_NEGATIVE_PILOT_DO_NOT_EXPAND`

Outcome: `4/4` technical completion, `0/4` provisional action success,
`0/4` provisional photo success, and `4/4` exact outside-mask preservation.
Both two-case workers loaded the pipeline exactly once. The candidate does not
warrant a seed or test-set expansion because it did not beat the Day 3 vanilla
control (`1/4` provisional action success).

The post-run overlap audit found that shorter semantic windows were partly or
fully shadowed by later-ending masks. In the soup case, all contact and material
pixels also belong to the rigid mask, so their nominal releases have no local
effect. The next candidate must use explicitly non-shadowing projection masks
and must be frozen before new stochastic outputs.

Frozen schedule: `configs/staged_schedule_v1.json`

Evidence: `results/day4_staged_v1/` and `results/DAY4_STAGED_REVIEW.md`
