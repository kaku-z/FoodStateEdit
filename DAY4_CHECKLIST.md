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

- [ ] Verify the per-GPU worker loads one pipeline for multiple cases.
- [ ] Run all four anchors at frozen seed `1` without case/seed replacement.
- [ ] Apply the same exact-protection 2D projection.
- [ ] Retain technical failures and timing evidence.

## Review gate

- [ ] Review utensil identity, contact topology, source repair, and photo quality.
- [ ] Compare against Day 3 vanilla/unified results without changing v1 outputs.
- [ ] Decide whether the candidate warrants a formal seed expansion.

Day 4 Gate status: `OPEN_RESIDENT_WORKER_PENDING`

Frozen schedule: `configs/staged_schedule_v1.json`
