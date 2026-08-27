# Day 3 checklist: baseline executability

Date opened: 2026-08-26; execution continued: 2026-08-27

## Asset and engine audit

- [x] Verify all eight remote A6000 GPUs are visible.
- [x] Verify existing Wan2.2-VACE-Fun-A14B component sizes and tokenizer files.
- [x] Recheck the four remote GeoEdit override hashes.
- [x] Import the GeoEdit CLI with downloads disabled.
- [x] Run all six GeoEdit mask/schedule tests through `unittest`.
- [x] Record complete SHA-256 hashes for all four Wan model components.

## Baseline freeze

- [x] Reuse the Day 1 frozen formal stochastic seeds: `1`, `2`, `3`.
- [x] Freeze `input_no_edit` as a deterministic primary control.
- [x] Freeze `vanilla_geoedit` as a primary baseline.
- [x] Freeze `geoedit_unified_action_mask` as a same-proxy primary baseline.
- [x] Keep `vace_direct_static_proxy` conditional on an offline launcher.
- [x] Record reproducible exclusion reasons for FreeFine, ObjectMorpher,
  PhysicEdit, and a cloud strong editor.

## Four-anchor batch evidence

- [x] Batch `input_no_edit` on all four anchors with valid run manifests (4/4,
  bit-exact; `results/day3_no_edit_v1/`).
- [x] Build `common_proxy_v1` through one family-dispatched command (4/4;
  exact protection outside the edit alpha; internal geometry review passed).
- [x] Batch `vanilla_geoedit` on all four anchors, frozen seed `1` (4/4
  complete; 0 technical failures; exact protection 4/4).
- [x] Batch `geoedit_unified_action_mask` on all four anchors, the same seed and
  proxy (4/4 complete; 0 technical failures; exact protection 4/4).
- [x] Record technical failures without replacing cases or seeds (none in the
  eight stochastic smoke runs; no reruns performed).

Day 3 Gate status: `CLOSED_COMPLETE`

Baseline registry: `configs/baselines_v1.json`

Audit report: `results/DAY3_BASELINE_AUDIT.md`
