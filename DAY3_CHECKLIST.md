# Day 3 checklist: baseline executability

Date opened: 2026-08-26

## Asset and engine audit

- [x] Verify all eight remote A6000 GPUs are visible.
- [x] Verify existing Wan2.2-VACE-Fun-A14B component sizes and tokenizer files.
- [x] Recheck the four remote GeoEdit override hashes.
- [x] Import the GeoEdit CLI with downloads disabled.
- [x] Run all six GeoEdit mask/schedule tests through `unittest`.
- [ ] Record complete SHA-256 hashes for all four Wan model components.

## Baseline freeze

- [x] Freeze formal stochastic seeds: `20260826`, `20260827`, `20260828`.
- [x] Freeze `input_no_edit` as a deterministic primary control.
- [x] Freeze `vanilla_geoedit` as a primary baseline.
- [x] Freeze `geoedit_unified_action_mask` as a same-proxy primary baseline.
- [x] Keep `vace_direct_static_proxy` conditional on an offline launcher.
- [x] Record reproducible exclusion reasons for FreeFine, ObjectMorpher,
  PhysicEdit, and a cloud strong editor.

## Four-anchor batch evidence

- [ ] Batch `input_no_edit` on all four anchors with run manifests.
- [ ] Build `common_proxy_v1` through one family-dispatched command.
- [ ] Batch `vanilla_geoedit` on all four anchors, one frozen smoke seed.
- [ ] Batch `geoedit_unified_action_mask` on all four anchors, same seed/proxy.
- [ ] Record technical failures without replacing cases or seeds.

Day 3 Gate status: `OPEN_AWAITING_COMMON_PROXY_V1`

Baseline registry: `configs/baselines_v1.json`

Audit report: `results/DAY3_BASELINE_AUDIT.md`
