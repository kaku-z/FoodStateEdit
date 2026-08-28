# Day 5 checklist: topology-preserving material render

Date opened: 2026-08-28

## Pre-output contract

- [x] Reuse the hash-locked common proxy, exact semantic masks, and anchor specs.
- [x] Use one deterministic 2.5D renderer with material-class rules only.
- [x] Preserve exact rigid/material silhouettes and chopstick z-order partition.
- [x] Add surface-normal shading, texture enhancement, shadows, and feathering.
- [x] Preserve the source exactly outside `edit_alpha`.
- [x] Refuse output-root reuse and fail on any hash/invariant mismatch.
- [x] Commit the implementation/config before generating v1 outputs.

## Execution and gate

- [ ] Generate all four outputs in a new remote root without a learned model.
- [ ] Verify topology-mask hashes and exact outside-mask preservation.
- [ ] Review action success and photo success independently.
- [ ] Compare against common proxy and geometry-lock v1.
- [ ] Decide whether material rendering should condition the next VACE run.

Status: `FROZEN_REMOTE_EXECUTION_PENDING`

Frozen config: `configs/material_render_v1.json`
