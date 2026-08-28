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

- [x] Generate all four outputs in a new remote root without a learned model.
- [x] Verify topology-mask hashes and exact outside-mask preservation.
- [x] Review action success and photo success independently.
- [x] Compare against common proxy and geometry-lock v1.
- [x] Decide whether material rendering should condition the next VACE run.

Status: `CLOSED_RIGID_APPEARANCE_IMPROVED_PAYLOAD_APPEARANCE_NEGATIVE`

Frozen config: `configs/material_render_v1.json`
