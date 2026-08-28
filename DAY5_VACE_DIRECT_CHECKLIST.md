# Day 5 checklist: VACE-direct static proxy

Date opened: 2026-08-28

## Pre-output contract

- [x] Reuse the same hash-locked common proxy, prompts, and source images.
- [x] Reuse the audited local Wan2.2-VACE model without downloads.
- [x] Freeze seed `1`, 21 frames, 20 steps, and VACE scale `1.0`.
- [x] Feed the repeated static proxy and edit alpha directly to VACE.
- [x] Disable TTM and all projection schedules.
- [x] Preserve the source exactly outside `edit_alpha` after decoding.
- [x] Use resident two-case workers and retain failures.
- [x] Commit the implementation/config before generating v1 outputs.

## Execution and gate

- [ ] Generate all four anchors in two resident workers.
- [ ] Verify one pipeline load per worker and exact outside-mask preservation.
- [ ] Review action success separately from photo success.
- [ ] Compare against unified GeoEdit, staged/exclusive, and geometry lock.
- [ ] Decide whether native VACE is a viable appearance backend.

Status: `FROZEN_REMOTE_EXECUTION_PENDING`

Frozen config: `configs/vace_direct_static_proxy_v1.json`
