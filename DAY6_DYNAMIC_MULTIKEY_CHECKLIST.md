# Day 6 checklist: deterministic dynamic multikey

Date opened: 2026-08-29

## Pre-output contract

- [x] Reuse the same four frozen anchors, common proxies, prompts, and sources.
- [x] Generate all 21 RGB/mask frames deterministically without ImageGen.
- [x] Use source, approach, contact, lift, and final-hold phases for every anchor.
- [x] Keep frame 0 exactly equal to the source and frames 17--20 exactly equal to the static proxy.
- [x] Freeze frame 18 as the selected 2D result before stochastic inference.
- [x] Reuse the audited local Wan2.2-VACE model without downloads.
- [x] Freeze seed `1`, 21 frames, 20 steps, and VACE scale `1.0`.
- [x] Disable TTM and preserve the source exactly outside final `edit_alpha`.
- [x] Use resident workers, refuse existing output directories, and retain failures.
- [x] Commit implementation, config, and deterministic-control hashes before inference.

## Execution and gate

- [ ] Run the four-anchor pilot only when two GPUs have safe free capacity.
- [ ] Verify one pipeline load per worker, artifact hashes, and exact protection.
- [ ] Review action success and photo success separately.
- [ ] Compare directly with static-proxy VACE-direct v1.
- [ ] Decide whether dynamic action phases justify appearance-backend development.

Status: `PRE_OUTPUT_FROZEN`

Frozen config: `configs/vace_direct_dynamic_multikey_v1.json`
