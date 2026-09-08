# Day 18 high-lift swept-support repair

## Scope and frozen comparison

- [x] Retain the Day 17 high-lift geometry and original failed projected output.
- [x] Audit actual alpha coverage, not only its bounding box.
- [x] Reproject the identical Day 17 raw video with old and expanded alpha.
- [x] Freeze the union of the full rendered trajectory, an 8 px opaque margin,
  and an 8 px feather ramp, combined with the old alpha by pixelwise maximum.
- [x] Keep prompts, seed 1, geometry, model/runtime bytes, 21 frames, 20 steps,
  VACE scale 1, LoRA off and TTM off unchanged from Day 17.
- [x] Disclose that fresh inference changes control raster support, VACE mask
  and final projection alpha together; it is not a three-stage causal isolation.
- [x] Do not reuse the old low-lift target as a high-lift target or train on it.

## Safety and execution

- [x] Audit gp38--gp42; exclude occupied A6000s, A40 and Blackwell.
- [x] Verify new gp40 dataset/runtime/output/preflight paths were absent.
- [x] Upload and verify all 12 dataset files and the frozen config/runner.
- [x] Pass the fresh runner resource gate on gp40 physical GPU 5.
- [x] Start exactly one new VACE inference; no process preemption or downloads.
- [x] Verify technical completion, one pipeline load, 21 decoded frames.
- [x] Pull into new local directories and check every artifact SHA-256.

## Review and claims

- [x] Correct the Day 17 attribution: old projection hid sticks present in raw.
- [x] Review native output, projected output, contact crops and final hold.
- [x] Judge two-stick persistence, pinch, strand continuity and bowl connection:
  raised utensil/handles improved, but the strict action gate did not pass.
- [x] Record photographic appearance separately from geometric action:
  overly straight/soft strand and weak contact shading remain.
- [x] Update result MD/JSON, inventory and README; full suite passed 171 tests
  with zero failures, errors or skips under bundled Python plus the isolated
  `jsonschema` validation dependency cache.
- [x] Keep independent review, multiple seeds, unseen images and real scenes
  unvalidated; do not claim a learned/novel algorithmic contribution from repair.
- [x] No blind fork, new training, balanced expansion or automatic heartbeat.

Frozen execution config:
`configs/day18_high_lift_swept_support_gp40_v1.json`
(SHA-256 `b497aaeac4d595c2938a632f6ac8dfe9bcc5c7a211de4dd84825b57f154136db`).

This host-specific variant differs from
`configs/day18_high_lift_swept_support_v1.json` only in the new output path.
The one-shot builder refuses to overwrite its package or named config.
