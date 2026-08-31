# Change log

## 2026-08-26 — scope v1

- froze the structured-control task and four material/action families;
- separated state, action, photo, and preservation success;
- froze the 60-case target and pilot/test policy;
- created the lightweight paper-release workspace.

## 2026-08-26 — benchmark provisional split v3

- reused the existing UECFOOD256 dataset without downloading or copying it;
- sampled 32 candidates per family with seed `20260826`;
- recorded SHA-256, dimensions, format, class, source path, and dHash;
- found no exact duplicates and no dHash-distance-4 near-duplicate pairs;
- verified the official non-commercial research-only usage restriction;
- removed omelet-rice cases whose granular payload was hidden;
- completed three visual review passes, recording every exclusion reason;
- provisionally selected 15 cases per family: 5 pilot and 10 test;
- pinned prior development cases `noodle_001` and `noodle_002` to pilot;
- retained rejected v0-v2 iterations under `benchmark/audit/day2/`.

## 2026-08-26 — benchmark and anchor freeze v1

- recorded human confirmation and froze 60 source cases at commit `3a512ad`;
- locked one existing OSEDiff x4-aligned editing input per source by SHA-256;
- froze four real pilot anchors with rigid/contact/material/hole/protect layers;
- closed the Day 2 gate at tag `paper-sprint-day2-final`.

## 2026-08-27 — Day 3 seed-policy consistency correction

- the initial baseline-registry commit accidentally introduced date-valued
  stochastic seeds despite `configs/paper_scope_v1.json` already freezing
  formal seeds `[1, 2, 3]` on Day 1;
- corrected the registry, checklist, report, and test back to `[1, 2, 3]`
  before any stochastic Day 3 inference was launched;
- the completed no-edit control is deterministic and uses seed `0`, so this
  correction was not selected from or influenced by a stochastic output.

## 2026-08-27 — Day 3 baseline execution close

- completed all four no-edit, four vanilla GeoEdit, and four unified-mask
  anchor runs without replacing a case or seed;
- verified all four existing Wan component hashes without downloading a model;
- built one deterministic common proxy for every action family;
- applied the same exact-protection 2D projection to both generated baselines;
- closed the execution gate at 12/12 technical completions and recorded the
  provisional visual diagnosis separately from technical success;
- identified repeated dual-denoiser loading as a scaling bottleneck and made a
  resident per-GPU worker the next engineering gate.

## 2026-08-30 — Day 7 VACE-LoRA infrastructure smoke

- expanded safe resource discovery from gp40 to gp38--gp42 and selected an idle
  gp39 A6000 without modifying another user's process;
- preserved three complete technical-failure stages and moved video decoding
  into preflight before expensive model loading;
- completed two high-noise VACE-LoRA optimizer steps over the frozen two-sample
  identity-target dataset without downloading a model;
- validated the final rank-8 checkpoint through the official offline VACE LoRA
  loader: 160 finite tensors, 80 complete A/B pairs, and 80 updated tensors;
- retained the strict claim limit: infrastructure success is not action,
  generalization, or photo-realism evidence.

## 2026-08-31 — Day 8 relative-3-D fork projection pilot

- made `3-D action -> camera/depth projection -> 2-D control -> learned render`
  the explicit FoodStateEdit method decomposition;
- added a deterministic normalized pinhole camera, 3-D curve primitives, and
  nearest-depth z-buffer splatting;
- produced a 21-frame pasta/fork control with one rigid fork, four 3-D helices,
  four plate-connected tails, and depth-tested front/back crossings;
- verified sub-pixel numerical reprojection, zero change outside motion
  support, stored 3-D arrays, and hash-locked review evidence;
- retained the strict evidence label `relative_3d`: this run does not replace
  the unavailable VGGT reconstruction and does not establish action success or
  photo realism.
- froze a one-anchor, same-seed VACE comparison against the Day 6 planar
  control; all gp38--42 candidates were occupied, so the fail-closed resource
  gate correctly prevented inference.
- launched only after gp39 became fully idle; the final preflight passed 36/36
  checks, the pipeline loaded once, all 21 frames decoded, and exact outside-
  support preservation passed;
- closed the render as a strict action/photo negative: relative 3-D recovered a
  recognizable fork and trajectory versus the Day 6 spoon-like blob, but VACE
  discarded the wrapped and lifted spaghetti payload;
- made non-identity action-target supervision the next renderer gate and
  rejected seed/test expansion for this single-anchor negative pilot.
