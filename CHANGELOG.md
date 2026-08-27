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
