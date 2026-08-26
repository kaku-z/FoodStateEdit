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
