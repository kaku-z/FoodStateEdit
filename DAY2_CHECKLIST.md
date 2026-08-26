# Day 2 checklist: benchmark freeze

Date opened: 2026-08-26

The benchmark target is 60 traceable inputs: 15 cases in each of four
families, split into 5 pilot and 10 held-out test cases per family.

- Frozen source manifest: `benchmark/data_manifest_v1.csv`
- Canonical editing inputs: `benchmark/canonical_input_manifest_v1.csv`
- Final-review boards: `artifacts/day2_frozen_review_v3/`

## Automatic inventory

- [x] Inventory 32 candidates per family (128 total).
- [x] Record immutable source path and SHA-256 for every candidate.
- [x] Record image width, height, format, and source kind.
- [x] Verify official UECFOOD256 non-commercial research-only terms.
- [x] Reject corrupt inputs and check exact/near duplicates (none in v1).

## Manual eligibility review

- [x] Complete Codex visual first pass for target-utensil absence.
- [x] Complete Codex visual first pass for target-action absence.
- [x] Complete Codex visual first pass for food/container visibility.
- [x] Complete Codex visual first pass for local edit space.
- [x] Record exclusion reasons instead of silently replacing cases.
- [x] Obtain final human confirmation from the four frozen review boards.

## Freeze and split

- [x] Provisionally select exactly 15 eligible cases per family.
- [x] Provisionally assign exactly 5 pilot and 10 test cases per family.
- [x] Change all 60 rows from `provisional_frozen` to `frozen` after human review.
- [x] Record deterministic SHA-256 split rule and seed `20260826`.
- [x] Check that no exact/near content duplicate crosses pilot/test.

## Canonical editing-input audit

- [x] Reuse the existing OSEDiff x4 tree without downloading or rerunning a model.
- [x] Match all 60 derived images to the frozen source paths and SHA-256 hashes.
- [x] Validate the x4/multiple-of-8 dimension rule for all 60 inputs.
- [x] Verify all canonical SHA-256 hashes are unique.
- [x] Require source/canonical dHash distance at most 8 (observed maximum: 5).

## Four primary anchors

- [x] `soup_spoon_001` -> `soup_002`: five-layer action annotation complete.
- [x] `fried_rice_spatula_001` -> `rice_007`: five-layer action annotation complete.
- [x] `ramen_chopsticks_001` -> `noodle_001`: five-layer annotation complete;
  target/source centerline length ratio `1.005146`.
- [x] `pasta_fork_001` -> `pasta_006`: source chosen and five-layer annotation complete.

Day 2 Gate status: `COMPLETE`

Provisional manifest commit: `a8c95a52f5e7975950e4fc3da19c5c1b097d26c7`

Final frozen manifest commit: `3a512ad1a996e7c0ae636854f3079cfe643282b2`
