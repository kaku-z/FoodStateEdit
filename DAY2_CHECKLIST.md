# Day 2 checklist: benchmark freeze

Date opened: 2026-08-26

The benchmark target is 60 traceable inputs: 15 cases in each of four
families, split into 5 pilot and 10 held-out test cases per family.

- Current manifest: `benchmark/data_manifest_provisional_v3.csv`
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
- [ ] Obtain final human confirmation from the four frozen review boards.

## Freeze and split

- [x] Provisionally select exactly 15 eligible cases per family.
- [x] Provisionally assign exactly 5 pilot and 10 test cases per family.
- [ ] Change all 60 rows from `provisional_frozen` to `frozen` after human review.
- [x] Record deterministic SHA-256 split rule and seed `20260826`.
- [x] Check that no exact/near content duplicate crosses pilot/test.

## Four primary anchors

- [ ] `soup_spoon_001`: complete five-layer structured action annotation.
- [ ] `fried_rice_spatula_001`: complete five-layer structured action annotation.
- [ ] `ramen_chopsticks_001`: complete five-layer structured action annotation.
- [ ] `pasta_fork_001`: choose source and complete five-layer annotation.

Day 2 Gate status: `PROVISIONAL_SPLIT_READY`

Provisional manifest commit: `a8c95a52f5e7975950e4fc3da19c5c1b097d26c7`

Final frozen manifest commit: `PENDING_HUMAN_CONFIRMATION`
