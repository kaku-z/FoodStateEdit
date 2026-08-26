# Day 2 checklist: benchmark freeze

Date opened: 2026-08-26

The benchmark target is 60 traceable inputs: 15 cases in each of four
families, split into 5 pilot and 10 held-out test cases per family.

## Automatic inventory

- [ ] Inventory 20 or more candidates per family.
- [ ] Record immutable source path and SHA-256 for every candidate.
- [ ] Record image width, height, format, and source kind.
- [ ] Record source URL/owner and usage license when applicable.
- [ ] Reject corrupt, duplicate, or too-small inputs.

## Manual eligibility review

- [ ] Confirm that the target manipulation utensil is absent.
- [ ] Confirm that the source does not already depict the target action.
- [ ] Confirm that the food and main container are sufficiently visible.
- [ ] Confirm that the intended local edit has usable empty space.
- [ ] Record exclusion reason instead of silently replacing a case.

## Freeze and split

- [ ] Select exactly 15 eligible cases per family.
- [ ] Assign exactly 5 pilot and 10 test cases per family.
- [ ] Mark all 60 selected rows `freeze_status=frozen`.
- [ ] Record the deterministic split rule and random seed.
- [ ] Check that no content duplicate crosses pilot/test.

## Four primary anchors

- [ ] `soup_spoon_001`: complete five-layer structured action annotation.
- [ ] `fried_rice_spatula_001`: complete five-layer structured action annotation.
- [ ] `ramen_chopsticks_001`: complete five-layer structured action annotation.
- [ ] `pasta_fork_001`: choose source and complete five-layer annotation.

Day 2 Gate status: `NOT_STARTED`

Frozen manifest commit: `PENDING`
