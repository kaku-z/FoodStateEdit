# Day 25 same-condition ablation result

Date: 2026-09-13

## Outcome

The four selected development cases are technically complete.  The full matrix
contains four cases, three fixed seeds and five conditions: 60 final-image
cells.  Forty-seven cells were newly generated and thirteen reused hash-locked
seed-1 evidence.  All 60 final-image hashes match their source records and are
unique.

The newly generated runs contain 282 verified files.  Every expected condition
completed with 21 frames, 20 inference steps, TTM and LoRA disabled, one
pipeline load per case worker, and zero maximum pixel difference outside the
declared support.  The byte-level record is
`day25_ablation_collection_verified_v1.json`; the complete matrix record is
`day25_ablation_matrix_verified_v1.json`.

## Internal development review

This review is unblinded and is not substituted for independent scoring.

| Case | Native vs explicit control | Planar vs relative3D | Material scale observation |
| --- | --- | --- | --- |
| Ramen | Native lacks the requested interaction; controls introduce chopsticks and a lifted strand | Pinch/contact is inconsistent and no stable relative3D advantage is visible | 0.6 is unstable; 0.8 avoids some deformation but does not clearly beat planar |
| Soup | Native lacks a spoon; controls consistently produce spoon-and-liquid interaction | Scale-1.0 planar and relative3D are visually similar | 0.6 deforms the spoon in multiple seeds; 1.0 is the conservative choice |
| Fried rice | Controls create a scoop and payload while native does not | Scale-1.0 variants remain nearly indistinguishable and schematic | 0.6 sometimes looks more metallic but increases local distortion |
| Synthetic cake | Controls lift the pre-cut block while native does not | Fixed variants often retain a rod-like utensil | 0.6 can make a fork but also damages or reshapes the block; no strict consistent gain |

The defensible positive observation is that an explicit phase-action motion
proxy is necessary on these selected cases.  The intended algorithmic claims
do not pass: clear relative3D gain over planar is `0/4` cases, and clear
material-adaptive gain over fixed relative3D is `0/4` cases.  A reviewer-driven
rollback remains a development procedure, not a validated automatic observer.

## Decision

Do not claim relative3D superiority and do not unlock held-out effectiveness
testing from Day 25.  Continue only with the frozen 20-image Day 32 pilot to
test whether the negative result persists beyond one anchor per material.  If
that broader pilot remains negative, preserve it and reshape the paper around
explicit process-control engineering and failure analysis rather than claiming
an unobserved 3-D benefit.

## Evidence locations

- complete collected roots: `artifacts/day25_same_condition_ablation_collected_20260913_v1`;
- fixed final/process grids: `artifacts/day25_same_condition_ablation_review_20260913_v1`;
- collection verification: `results/day25_ablation_collection_verified_v1.json`;
- 60-cell final-image verification: `results/day25_ablation_matrix_verified_v1.json`;
- internal decision record: `results/day25_same_condition_ablation_result_v1.json`.

## Claim boundary

This is a selected four-case development ablation.  It supports explicit-control
necessity, not relative3D superiority, held-out generalization, physical
correctness, photo realism, or fully automatic rollback.  Cake is synthetic
supplementary evidence and is not counted as real-data performance.
