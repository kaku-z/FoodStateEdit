# Day 5 material-render review

Review date: 2026-08-28 (Asia/Tokyo)

## Outcome

The frozen deterministic material renderer completed all four anchors without a
learned model. It reuses the exact common-proxy topology masks and applies one
material-class renderer: surface-normal metal/wood shading, source-reflection
mixing, food detail enhancement, contact/cast shadows, and feathered boundaries.

All four actions remain readable and all four rigid appearances improve over the
flat outlined proxy. Photo success nevertheless remains `0/4`. The dominant
artifact is no longer only utensil shading; it is the warped food payload and
incorrect fine occlusion at the utensil-food interface.

| Anchor | Action topology | Photo | Rigid appearance vs proxy | Main remaining defect |
| --- | --- | --- | --- | --- |
| Soup + spoon | pass | fail | improved | Bowl depth and liquid/metal occlusion |
| Fried rice + spatula | pass | fail | improved | Stretched payload and source repair |
| Udon + chopsticks | pass | fail | improved | Segmented lifted noodle |
| Pasta + fork | pass | fail | improved | Twirl texture, tine occlusion, dark contact artifacts |

## Invariants

The renderer uses the frozen rigid/material mask hashes for every anchor. The
rear/material/front chopstick partition covers the exact rigid mask, and every
pixel outside `edit_alpha` matches the source with maximum channel difference
`0`. All output hashes point to pre-output commit
`019b3f8eaaaa1f749e38098f74e85e682dcf694a`.

## Decision

Close this as a useful appearance ablation and do not continue adding manual
shading rules. The result narrows the problem to local deformable-food texture
and contact occlusion. The next learned experiment remains the independently
frozen `vace_direct_static_proxy` baseline; only after that control is reviewed
should the materialized proxy be tested as a separate condition.
