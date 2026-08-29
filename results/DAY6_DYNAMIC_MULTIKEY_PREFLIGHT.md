# Day 6 deterministic dynamic-multikey preflight

Preflight date: 2026-08-29 (Asia/Tokyo)

## Frozen intervention

This pilot changes one variable from VACE-direct static-proxy v1: the repeated
static RGB/mask inputs are replaced by a deterministic 21-frame action
trajectory. Every anchor uses the same five phases: exact source anchor,
utensil approach, contact hold, coupled lift, and a four-frame exact final
hold. Frame `18` is selected before inference. Seed `1`, 20 steps, VACE scale
`1.0`, source reference, no TTM, and final exact-protection projection remain
unchanged.

No ImageGen output, generated semantic keyframe, learned depth, learned flow,
or output-dependent frame choice is used. The geometry builder uses only the
frozen common proxy, five-layer masks, and anchor primitives.

## Deterministic validation

The controls were rebuilt on `gp40` from the original frozen remote common
proxy. All four source-anchor frames match their sources exactly; all four
final holds match the corresponding static proxies exactly; and the frozen
selected controls are unchanged outside the final `edit_alpha`.

| Anchor | Motion support | Source exact | Final hold exact | Selected outside final alpha |
| --- | ---: | ---: | ---: | ---: |
| Soup + spoon | 0.126429 | pass | pass | pass |
| Fried rice + spatula | 0.158675 | pass | pass | pass |
| Udon + chopsticks | 0.258641 | pass | pass | pass |
| Pasta + fork | 0.147668 | pass | pass | pass |

Remote summary SHA-256:
`95993c77a82188f6b22a65b38e5a6a480a328952b2f0626af2963909dea15c60`.

The non-blind control review confirms visible motion in all four anchors and
the intended rear-chopstick, noodle, front-chopstick order for udon. This is a
geometry-control review, not a claim of photographic realism.

## Decision rule

Proceed with exactly one four-anchor seed-1 pilot. Compare it directly with
static-proxy v1. Continue developing VACE as an appearance backend only if the
dynamic intervention improves action topology without weakening exact scene
protection; photo realism remains a separate gate.
