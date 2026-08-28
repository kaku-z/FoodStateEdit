# Day 5 geometry-lock review

Review date: 2026-08-28 (Asia/Tokyo)

## Outcome

The frozen deterministic geometry-lock pilot completed all four anchors without
rerunning or downloading a model. It used the hash-locked common proxy as the
semantic foreground, the failed exclusive seed-1 output only as a weak contact
halo donor, and one global parameter set for every family.

The representation change recovers readable action topology in `4/4` anchors,
up from `0/4` for both stochastic staged candidates. It does **not** recover
photorealism: the single internal, non-blind diagnostic review remains `0/4`
for photo success. This is a geometry-positive, appearance-negative result.

| Anchor | Action topology | Photo | Exact proxy core | Exact source protection | Main diagnostic |
| --- | --- | --- | --- | --- | --- |
| Soup + spoon | pass | fail | pass | pass | Liquid-bearing spoon is readable; metal remains flat and outlined |
| Fried rice + spatula | pass | fail | pass | pass | Spatula/payload/hole relation survives; blade and repair boundaries are synthetic |
| Udon + chopsticks | pass | fail | pass | pass | Two sticks and a continuous lifted strand survive; strand/sticks remain rendered |
| Pasta + fork | pass | fail | pass | pass | Four tines and twirl survive; shading and occlusion remain artificial |

## Hard invariants

All four semantic cores match `motion_signal.png` with maximum channel
difference `0`. All pixels outside `edit_alpha` match `first_frame.png` with
maximum channel difference `0`. Generated donor influence is restricted to a
non-core contact halo of 1,172--2,556 pixels per anchor. The deterministic
output hashes are recorded in `geometry_lock_summary.json` and each run points
to pre-output commit `29531326cf74aa1f9533a42cf0cb154baf4e011a`.

## Comparison and decision

Proxy-only and geometry-lock v1 both retain the target relation; stochastic
GeoEdit destroys utensil identity or contact in every anchor. Geometry-lock v1
therefore isolates the central bottleneck: topology must be represented and
preserved explicitly, while appearance must be optimized under that constraint.

Keep geometry locking as the method direction and keep this run as an ablation.
Do not expand v1 to more seeds or the 60-case set. The next pilot should rebuild
utensil material, local illumination, cast/contact shadows, and anti-aliased
boundaries while preserving the locked silhouettes and source background.
