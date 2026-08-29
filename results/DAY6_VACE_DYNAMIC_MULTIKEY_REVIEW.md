# Day 6 VACE-direct dynamic-multikey review

Review date: 2026-08-29 (Asia/Tokyo)

## Outcome

The frozen dynamic-multikey pilot completed all four anchors at seed `1`.
Relative to static-proxy v1 it changed only the VACE control from one repeated
image/mask to a deterministic source, approach, contact, lift, and final-hold
trajectory. Frame `18` was selected before inference and projected through the
original final `edit_alpha`.

Technical completion is `4/4`, but the reported projected edits remain `0/4`
for action success and `0/4` for strict photo success. Exact source protection
is `4/4`.

| Anchor | Action | Photo | Exact protection | Main diagnostic |
| --- | --- | --- | --- | --- |
| Soup + spoon | fail | fail | pass | Solid spoon identity, but no visibly lifted/contained soup payload |
| Fried rice + spatula | fail | fail | pass | Persistent handle; blade and grains melt into a glossy patch |
| Udon + chopsticks | fail | fail | pass | Raw frame recovers two sticks plus lifted noodle; final mask projection removes displaced sticks |
| Pasta + fork | fail | fail | pass | Solid metal handle, but malformed spoon-like head and no four-tine twirl |

## What improved over the static control

The aggregate success count is unchanged, but the failure mode is materially
different. Static-proxy v1 usually erased or ghosted the utensil immediately.
Dynamic v1 keeps a solid utensil through much of all four sequences. Most
importantly, the unprojected ramen frame at frozen index `18` contains exactly
two separated chopsticks and one visibly lifted noodle connected to the bowl.
This is the first controlled indication in the current four-anchor workspace
that deterministic action phases can recover the intended raw topology without
ImageGen keyframes.

It is not a reported action success: the generated sticks lie partly outside
the original final semantic support. Exact projection correctly restores those
pixels to the source and thereby removes most of both sticks. This isolates a
new mismatch between the deterministic motion support and the narrower final
projection support.

## Runtime and reproducibility

Both two-case workers loaded the audited pipeline once. First cases took
380.380 and 386.479 seconds; resident second cases took 264.524 and 251.695
seconds. All twelve returned artifacts matched their manifest hashes after
transfer. Every pixel outside final `edit_alpha` has maximum channel difference
`0`.

An initial launcher attempt failed before output-directory creation because
the remote bundle omitted the already committed `run_geoedit_anchor.py`
dependency. Both identical logs were retained with SHA-256
`7cce29ad7196847c8a27d3241e8fedf4bef6f8a85272a1f2dfe8e9c274ca3311`.
No model was loaded and no stochastic output was produced. Adding that existing
dependency and restarting with the same frozen parameters was a technical
rerun, not seed or result replacement.

## Decision

Close projected dynamic-multikey v1 and do not expand it. Do not discard the
dynamic direction: unlike static conditioning, it provides a positive raw
topology signal. The next experiment must isolate support handling using the
already frozen deterministic motion-union support, while keeping the fixed
frame, seed, controls, and exact outside-support projection unchanged. This
follow-up is diagnostic and must be labeled post-hoc unless confirmed by a new
pre-registered seed or held-out case.
