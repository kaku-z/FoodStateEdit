# Day 6 motion-union projection diagnostic

Review date: 2026-08-29 (Asia/Tokyo)

## Status and question

This is explicitly a post-hoc exploratory diagnostic, not a pre-registered
method result. It was motivated by observing that the frozen raw ramen frame
contained two chopsticks that the narrower final `edit_alpha` removed.

No model was rerun. All four seed-1 raw frames, frozen frame index `18`, and the
pre-inference deterministic `motion_union_alpha` were reused without case or
frame replacement. The diagnostic asks only whether action geometry was
generated inside the motion support but outside the final support.

## Invariants and support cost

All four outputs are exact copies of their source wherever
`motion_union_alpha == 0`.

| Anchor | Final support px | Motion support px | Extra support px | Changed px recovered outside final support |
| --- | ---: | ---: | ---: | ---: |
| Soup + spoon | 20,546 | 52,109 | 31,563 | 27,719 |
| Fried rice + spatula | 35,428 | 63,531 | 28,103 | 24,840 |
| Udon + chopsticks | 53,851 | 112,693 | 58,842 | 53,720 |
| Pasta + fork | 23,507 | 59,124 | 35,617 | 32,995 |

Summary SHA-256:
`225dffe81f6cbb0165d337fd1eaaa0de40deff31d65758cd63f6db0ebe6cfedd`.

## Visual result

The support hypothesis is confirmed for ramen only. The motion-union result
retains exactly two separated wooden chopsticks contacting a lifted noodle that
returns to the bowl. The noodle is still soft/ghosted, and the enlarged support
was chosen after observing the failure, so this is a diagnostic topology
near-pass rather than a formal action success.

Soup, fried rice, and pasta remain failures under the larger support. Their
problems are not projection clipping: liquid payload, distinct supported rice
grains, and four fork tines/twirl are absent in the raw model output itself.

## Decision

Do not promote the full motion union to the formal edit region. It preserves
28k--59k additional pixels per case and includes the entire approach path,
which is unnecessarily broad for a single final image.

The next prospective test should freeze a narrower final-geometry tolerance
corridor before inference, retain exact protection outside it, and validate on
a new seed or held-out case. This can address ramen's projection mismatch. A
separate appearance/topology mechanism remains necessary for soup payloads,
granular support, and fork tines.
