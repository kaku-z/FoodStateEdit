# Day 4 exclusive-mask follow-up review

Review date: 2026-08-27 (Asia/Tokyo)

## Outcome

The pre-registered exclusive-mask follow-up completed all four anchors at seed
`1`. It held the common proxy, model, prompt, seed, 21 frames, 20 steps,
`tweak=3`, endpoints `18/15/8/8`, no-warm setting, and exact 2D protection
projection fixed. The only intended change was pairwise-disjoint projection-mask
ownership (`hole > contact > material > rigid`).

Technical completion is not editing success. The single internal, non-blind
diagnostic review gives `0/4` provisional action successes and `0/4`
provisional photo successes. Exact outside-mask preservation remains `4/4`.

| Anchor | Action | Photo | Exact protection | Main diagnostic |
| --- | --- | --- | --- | --- |
| Soup + spoon | fail | fail | pass | duplicate spoon-bowl fragment remains |
| Fried rice + spatula | fail | fail | pass | blade/handle topology is broken and the source repair is blurred |
| Udon + chopsticks | fail | fail | pass | transparent sticks and no continuous lifted strand |
| Pasta + fork | fail | fail | pass | no readable four tines or twirl contact |

## Did exclusive ownership affect the generated pixels?

Yes, but not enough to alter the semantic verdict. All four v2 image hashes
differ from staged v1. Inside `edit_alpha`, 67.0%--85.5% of pixels change by at
least one channel value, but the average RGB changes are small and the visible
failure topology remains the same:

| Anchor | Edit-support RGB MAE vs staged v1 | Changed edit pixels | Full-image PSNR |
| --- | ---: | ---: | ---: |
| Soup + spoon | 2.072 | 74.4% | 49.18 dB |
| Fried rice + spatula | 5.300 | 85.5% | 37.04 dB |
| Udon + chopsticks | 1.147 | 67.0% | 48.19 dB |
| Pasta + fork | 4.329 | 83.0% | 41.68 dB |

Both methods are byte-exact outside the editable support, so the comparison is
not diluted by background changes. The experiment supports two separate
conclusions: staged v1 did contain a real mask-shadowing bug, and removing that
bug alone is insufficient to recover utensil identity/contact topology from the
current procedural proxy plus Wan/GeoEdit appearance backend.

## Runtime

Both two-case workers loaded the pipeline once. Shared-filesystem cold loading
made the first cases unusually slow (1,410.222 and 1,423.624 seconds); resident
second cases took 262.723 and 261.222 seconds. There were no downloads, seed
replacements, inference failures, or output overwrites.

## Gate decision

Close this candidate and do not expand to seeds 2/3 or the 60-case set. The next
research step should change the representation/backend interface rather than
continue endpoint tuning: preserve an explicit utensil/contact render as a
stronger spatial condition, or composite a geometry-faithful foreground before
a bounded harmonization pass. Any next candidate must again be frozen and
tested on the same four anchors before expansion.
