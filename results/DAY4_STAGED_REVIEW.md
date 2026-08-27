# Day 4 four-layer staged pilot review

Review date: 2026-08-27 (Asia/Tokyo)

## Outcome

All four stochastic runs completed at the pre-registered seed `1` and schedule
`rigid=18, contact=15, material=8, hole=8`. There were no model/inference
failures, no seed replacements, and the final exact-protection projection had
maximum pixel difference `0` outside `edit_alpha` in every case.

Technical completion is not editing success. The single internal, non-blind
diagnostic review gives the staged candidate `0/4` provisional action successes
and `0/4` provisional photo successes. It therefore does not beat the Day 3
vanilla GeoEdit control (`1/4` action, `0/4` photo) or justify a formal seed/test
expansion.

| Anchor | Action | Photo | Exact protection | Main diagnostic |
| --- | --- | --- | --- | --- |
| Soup + spoon | fail | fail | pass | duplicate spoon-bowl fragment remains on the soup surface |
| Fried rice + spatula | fail | fail | pass | handle/blade topology breaks and the source region becomes a broad blur |
| Udon + chopsticks | fail | fail | pass | sticks remain transparent and no stable continuous lifted strand is visible |
| Pasta + fork | fail | fail | pass | no readable four-tine fork or twirl; contact region is blurred |

Machine-readable visual labels are in
`day4_staged_v1/internal_pilot_review_v1.json`. Generated images remain in the
ignored `artifacts/day4_staged_v1/` directory because the source dataset is
restricted.

## Technical result

Two workers ran concurrently, with two cases per A6000. Each worker loaded one
pipeline for both cases (`pipeline_load_count=1`), so the resident-worker
contract passed. Per-case wall times were 257.043--275.621 seconds. The combined
summary verifies all four copied final-image hashes against their remote run
manifests.

An initial launch under the quota-limited result filesystem was stopped before
an output was produced when the second log could not be created. Its partial
manifest was retained, and the same fixed seed/protocol was rerun in two unused
`/tmp` roots. This infrastructure incident is documented under
`day4_staged_v1/preflight_quota_abort/`; it is not counted as a stochastic
candidate or as an inference failure.

## Why the four nominal windows did not behave as four independent windows

The post-run audit found extensive overlap between semantic masks. Union
composition means a shorter-window pixel is still projected whenever it also
belongs to a later-ending layer. The fractions below are the shorter layer's
pixels that are shadowed by any later endpoint:

| Anchor | Contact shadowed | Material shadowed | Hole shadowed |
| --- | ---: | ---: | ---: |
| Soup + spoon | 100% | 100% | 0% |
| Fried rice + spatula | 93% | 71% | 3% |
| Udon + chopsticks | 26% | 5% | 0% |
| Pasta + fork | 26% | 64% | 5% |

For soup, the active union contains 4,733 pixels during both `[8,15)` and
`[15,18)`: releasing contact at step 15 changes no pixel at all. This audit
provides a concrete mechanism consistent with the staged result remaining close
to the unified-mask failure. It is diagnostic evidence, not by itself a causal
proof of every visual artifact.

## Gate decision

Close staged v1 as a reproducible negative pilot. Do not run seeds 2/3 or the
60-case set. The next candidate should keep the same proxy images and frozen
seed while using non-shadowing/exclusive projection masks; its construction,
hashes, and schedule must be committed before any new stochastic output.
