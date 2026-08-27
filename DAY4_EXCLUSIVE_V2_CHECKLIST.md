# Day 4 follow-up: exclusive projection masks v2

Date opened: 2026-08-27

## Pre-inference contract

- [x] Freeze staged v1 as a negative pilot and preserve its tag/results.
- [x] Keep the common proxy, model, seed, steps, frames, endpoints, and 2D
  protection projection unchanged.
- [x] Isolate one factor: semantic projection-mask ownership.
- [x] Define global ownership priority `hole > contact > material > rigid`.
- [x] Require pairwise-disjoint masks with exactly preserved union coverage.
- [x] Add a deterministic builder, fail-closed manifest checks, and unit tests.
- [x] Commit this contract before the first v2 stochastic output.

## Remote preparation

- [x] Build four exclusive mask packages in a new output root.
- [x] Verify source proxy hashes, pairwise overlap `0`, union preservation, and
  nonempty masks for every anchor.
- [x] Deploy the exact committed worker to a new launcher directory.

## Pilot and gate

- [x] Run the same four anchors at seed `1` with two resident workers.
- [x] Verify `4/4` technical status, exact protection, and one pipeline load per
  worker.
- [x] Review action/photo success against vanilla, unified, and staged v1.
- [x] Expand only if v2 improves action success without losing preservation.

Status: `CLOSED_NEGATIVE_PILOT_DO_NOT_EXPAND`

Outcome: `4/4` technical completion, `0/4` provisional action success,
`0/4` provisional photo success, and `4/4` exact outside-mask preservation.
Exclusive ownership changed every output hash and produced edit-support RGB MAE
`1.147--5.300` relative to staged v1, but did not change any hard-action verdict.
Do not run more seeds or test cases for this candidate.

Frozen candidate: `configs/staged_schedule_v2_exclusive.json`

Evidence: `results/day4_staged_exclusive_v1/` and
`results/DAY4_EXCLUSIVE_REVIEW.md`
