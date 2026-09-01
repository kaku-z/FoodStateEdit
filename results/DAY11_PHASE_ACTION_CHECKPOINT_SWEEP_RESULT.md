# Day 11 phase-action checkpoint-sweep result

Date: 2026-09-01

## Outcome

Both seen synthetic pseudo-motion samples completed the frozen same-seed sweep
on gp39. Each sample used one pipeline load and the five predeclared conditions
(`lora_off`, `step_16`, `step_32`, `step_48`, and `step_64`). Every condition
decoded 21 frames, preserved every pixel outside the declared support exactly,
and produced nonempty hash-recorded outputs.

The capacity gate is negative. Some checkpoints reduce target-support MAE, but
neither sample shows a clear improvement in phase action/contact semantics over
LoRA-off, and neither shows a clear photo-realism improvement. Balanced
multi-family expansion and another blind fork run therefore remain blocked.

## Numerical checkpoint comparison

| sample | LoRA-off | step 16 | step 32 | step 48 | step 64 | numeric best | clear semantic gain |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| udon / chopsticks | 13.6511 | 13.4942 | 12.6833 | 12.1415 | 11.9998 | step 64 | no |
| clear broth / spoon | 21.2355 | 20.9574 | 21.3260 | 25.7896 | 23.8928 | step 16 | no |

These are mean RGB MAEs inside the frozen support against the seen synthetic
target sequence. They are diagnostic distances, not physical-success or
photo-realism metrics.

## Phase action and contact review

The reviewed frames were frozen before inference at 0, 3, 6, 10, 15, and 20.

- Udon: the checkpoints retain nearly the same approach and final
  chopstick/noodle configuration as LoRA-off. The monotonic MAE reduction does
  not yield a visibly clearer pinch, payload acquisition, bowl-connected lift,
  or final hold.
- Broth: LoRA-off already renders the same spoon approach, contained broth, and
  hold sequence. Step 16 is slightly closer numerically, but it does not add or
  clarify a phase or contact event.

The conservative semantic verdict is `0/2` clear checkpoint gains.

## Photo-realism review

- Udon remains a broadly coherent food photograph, but none of the checkpoints
  clearly improves the soft/composited utensil-contact and lifted-noodle
  geometry over LoRA-off.
- Broth step 16 is visually similar to LoRA-off. Steps 48 and 64 are worse:
  they introduce a conspicuous pale or translucent rim around the spoon bowl.

The conservative photo verdict is `0/2` clear checkpoint gains.

## Evidence integrity

The successful output root contains 40 files. Both manifests record 18 hashed
outputs, for 36 output records total. After pullback, every recorded file was
rehashed locally and matched its remote manifest; the mismatch count is zero.
The two run-manifest hashes are:

- udon: `4befd5608e42b1f685aa666469cc070b0cf3614da616a90f86ac41c15d530bfa`;
- broth: `222bc1be2e0cb8d6ffadf6f3d1b25834602f39d7a7a619156a1ea09c93ffc94d`.

The gp39-derived configuration differs from the original frozen configuration
only in immutable checkpoint-copy and new output paths. Its SHA-256 is
`25b85a4d824443d9c51b040f871771f70004f64a5bbb7c4afd041921c8c6d7e0`.
The correct gp40 checkpoint bytes were copied into a new gp39 directory rather
than overwriting the pre-existing mismatched gp39 LoRA directory.

## Claim boundary and decision

This is a seen-sample synthetic overfit-capacity diagnostic. It does not
establish held-out generalization, real-data performance, physical correctness,
or paper-level photo realism. Because both samples failed the clear semantic
gain gate, no balanced expansion is frozen and no blind fork evaluation is
launched. Machine-readable evidence is in
`results/day11_phase_action_checkpoint_sweep_result_v1.json`.
